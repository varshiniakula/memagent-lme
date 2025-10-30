# run_pipeline_lme.py
# Evaluate multiple retrievers per question over your Chroma per-question collections.

import os
import json
import orjson
import argparse
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from pathlib import Path
import warnings

import chromadb
from sentence_transformers import SentenceTransformer
from langchain_core.documents import Document

from all_retrievers import RetrieverRegistry


# ---------------------- utils ----------------------
def ts() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def read_json_any(path: Path) -> Any:
    try:
        return orjson.loads(path.read_bytes())
    except Exception:
        return json.loads(path.read_text())

def try_load_manifest(persist_dir: Path) -> Optional[Dict[str, Any]]:
    mf = persist_dir / "subset_manifest.json"
    return read_json_any(mf) if mf.exists() else None

def first_nonempty(*vals, default: str = "") -> str:
    for v in vals:
        if isinstance(v, str) and v.strip():
            return v
    return default


# ---------------------- config ----------------------
DEFAULT_NAMES = [
    "bm25","tfidf","faiss","knn","svm","time_weighted","nanopq","colbert",
    "multiquery","contextual_compression","self_query"
]

@dataclass
class RunnerConfig:
    persist_dir: str = "./chroma_longmemeval_s_subset"
    dataset_json: Optional[str] = "./longmemeval_s_cleaned.json"  # optional fallback for gold
    prefix: str = "vectordb_collection_"
    top_k: int = 5
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    retrievers: List[str] = None
    results_dir: str = "./results_lme"

    # LLM
    llm_provider: Optional[str] = None  # "google"|"openai"|"anthropic"|...
    llm_model: Optional[str] = None
    llm_temperature: float = 0.0

# ---------------------- LLM factory ----------------------
def build_llm(provider: Optional[str], model: Optional[str], temperature: float = 0.0):
    if not provider or not model:
        return None
    provider = provider.lower()
    try:
        if provider == "google":
            # pip install langchain-google-genai google-generativeai
            from langchain_google_genai import ChatGoogleGenerativeAI
            return ChatGoogleGenerativeAI(model=model, temperature=temperature)
        if provider == "openai":
            # pip install langchain-openai
            from langchain_openai import ChatOpenAI
            return ChatOpenAI(model=model, temperature=temperature)
        if provider == "anthropic":
            # pip install langchain-anthropic
            from langchain_anthropic import ChatAnthropic
            return ChatAnthropic(model=model, temperature=temperature)
    except Exception as e:
        warnings.warn(f"[LLM] Failed to initialize {provider}:{model}: {e}")
        return None
    warnings.warn(f"[LLM] Unsupported provider: {provider}")
    return None

# ---------------------- Chroma helpers ----------------------
def list_target_questions(client: chromadb.ClientAPI, cfg: RunnerConfig) -> List[Tuple[str,str]]:
    """
    Returns list of (qid, collection_name) for the 10-question subset.
    Prefer subset_manifest.json; else first 10 matching prefix.
    """
    persist_path = Path(cfg.persist_dir)
    manifest = try_load_manifest(persist_path)
    out: List[Tuple[str,str]] = []

    if manifest and "collections" in manifest:
        for item in manifest["collections"]:
            coll = item["collection"]
            qid = item.get("question_id") or coll.replace(cfg.prefix, "")
            out.append((qid, coll))
        return out

    cols = client.list_collections()
    for c in cols:
        name = getattr(c, "name", str(c))
        if name.startswith(cfg.prefix):
            qid = name.replace(cfg.prefix, "")
            out.append((qid, name))
    return out[:10]

def docs_from_collection(coll) -> List[Document]:
    """Convert a per-question Chroma collection into LangChain Documents."""
    got = coll.get(include=["metadatas", "documents"])  # 'ids' are always returned
    ids = got.get("ids", [])
    metas = got.get("metadatas", [])
    docs = got.get("documents", [])
    out: List[Document] = []
    for sid, meta, txt in zip(ids, metas, docs):
        # ensure required metadata present
        session_id = meta.get("session_id") or meta.get("evidence") or sid
        date = meta.get("date", "")
        m = dict(meta)  # copy
        m.setdefault("session_id", session_id)
        m.setdefault("evidence", session_id)
        m.setdefault("date", date)
        out.append(Document(page_content=txt or "", metadata=m))
    return out

def load_dataset_index(dataset_json: Optional[str]) -> Dict[str, Dict[str, Any]]:
    """Map question_id -> raw JSON row (optional)."""
    idx: Dict[str, Dict[str, Any]] = {}
    if not dataset_json:
        return idx
    p = Path(dataset_json)
    if not p.exists():
        return idx
    rows = read_json_any(p)
    for r in rows:
        qid = r.get("question_id")
        if qid:
            idx[qid] = r
    return idx

def get_question_text(coll, dataset_idx: Dict[str, Dict[str, Any]], qid: str) -> str:
    # Try first doc meta
    try:
        got = coll.get(include=["metadatas"], limit=1)
        if got and got.get("metadatas"):
            meta = got["metadatas"][0]
            q = meta.get("question") or meta.get("q") or ""
            if q:
                return q
    except Exception:
        pass
    row = dataset_idx.get(qid)
    if row:
        return first_nonempty(row.get("question",""), default=f"[QID {qid}]")
    return f"[QID {qid}]"

def gold_sessions_for_question(coll, dataset_idx: Dict[str, Dict[str, Any]], qid: str) -> List[str]:
    """
    In order:
      1) collection.metadata['gold_session_ids_json'] (JSON string) if present
      2) any doc with meta['is_answer_session'] == True
      3) dataset row['answer_session_ids'] ∩ {collection session ids}
    """
    got_all = coll.get(include=["metadatas"])  # ids always present
    coll_ids = list(got_all.get("ids", [])) if got_all else []

    # 1) collection-level JSON string
    try:
        m = getattr(coll, "metadata", {}) or {}
        j = m.get("gold_session_ids_json") or m.get("answer_session_ids_json") or ""
        if isinstance(j, str) and j.strip():
            parsed = json.loads(j)
            if isinstance(parsed, list):
                return [s for s in map(str, parsed) if s in set(coll_ids)]
    except Exception:
        pass

    # 2) per-doc flags
    metas = got_all.get("metadatas", []) if got_all else []
    gold_meta = [ (metas[i].get("session_id") or metas[i].get("evidence") or coll_ids[i])
                  for i in range(min(len(metas), len(coll_ids)))
                  if bool(metas[i].get("is_answer_session", False)) ]
    if gold_meta:
        return gold_meta

    # 3) dataset intersection
    row = dataset_idx.get(qid, {})
    ds_gold = row.get("answer_session_ids") or []
    if ds_gold:
        ds_gold = [str(x) for x in ds_gold]
        return [g for g in ds_gold if g in set(coll_ids)]
    return []


# ---------------------- evaluation helpers ----------------------
def topk_to_session_ids(docs: List[Document]) -> List[str]:
    out = []
    for d in docs:
        sid = d.metadata.get("session_id") or d.metadata.get("evidence")
        out.append(str(sid))
    return out

def eval_hit_and_recall(pred_ids: List[str], gold_ids: List[str]) -> Dict[str, Any]:
    gold = set(map(str, gold_ids or []))
    preds = list(map(str, pred_ids or []))
    hits = [i for i, sid in enumerate(preds, start=1) if sid in gold]
    n_gold = len(gold)
    n_hits = len(hits)
    return {
        "hit@k": 1 if n_hits > 0 else 0,
        "hit_ranks": hits,
        "recall@k": (n_hits / n_gold) if n_gold > 0 else None,
        "gold_count": n_gold,
        "pred_count": len(preds),
    }

def greedy_set_cover(success: Dict[str, List[str]], universe: List[str]) -> Dict[str, Any]:
    """Greedy cover of questions using retrievers that succeed (hit@k==1)."""
    remaining = set(universe)
    order: List[str] = []
    cover_steps: List[Dict[str, Any]] = []
    # convert to sets
    succ_sets = {r: set(v) for r, v in success.items()}
    while remaining:
        # select retriever with max new coverage
        best_r = None
        best_gain = -1
        for r, s in succ_sets.items():
            gain = len(remaining & s)
            if gain > best_gain:
                best_gain = gain
                best_r = r
        if best_r is None or best_gain <= 0:
            break
        order.append(best_r)
        newly = sorted(list(remaining & succ_sets[best_r]))
        remaining -= succ_sets[best_r]
        cover_steps.append({
            "pick": best_r,
            "newly_covered": newly,
            "covered_so_far": sorted(list(set().union(*[succ_sets[x] for x in order]))),
            "remaining": sorted(list(remaining)),
        })
    covered = sorted(list(set().union(*[succ_sets[x] for x in order]))) if order else []
    return {
        "order": order,
        "covered": covered,
        "coverage_fraction": (len(covered) / len(universe)) if universe else 0.0,
        "steps": cover_steps,
        "uncovered": sorted(list(remaining)),
    }


# ---------------------- main ----------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--persist_dir", default="./chroma_longmemeval_s_subset")
    ap.add_argument("--dataset_json", default="./longmemeval_s_cleaned.json")
    ap.add_argument("--prefix", default="vectordb_collection_")
    ap.add_argument("--top_k", type=int, default=5)
    ap.add_argument("--embedding_model", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--retrievers", default=",".join(DEFAULT_NAMES),
                    help="comma list: bm25,tfidf,faiss,knn,svm,time_weighted,nanopq,colbert,multiquery,contextual_compression,self_query")
    ap.add_argument("--results_dir", default="./results_lme")
    # LLM (optional)
    ap.add_argument("--llm_provider", default=None)
    ap.add_argument("--llm_model", default=None)
    ap.add_argument("--llm_temperature", type=float, default=0.0)
    # NanoPQ / ColBERT toggles
    ap.add_argument("--nanopq", action="store_true", help="enable IVF+PQ (requires faiss)")
    ap.add_argument("--colbert", action="store_true", help="enable ColBERT (requires ragatouille)")
    args = ap.parse_args()

    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")

    cfg = RunnerConfig(
        persist_dir=args.persist_dir,
        dataset_json=args.dataset_json,
        prefix=args.prefix,
        top_k=args.top_k,
        embedding_model=args.embedding_model,
        retrievers=[r.strip().lower() for r in args.retrievers.split(",") if r.strip()],
        results_dir=args.results_dir,
        llm_provider=args.llm_provider,
        llm_model=args.llm_model,
        llm_temperature=args.llm_temperature,
    )

    ensure_dir(Path(cfg.results_dir))
    client = chromadb.PersistentClient(path=cfg.persist_dir)
    qlist = list_target_questions(client, cfg)
    if not qlist:
        raise RuntimeError("No matching collections found; check --persist_dir and --prefix")

    dataset_idx = load_dataset_index(cfg.dataset_json)

    # LLM (optional, only for LLM retrievers)
    llm = build_llm(cfg.llm_provider, cfg.llm_model, cfg.llm_temperature)

    retriever_cfg = {
        "retrievers": {
            "top_k": cfg.top_k,
            "embedding_model": cfg.embedding_model,
            # optional knobs
            "nanopq": {"enabled": bool(args.nanopq), "ivf_nlist": 1024, "pq_m": 64, "opq_m": 16},
            "colbert": {"enabled": bool(args.colbert), "index_dir": "indexes/colbert", "overwrite": False}
        }
    }

    per_question_rows: List[Dict[str, Any]] = []
    # retriever -> {hits, evaluable, recall_sum, correct_qids, wrong_qids}
    scoreboard: Dict[str, Dict[str, Any]] = {}

    # for orchestrator analysis
    hit_by_retriever: Dict[str, List[str]] = {name: [] for name in cfg.retrievers}
    universe_qids: List[str] = []

    for qid, coll_name in qlist:
        coll = client.get_collection(coll_name)

        # build docs
        docs = docs_from_collection(coll)
        if not docs:
            warnings.warn(f"[{qid}] no docs in collection {coll_name}, skipping")
            continue

        # question text & gold
        qtext = get_question_text(coll, dataset_idx, qid)
        gold = gold_sessions_for_question(coll, dataset_idx, qid)
        evaluatable = len(gold) > 0
        universe_qids.append(qid)

        # build registry for this question
        registry = RetrieverRegistry(docs, retriever_cfg, llm=llm)

        retriever_rows = []
        for name in cfg.retrievers:
            try:
                out_docs = registry.invoke(name, qtext, k=cfg.top_k) or []
            except Exception as e:
                warnings.warn(f"[{name}] failed on qid={qid}: {e}")
                out_docs = []

            pred_ids = topk_to_session_ids(out_docs)
            metrics = eval_hit_and_recall(pred_ids, gold) if evaluatable else {
                "hit@k": None, "hit_ranks": [], "recall@k": None, "gold_count": 0, "pred_count": len(pred_ids)
            }

            retriever_rows.append({
                "retriever": name,
                "topk_session_ids": pred_ids,
                "metrics": metrics
            })

            sb = scoreboard.setdefault(name, {"hits": 0, "evaluable": 0, "recall_sum": 0.0, "correct_qids": [], "wrong_qids": []})
            if evaluatable:
                sb["evaluable"] += 1
                if metrics["hit@k"] == 1:
                    sb["hits"] += 1
                    sb["correct_qids"].append(qid)
                    hit_by_retriever[name].append(qid)
                else:
                    sb["wrong_qids"].append(qid)
                if metrics["recall@k"] is not None:
                    sb["recall_sum"] += float(metrics["recall@k"])

        per_question_rows.append({
            "question_id": qid,
            "collection": coll_name,
            "question_text": qtext,
            "gold_session_ids": gold,
            "retrievers": retriever_rows,
            "evaluatable": evaluatable
        })

    # Aggregate per retriever
    leaderboard = []
    for name, s in scoreboard.items():
        ev = s["evaluable"]
        acc = (s["hits"]/ev) if ev>0 else None
        mean_rec = (s["recall_sum"]/ev) if ev>0 else None
        leaderboard.append({
            "retriever": name,
            "evaluable": ev,
            "hits": s["hits"],
            "accuracy_hit@k": acc,
            "mean_recall@k": mean_rec,
            "correct_qids": sorted(s["correct_qids"]),
            "wrong_qids": sorted(s["wrong_qids"]),
        })
    leaderboard.sort(key=lambda r: (r["accuracy_hit@k"] or 0.0, r["mean_recall@k"] or 0.0), reverse=True)

    # Orchestrator: union coverage + greedy set-cover
    cover = greedy_set_cover(hit_by_retriever, universe_qids)

    ts_now = ts()
    out_dir = Path(cfg.results_dir); ensure_dir(out_dir)

    with open(out_dir / f"per_question_{ts_now}.json", "w", encoding="utf-8") as f:
        json.dump(per_question_rows, f, indent=2, ensure_ascii=False)

    with open(out_dir / f"leaderboard_{ts_now}.json", "w", encoding="utf-8") as f:
        json.dump(leaderboard, f, indent=2, ensure_ascii=False)

    with open(out_dir / f"orchestrator_{ts_now}.json", "w", encoding="utf-8") as f:
        json.dump({
            "universe_qids": universe_qids,
            "hit_by_retriever": hit_by_retriever,
            "union_coverage_fraction": cover["coverage_fraction"],
            "greedy_order": cover["order"],
            "greedy_steps": cover["steps"],
            "uncovered": cover["uncovered"],
        }, f, indent=2, ensure_ascii=False)

    print("\nDONE.")
    print(f"  Per-question : {out_dir / f'per_question_{ts_now}.json'}")
    print(f"  Leaderboard  : {out_dir / f'leaderboard_{ts_now}.json'}")
    print(f"  Orchestrator : {out_dir / f'orchestrator_{ts_now}.json'}")
    print("\nKey numbers:")
    if leaderboard:
        for row in leaderboard:
            print(f"  {row['retriever']:>23s}  acc@{cfg.top_k}={row['accuracy_hit@k']}  meanR@{cfg.top_k}={row['mean_recall@k']}")
    print(f"\nUnion (any retriever succeeds) covers {cover['coverage_fraction']*100:.1f}% of questions.")
    if cover["order"]:
        print("Greedy set-cover order:", " → ".join(cover["order"]))


if __name__ == "__main__":
    main()
