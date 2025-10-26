"""
run_pipeline_lme.py
LongMemEval runner (question-wise) with multi-retriever support via RetrieverRegistry.

What this script produces (timestamps in filenames):
  1) results_lme/retriever_all_<ts>.json      # per-question per-retriever rows
  2) results_lme/summary_overall_<ts>.json    # overall aggregate per retriever
  3) results_lme/by_question_<ts>.json        # QUESTION-WISE report you asked for

Notes
- No category logic at all.
- NumPy-accelerated rank fusion & metric calc.
- Optional prefiltering that ONLY uses official dataset signals:
    data.prefilter_answer_only: true|false
      -> keep only sessions that contain has_answer=True OR are listed in answer_session_ids
    data.max_sessions_per_q: int
      -> cap sessions per question (helps speed/memory)
- CLI overrides so you don’t need to edit YAML while iterating.
"""

from __future__ import annotations

import json
import yaml
import argparse
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from pathlib import Path
from collections import defaultdict

import numpy as np
from langchain_core.documents import Document

from dataloader_lme import LMEDataLoader
from all_retrievers import RetrieverRegistry  # bm25, tfidf, faiss, knn, svm, colbert, time_weighted, nanopq
from answer_evaluator import evaluate_retriever_results  # (overall aggregation only)

# -------------------------------------------------------------------------------------
# Utilities
# -------------------------------------------------------------------------------------
def get_timestamp(fmt: str = "datetime") -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S") if fmt == "datetime" else str(int(datetime.now().timestamp()))

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def sanitize_docs(docs: List[Document]) -> List[Document]:
    """Keep any non-empty text (even punctuation-only)."""
    out = []
    for d in docs or []:
        text = (d.page_content or "").strip()
        if text:
            out.append(d)
    return out

# -------------------------------------------------------------------------------------
# NumPy helpers (speedups)
# -------------------------------------------------------------------------------------
def _docs_to_ids_np(docs: List[Document]) -> np.ndarray:
    if not docs:
        return np.empty((0,), dtype=object)
    return np.fromiter((d.metadata.get("dia_id") for d in docs), dtype=object, count=len(docs))

def rank_fusion_numpy(vec_ids: np.ndarray,
                      bm_ids:  np.ndarray,
                      k: int,
                      vec_w: float,
                      bm_w:  float) -> np.ndarray:
    """
    Weighted reciprocal-rank fusion (vectorized).
    score = vec_w * 1/(1+rank_vec) + bm_w * 1/(1+rank_bm)
    """
    if vec_ids.size == 0 and bm_ids.size == 0:
        return np.empty((0,), dtype=object)

    cand = np.unique(np.concatenate([vec_ids, bm_ids]))
    big = 10_000

    vec_pos = {id_: i for i, id_ in enumerate(vec_ids.tolist())}
    bm_pos  = {id_: i for i, id_ in enumerate(bm_ids.tolist())}

    vec_ranks = np.fromiter((vec_pos.get(x, big) for x in cand), dtype=np.int32, count=cand.size)
    bm_ranks  = np.fromiter((bm_pos.get(x, big)  for x in cand), dtype=np.int32, count=cand.size)

    scores = vec_w / (1.0 + vec_ranks) + bm_w / (1.0 + bm_ranks)

    if cand.size <= k:
        order = np.argsort(-scores)
    else:
        topk_idx = np.argpartition(-scores, k-1)[:k]
        order = topk_idx[np.argsort(-scores[topk_idx])]

    return cand[order]

def metrics_from_ids_np(gold_ids: List[str], retrieved_ids: np.ndarray) -> Dict[str, Any]:
    if not gold_ids:
        return {"evidence_found": 0, "evidence_total": 0, "recall_pct": 0.0}
    gold = np.array(gold_ids, dtype=object)
    if retrieved_ids.size == 0:
        tot = int(gold.size)
        return {"evidence_found": 0, "evidence_total": tot, "recall_pct": 0.0}
    found = int(np.isin(gold, retrieved_ids).sum())
    tot = int(gold.size)
    rec = 100.0 * (found / tot) if tot else 0.0
    return {"evidence_found": found, "evidence_total": tot, "recall_pct": rec}

# -------------------------------------------------------------------------------------
# Config
# -------------------------------------------------------------------------------------
@dataclass
class RunnerConfig:
    top_k: int
    ensemble_weights: Tuple[float, float]

# -------------------------------------------------------------------------------------
# Prefilter (uses ONLY official fields the loader exposes)
# -------------------------------------------------------------------------------------
def apply_prefilter(
        all_docs: List[Document],
        qa_list: List[Dict[str, Any]],
        prefilter_answer_only: bool,
        max_sessions_per_q: Optional[int],
) -> List[Document]:
    """Shrink document set before indexing, using official LongMemEval signals only."""
    if not (prefilter_answer_only or max_sessions_per_q):
        return all_docs

    docs_by_sample_session = defaultdict(lambda: defaultdict(list))
    for d in all_docs:
        sid = d.metadata.get("sample_id")
        s_rank = d.metadata.get("session_rank")
        docs_by_sample_session[sid][s_rank].append(d)

    allowed_sessions: Dict[str, set] = {}
    for q in qa_list:
        sample_id = q["sample_id"]
        allow = set()

        if prefilter_answer_only:
            # 1) sessions with has_answer turns
            for dia in q.get("has_answer_turn_ids", []):
                try:
                    s = int(dia.split(":")[0].lstrip("S"))
                    allow.add(s)
                except Exception:
                    pass
            # 2) answer_session_ranks
            for s in q.get("answer_session_ranks", []):
                try:
                    allow.add(int(s))
                except Exception:
                    pass

        if max_sessions_per_q:
            cap = int(max_sessions_per_q)
            if not allow:
                # keep earliest N sessions if we have no labels
                all_s = sorted(docs_by_sample_session.get(sample_id, {}).keys())[:cap]
                allow.update(all_s)
            else:
                allow = set(sorted(list(allow))[:cap])

        if allow:
            allowed_sessions[sample_id] = allow

    if not allowed_sessions:
        return all_docs

    filtered_docs = []
    for d in all_docs:
        sid = d.metadata.get("sample_id")
        s_rank = d.metadata.get("session_rank")
        if sid in allowed_sessions:
            if s_rank in allowed_sessions[sid]:
                filtered_docs.append(d)
        else:
            # sample has no allow-list => keep (conservative)
            filtered_docs.append(d)

    print(f"[DEBUG] prefilter applied: kept {len(filtered_docs)}/{len(all_docs)} docs")
    return filtered_docs

# -------------------------------------------------------------------------------------
# Question-wise report
# -------------------------------------------------------------------------------------
def build_by_question_report(
        detailed_rows: List[Dict[str, Any]],
        retrievers_tested: List[str]
) -> Dict[str, Any]:
    """
    Build question-wise view:
      - success_by_retriever: {name: bool}
      - winner_list: [retrievers that hit any gold id]
      - oracle_success: True if any retriever hits
      - union_retrieved_dia_ids: union of top-k IDs across retrievers
    """
    # group rows by (sample_id, question)
    by_q: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    for r in detailed_rows:
        key = (r["sample_id"], r["question"])
        by_q.setdefault(key, []).append(r)

    questions_out = []
    for (sample_id, question), rows in by_q.items():
        gold = rows[0].get("gold_evidence", []) or []

        success_by_retriever = {name: False for name in retrievers_tested}
        union_ids = set()

        for r in rows:
            name = r["retriever"]
            ret_ids = r.get("retrieved_dia_ids", []) or []
            union_ids.update(ret_ids)
            if gold and any(g in ret_ids for g in gold):
                success_by_retriever[name] = True

        winner_list = [n for n, ok in success_by_retriever.items() if ok]
        oracle_success = any(winner_list)

        questions_out.append({
            "sample_id": sample_id,
            "question": question,
            "gold_evidence": gold,
            "success_by_retriever": success_by_retriever,
            "winner_list": winner_list,
            "oracle_success": oracle_success,
            "union_retrieved_dia_ids": sorted(union_ids),
        })

    return {
        "timestamp": get_timestamp("datetime"),
        "questions": questions_out
    }

# -------------------------------------------------------------------------------------
# Printing
# -------------------------------------------------------------------------------------
def print_table(overall: Dict[str, Dict[str, Any]]):
    print("-" * 80)
    print(f"{'Retriever':<18} {'Questions':>10} {'Evidence Found':>18} {'Recall %':>10}")
    print("-" * 80)
    for r, s in sorted(overall.items()):
        qn = s["questions"]; ef = s["evidence_found"]; et = s["evidence_total"]; rc = s["recall_pct"]
        print(f"{r:<18} {qn:>10} {f'{ef}/{et}':>18} {rc:>9.1f}%")
    print("-" * 80)

# -------------------------------------------------------------------------------------
# Main pipeline
# -------------------------------------------------------------------------------------
@dataclass
class RunnerConfig:
    top_k: int
    ensemble_weights: Tuple[float, float]

def run_pipeline(config: Dict[str, Any]) -> Dict[str, Any]:
    # safety defaults
    config.setdefault("data", {})
    config.setdefault("evaluation", {})
    config["data"]["limit"] = config["data"].get("limit", None)
    config["evaluation"]["question_limit"] = config["evaluation"].get("question_limit", None)

    # 1) Load data
    data_cfg = config.get("data") or {}
    json_path = data_cfg.get("json_path") or "data/longmemeval_s_cleaned.json"
    print("[DEBUG] using data file:", json_path)

    if not Path(json_path).exists():
        raise FileNotFoundError(
            f"Data file not found at '{json_path}'. "
            "Set data.json_path in config_lme.yaml or provide --json_path"
        )

    loader = LMEDataLoader(json_path=json_path)
    all_docs, qa_list, meta = loader.load_data(
        sample_ids=data_cfg.get("sample_ids"),
        limit=data_cfg.get("limit"),
    )
    if not all_docs:
        raise ValueError("No usable documents found. Check dataset fields or preprocessing.")
    if not qa_list:
        raise ValueError("No questions found. Ensure JSON has non-empty 'qa' per instance.")

    print(f"[DEBUG] loaded docs={len(all_docs)}, qas={len(qa_list)} (before filters)")

    # Optional cap on number of QA items
    q_limit = config.get("evaluation", {}).get("question_limit")
    if q_limit:
        qa_list = qa_list[: int(q_limit)]
        print(f"[DEBUG] applied question_limit={q_limit}: qas={len(qa_list)}")

    # 2) Prefilter docs (optional; speeds up indexing)
    all_docs = sanitize_docs(all_docs)
    all_docs = apply_prefilter(
        all_docs,
        qa_list,
        prefilter_answer_only=bool(data_cfg.get("prefilter_answer_only", False)),
        max_sessions_per_q=data_cfg.get("max_sessions_per_q", None),
    )

    # 3) Runner config
    ret_cfg = config.get("retrievers", {})
    to_run: List[str] = list(ret_cfg.get("to_run", ["bm25", "tfidf", "faiss"]))
    rcfg = RunnerConfig(
        top_k=int(ret_cfg.get("top_k", 5)),
        ensemble_weights=tuple(ret_cfg.get("ensemble_weights", [0.75, 0.25])),
    )

    # 4) Build unified registry (handles availability gracefully)
    print("[DEBUG] building retrievers…")
    registry = RetrieverRegistry(all_docs, config)
    available = set(registry.list_available())
    print("[DEBUG] available retrievers:", sorted(available))

    # 5) Evaluate
    detailed: List[Dict[str, Any]] = []
    tested = set()

    def run_ensemble(query: str, k: int) -> List[Document]:
        vec_name = "faiss" if "faiss" in available else ("knn" if "knn" in available else None)
        if not vec_name or "bm25" not in available:
            return []

        vec_docs = registry.invoke(vec_name, query, k) or []
        bm_docs  = registry.invoke("bm25",  query, k) or []
        if not vec_docs and not bm_docs:
            return []

        vec_ids = _docs_to_ids_np(vec_docs)
        bm_ids  = _docs_to_ids_np(bm_docs)
        fused_ids = rank_fusion_numpy(vec_ids, bm_ids, k, rcfg.ensemble_weights[0], rcfg.ensemble_weights[1])

        # id -> doc cache (prefer vector doc)
        cache = {d.metadata.get("dia_id"): d for d in vec_docs}
        for d in bm_docs:
            cache.setdefault(d.metadata.get("dia_id"), d)
        return [cache[i] for i in fused_ids.tolist() if i in cache]

    # group QAs by (sample) for stable iteration
    q_by_sample: Dict[str, List[Dict[str, Any]]] = {}
    for q in qa_list:
        q_by_sample.setdefault(q["sample_id"], []).append(q)

    for sid, qas in q_by_sample.items():
        for qa in qas:
            qtext = qa["question"]
            gold = qa.get("evidence", []) or []

            for name in to_run:
                name_l = name.lower()

                if name_l == "ensemble":
                    docs = run_ensemble(qtext, rcfg.top_k)
                else:
                    if name_l not in available:
                        continue
                    docs = registry.invoke(name_l, qtext, k=rcfg.top_k)

                docs = docs or []
                metrics = metrics_from_ids_np(gold, _docs_to_ids_np(docs))
                detailed.append({
                    "sample_id": sid,
                    "retriever": name_l,
                    "question": qtext,
                    "gold_evidence": gold,
                    "retrieved_dia_ids": [d.metadata.get("dia_id") for d in docs],
                    "metrics": metrics
                })
                tested.add(name_l)

    # 6) Summaries
    overall = evaluate_retriever_results(detailed)  # uses your existing evaluator to aggregate

    # 7) Save (including QUESTION-WISE)
    out_cfg = config.get("output", {})
    ts = get_timestamp(out_cfg.get("timestamp_format", "datetime"))
    out_dir = Path(out_cfg.get("results_dir", "./results_lme"))
    ensure_dir(out_dir)

    detailed_path = out_dir / f"retriever_all_{ts}.json"
    overall_path  = out_dir / f"summary_overall_{ts}.json"
    byq_path      = out_dir / f"by_question_{ts}.json"

    with open(detailed_path, "w", encoding="utf-8") as f:
        json.dump({"timestamp": ts, "retrievers_tested": sorted(tested), "results": detailed},
                  f, indent=2, ensure_ascii=False)

    with open(overall_path, "w", encoding="utf-8") as f:
        json.dump({"timestamp": ts, "retrievers_tested": sorted(tested), "summary": overall},
                  f, indent=2, ensure_ascii=False)

    by_q = build_by_question_report(detailed, sorted(tested))
    with open(byq_path, "w", encoding="utf-8") as f:
        json.dump(by_q, f, indent=2, ensure_ascii=False)

    # 8) Print table
    print("\nOVERALL")
    print_table(overall)
    print(f"\nDetailed : {detailed_path}")
    print(f"Overall  : {overall_path}")
    print(f"ByQuestion: {byq_path}")

    return {
        "detailed_path": str(detailed_path),
        "overall_path": str(overall_path),
        "by_question_path": str(byq_path),
        "retrievers_tested": sorted(tested),
        "overall": overall,
    }

# -------------------------------------------------------------------------------------
# CLI overrides (so you don’t have to touch YAML while iterating)
# -------------------------------------------------------------------------------------
def _apply_overrides(cfg: Dict[str, Any]) -> Dict[str, Any]:
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--json_path")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--question_limit", type=int)
    ap.add_argument("--top_k", type=int)
    ap.add_argument("--retrievers", help="comma-separated, e.g. bm25,tfidf,faiss")
    # Optional prefilters
    ap.add_argument("--prefilter_answer_only", action="store_true")
    ap.add_argument("--max_sessions_per_q", type=int)
    args, _ = ap.parse_known_args()

    if args.json_path:
        cfg.setdefault("data", {})["json_path"] = args.json_path
    if args.limit is not None:
        cfg.setdefault("data", {})["limit"] = args.limit
    if args.question_limit is not None:
        cfg.setdefault("evaluation", {})["question_limit"] = args.question_limit
    if args.top_k is not None:
        cfg.setdefault("retrievers", {})["top_k"] = args.top_k
    if args.retrievers:
        names = [x.strip() for x in args.retrievers.split(",") if x.strip()]
        cfg.setdefault("retrievers", {})["to_run"] = names
    if args.prefilter_answer_only:
        cfg.setdefault("data", {})["prefilter_answer_only"] = True
    if args.max_sessions_per_q is not None:
        cfg.setdefault("data", {})["max_sessions_per_q"] = int(args.max_sessions_per_q)
    return cfg

def main():
    config_path = Path("config_lme.yaml")
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    config = _apply_overrides(config)
    run_pipeline(config)

if __name__ == "__main__":
    main()
