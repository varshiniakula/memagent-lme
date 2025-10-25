"""
run_pipeline_lme_simple.py
Minimal, Locomo-style runner for LongMemEval:
- Global index only
- FAISS (semantic) first; safe fallback to BM25 if FAISS can't build
- No ensembles/MMR/LLM
- Uses unified evaluator (compare_evidence_ids, evaluate_retriever_results)
"""

import json
import yaml
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from datetime import datetime
from pathlib import Path

from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS

from dataloader_lme import LMEDataLoader
from answer_evaluator import compare_evidence_ids, evaluate_retriever_results


# -------------------------
# Utilities
# -------------------------
def get_timestamp(fmt: str = "datetime") -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S") if fmt == "datetime" else str(int(datetime.now().timestamp()))

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def sanitize_docs(docs: List[Document]) -> List[Document]:
    out = []
    for d in docs or []:
        text = (d.page_content or "").strip()
        if any(ch.isalnum() for ch in text):
            out.append(d)
    return out


# -------------------------
# Single-retriever builder (FAISS with fallback)
# -------------------------
@dataclass
class SimpleConfig:
    top_k: int
    embedding_model: str


def build_faiss_retriever(documents: List[Document], embedding_model: str):
    """Return a retriever (vs.as_retriever) or None."""
    docs = sanitize_docs(documents)
    if not docs:
        print("[WARN] FAISS: no usable documents.")
        return None

    # Prefer the modern import to avoid deprecation warnings
    try:
        from langchain_huggingface import HuggingFaceEmbeddings as HFEmb
    except Exception:
        from langchain_community.embeddings import HuggingFaceEmbeddings as HFEmb

    try:
        emb = HFEmb(model_name=embedding_model)
        vs = FAISS.from_documents(docs, emb)
        return vs.as_retriever(search_kwargs={"k": 5})  # k overridden later per query
    except Exception as e:
        print(f"[WARN] FAISS build failed: {e}")
        return None


def build_bm25_retriever(documents: List[Document]):
    """Return a BM25 retriever or None."""
    docs = sanitize_docs(documents)
    if not docs:
        print("[WARN] BM25: no usable documents.")
        return None
    try:
        return BM25Retriever.from_documents(docs)
    except Exception as e:
        print(f"[WARN] BM25 build failed: {e}")
        return None


# -------------------------
# Pipeline
# -------------------------
def run_pipeline(config: Dict[str, Any]) -> Dict[str, Any]:
    # 1) Load data
    data_cfg = config["data"]
    loader = LMEDataLoader(json_path=data_cfg["json_path"])
    all_docs, qa_list, meta = loader.load_data(
        sample_ids=data_cfg.get("sample_ids"),
        limit=data_cfg.get("limit")
    )

    # 2) Basic validation
    all_docs = sanitize_docs(all_docs)
    if not all_docs:
        raise ValueError("No usable documents found. Check your dataset fields or preprocessing.")
    if not qa_list:
        raise ValueError("No questions found. Ensure your JSON has a non-empty 'qa' list per conversation.")

    # 3) Build simple retriever: FAISS → fallback to BM25
    ret_cfg = config["retrievers"]
    simp = SimpleConfig(
        top_k=int(ret_cfg.get("top_k", 5)),
        embedding_model=str(ret_cfg.get("embedding_model", "sentence-transformers/all-mpnet-base-v2")),
    )

    faiss_ret = build_faiss_retriever(all_docs, simp.embedding_model)
    bm25_ret = None if faiss_ret else build_bm25_retriever(all_docs)

    if not faiss_ret and not bm25_ret:
        raise RuntimeError("Could not build any retriever (FAISS and BM25 both unavailable).")

    # 4) Iterate QAs (global index), compute metrics
    detailed_results = []
    tested = set()
    q_limit = config.get("evaluation", {}).get("question_limit")
    k = simp.top_k

    # group QAs by sample_id (optional; for clarity)
    q_by_sample: Dict[str, List[Dict[str, Any]]] = {}
    for q in qa_list:
        q_by_sample.setdefault(q["sample_id"], []).append(q)

    for sid, qas in q_by_sample.items():
        if q_limit is not None:
            qas = qas[: int(q_limit)]

        for qa in qas:
            qtext = qa["question"]
            gold_evidence = qa.get("evidence", []) or []

            # choose available retriever
            if faiss_ret:
                # set k per query if supported
                if hasattr(faiss_ret, "search_kwargs"):
                    faiss_ret.search_kwargs["k"] = k
                retrieved = faiss_ret.invoke(qtext)
                retriever_name = "faiss"
            else:
                if hasattr(bm25_ret, "k"):
                    bm25_ret.k = k
                retrieved = bm25_ret.invoke(qtext)
                retriever_name = "bm25"

            metrics = compare_evidence_ids(gold_evidence, retrieved)
            detailed_results.append({
                "sample_id": sid,
                "retriever": retriever_name,
                "question": qtext,
                "gold_evidence": gold_evidence,
                "retrieved_dia_ids": [d.metadata.get("dia_id") for d in retrieved],
                "metrics": metrics
            })
            tested.add(retriever_name)

    # 5) Summarize + save
    summary = evaluate_retriever_results(detailed_results)
    out_cfg = config["output"]
    ts = get_timestamp(out_cfg.get("timestamp_format", "datetime"))
    out_dir = Path(out_cfg.get("results_dir", "./results_lme"))
    ensure_dir(out_dir)

    detailed_path = out_dir / f"retriever_all_{ts}.json"
    summary_path = out_dir / f"summary_{ts}.json"
    with open(detailed_path, "w", encoding="utf-8") as f:
        json.dump({"timestamp": ts, "retrievers_tested": sorted(tested), "results": detailed_results},
                  f, indent=2, ensure_ascii=False)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump({"timestamp": ts, "retrievers_tested": sorted(tested), "summary": summary},
                  f, indent=2, ensure_ascii=False)

    # print a tiny table
    print("\nFINAL COMPARISON")
    print("-" * 80)
    print(f"{'Retriever':<18} {'Questions':>10} {'Evidence Found':>18} {'Recall %':>10}")
    print("-" * 80)
    for r, s in sorted(summary.items()):
        qn = s["questions"]
        ef = s["evidence_found"]
        et = s["evidence_total"]
        rc = s["recall_pct"]
        print(f"{r:<18} {qn:>10} {f'{ef}/{et}':>18} {rc:>9.1f}%")
    print("-" * 80)
    print(f"Detailed: {detailed_path}")
    print(f"Summary : {summary_path}")

    return {
        "detailed_path": str(detailed_path),
        "summary_path": str(summary_path),
        "retrievers_tested": sorted(tested),
        "summary": summary
    }


def main():
    config_path = Path("config_lme.yaml")
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    run_pipeline(config)


if __name__ == "__main__":
    main()
