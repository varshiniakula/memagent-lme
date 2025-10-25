"""
run_pipeline_lme.py (Category-aware, multi-retriever)
- Global index only (robust)
- Retrievers: bm25, tfidf, faiss, ensemble, svm(optional), time_weighted(proxy)
- Category-wise evaluation (defaults to categories [1, 2])
"""

import json
import yaml
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from pathlib import Path

from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever, TFIDFRetriever
from langchain_community.vectorstores import FAISS

from dataloader_lme import LMEDataLoader
from answer_evaluator import compare_evidence_ids, evaluate_retriever_results

# -------------------------

CATEGORIES_TO_EVAL = [1, 2, 3, 4, 5, None]


# -------------------------
# Utilities
# -------------------------
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

# -------------------------
# Retrieval builders
# -------------------------
@dataclass
class SimpleConfig:
    top_k: int
    embedding_model: str
    ensemble_weights: Tuple[float, float] = (0.75, 0.25)  # [vector, keyword]

def build_faiss_retriever(documents: List[Document], embedding_model: str):
    docs = sanitize_docs(documents)
    if not docs:
        print("[WARN] FAISS: no usable documents.")
        return None
    try:
        from langchain_huggingface import HuggingFaceEmbeddings as HFEmb
    except Exception:
        from langchain_community.embeddings import HuggingFaceEmbeddings as HFEmb
    try:
        emb = HFEmb(model_name=embedding_model)
        vs = FAISS.from_documents(docs, emb)
        return vs
    except Exception as e:
        print(f"[WARN] FAISS build failed: {e}")
        return None

def build_bm25(documents: List[Document]):
    docs = sanitize_docs(documents)
    if not docs:
        print("[WARN] BM25: no usable documents.")
        return None
    try:
        return BM25Retriever.from_documents(docs)
    except Exception as e:
        print(f"[WARN] BM25 build failed: {e}")
        return None

def build_tfidf(documents: List[Document]):
    docs = sanitize_docs(documents)
    if not docs:
        print("[WARN] TF-IDF: no usable documents.")
        return None
    try:
        return TFIDFRetriever.from_documents(docs)
    except Exception as e:
        print(f"[WARN] TF-IDF build failed (likely empty vocab): {e}")
        return None

def try_build_svm(documents: List[Document], embedding_model: str, top_k: int):
    try:
        from langchain_community.retrievers import SVMRetriever
    except Exception:
        print("[INFO] SVM retriever not available. Skipping.")
        return None
    docs = sanitize_docs(documents)
    if not docs:
        print("[WARN] SVM: no usable documents.")
        return None
    try:
        try:
            from langchain_huggingface import HuggingFaceEmbeddings as HFEmb
        except Exception:
            from langchain_community.embeddings import HuggingFaceEmbeddings as HFEmb
        emb = HFEmb(model_name=embedding_model)
        ret = SVMRetriever.from_documents(docs, emb, k=top_k)
        return ret
    except Exception as e:
        print(f"[WARN] SVM build failed: {e}")
        return None

# -------------------------
# Invoke helpers
# -------------------------
def inv_keyword(retriever, query: str, k: int) -> List[Document]:
    if hasattr(retriever, "k"):
        retriever.k = k
    return retriever.invoke(query)

def inv_faiss(vs: FAISS, query: str, k: int) -> List[Document]:
    return vs.similarity_search(query, k=k)

def inv_faiss_mmr(vs: FAISS, query: str, k: int, fetch_k: int, lambda_mult: float) -> List[Document]:
    return vs.max_marginal_relevance_search(query, k=k, fetch_k=fetch_k, lambda_mult=lambda_mult)

def reorder_long_context(docs: List[Document]) -> List[Document]:
    return sorted(docs, key=lambda d: len(d.page_content or ""))

# -------------------------
# Main pipeline
# -------------------------
def run_pipeline(config: Dict[str, Any]) -> Dict[str, Any]:
    # 1) Load
    data_cfg = config["data"]
    loader = LMEDataLoader(json_path=data_cfg["json_path"])
    all_docs, qa_list, meta = loader.load_data(
        sample_ids=data_cfg.get("sample_ids"),
        limit=data_cfg.get("limit")
    )
    all_docs = sanitize_docs(all_docs)
    if not all_docs:
        raise ValueError("No usable documents found. Check dataset fields or preprocessing.")
    if not qa_list:
        raise ValueError("No questions found. Ensure JSON has non-empty 'qa' per conversation.")

    # Filter categories (your Locomo focus: 1 & 2)
    qa_list = [q for q in qa_list if (q.get("category") in CATEGORIES_TO_EVAL)]
    if not qa_list:
        raise ValueError(f"No QA found for categories {CATEGORIES_TO_EVAL}. Add category labels in your JSON.")

    # 2) Config
    ret_cfg = config["retrievers"]
    simp = SimpleConfig(
        top_k=int(ret_cfg.get("top_k", 5)),
        embedding_model=str(ret_cfg.get("embedding_model", "sentence-transformers/all-mpnet-base-v2")),
        ensemble_weights=tuple(ret_cfg.get("ensemble_weights", [0.75, 0.25])),
    )

    # 3) Build retrievers (global)
    faiss_vs = build_faiss_retriever(all_docs, simp.embedding_model)
    bm25 = build_bm25(all_docs)
    tfidf = build_tfidf(all_docs)
    svm = try_build_svm(all_docs, simp.embedding_model, simp.top_k)

    # 4) Evaluate
    detailed: List[Dict[str, Any]] = []
    tested = set()

    # Small lambda to score ensemble when components exist
    def run_ensemble(query: str, k: int) -> List[Document]:
        if not faiss_vs and not bm25:
            return []
        faiss_list = inv_faiss(faiss_vs, query, k) if faiss_vs else []
        bm25_list = inv_keyword(bm25, query, k) if bm25 else []
        if not faiss_list and not bm25_list:
            return []
        def rmap(lst): return {d.metadata.get("dia_id"): i for i, d in enumerate(lst)}
        rf, rb = rmap(faiss_list), rmap(bm25_list)
        cand: Dict[str, Tuple[float, Document]] = {}
        for doc in faiss_list + bm25_list:
            key = doc.metadata.get("dia_id")
            sf = (1.0 / (1 + rf.get(key, 9999)))
            sb = (1.0 / (1 + rb.get(key, 9999)))
            score = simp.ensemble_weights[0] * sf + simp.ensemble_weights[1] * sb
            if key not in cand or score > cand[key][0]:
                cand[key] = (score, doc)
        return [d for _, d in sorted(cand.values(), key=lambda x: x[0], reverse=True)[:k]]

    # Group QAs by sample for reporting (not required)
    q_by_sample: Dict[str, List[Dict[str, Any]]] = {}
    for q in qa_list:
        q_by_sample.setdefault(q["sample_id"], []).append(q)

    for sid, qas in q_by_sample.items():
        for qa in qas:
            qtext = qa["question"]
            gold = qa.get("evidence", []) or []
            cat = qa.get("category")

            # Run each retriever if available
            # 1) BM25
            if bm25:
                docs = inv_keyword(bm25, qtext, simp.top_k)
                m = compare_evidence_ids(gold, docs)
                detailed.append(_row(sid, "bm25", qtext, gold, docs, m, cat))
                tested.add("bm25")
            # 2) TF-IDF
            if tfidf:
                docs = inv_keyword(tfidf, qtext, simp.top_k)
                m = compare_evidence_ids(gold, docs)
                detailed.append(_row(sid, "tfidf", qtext, gold, docs, m, cat))
                tested.add("tfidf")
            # 3) FAISS (KNN)
            if faiss_vs:
                docs = inv_faiss(faiss_vs, qtext, simp.top_k)
                m = compare_evidence_ids(gold, docs)
                detailed.append(_row(sid, "faiss", qtext, gold, docs, m, cat))
                tested.add("faiss")
            # 4) SVM (optional)
            if svm:
                if hasattr(svm, "k"): svm.k = simp.top_k
                docs = svm.invoke(qtext)
                m = compare_evidence_ids(gold, docs)
                detailed.append(_row(sid, "svm", qtext, gold, docs, m, cat))
                tested.add("svm")
            # 5) Time-weighted (proxy): FAISS then recency by dia_id
            if faiss_vs:
                base = inv_faiss(faiss_vs, qtext, simp.top_k * 3)
                docs = sorted(base, key=lambda d: d.metadata.get("dia_id", ""), reverse=True)[:simp.top_k]
                m = compare_evidence_ids(gold, docs)
                detailed.append(_row(sid, "time_weighted", qtext, gold, docs, m, cat))
                tested.add("time_weighted")
            # 6) Ensemble (FAISS + BM25)
            ens_docs = run_ensemble(qtext, simp.top_k)
            if ens_docs:
                m = compare_evidence_ids(gold, ens_docs)
                detailed.append(_row(sid, "ensemble", qtext, gold, ens_docs, m, cat))
                tested.add("ensemble")

    # 5) Summaries
    overall = evaluate_retriever_results(detailed)
    by_cat = summarize_by_category(detailed)  # retriever -> {cat -> metrics}

    # 6) Save
    out_cfg = config["output"]
    ts = get_timestamp(out_cfg.get("timestamp_format", "datetime"))
    out_dir = Path(out_cfg.get("results_dir", "./results_lme"))
    ensure_dir(out_dir)

    detailed_path = out_dir / f"retriever_all_{ts}.json"
    overall_path  = out_dir / f"summary_overall_{ts}.json"
    bycat_path    = out_dir / f"summary_by_category_{ts}.json"

    with open(detailed_path, "w", encoding="utf-8") as f:
        json.dump({"timestamp": ts, "retrievers_tested": sorted(tested), "results": detailed}, f, indent=2, ensure_ascii=False)

    with open(overall_path, "w", encoding="utf-8") as f:
        json.dump({"timestamp": ts, "retrievers_tested": sorted(tested), "summary": overall}, f, indent=2, ensure_ascii=False)

    with open(bycat_path, "w", encoding="utf-8") as f:
        json.dump({"timestamp": ts, "retrievers_tested": sorted(tested), "summary_by_category": by_cat}, f, indent=2, ensure_ascii=False)

    # 7) Print nice tables
    print("\nOVERALL (categories {} only)".format(CATEGORIES_TO_EVAL))
    print_table(overall)
    print("\nBY CATEGORY")
    print_table_by_category(by_cat)

    print(f"\nDetailed: {detailed_path}")
    print(f"Overall : {overall_path}")
    print(f"By Cat  : {bycat_path}")

    return {
        "detailed_path": str(detailed_path),
        "overall_path": str(overall_path),
        "bycat_path": str(bycat_path),
        "retrievers_tested": sorted(tested),
        "overall": overall,
        "by_category": by_cat
    }

# -------------------------
# Helpers: rows & summaries
# -------------------------
def _row(sample_id, retriever, question, gold, docs, metrics, category):
    return {
        "sample_id": sample_id,
        "retriever": retriever,
        "category": category,
        "question": question,
        "gold_evidence": gold,
        "retrieved_dia_ids": [d.metadata.get("dia_id") for d in docs],
        "metrics": metrics
    }

def summarize_by_category(detailed_rows: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """
    Returns: {retriever: {category: {questions, evidence_found, evidence_total, recall_pct}}}
    """
    agg: Dict[str, Dict[int, Dict[str, float]]] = {}
    for row in detailed_rows:
        r = row["retriever"]
        cat = int(row.get("category") or 0)
        m = row["metrics"]
        agg.setdefault(r, {})
        s = agg[r].setdefault(cat, {"questions": 0, "evidence_found": 0, "evidence_total": 0})
        s["questions"] += 1
        s["evidence_found"] += m["evidence_found"]
        s["evidence_total"] += m["evidence_total"]

    # add recall_pct
    out: Dict[str, Dict[str, Any]] = {}
    for r, percat in agg.items():
        out[r] = {}
        for cat, s in percat.items():
            tot = max(s["evidence_total"], 1)
            s["recall_pct"] = (s["evidence_found"] / tot) * 100.0
            out[r][str(cat)] = s
    return out

def print_table(overall: Dict[str, Dict[str, Any]]):
    print("-" * 80)
    print(f"{'Retriever':<18} {'Questions':>10} {'Evidence Found':>18} {'Recall %':>10}")
    print("-" * 80)
    for r, s in sorted(overall.items()):
        qn = s["questions"]; ef = s["evidence_found"]; et = s["evidence_total"]; rc = s["recall_pct"]
        print(f"{r:<18} {qn:>10} {f'{ef}/{et}':>18} {rc:>9.1f}%")
    print("-" * 80)

def print_table_by_category(by_cat: Dict[str, Dict[str, Any]]):
    cats = sorted({int(c) for r in by_cat.values() for c in r.keys()})
    header = "Retriever".ljust(18) + "".join([f"  Cat{c} Q  Cat{c} Rec%".rjust(16) for c in cats])
    print("-" * max(80, len(header)))
    print(header)
    print("-" * max(80, len(header)))
    for r in sorted(by_cat.keys()):
        row = r.ljust(18)
        for c in cats:
            s = by_cat[r].get(str(c), {"questions": 0, "recall_pct": 0.0})
            row += f"  {s['questions']:>5}  {s['recall_pct']:>9.1f}%"
        print(row)
    print("-" * max(80, len(header)))

def main():
    config_path = Path("config_lme.yaml")
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    run_pipeline(config)

if __name__ == "__main__":
    main()
