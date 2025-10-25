"""
answer_evaluator.py
Unified evaluator for retrieval QA tasks (Locomo + LongMemEval).
Computes per-question recall and dataset-level summaries based on dia_id matching.
"""

from typing import List, Dict, Any
from langchain_core.documents import Document

def normalize_id(raw_id: str) -> str:
    if not raw_id:
        return ""
    return str(raw_id).strip().lower()

def compare_evidence_ids(gold_evidence: List[str], retrieved_docs: List[Document]) -> Dict[str, Any]:
    gold_ids = {normalize_id(e) for e in (gold_evidence or []) if e}
    retrieved_ids = {normalize_id(d.metadata.get("dia_id")) for d in (retrieved_docs or []) if d}
    matched = gold_ids.intersection(retrieved_ids)
    total = max(len(gold_ids), 1)
    return {
        "evidence_found": len(matched),
        "evidence_total": len(gold_ids),
        "recall_pct": (len(matched) / total) * 100.0
    }

def evaluate_retriever_results(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    summary: Dict[str, Dict[str, Any]] = {}
    for entry in results:
        r = entry.get("retriever")
        m = entry.get("metrics", {})
        if not r:
            continue
        s = summary.setdefault(r, {"questions": 0, "evidence_found": 0, "evidence_total": 0})
        s["questions"] += 1
        s["evidence_found"] += m.get("evidence_found", 0)
        s["evidence_total"] += m.get("evidence_total", 0)
    for r, s in summary.items():
        total = max(s["evidence_total"], 1)
        s["recall_pct"] = (s["evidence_found"] / total) * 100.0
    return summary

def evaluate_question(gold_evidence: List[str], retrieved_docs: List[Document]) -> Dict[str, Any]:
    return compare_evidence_ids(gold_evidence, retrieved_docs)
