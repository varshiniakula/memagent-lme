"""
answer_evaluator.py
-------------------
Unified evaluator for retrieval QA tasks (Locomo + LongMemEval).
Computes per-question recall and dataset-level summaries based on dia_id matching.
"""

from typing import List, Dict, Any
from langchain_core.documents import Document

# ---------------------------------------------------------------------
# Normalization helpers
# ---------------------------------------------------------------------
def normalize_id(raw_id: str) -> str:
    """
    Normalize dialogue/evidence identifiers to a canonical form.
    Works for both Locomo (D#:#) and LongMemEval (T#:#) styles.
    """
    if not raw_id:
        return ""
    return str(raw_id).strip().lower()

# ---------------------------------------------------------------------
# Evidence comparison core
# ---------------------------------------------------------------------
def compare_evidence_ids(gold_evidence: List[str], retrieved_docs: List[Document]) -> Dict[str, Any]:
    """
    Compare ground-truth evidence IDs to retrieved document metadata dia_ids.

    Parameters
    ----------
    gold_evidence : list of str
        Example: ["D3:11", "D2:14"] or ["T1:3"]
    retrieved_docs : list of langchain_core.documents.Document
        Documents returned by retriever; must have metadata["dia_id"]

    Returns
    -------
    dict
        {
          "evidence_found": <int>,
          "evidence_total": <int>,
          "recall_pct": <float>
        }
    """
    gold_ids = {normalize_id(e) for e in (gold_evidence or []) if e}
    retrieved_ids = {normalize_id(d.metadata.get("dia_id")) for d in (retrieved_docs or []) if d}

    matched = gold_ids.intersection(retrieved_ids)
    total = max(len(gold_ids), 1)

    return {
        "evidence_found": len(matched),
        "evidence_total": len(gold_ids),
        "recall_pct": (len(matched) / total) * 100.0
    }

# ---------------------------------------------------------------------
# Main evaluation entry point
# ---------------------------------------------------------------------
def evaluate_retriever_results(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Aggregate detailed per-question results into a summary per retriever.

    Each element in `results` should have:
      {
        "retriever": "bm25",
        "metrics": {"evidence_found": 3, "evidence_total": 5, "recall_pct": 60.0},
        ...
      }

    Returns
    -------
    summary : dict
        {
          "bm25": {"questions": 10, "evidence_found": 50, "evidence_total": 100, "recall_pct": 50.0},
          ...
        }
    """
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

    # Compute recall %
    for r, s in summary.items():
        total = max(s["evidence_total"], 1)
        s["recall_pct"] = (s["evidence_found"] / total) * 100.0

    return summary

# ---------------------------------------------------------------------
# Optional helper for per-question computation
# ---------------------------------------------------------------------
def evaluate_question(gold_evidence: List[str], retrieved_docs: List[Document]) -> Dict[str, Any]:
    """
    Shortcut for evaluating a single QA pair.
    """
    return compare_evidence_ids(gold_evidence, retrieved_docs)

# ---------------------------------------------------------------------
# Example usage
# ---------------------------------------------------------------------
if __name__ == "__main__":
    # dummy example
    from langchain_core.documents import Document

    docs = [
        Document(page_content="hi", metadata={"dia_id": "D3:11"}),
        Document(page_content="there", metadata={"dia_id": "T1:3"}),
    ]
    gold = ["D3:11", "D2:1"]
    metrics = compare_evidence_ids(gold, docs)
    print("Per-question metrics:", metrics)

    summary = evaluate_retriever_results([
        {"retriever": "bm25", "metrics": metrics},
        {"retriever": "bm25", "metrics": metrics},
    ])
    print("Summary:", summary)
