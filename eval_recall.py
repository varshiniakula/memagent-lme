#!/usr/bin/env python3
"""
Evaluate retrieval results with a simple recall@k metric against LongMemEval.

Assumption (fast baseline):
- Using the *oracle* JSON (recommended for quick iteration), every turn (or session,
  if you choose session granularity) in `haystack_sessions` is considered relevant
  for its question_id. This lets you compare retrievers consistently even without
  per-turn gold labels.

What it computes:
- recall@k = (# unique relevant doc_ids retrieved in top-k) / (total relevant doc_ids)
- any_hit%  = % of questions with at least one relevant hit in top-k
- Also prints a small table and writes a JSON summary.

Usage:
  python eval_recall.py \
    --data_file data/longmemeval_oracle.json \
    --outputs_dir outputs/retrieval_turn \
    --granularity turn \
    --topk 5 \
    --summary_out results/summary_recall_turn_k5.json
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Set

from src.datasets.longmemeval_loader import load_instances, turns_from_instance


def build_gold_sets(data_file: str, granularity: str) -> Dict[str, Set[str]]:
    """
    Build gold doc_id sets per question_id using the same doc-iding scheme as the pipeline.
    For granularity="turn": gold are all turn doc_ids (qid|sX|tY)
    For granularity="session": gold are all session doc_ids (qid|sX)
    """
    instances = load_instances(data_file)
    gold: Dict[str, Set[str]] = {}
    for inst in instances:
        qid = inst["question_id"]
        gold.setdefault(qid, set())
        for d in turns_from_instance(inst, granularity=granularity):
            gold[qid].add(d["doc_id"])
    return gold


def load_retrieval_file(path: Path) -> Dict[str, List[str]]:
    """
    Load one retriever's JSONL output and return:
      qid -> list of doc_ids in the order they appear (top-k)
    """
    qid_to_docs: Dict[str, List[str]] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            qid = rec["question_id"]
            hits = rec.get("retrieval_results", [])
            doc_ids = [h.get("doc_id", "") for h in hits if h.get("doc_id")]
            qid_to_docs[qid] = doc_ids
    return qid_to_docs


def evaluate_one(
        retr_file: Path,
        gold_sets: Dict[str, Set[str]],
        topk: int,
) -> Dict:
    """
    Compute recall@k and any_hit% over the intersection of questions present
    in both the gold and the retrieval file.
    """
    preds = load_retrieval_file(retr_file)
    common_qids = sorted(set(preds.keys()) & set(gold_sets.keys()))
    if not common_qids:
        return {
            "retriever": retr_file.stem,
            "total_questions": 0,
            "questions_with_any_hit": 0,
            "any_hit_pct": 0.0,
            "mean_recall_at_k": 0.0,
            "details": [],
        }

    per_q = []
    any_hits = 0
    recalls = []
    for qid in common_qids:
        gold = gold_sets[qid]
        pred = preds.get(qid, [])[:topk]
        pred_set = set(pred)
        inter = pred_set & gold
        hit = len(inter) > 0
        if hit:
            any_hits += 1
        denom = max(len(gold), 1)
        recall = len(inter) / denom
        recalls.append(recall)
        per_q.append(
            {
                "question_id": qid,
                "gold_count": len(gold),
                "retrieved_k": len(pred),
                "hits": len(inter),
                "recall_at_k": recall,
                "any_hit": hit,
            }
        )

    return {
        "retriever": retr_file.stem,
        "total_questions": len(common_qids),
        "questions_with_any_hit": any_hits,
        "any_hit_pct": 100.0 * any_hits / max(len(common_qids), 1),
        "mean_recall_at_k": sum(recalls) / max(len(recalls), 1),
        "details": per_q,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_file", required=True, help="LongMemEval JSON (oracle/S/M cleaned)")
    ap.add_argument("--outputs_dir", required=True, help="Directory with *.jsonl retriever outputs")
    ap.add_argument("--granularity", choices=["turn", "session"], default="turn")
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--summary_out", default="results/summary_recall.json")
    args = ap.parse_args()

    outputs_dir = Path(args.outputs_dir)
    files = sorted(outputs_dir.glob("*.jsonl"))
    if not files:
        raise SystemExit(f"No JSONL files found in: {outputs_dir}")

    # Build gold sets consistent with the retrieval granularity
    gold_sets = build_gold_sets(args.data_file, args.granularity)

    # Evaluate each retriever file
    summaries = []
    for f in files:
        res = evaluate_one(f, gold_sets, args.topk)
        summaries.append(res)

    # Pretty print small table
    print("\nRECALL@{} (granularity: {})".format(args.topk, args.granularity))
    print("-" * 78)
    print("{:<32} {:>10} {:>12} {:>12}".format("Retriever", "Questions", "AnyHit %", "Mean R@k"))
    print("-" * 78)
    for s in sorted(summaries, key=lambda x: x["mean_recall_at_k"], reverse=True):
        print(
            "{:<32} {:>10} {:>11.1f}% {:>11.3f}".format(
                s["retriever"],
                s["total_questions"],
                s["any_hit_pct"],
                s["mean_recall_at_k"],
            )
        )
    print("-" * 78)

    # Write JSON summary
    out_path = Path(args.summary_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as w:
        json.dump(
            {
                "granularity": args.granularity,
                "topk": args.topk,
                "results": summaries,
            },
            w,
            ensure_ascii=False,
            indent=2,
        )
    print(f"[+] Wrote summary to {out_path}")


if __name__ == "__main__":
    main()
