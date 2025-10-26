#!/usr/bin/env python3
# tools/summarize_winners.py
"""
Summarize per-conversation retriever winners from a by_question_*.json file.

Input structure (by_question JSON from run_pipeline_lme.py):
{
  "timestamp": "...",
  "questions": [
    {
      "sample_id": "conv-0041",
      "question": "When is the meeting scheduled?",
      "gold_evidence": ["T41:1"],
      "success_by_retriever": {"bm25": true/false, "faiss": true/false, ...},
      "winner_list": ["tfidf", ...],
      "oracle_success": true/false,
      "union_retrieved_dia_ids": [...]
    },
    ...
  ]
}

Output structure (winners JSON we produce):
{
  "timestamp": "...",
  "conversations": {
    "conv-0041": {
      "q_ids": ["q1","q2","q3",...],
      "questions": ["...", "...", ...],
      "winners": {
        "bm25": ["q1","q2","q5"],
        "faiss": ["q3","q5"],
        "tfidf": []
      },
      "oracle": ["q1","q2","q3","q5"]
    },
    ...
  }
}

Usage:
  python3 tools/summarize_winners.py --by_question results_lme/by_question_2025....json
  # Optional:
  python3 tools/summarize_winners.py --by_question results_lme/by_question_....json --filter conv-0041 --pretty
  python3 tools/summarize_winners.py --by_question results_lme/by_question_....json --out results_lme/winners_custom.json
"""

import json
import argparse
from collections import OrderedDict
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List


def _load_by_question(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Input not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if "questions" not in data or not isinstance(data["questions"], list):
        raise ValueError("Invalid by_question JSON: missing top-level 'questions' array.")
    return data


def _build_winners(by_question: Dict[str, Any]) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = by_question.get("questions", [])
    # Preserve arrival order per conversation
    grouped: Dict[str, List[Dict[str, Any]]] = OrderedDict()
    for r in rows:
        sid = r.get("sample_id")
        if sid is None:
            # skip malformed row
            continue
        grouped.setdefault(sid, []).append(r)

    # Collect full retriever set across all questions
    all_retrievers = set()
    for r in rows:
        succ = r.get("success_by_retriever") or {}
        all_retrievers.update(succ.keys())
    all_retrievers = sorted(all_retrievers)

    conversations: Dict[str, Any] = {}
    for sid, qrows in grouped.items():
        # stable q1..qN labels following appearance order
        q_ids = [f"q{i+1}" for i in range(len(qrows))]
        questions = [qr.get("question", "") for qr in qrows]

        winners = {name: [] for name in all_retrievers}
        oracle: List[str] = []

        for i, qr in enumerate(qrows):
            qi = q_ids[i]
            succ = qr.get("success_by_retriever") or {}
            any_win = False
            for name in all_retrievers:
                if succ.get(name) is True:
                    winners[name].append(qi)
                    any_win = True
            if any_win:
                oracle.append(qi)

        conversations[sid] = {
            "q_ids": q_ids,
            "questions": questions,
            "winners": winners,
            "oracle": oracle
        }

    out = {
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "conversations": conversations
    }
    return out


def _print_human(conversations: Dict[str, Any], only_conv: str | None):
    """Pretty console output like: 'conv-0041 — bm25: q1,q2,q5 | faiss: q3,q5 | oracle: q1,q2,q3,q5'"""
    items = conversations.items()
    if only_conv:
        items = [(only_conv, conversations.get(only_conv))] if only_conv in conversations else []
        if not items:
            print(f"[WARN] conversation '{only_conv}' not found.")
            return
    for sid, block in items:
        if block is None:
            continue
        winners = block.get("winners", {})
        oracle = block.get("oracle", [])
        parts = []
        for name, qs in sorted(winners.items()):
            parts.append(f"{name}: {','.join(qs) if qs else '-'}")
        line = " | ".join(parts)
        print(f"{sid} — {line} | oracle: {','.join(oracle) if oracle else '-'}")


def main():
    ap = argparse.ArgumentParser(description="Summarize per-conversation winners from by_question_*.json")
    ap.add_argument("--by_question", required=True, help="Path to by_question_*.json (from run_pipeline_lme.py)")
    ap.add_argument("--out", default=None, help="Output path for winners JSON (defaults next to input)")
    ap.add_argument("--filter", dest="filter_conv", default=None, help="Only print/show a specific conversation ID")
    ap.add_argument("--pretty", action="store_true", help="Print a human-readable summary to the console")
    args = ap.parse_args()

    byq_path = Path(args.by_question)
    data = _load_by_question(byq_path)
    winners = _build_winners(data)

    # Write winners JSON
    if args.out:
        out_path = Path(args.out)
    else:
        # results_lme/by_question_<ts>.json -> results_lme/winners_<ts>.json
        stem = byq_path.stem  # e.g., 'by_question_20251025_193015'
        ts = stem.replace("by_question_", "")
        out_path = byq_path.with_name(f"winners_{ts}.json")

    out_path.write_text(json.dumps(winners, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[OK] wrote {out_path}")

    # Optional console pretty print
    if args.pretty:
        _print_human(winners["conversations"], args.filter_conv)


if __name__ == "__main__":
    main()
