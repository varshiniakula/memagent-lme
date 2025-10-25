"""
extend_lme_dataset.py
Extend LongMemEval with more samples and/or merge files.

Changes vs previous version:
- Do NOT default missing categories to 1 inside ensure_ids()
- Add optional --auto_categorize heuristic (temporal/inference)
- Optional --default_cat1_missing to coerce remaining None to 1 (if desired)

Usage:
  # Add synthetic Cat1 and Cat2
  python tools/extend_lme_dataset.py --add_cat1 20 --add_cat2 20

  # Merge external JSON(s)
  python tools/extend_lme_dataset.py --merge path/to/extra.json --merge path2.json

  # Heuristically categorize missing categories from question text
  python tools/extend_lme_dataset.py --merge raw.json --auto_categorize

  # If you still want everything without a category to become Cat-1:
  python tools/extend_lme_dataset.py --default_cat1_missing
"""

import json
import argparse
import re
from pathlib import Path
from typing import List, Dict, Any

DATA_PATH = Path("data/longmemeval.json")

# --- Simple heuristics for category tagging ---
TEMPORAL_PAT = re.compile(r"\b(when|date|year|month|time|day|schedule|scheduled|deadline|tomorrow|today|yesterday)\b", re.I)
INFERENCE_PAT = re.compile(r"\b(would|could|should|why|if|likely|probably|might)\b", re.I)


def load_base() -> List[Dict[str, Any]]:
    if not DATA_PATH.exists():
        return []
    try:
        data = json.loads(DATA_PATH.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            data = data.get("data") or data
        if not isinstance(data, list):
            raise ValueError("Input JSON must be a list of conversations.")
        return data
    except Exception as e:
        raise RuntimeError(f"Failed to read {DATA_PATH}: {e}")


def save_data(data: List[Dict[str, Any]]):
    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    DATA_PATH.write_text(json.dumps(data, indent=2), encoding="utf-8")
    print(f"[OK] Saved {len(data)} conversations to {DATA_PATH}")


def synth_cat1(start_idx: int, n: int) -> List[Dict[str, Any]]:
    """Category 1: Multi-part factual answers; evidence points to one turn."""
    out: List[Dict[str, Any]] = []
    for i in range(n):
        ci = start_idx + i + 1
        conv_id = f"conv-{ci:04d}"
        turns = [
            {"speaker": "A", "text": f"City: Alpha-{ci}, Country: Zeta", "dia_id": f"T{ci}:1"},
            {"speaker": "B", "text": f"Population: {1000+ci}, Landmarks: L{ci}-1, L{ci}-2", "dia_id": f"T{ci}:2"},
        ]
        qa = [{
            "question": "List the city and country.",
            "answer": f"Alpha-{ci}, Zeta",
            "evidence": [f"T{ci}:1"],
            "category": 1
        }]
        out.append({"id": conv_id, "turns": turns, "qa": qa})
    return out


def synth_cat2(start_idx: int, n: int) -> List[Dict[str, Any]]:
    """Category 2: Temporal question about a date referenced in the dialog."""
    out: List[Dict[str, Any]] = []
    for i in range(n):
        ci = start_idx + i + 1
        conv_id = f"conv-{ci:04d}"
        month = (i % 9) + 1
        day = (i % 9) + 10
        date_str = f"2024-0{month}-{day}"
        turns = [
            {"speaker": "A", "text": f"Meeting scheduled on {date_str}.", "dia_id": f"T{ci}:1"},
            {"speaker": "B", "text": "Okay, noted.", "dia_id": f"T{ci}:2"},
        ]
        qa = [{
            "question": "When is the meeting scheduled?",
            "answer": date_str,
            "evidence": [f"T{ci}:1"],
            "category": 2
        }]
        out.append({"id": conv_id, "turns": turns, "qa": qa})
    return out


def merge_file(path: Path) -> List[Dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        data = data.get("data") or data
    if not isinstance(data, list):
        raise ValueError(f"Merged file {path} must be a list of conversations.")
    return data


def ensure_ids(data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Normalize minimal schema and leave 'category' untouched if missing (None).
    - Keep any non-empty text turn
    - Ensure dia_id exists
    """
    out: List[Dict[str, Any]] = []
    for idx, conv in enumerate(data, start=1):
        cid = conv.get("id") or f"conv-{idx:04d}"
        turns = conv.get("turns", [])
        fixed_turns = []
        ti = 0
        for t in turns:
            text = (t.get("text") or t.get("content") or t.get("message") or t.get("utterance") or "").strip()
            if not text:
                continue
            ti += 1
            dia_id = t.get("dia_id") or t.get("id") or f"T{idx}:{ti}"
            fixed_turns.append({"speaker": t.get("speaker"), "text": text, "dia_id": dia_id})

        qas_in = conv.get("qa", [])
        qas_out = []
        for q in qas_in:
            qas_out.append({
                "question": (q.get("question") or "").strip(),
                "answer": q.get("answer"),
                "evidence": q.get("evidence", []) or [],
                "category": q.get("category")  # <- keep None if missing
            })

        out.append({"id": cid, "turns": fixed_turns, "qa": qas_out})
    return out


def auto_categorize(data: List[Dict[str, Any]], warn: bool = True) -> None:
    """
    Heuristically set category if it's None:
      - Cat-2 if temporal keywords in question
      - Cat-3 if inference/reasoning keywords in question
      - Otherwise Cat-1
    Modifies data in place.
    """
    for conv in data:
        for qa in conv.get("qa", []):
            if qa.get("category") is not None:
                continue
            q = (qa.get("question") or "").lower()
            if TEMPORAL_PAT.search(q):
                qa["category"] = 2
            elif INFERENCE_PAT.search(q):
                qa["category"] = 3
            else:
                qa["category"] = 1
            if warn:
                print(f"[INFO] Auto-categorized: '{qa.get('question')}' -> {qa['category']}")


def default_missing_to_cat1(data: List[Dict[str, Any]], warn: bool = True) -> None:
    """
    Coerce any remaining None categories to 1.
    Modifies data in place.
    """
    for conv in data:
        for qa in conv.get("qa", []):
            if qa.get("category") is None:
                qa["category"] = 1
                if warn:
                    print(f"[INFO] Defaulted missing category to 1 for question: '{qa.get('question')}'")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--add_cat1", type=int, default=0, help="How many synthetic Category-1 samples to add")
    ap.add_argument("--add_cat2", type=int, default=0, help="How many synthetic Category-2 samples to add")
    ap.add_argument("--merge", type=str, action="append", help="Path(s) to JSON files to merge")
    ap.add_argument("--auto_categorize", action="store_true", help="Heuristically set category when missing")
    ap.add_argument("--default_cat1_missing", action="store_true", help="Force remaining missing categories to 1")
    args = ap.parse_args()

    # Load existing and normalize
    base = load_base()
    base = ensure_ids(base)

    start_idx = len(base)
    added: List[Dict[str, Any]] = []

    # Add synthetic data
    if args.add_cat1 > 0:
        added += synth_cat1(start_idx, args.add_cat1)
        start_idx += args.add_cat1

    if args.add_cat2 > 0:
        added += synth_cat2(start_idx, args.add_cat2)
        start_idx += args.add_cat2

    # Merge external files
    if args.merge:
        for p in args.merge:
            extra = ensure_ids(merge_file(Path(p)))
            added += extra

    # Combine and optionally categorize
    new_data = base + added

    if args.auto_categorize:
        auto_categorize(new_data, warn=True)

    if args.default_cat1_missing:
        default_missing_to_cat1(new_data, warn=True)

    save_data(new_data)


if __name__ == "__main__":
    main()
