"""
Prepare LongMemEval into the canonical schema expected by LMEDataLoader.
This is a toy generator to help you validate the pipeline quickly.

Edit `ingest_real_lme()` to map your actual raw LME sources.
"""

import json
from pathlib import Path
from typing import List, Dict


def ingest_toy() -> List[Dict]:
    # Works even with punctuation-only turns thanks to relaxed sanitization
    return [
        {
            "id": "conv-001",
            "turns": [
                {"speaker": "A", "text": "...", "dia_id": "T1:1"},
                {"speaker": "B", "text": "Hi there.", "dia_id": "T1:2"},
            ],
            "qa": [
                {"question": "Who greeted first?", "answer": "A", "evidence": ["T1:1"], "category": 1}
            ],
        }
    ]


def ingest_real_lme() -> List[Dict]:
    """
    TODO: Replace this stub with your real LongMemEval ingestion.
    Must return a list of conversation dicts shaped like `ingest_toy()`.
    """
    return ingest_toy()


def main():
    out = Path("data/longmemeval.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    data = ingest_real_lme()
    out.write_text(json.dumps(data, indent=2), encoding="utf-8")
    print(f"Wrote {out} with {len(data)} conversation(s).")


if __name__ == "__main__":
    main()
