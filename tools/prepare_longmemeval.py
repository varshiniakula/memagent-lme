"""
Prepare LongMemEval into the canonical schema expected by LMEDataLoader.
- Reads raw LME files (you provide paths)
- Writes data/longmemeval.json in the shape:
  [
    {
      "id": "conv-001",
      "turns": [{"speaker": "A", "text": "..." , "dia_id": "T1:1"}, ...],
      "qa": [{"question": "...", "answer": "...", "evidence": ["T1:3","T2:1"], "category": 1}]
    },
    ...
  ]
"""

import json
from pathlib import Path
from typing import List, Dict

def ingest_example_raw() -> List[Dict]:


    """Replace this with actual ingestion of LongMemEval raw format."""
    # Minimal toy example
    return [
        {
            "id": "conv-001",
            "turns": [
                {"speaker": "A", "text": "...", "dia_id": "T1:1"},
                {"speaker": "B", "text": "...", "dia_id": "T1:2"}
            ],
            "qa": [
                {"question": "Who greeted first?", "answer": "A", "evidence": ["T1:1"], "category": 1}
            ]
        }
    ]


def main():
    out = Path("data/longmemeval.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    data = ingest_example_raw()
    out.write_text(json.dumps(data, indent=2), encoding="utf-8")
    print(f"Wrote {out} with {len(data)} conversation(s).")

if __name__ == "__main__":
    main()
