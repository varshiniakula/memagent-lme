"""
dataloader_lme.py (simple & tolerant)
- Accepts LongMemEval canonical list OR {"data": [...]}
- Tolerates text fields: text | content | message | utterance
- Auto-fills dia_id if missing (T{conv_index}:{turn_index})
"""

import json
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional, Union
from langchain_core.documents import Document


TextKeys = ("text", "content", "message", "utterance")


@dataclass
class QAPair:
    sample_id: str
    question: str
    answer: Optional[str]
    evidence: List[str]  # list of dia_id strings like "T1:3"
    category: Optional[int] = None


class LMEDataLoader:
    """
    Expected canonical schema (per conversation item):
    {
        "id": "conv-001",
        "turns": [{"speaker": "A|B", "text": "...", "dia_id": "T1:1"}, ...],
        "qa": [
            {"question": "...", "answer": "...", "evidence": ["T1:7","T2:4"], "category": 1}
        ]
    }
    """

    def __init__(self, json_path: str):
        self.json_path = json_path
        self.documents: List[Document] = []
        self.qa_pairs: List[QAPair] = []
        self.conversation_index: Dict[str, List[int]] = {}  # sample_id -> indices in self.documents

    def _read_json(self) -> List[Dict]:
        with open(self.json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            data = data.get("data") or data.get("conversations") or data.get("items") or data
        if not isinstance(data, list):
            raise ValueError("Expected a list of conversations in LongMemEval JSON.")
        return data

    @staticmethod
    def _extract_text(turn: Dict) -> str:
        for k in TextKeys:
            if k in turn and isinstance(turn[k], str):
                return turn[k].strip()
        return ""

    def load_data(
            self,
            sample_ids: Optional[Union[List[str], str]] = None,
            limit: Optional[int] = None
    ) -> Tuple[List[Document], List[Dict], Dict]:
        raw = self._read_json()

        # Filter by sample_ids
        if sample_ids and sample_ids != "all":
            if isinstance(sample_ids, str):
                sample_ids = [sample_ids]
            sel = set(sample_ids)
            raw = [c for c in raw if c.get("id") in sel]

        # Apply limit
        if limit is not None:
            raw = raw[: int(limit)]

        # Build documents and QA list
        self.documents.clear()
        self.qa_pairs.clear()
        self.conversation_index.clear()

        for ci, conv in enumerate(raw, start=1):
            sid = conv.get("id") or f"conv-{ci:04d}"
            turns = conv.get("turns", [])
            start_idx = len(self.documents)
            ti = 0
            for t in turns:
                text = self._extract_text(t)
                # Skip empty/near-empty turns (no letters or digits)
                if not any(ch.isalnum() for ch in text):
                    continue
                ti += 1
                dia_id = t.get("dia_id") or t.get("id") or f"T{ci}:{ti}"
                meta = {"sample_id": sid, "dia_id": dia_id, "speaker": t.get("speaker")}
                self.documents.append(Document(page_content=text, metadata=meta))
            end_idx = len(self.documents)
            self.conversation_index[sid] = list(range(start_idx, end_idx))

            for qa in conv.get("qa", []):
                self.qa_pairs.append(QAPair(
                    sample_id=sid,
                    question=qa.get("question", "") or "",
                    answer=qa.get("answer"),
                    evidence=qa.get("evidence", []) or [],
                    category=qa.get("category")
                ))

        qa_records = [q.__dict__ for q in self.qa_pairs]
        meta = {
            "num_conversations": len(raw),
            "num_documents": len(self.documents),
            "num_questions": len(self.qa_pairs),
            "conversation_index": self.conversation_index
        }
        return self.documents, qa_records, meta

    def get_documents_for_sample(self, sample_id: str) -> List[Document]:
        idxs = self.conversation_index.get(sample_id, [])
        return [self.documents[i] for i in idxs]
