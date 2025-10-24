from typing import List, Dict, Any
from rank_bm25 import BM25Okapi
from .base import Retriever

class BM25Retriever(Retriever):
    def __init__(self):
        self.docs = []
        self._name = "bm25"

    @property
    def name(self) -> str:
        return self._name

    def build(self, docs: List[Dict[str, Any]]) -> None:
        self.docs = docs
        texts = [d["text"] for d in docs]
        self.tokenized = [t.lower().split() for t in texts]
        self.bm25 = BM25Okapi(self.tokenized)

    def query(self, q: str, topk: int = 5) -> List[Dict[str, Any]]:
        if not q.strip():
            return []
        scores = self.bm25.get_scores(q.lower().split())
        idxs = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:topk]
        out = []
        for i in idxs:
            d = self.docs[i]
            out.append({
                "doc_id": d["doc_id"],
                "text": d["text"],
                "score": float(scores[i]),
                "meta": d["meta"]
            })
        return out
