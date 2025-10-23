from typing import List, Dict, Any
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from .base import Retriever

class TfidfRetriever(Retriever):
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=100000, ngram_range=(1, 2))
        self.X = None
        self.docs = []
        self._name = "tfidf"

    @property
    def name(self) -> str:
        return self._name

    def build(self, docs: List[Dict[str, Any]]) -> None:
        self.docs = docs
        texts = [d["text"] for d in docs]
        self.X = self.vectorizer.fit_transform(texts)

    def query(self, q: str, topk: int = 5) -> List[Dict[str, Any]]:
        if not q.strip():
            return []
        qv = self.vectorizer.transform([q])
        sims = cosine_similarity(qv, self.X)[0]
        idxs = np.argsort(-sims)[:topk]
        out = []
        for i in idxs:
            d = self.docs[i]
            out.append({
                "doc_id": d["doc_id"],
                "text": d["text"],
                "score": float(sims[i]),
                "meta": d["meta"]
            })
        return out
