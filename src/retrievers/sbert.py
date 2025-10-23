# src/retrievers/sbert.py
from typing import List, Dict, Any
import os
import numpy as np
from sentence_transformers import SentenceTransformer
from .base import Retriever

def _l2norm(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True) + 1e-12
    return x / n

class SBertFaissRetriever(Retriever):
    def __init__(self, model_name: str):
        self.model_name = model_name
        # Use CPU for stability; switch to "mps" on Apple Silicon only if you know PyTorch MPS is stable for you
        self.model = SentenceTransformer(model_name, device="cpu")
        self.docs: List[Dict[str, Any]] = []
        self.emb: np.ndarray = None

        # Try FAISS unless user disables it
        self.use_faiss = os.environ.get("USE_FAISS", "1") == "1"
        self.faiss = None
        self.index = None
        if self.use_faiss:
            try:
                import faiss  # noqa
                self.faiss = faiss
            except Exception as e:
                print(f"[warn] FAISS unavailable or unstable: {e}. Falling back to NumPy search.")
                self.use_faiss = False

        safe = self.model_name.split("/")[-1].replace(":", "_")
        self._name = f"dense-{safe}"

    @property
    def name(self) -> str:
        return self._name

    def _encode(self, texts: List[str]) -> np.ndarray:
        v = self.model.encode(texts, batch_size=64, show_progress_bar=False, normalize_embeddings=False)
        v = np.asarray(v, dtype="float32")
        v = _l2norm(v)  # normalize so dot == cosine
        return v

    def build(self, docs: List[Dict[str, Any]]) -> None:
        self.docs = docs
        texts = [d["text"] for d in docs]
        self.emb = self._encode(texts)

        if self.use_faiss:
            dim = self.emb.shape[1]
            self.index = self.faiss.IndexFlatIP(dim)  # cosine with normalized vectors
            self.index.add(self.emb)
        # else: no index; we’ll do numpy dot at query time

    def query(self, q: str, topk: int = 5) -> List[Dict[str, Any]]:
        if not q.strip():
            return []
        qv = self._encode([q])[0:1]  # shape (1, d)

        if self.use_faiss and self.index is not None:
            D, I = self.index.search(qv, topk)
            I = I[0]; D = D[0]
        else:
            # pure NumPy cosine similarity
            sims = (qv @ self.emb.T).ravel()  # (N,)
            I = np.argsort(-sims)[:topk]
            D = sims[I]

        out = []
        for score, idx in zip(D, I):
            if idx is None or idx < 0:
                continue
            d = self.docs[int(idx)]
            out.append({"doc_id": d["doc_id"], "text": d["text"], "score": float(score), "meta": d["meta"]})
        return out
