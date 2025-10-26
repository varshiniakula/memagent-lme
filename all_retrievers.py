"""
retrievers_all.py
All 8 retrievers for LongMemEval with graceful fallbacks.

Retrievers:
1) bm25 (lexical)
2) tfidf (lexical)
3) knn (dense; alias of faiss)
4) svm (ML)
5) colbert (RAGatouille, optional)
6) faiss (dense)
7) time_weighted (recency proxy)
8) nanopq (FAISS IVF+PQ, optional)

Usage in your runner:
    registry = RetrieverRegistry(documents, cfg)
    results = registry.invoke("faiss", query, k=cfg["retrievers"]["top_k"])
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
import os
import warnings

import numpy as np
from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever, TFIDFRetriever

# -------------------------
# Optional deps (import guarded)
# -------------------------
try:
    from langchain_huggingface import HuggingFaceEmbeddings as HFEmb
except Exception:
    try:
        from langchain_community.embeddings import HuggingFaceEmbeddings as HFEmb
    except Exception:
        HFEmb = None

try:
    from langchain_community.retrievers import SVMRetriever
except Exception:
    SVMRetriever = None

try:
    from langchain_community.vectorstores import FAISS
except Exception:
    FAISS = None

# FAISS raw API (for NanoPQ)
try:
    import faiss  # type: ignore
except Exception:
    faiss = None

# RAGatouille / ColBERT
try:
    from ragatouille import RAGPretrainedModel  # type: ignore
except Exception:
    RAGPretrainedModel = None


# -------------------------
# Small helpers
# -------------------------
def _sanitize_docs(docs: List[Document]) -> List[Document]:
    out = []
    for d in docs or []:
        txt = (d.page_content or "").strip()
        if txt:
            out.append(d)
    return out

def _embedder(model_name: str):
    if HFEmb is None:
        raise RuntimeError(
            "HuggingFaceEmbeddings not available. Install either "
            "`langchain-huggingface` or `langchain-community`."
        )
    return HFEmb(model_name=model_name)

def _as_array(vecs: List[List[float]]) -> np.ndarray:
    arr = np.asarray(vecs, dtype="float32")
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    return arr


@dataclass
class Built:
    name: str
    impl: Any           # retriever or object
    kind: str           # "keyword" | "vector" | "ml" | "colbert" | "pq" | "proxy"


class RetrieverRegistry:
    """
    Central builder & invoker for all 8 retrievers.
    - build_* methods construct and cache components
    - invoke(name, query, k) returns List[Document]
    """
    def __init__(self, documents: List[Document], config: Dict[str, Any]):
        self.documents = _sanitize_docs(documents)
        self.cfg = config.get("retrievers", {})
        self.k_default = int(self.cfg.get("top_k", 5))
        self.model_name = str(self.cfg.get("embedding_model", "sentence-transformers/all-mpnet-base-v2"))

        # caches
        self._built: Dict[str, Built] = {}
        self._emb = None
        self._faiss_vs = None
        self._bm25 = None
        self._tfidf = None
        self._svm = None
        self._colbert = None
        self._nanopq = None

        if not self.documents:
            warnings.warn("RetrieverRegistry initialized with 0 usable documents.")

    # ---------------------
    # Public API
    # ---------------------
    def list_available(self) -> List[str]:
        avail = []
        for name in ["bm25","tfidf","faiss","knn","svm","colbert","time_weighted","nanopq"]:
            try:
                if self._ensure(name):
                    avail.append(name)
            except Exception:
                pass
        return sorted(set(avail))

    def invoke(self, name: str, query: str, k: Optional[int] = None) -> List[Document]:
        k = int(k or self.k_default)
        built = self._ensure(name)
        if not built:
            return []

        if built.name in ("bm25","tfidf","svm"):
            ret = built.impl
            # langchain retrievers accept k via attribute
            if hasattr(ret, "k"):
                ret.k = k
            return ret.invoke(query)

        if built.name in ("faiss","knn"):
            vs = built.impl
            return vs.similarity_search(query, k=k)

        if built.name == "time_weighted":
            # proxy: FAISS then reorder by dia_id (descending)
            vs = self._ensure("faiss").impl if self._ensure("faiss") else None
            if not vs:
                return []
            base = vs.similarity_search(query, k=max(k * 3, k))
            return sorted(base, key=lambda d: d.metadata.get("dia_id",""), reverse=True)[:k]

        if built.name == "colbert":
            col = built.impl
            hits = col.search(query, k=k) or []
            # RAGatouille returns a list of dicts with 'doc_id' or 'document'
            out: List[Document] = []
            for h in hits:
                # Prefer doc_id mapping if present
                idx = h.get("doc_id")
                if idx is not None and 0 <= idx < len(self.documents):
                    out.append(self.documents[idx])
                else:
                    # fallback: return the text as a raw Document
                    txt = h.get("document") or h.get("content") or ""
                    out.append(Document(page_content=str(txt), metadata={"source":"colbert"}))
            return out[:k]

        if built.name == "nanopq":
            npq = built.impl  # (index, xb matrix, id_map, emb)
            return self._nanopq_search(npq, query, k)

        return []

    # ---------------------
    # Builders
    # ---------------------
    def _ensure(self, name: str) -> Optional[Built]:
        name = name.lower()
        if name in self._built:
            return self._built[name]

        # Dispatch
        builder = {
            "bm25": self._build_bm25,
            "tfidf": self._build_tfidf,
            "faiss": self._build_faiss,
            "knn": self._build_faiss,            # alias
            "svm": self._build_svm,
            "colbert": self._build_colbert,
            "time_weighted": self._build_time_weighted,
            "nanopq": self._build_nanopq,
        }.get(name)

        if not builder:
            return None
        built = builder()
        if built:
            self._built[name] = built
        return built

    def _get_emb(self):
        if self._emb is None:
            self._emb = _embedder(self.model_name)
        return self._emb

    def _build_bm25(self) -> Optional[Built]:
        if not self.documents:
            return None
        try:
            self._bm25 = BM25Retriever.from_documents(self.documents)
            return Built("bm25", self._bm25, "keyword")
        except Exception as e:
            warnings.warn(f"[bm25] unavailable: {e}")
            return None

    def _build_tfidf(self) -> Optional[Built]:
        if not self.documents:
            return None
        try:
            self._tfidf = TFIDFRetriever.from_documents(self.documents)
            return Built("tfidf", self._tfidf, "keyword")
        except Exception as e:
            warnings.warn(f"[tfidf] unavailable (maybe empty vocabulary): {e}")
            return None

    def _build_faiss(self) -> Optional[Built]:
        if not self.documents or FAISS is None:
            return None
        try:
            vs = FAISS.from_documents(self.documents, self._get_emb())
            self._faiss_vs = vs
            return Built("faiss", vs, "vector")
        except Exception as e:
            warnings.warn(f"[faiss] build failed: {e}")
            return None

    def _build_svm(self) -> Optional[Built]:
        if not self.documents or SVMRetriever is None:
            warnings.warn("[svm] not available; install langchain-community>=0.2")
            return None
        try:
            ret = SVMRetriever.from_documents(self.documents, self._get_emb(), k=self.k_default)
            self._svm = ret
            return Built("svm", ret, "ml")
        except Exception as e:
            warnings.warn(f"[svm] build failed: {e}")
            return None

    def _build_colbert(self) -> Optional[Built]:
        cfg = self.cfg.get("colbert", {}) or {}
        if not cfg.get("enabled", False):
            return None
        if RAGPretrainedModel is None:
            warnings.warn("[colbert] RAGatouille not installed. `pip install ragatouille`")
            return None
        # Prepare collection as list of strings
        texts = [(d.page_content or "") for d in self.documents]
        # Build or load index
        model_id = cfg.get("model", "colbert-ir/colbertv2.0")
        index_dir = Path(cfg.get("index_dir", "indexes/colbert"))
        index_dir.mkdir(parents=True, exist_ok=True)
        overwrite = bool(cfg.get("overwrite", False))
        try:
            rag = RAGPretrainedModel.from_pretrained(model_id)
            # Use a deterministic index name based on count + model
            index_name = f"lme_{len(texts)}_{model_id.replace('/','_')}"
            rag.index(
                collection=texts,
                index_name=index_name,
                save_index_dir=str(index_dir),
                overwrite=overwrite,
            )
            # Attach a tiny wrapper with .search(query,k) that returns doc_ids
            class _ColWrap:
                def __init__(self, rag, n):
                    self.rag = rag
                    self.n = n
                def search(self, q, k=5):
                    hits = self.rag.search(q, k=k)
                    # Attach doc_id when missing
                    out = []
                    for h in hits:
                        if "doc_id" not in h and "rank" in h:
                            # fallback: trust order; unsafe but keeps flow
                            h["doc_id"] = h.get("document_id", h.get("pid", h.get("rank", 0))) % self.n
                        out.append(h)
                    return out
            self._colbert = _ColWrap(rag, len(texts))
            return Built("colbert", self._colbert, "colbert")
        except Exception as e:
            warnings.warn(f"[colbert] build failed: {e}")
            return None

    def _build_time_weighted(self) -> Optional[Built]:
        # Proxy built on FAISS availability
        f = self._ensure("faiss")
        if not f:
            return None
        return Built("time_weighted", object(), "proxy")

    def _build_nanopq(self) -> Optional[Built]:
        cfg = self.cfg.get("nanopq", {}) or {}
        if not cfg.get("enabled", False):
            return None
        if faiss is None:
            warnings.warn("[nanopq] FAISS raw API not installed. `pip install faiss-cpu` or `faiss-gpu`")
            return None
        if not self.documents:
            return None
        try:
            emb = self._get_emb()
            texts = [d.page_content for d in self.documents]
            X = _as_array(emb.embed_documents(texts))  # (N, d)

            d = X.shape[1]
            nlist = int(cfg.get("ivf_nlist", 2048))
            m = int(cfg.get("pq_m", 64))
            opq_m = int(cfg.get("opq_m", 16))

            quantizer = faiss.IndexFlatL2(d)
            if opq_m and opq_m > 0:
                # OPQ -> IVF,PQ
                opq = faiss.OPQMatrix(d, opq_m)
                index_ivfpq = faiss.IndexIVFPQ(quantizer, d, nlist, m, 8)
                index = faiss.IndexPreTransform(opq, index_ivfpq)
            else:
                index = faiss.IndexIVFPQ(quantizer, d, nlist, m, 8)

            # Train & add
            if not index.is_trained:
                index.train(X)
            index.add(X)

            id_map = list(range(len(self.documents)))
            self._nanopq = (index, X, id_map, emb)
            return Built("nanopq", self._nanopq, "pq")
        except Exception as e:
            warnings.warn(f"[nanopq] build failed: {e}")
            return None

    # ---------------------
    # NanoPQ search
    # ---------------------
    def _nanopq_search(self, npq_tuple, query: str, k: int) -> List[Document]:
        if faiss is None:
            return []
        index, X, id_map, emb = npq_tuple
        qv = _as_array(emb.embed_query(query))  # (1, d)
        # IVF needs nprobe to control recall-speed tradeoff
        try:
            index.nprobe = min(32, index.nlist)  # type: ignore[attr-defined]
        except Exception:
            pass
        D, I = index.search(qv, k)
        out = []
        for idx in I[0]:
            if idx == -1:
                continue
            doc_id = id_map[idx]
            out.append(self.documents[doc_id])
        return out
