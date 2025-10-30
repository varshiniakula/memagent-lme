# all_retrievers.py
"""
Per-question retriever registry over a list of LangChain Documents.
Supported names (case-insensitive):
  "bm25","tfidf","faiss","knn","svm","time_weighted","nanopq","colbert",
  "multiquery","contextual_compression","self_query"

Graceful fallbacks: if a dependency is missing, that retriever is skipped.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Tuple
import warnings
from datetime import datetime
from pathlib import Path
import numpy as np

from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever, TFIDFRetriever

# Embeddings (HF)
try:
    from langchain_huggingface import HuggingFaceEmbeddings as HFEmb
except Exception:
    try:
        from langchain_community.embeddings import HuggingFaceEmbeddings as HFEmb
    except Exception:
        HFEmb = None  # will be checked

# Vector store
try:
    from langchain_community.vectorstores import FAISS
except Exception:
    FAISS = None

# SVM
try:
    from langchain_community.retrievers import SVMRetriever
except Exception:
    SVMRetriever = None

# FAISS raw API for NanoPQ
try:
    import faiss  # type: ignore
except Exception:
    faiss = None

# RAGatouille / ColBERT
try:
    from ragatouille import RAGPretrainedModel  # type: ignore
except Exception:
    RAGPretrainedModel = None

# LLM-based retrievers (with version fallbacks)
try:
    from langchain.retrievers.multi_query import MultiQueryRetriever  # LC >= 0.2
except Exception:
    try:
        from langchain_classic.retrievers import MultiQueryRetriever  # older LC
    except Exception:
        MultiQueryRetriever = None

try:
    from langchain.retrievers import ContextualCompressionRetriever
    from langchain.retrievers.document_compressors import LLMChainExtractor
except Exception:
    try:
        from langchain_classic.retrievers import ContextualCompressionRetriever
        from langchain_classic.retrievers.document_compressors import LLMChainExtractor
    except Exception:
        ContextualCompressionRetriever = None
        LLMChainExtractor = None

try:
    from langchain.retrievers.self_query.base import SelfQueryRetriever
    from langchain.chains.query_constructor.base import AttributeInfo
except Exception:
    try:
        from langchain_classic.retrievers.self_query.base import SelfQueryRetriever
        from langchain.chains.query_constructor.base import AttributeInfo
    except Exception:
        SelfQueryRetriever = None
        AttributeInfo = None


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
            "HuggingFaceEmbeddings not available. Install `langchain-huggingface` "
            "or `langchain-community` (>=0.2)."
        )
    return HFEmb(model_name=model_name)

def _as_array(vecs: List[List[float]]) -> np.ndarray:
    arr = np.asarray(vecs, dtype="float32")
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    return arr

def _parse_dt(s: str) -> float:
    # return unix timestamp for sorting; if parse fails, 0
    if not s:
        return 0.0
    try:
        # prefer ISO from your builder; fallback to loose parsing
        return datetime.fromisoformat(str(s).replace("Z","")).timestamp()
    except Exception:
        return 0.0


@dataclass
class Built:
    name: str
    impl: Any
    kind: str  # "keyword" | "vector" | "ml" | "pq" | "colbert" | "proxy" | "llm"


class RetrieverRegistry:
    def __init__(
            self,
            documents: List[Document],
            config: Dict[str, Any],
            llm: Optional[Any] = None,
    ):
        self.documents = _sanitize_docs(documents)
        self.cfg = config.get("retrievers", {}) if isinstance(config, dict) else {}
        self.k_default = int(self.cfg.get("top_k", 5))
        self.model_name = str(self.cfg.get("embedding_model", "sentence-transformers/all-MiniLM-L6-v2"))
        self.llm = llm

        self._built: Dict[str, Built] = {}
        self._emb = None
        self._faiss_vs = None
        self._bm25 = None
        self._tfidf = None
        self._svm = None
        self._nanopq = None
        self._colbert = None
        self._multiquery = None
        self._compress = None
        self._selfquery = None

        if not self.documents:
            warnings.warn("RetrieverRegistry initialized with 0 usable documents.")

    # ---------------- Public API ----------------
    def list_available(self) -> List[str]:
        names = [
            "bm25","tfidf","faiss","knn","svm","time_weighted","nanopq","colbert",
            "multiquery","contextual_compression","self_query",
        ]
        avail = []
        for n in names:
            try:
                if self._ensure(n):
                    avail.append(n)
            except Exception:
                pass
        return sorted(set(avail))

    def invoke(self, name: str, query: str, k: Optional[int] = None) -> List[Document]:
        k = int(k or self.k_default)
        built = self._ensure(name)
        if not built:
            return []

        # LLM-based
        if built.name in ("multiquery","contextual_compression","self_query"):
            ret = built.impl
            try:
                docs = ret.invoke(query) if hasattr(ret, "invoke") else ret.get_relevant_documents(query)
                return list(docs)[:k]
            except Exception as e:
                warnings.warn(f"[{built.name}] LLM call failed: {e}. Skipping.")
                return []

        # Keyword / ML
        if built.name in ("bm25","tfidf","svm"):
            ret = built.impl
            if hasattr(ret, "k"):
                ret.k = k
            return ret.invoke(query)

        # Vector
        if built.name in ("faiss","knn"):
            return built.impl.similarity_search(query, k=k)

        # Time-weighted proxy: expand then rerank by recency (uses 'date' meta)
        if built.name == "time_weighted":
            faiss_b = self._ensure("faiss")
            if not faiss_b:
                return []
            base = faiss_b.impl.similarity_search(query, k=max(3*k, k))
            return sorted(base, key=lambda d: _parse_dt(d.metadata.get("date","")), reverse=True)[:k]

        # ColBERT (via RAGatouille)
        if built.name == "colbert":
            col = built.impl
            hits = col.search(query, k=k) or []
            out: List[Document] = []
            for h in hits:
                idx = h.get("doc_id")
                if idx is not None and 0 <= idx < len(self.documents):
                    out.append(self.documents[idx])
                else:
                    # fallback to content
                    txt = h.get("document") or h.get("content") or ""
                    out.append(Document(page_content=str(txt), metadata={"source":"colbert"}))
            return out[:k]

        # NanoPQ IVF+PQ
        if built.name == "nanopq":
            return self._nanopq_search(built.impl, query, k)

        return []

    # ---------------- Builders ----------------
    def _ensure(self, name: str) -> Optional[Built]:
        name = name.lower()
        if name in self._built:
            return self._built[name]

        builder = {
            "bm25": self._build_bm25,
            "tfidf": self._build_tfidf,
            "faiss": self._build_faiss,
            "knn": self._build_faiss,
            "svm": self._build_svm,
            "time_weighted": self._build_time_weighted,
            "nanopq": self._build_nanopq,
            "colbert": self._build_colbert,
            "multiquery": self._build_multiquery,
            "contextual_compression": self._build_contextual_compression,
            "self_query": self._build_self_query,
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
            warnings.warn(f"[tfidf] unavailable: {e}")
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
            warnings.warn("[svm] not available; requires langchain-community and scikit-learn")
            return None
        try:
            ret = SVMRetriever.from_documents(self.documents, self._get_emb(), k=self.k_default)
            self._svm = ret
            return Built("svm", ret, "ml")
        except Exception as e:
            warnings.warn(f"[svm] build failed: {e}")
            return None

    def _build_time_weighted(self) -> Optional[Built]:
        f = self._ensure("faiss")
        if not f:
            return None
        # proxy object; scoring happens in invoke
        return Built("time_weighted", object(), "proxy")

    def _build_nanopq(self) -> Optional[Built]:
        cfg = self.cfg.get("nanopq", {}) or {}
        if not cfg.get("enabled", False):
            return None
        if faiss is None:
            warnings.warn("[nanopq] requires `faiss-cpu` or `faiss-gpu`.")
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
                opq = faiss.OPQMatrix(d, opq_m)
                index_ivfpq = faiss.IndexIVFPQ(quantizer, d, nlist, m, 8)
                index = faiss.IndexPreTransform(opq, index_ivfpq)
            else:
                index = faiss.IndexIVFPQ(quantizer, d, nlist, m, 8)

            if not index.is_trained:
                index.train(X)
            index.add(X)

            id_map = list(range(len(self.documents)))
            self._nanopq = (index, id_map, emb)
            return Built("nanopq", self._nanopq, "pq")
        except Exception as e:
            warnings.warn(f"[nanopq] build failed: {e}")
            return None

    def _build_colbert(self) -> Optional[Built]:
        cfg = self.cfg.get("colbert", {}) or {}
        if not cfg.get("enabled", False):
            return None
        if RAGPretrainedModel is None:
            warnings.warn("[colbert] RAGatouille not installed. `pip install ragatouille`")
            return None
        try:
            texts = [(d.page_content or "") for d in self.documents]
            model_id = cfg.get("model", "colbert-ir/colbertv2.0")
            index_dir = Path(cfg.get("index_dir", "indexes/colbert"))
            index_dir.mkdir(parents=True, exist_ok=True)
            overwrite = bool(cfg.get("overwrite", False))

            rag = RAGPretrainedModel.from_pretrained(model_id)
            index_name = f"lme_{len(texts)}_{model_id.replace('/','_')}"
            rag.index(
                collection=texts,
                index_name=index_name,
                save_index_dir=str(index_dir),
                overwrite=overwrite,
            )

            class _ColWrap:
                def __init__(self, rag, n):
                    self.rag = rag
                    self.n = n
                def search(self, q, k=5):
                    hits = self.rag.search(q, k=k)
                    out = []
                    for i, h in enumerate(hits):
                        if "doc_id" not in h:
                            h["doc_id"] = h.get("document_id", h.get("pid", i)) % self.n
                        out.append(h)
                    return out

            self._colbert = _ColWrap(rag, len(texts))
            return Built("colbert", self._colbert, "colbert")
        except Exception as e:
            warnings.warn(f"[colbert] build failed: {e}")
            return None

    # ---------- LLM-based ----------
    def _build_multiquery(self) -> Optional[Built]:
        cfg = self.cfg.get("multiquery", {}) or {}
        if not cfg.get("enabled", True):
            return None
        if MultiQueryRetriever is None or self.llm is None:
            warnings.warn("[multiquery] requires an LLM and compatible langchain.")
            return None
        base = self._ensure("faiss") or self._ensure("bm25")
        if not base:
            warnings.warn("[multiquery] needs a base retriever (faiss or bm25).")
            return None
        try:
            base_ret = base.impl.as_retriever(search_kwargs={"k": self.k_default}) if base.name in ("faiss","knn") else base.impl
            mq = MultiQueryRetriever.from_llm(retriever=base_ret, llm=self.llm)
            self._multiquery = mq
            return Built("multiquery", mq, "llm")
        except Exception as e:
            warnings.warn(f"[multiquery] build failed: {e}")
            return None

    def _build_contextual_compression(self) -> Optional[Built]:
        cfg = self.cfg.get("contextual_compression", {}) or {}
        if not cfg.get("enabled", True):
            return None
        if ContextualCompressionRetriever is None or LLMChainExtractor is None or self.llm is None:
            warnings.warn("[contextual_compression] requires an LLM and compatible langchain.")
            return None
        base = self._ensure("faiss") or self._ensure("bm25")
        if not base:
            warnings.warn("[contextual_compression] needs a base retriever (faiss or bm25).")
            return None
        try:
            base_ret = base.impl.as_retriever(search_kwargs={"k": self.k_default}) if base.name in ("faiss","knn") else base.impl
            compressor = LLMChainExtractor.from_llm(self.llm)
            cc = ContextualCompressionRetriever(base_compressor=compressor, base_retriever=base_ret)
            self._compress = cc
            return Built("contextual_compression", cc, "llm")
        except Exception as e:
            warnings.warn(f"[contextual_compression] build failed: {e}")
            return None

    def _build_self_query(self) -> Optional[Built]:
        cfg = self.cfg.get("self_query", {}) or {}
        if not cfg.get("enabled", True):
            return None
        if SelfQueryRetriever is None or AttributeInfo is None or self.llm is None:
            warnings.warn("[self_query] requires an LLM and compatible langchain.")
            return None
        base = self._ensure("faiss")
        if not base:
            warnings.warn("[self_query] requires a vector store (faiss).")
            return None
        try:
            metadata_field_info = [
                AttributeInfo(name="session_id", description="ID of the session (evidence)", type="string"),
                AttributeInfo(name="date", description="Session timestamp (ISO string)", type="string"),
                AttributeInfo(name="session_idx", description="Index of session in haystack list", type="integer"),
            ]
            doc_desc = "Chat sessions from multi-session conversations (LongMemEval)"
            sq = SelfQueryRetriever.from_llm(
                llm=self.llm,
                vectorstore=base.impl,
                document_contents=doc_desc,
                metadata_field_info=metadata_field_info,
                verbose=False
            )
            self._selfquery = sq
            return Built("self_query", sq, "llm")
        except Exception as e:
            warnings.warn(f"[self_query] build failed: {e}")
            return None

    # ---------- NanoPQ search ----------
    def _nanopq_search(self, npq_tuple, query: str, k: int) -> List[Document]:
        if faiss is None:
            return []
        index, id_map, emb = npq_tuple
        qv = _as_array(emb.embed_query(query))  # (1, d)
        try:
            # broader search than default
            if hasattr(index, "nlist"):
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
