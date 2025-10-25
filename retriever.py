"""
Unified Retriever Class (robust)
Provides a consistent interface for all retriever types with property-based access,
and guards against empty/stopword-only corpora & missing optional deps.
"""

from typing import List, Optional, Dict, Any
from langchain_core.documents import Document

# Core vector + embeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Lexical retrievers (guard build to avoid empty-vocab errors)
from langchain_community.retrievers import BM25Retriever, TFIDFRetriever

# Optional: ML lexical
try:
    from langchain_community.retrievers import SVMRetriever
    HAS_SVM = True
except Exception:
    HAS_SVM = False

# Advanced retrievers - with compatibility checks
try:
    from langchain_classic.retrievers import EnsembleRetriever
    HAS_ENSEMBLE = True
except Exception:
    HAS_ENSEMBLE = False

try:
    from langchain_classic.retrievers import MultiQueryRetriever
    HAS_MULTIQUERY = True
except Exception:
    HAS_MULTIQUERY = False

try:
    from langchain_classic.retrievers import ContextualCompressionRetriever
    from langchain_classic.retrievers.document_compressors import LLMChainExtractor
    HAS_COMPRESSION = True
except Exception:
    HAS_COMPRESSION = False

try:
    from langchain_classic.retrievers import ParentDocumentRetriever
    from langchain_core.stores import InMemoryStore
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    HAS_PARENT = True
except Exception:
    HAS_PARENT = False

try:
    from langchain_classic.retrievers.self_query.base import SelfQueryRetriever
    from langchain.chains.query_constructor.base import AttributeInfo
    HAS_SELF_QUERY = True
except Exception:
    HAS_SELF_QUERY = False

try:
    from langchain_classic.retrievers import MultiVectorRetriever
    from langchain.storage import InMemoryStore as MVInMemoryStore
    HAS_MULTIVECTOR = True
except Exception:
    HAS_MULTIVECTOR = False

try:
    from langchain_community.document_transformers import LongContextReorder
    from langchain_classic.retrievers.document_compressors import DocumentCompressorPipeline
    HAS_REORDER = True
except Exception:
    HAS_REORDER = False


# ======================================================================
# Helpers: sanitization & safe builders
# ======================================================================
def _sanitize_documents(documents: List[Document]) -> List[Document]:
    """Keep only docs with at least one alphanumeric character."""
    safe = []
    for d in documents or []:
        text = (d.page_content or "").strip()
        if any(ch.isalnum() for ch in text):
            safe.append(d)
    return safe


def _set_k_if_supported(retriever: Any, k: int) -> None:
    """Many LC retrievers expose `k`. Set it if present."""
    try:
        if hasattr(retriever, "k"):
            retriever.k = k
    except Exception:
        pass


def _safe_build_bm25_tfidf(documents: List[Document], top_k: int) -> Dict[str, Optional[Any]]:
    """
    Try to build BM25 and TF-IDF retrievers; if they fail (e.g., empty vocabulary),
    return None entries so callers can skip gracefully.
    """
    docs = _sanitize_documents(documents)
    if not docs:
        return {"bm25": None, "tfidf": None}

    bm25 = None
    tfidf = None

    try:
        bm25 = BM25Retriever.from_documents(docs)
        _set_k_if_supported(bm25, top_k)
    except Exception:
        bm25 = None

    try:
        tfidf = TFIDFRetriever.from_documents(docs)
        _set_k_if_supported(tfidf, top_k)
    except Exception:
        # e.g., ValueError: empty vocabulary; perhaps the documents only contain stop words
        tfidf = None

    return {"bm25": bm25, "tfidf": tfidf}


# ======================================================================
# Main class
# ======================================================================
class Retriever:
    """
    Unified retriever class providing consistent access to all retriever types.

    Usage:
        r = Retriever(documents, llm=llm, config=config)
        docs = r.bm25.invoke(query)
        docs = r.faiss.invoke(query)

    All retrievers follow LangChain's standard interface:
        - invoke(query: str) -> List[Document]
        - get_relevant_documents(query: str) -> List[Document]
    """

    def __init__(
            self,
            documents: List[Document],
            llm=None,
            session_docs: Optional[List[Document]] = None,
            config: Optional[dict] = None
    ):
        """
        Args:
            documents: List of Document objects (utterance-level)
            llm: Optional LLM instance for advanced retrievers
            session_docs: Optional session-level documents for Parent Document retriever
            config: Optional configuration dict (uses defaults if None)
        """
        self.documents = _sanitize_documents(documents or [])
        self.llm = llm
        self.session_docs = _sanitize_documents(session_docs or [])
        self.config = config or self._default_config()

        # Lazy-loaded components
        self._embeddings = None
        self._vectorstore = None

        # Cached retrievers
        self._retrievers: Dict[str, Any] = {}

    # -----------------------
    # Config / lazy builders
    # -----------------------
    def _default_config(self) -> dict:
        return {
            "embedding_model": "sentence-transformers/all-mpnet-base-v2",
            "top_k": 10,
            "ensemble_weights": [0.5, 0.5],  # [vector, keyword]
            "multiquery_variations": 3,
            "parent_chunk_size": 400,
            "parent_chunk_overlap": 50,
            "mmr_diversity": 0.5,
            "mmr_fetch_k": 20,
        }

    def _get_embeddings(self):
        if self._embeddings is None:
            print(f"  Loading embeddings: {self.config['embedding_model']}")
            self._embeddings = HuggingFaceEmbeddings(
                model_name=self.config["embedding_model"]
            )
        return self._embeddings

    def _get_vectorstore(self):
        if self._vectorstore is None:
            if not self.documents:
                raise ValueError("No valid documents available to build FAISS index.")
            print("  Building FAISS vector index...")
            self._vectorstore = FAISS.from_documents(
                self.documents, self._get_embeddings()
            )
        return self._vectorstore

    # ======================================================================
    # RETRIEVER PROPERTIES - Access via retriever.bm25, retriever.faiss, etc.
    # ======================================================================
    @property
    def bm25(self):
        """BM25 Retriever - Keyword-based probabilistic ranking (guarded)."""
        if "bm25" not in self._retrievers:
            print("[BM25] Initializing keyword-based retriever...")
            built = _safe_build_bm25_tfidf(self.documents, self.config["top_k"])
            self._retrievers["bm25"] = built["bm25"]
            if self._retrievers["bm25"] is None:
                print("  [BM25] Skipped (empty or stopword-only corpus).")
        return self._retrievers.get("bm25")

    @property
    def tfidf(self):
        """TF-IDF Retriever - Traditional information retrieval (guarded)."""
        if "tfidf" not in self._retrievers:
            print("[TF-IDF] Initializing IR retriever...")
            built = _safe_build_bm25_tfidf(self.documents, self.config["top_k"])
            self._retrievers["tfidf"] = built["tfidf"]
            if self._retrievers["tfidf"] is None:
                print("  [TF-IDF] Skipped (empty vocabulary).")
        return self._retrievers.get("tfidf")

    @property
    def faiss(self):
        """FAISS Retriever - Semantic vector similarity search."""
        if "faiss" not in self._retrievers:
            print("[FAISS] Initializing semantic vector retriever...")
            vectorstore = self._get_vectorstore()
            self._retrievers["faiss"] = vectorstore.as_retriever(
                search_kwargs={"k": self.config["top_k"]}
            )
        return self._retrievers["faiss"]

    @property
    def ensemble(self):
        """Ensemble Retriever - Hybrid (FAISS + BM25) fusion (graceful fallbacks)."""
        if not HAS_ENSEMBLE:
            print("[Ensemble] Not available - install `langchain-classic`.")
            return None
        if "ensemble" not in self._retrievers:
            print("[Ensemble] Initializing hybrid retriever (FAISS + BM25)...")
            base_vector = self.faiss
            base_lex = self.bm25
            if base_vector is None and base_lex is None:
                print("  [Ensemble] Skipped (no components available).")
                self._retrievers["ensemble"] = None
            else:
                # If one side is None, EnsembleRetriever will still accept a single retriever list.
                comps = [r for r in [base_vector, base_lex] if r is not None]
                self._retrievers["ensemble"] = EnsembleRetriever(
                    retrievers=comps,
                    weights=self.config["ensemble_weights"][: len(comps)]
                )
        return self._retrievers.get("ensemble")

    @property
    def multiquery(self):
        """MultiQuery Retriever - Query expansion with LLM."""
        if not HAS_MULTIQUERY:
            print("[MultiQuery] Not available - install `langchain-classic`.")
            return None
        if self.llm is None:
            print("[MultiQuery] Not available - requires LLM.")
            return None
        if "multiquery" not in self._retrievers:
            print("[MultiQuery] Initializing query expansion retriever...")
            self._retrievers["multiquery"] = MultiQueryRetriever.from_llm(
                retriever=self.faiss, llm=self.llm
            )
        return self._retrievers["multiquery"]

    @property
    def contextual_compression(self):
        """Contextual Compression Retriever - LLM-based filtering."""
        if not HAS_COMPRESSION:
            print("[ContextualCompression] Not available - install `langchain-classic`.")
            return None
        if self.llm is None:
            print("[ContextualCompression] Not available - requires LLM.")
            return None
        if "contextual_compression" not in self._retrievers:
            print("[ContextualCompression] Initializing reranking retriever...")
            compressor = LLMChainExtractor.from_llm(self.llm)
            self._retrievers["contextual_compression"] = ContextualCompressionRetriever(
                base_compressor=compressor,
                base_retriever=self.faiss
            )
        return self._retrievers["contextual_compression"]

    @property
    def parent_document(self):
        """Parent Document Retriever - Session-level context retrieval."""
        if not HAS_PARENT:
            print("[ParentDocument] Not available - install `langchain-classic`.")
            return None
        if not self.session_docs:
            print("[ParentDocument] Not available - requires `session_docs`.")
            return None
        if "parent_document" not in self._retrievers:
            print("[ParentDocument] Initializing context-aware retriever...")
            child_splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.config["parent_chunk_size"],
                chunk_overlap=self.config["parent_chunk_overlap"],
            )
            store = InMemoryStore()
            retriever = ParentDocumentRetriever(
                vectorstore=self._get_vectorstore(),
                docstore=store,
                child_splitter=child_splitter,
                search_kwargs={"k": self.config["top_k"]},
            )
            retriever.add_documents(self.session_docs)
            self._retrievers["parent_document"] = retriever
        return self._retrievers["parent_document"]

    @property
    def self_query(self):
        """Self-Query Retriever - Metadata-aware natural language filtering."""
        if not HAS_SELF_QUERY:
            print("[SelfQuery] Not available - install `langchain-classic`.")
            return None
        if self.llm is None:
            print("[SelfQuery] Not available - requires LLM.")
            return None
        if "self_query" not in self._retrievers:
            print("[SelfQuery] Initializing metadata-aware retriever...")
            metadata_field_info = [
                AttributeInfo(
                    name="speaker",
                    description="The person speaking (e.g., A, B, Caroline, etc.)",
                    type="string",
                ),
                AttributeInfo(
                    name="session",
                    description="Conversation session (e.g., session_1)",
                    type="string",
                ),
                AttributeInfo(
                    name="dia_id",
                    description="Dialogue/utterance ID (e.g., D1:3 or T1:3)",
                    type="string",
                ),
            ]
            document_content_description = "Utterances from conversations between speakers"
            self._retrievers["self_query"] = SelfQueryRetriever.from_llm(
                llm=self.llm,
                vectorstore=self._get_vectorstore(),
                document_contents=document_content_description,
                metadata_field_info=metadata_field_info,
                verbose=True,
            )
        return self._retrievers["self_query"]

    @property
    def svm(self):
        """SVM Retriever - ML-based lexical ranking (optional)."""
        if not HAS_SVM:
            print("[SVM] Not available - install required extras.")
            return None
        if "svm" not in self._retrievers:
            print("[SVM] Initializing ML-based retriever...")
            self._retrievers["svm"] = SVMRetriever.from_documents(
                self.documents, self._get_embeddings(), k=self.config["top_k"]
            )
        return self._retrievers["svm"]

    @property
    def mmr(self):
        """MMR Retriever - Max Marginal Relevance (diversity-aware)."""
        if "mmr" not in self._retrievers:
            print("[MMR] Initializing diversity-aware retriever...")
            vectorstore = self._get_vectorstore()
            self._retrievers["mmr"] = vectorstore.as_retriever(
                search_type="mmr",
                search_kwargs={
                    "k": self.config["top_k"],
                    "fetch_k": self.config["mmr_fetch_k"],
                    "lambda_mult": self.config["mmr_diversity"],
                },
            )
        return self._retrievers["mmr"]

    @property
    def time_weighted(self):
        """
        Time-Weighted Retriever - Recency-aware retrieval (lightweight proxy).
        NOTE: This uses a standard FAISS retriever; if you want true temporal scoring,
        swap to TimeWeightedVectorStoreRetriever and add timestamps in metadata.
        """
        if "time_weighted" not in self._retrievers:
            print("[TimeWeighted] Initializing temporal-aware (proxy) retriever...")
            vectorstore = self._get_vectorstore()
            self._retrievers["time_weighted"] = vectorstore.as_retriever(
                search_kwargs={"k": self.config["top_k"]}
            )
        return self._retrievers["time_weighted"]

    @property
    def multivector(self):
        """Multi-Vector Retriever - Multiple representations per document."""
        if not HAS_MULTIVECTOR:
            print("[MultiVector] Not available - install required extras.")
            return None
        if self.llm is None:
            print("[MultiVector] Not available - requires LLM.")
            return None
        if "multivector" not in self._retrievers:
            print("[MultiVector] Initializing multi-representation retriever...")
            from uuid import uuid4

            store = MVInMemoryStore()
            id_key = "doc_id"
            vectorstore = self._get_vectorstore()

            retriever = MultiVectorRetriever(
                vectorstore=vectorstore,
                docstore=store,
                id_key=id_key,
            )

            doc_ids = [str(uuid4()) for _ in self.documents]
            retriever.vectorstore.add_documents(self.documents)
            retriever.docstore.mset(list(zip(doc_ids, self.documents)))

            print("  [MultiVector] Using original utterances as vectors.")
            self._retrievers["multivector"] = retriever
        return self._retrievers["multivector"]

    @property
    def long_context_reorder(self):
        """Long Context Reorder Retriever - Optimizes doc ordering for LLMs."""
        if not HAS_REORDER:
            print("[LongContextReorder] Not available - install required extras.")
            return None
        if "long_context_reorder" not in self._retrievers:
            print("[LongContextReorder] Initializing context-reordering retriever...")
            reordering = LongContextReorder()
            pipeline = DocumentCompressorPipeline(transformers=[reordering])
            self._retrievers["long_context_reorder"] = ContextualCompressionRetriever(
                base_compressor=pipeline,
                base_retriever=self.faiss,
            )
        return self._retrievers["long_context_reorder"]

    # ======================================================================
    # UTILITY METHODS
    # ======================================================================
    def list_available(self) -> List[str]:
        """List all available retriever names (given current deps & inputs)."""
        retrievers = []

        # Core (if docs exist)
        if self.documents:
            retrievers += ["faiss", "mmr", "time_weighted"]

            # Lexical guarded builders might still be None, so list but warn later
            retrievers += ["bm25", "tfidf"]

            if HAS_SVM:
                retrievers.append("svm")

            if HAS_ENSEMBLE:
                retrievers.append("ensemble")
            if HAS_MULTIQUERY and self.llm:
                retrievers.append("multiquery")
            if HAS_COMPRESSION and self.llm:
                retrievers.append("contextual_compression")
            if HAS_PARENT and self.session_docs:
                retrievers.append("parent_document")
            if HAS_SELF_QUERY and self.llm:
                retrievers.append("self_query")
            if HAS_MULTIVECTOR and self.llm:
                retrievers.append("multivector")
            if HAS_REORDER:
                retrievers.append("long_context_reorder")

        return retrievers

    def get(self, name: str):
        """Get a retriever by name (or None if unavailable)."""
        return getattr(self, name, None)


# Registry of retriever metadata
RETRIEVER_INFO = {
    "bm25": {"type": "keyword", "requires_llm": False, "requires_sessions": False},
    "tfidf": {"type": "keyword", "requires_llm": False, "requires_sessions": False},
    "faiss": {"type": "vector", "requires_llm": False, "requires_sessions": False},
    "ensemble": {"type": "hybrid", "requires_llm": False, "requires_sessions": False},
    "multiquery": {"type": "query_expansion", "requires_llm": True, "requires_sessions": False},
    "contextual_compression": {"type": "reranking", "requires_llm": True, "requires_sessions": False},
    "parent_document": {"type": "hierarchical", "requires_llm": False, "requires_sessions": True},
    "self_query": {"type": "metadata_aware", "requires_llm": True, "requires_sessions": False},
    "svm": {"type": "ml_lexical", "requires_llm": False, "requires_sessions": False},
    "mmr": {"type": "diversification", "requires_llm": False, "requires_sessions": False},
    "time_weighted": {"type": "temporal", "requires_llm": False, "requires_sessions": False},
    "multivector": {"type": "multi_representation", "requires_llm": True, "requires_sessions": False},
    "long_context_reorder": {"type": "reordering", "requires_llm": False, "requires_sessions": False},
}
