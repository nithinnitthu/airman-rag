"""
Hybrid Retriever — Level 2: BM25 + Vector Search + Cross-Encoder Reranker
Combines keyword (BM25) and semantic (FAISS) retrieval, then reranks with
a cross-encoder for maximum precision.
"""

import json
import logging
from typing import List, Dict

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
INDEX_PATH = "data/faiss.index"
CHUNKS_PATH = "data/chunks.json"


class HybridRetriever:
    """
    Retrieval Flow:
    1. BM25 → top-20 keyword candidates
    2. FAISS vector → top-20 semantic candidates
    3. Union of both → deduplicated candidate pool
    4. Cross-encoder reranker → top-K final results
    """

    def __init__(self, chunks: List[Dict] = None):
        self.chunks = chunks
        self.embed_model = SentenceTransformer(EMBEDDING_MODEL)
        self.reranker = CrossEncoder(RERANKER_MODEL)
        self.index = None
        self.bm25 = None

        if chunks:
            self._build_indices(chunks)
        else:
            self._load_indices()

    def _build_indices(self, chunks: List[Dict]):
        """Build FAISS and BM25 from provided chunks."""
        self.chunks = chunks
        # BM25
        tokenized = [c["text"].lower().split() for c in chunks]
        self.bm25 = BM25Okapi(tokenized)
        # FAISS
        texts = [c["text"] for c in chunks]
        embeddings = self.embed_model.encode(texts, batch_size=32, show_progress_bar=False)
        embeddings = np.array(embeddings, dtype=np.float32)
        faiss.normalize_L2(embeddings)
        self.index = faiss.IndexFlatIP(embeddings.shape[1])
        self.index.add(embeddings)
        logger.info(f"Built hybrid index: {len(chunks)} chunks")

    def _load_indices(self):
        """Load persisted indices from disk."""
        try:
            self.index = faiss.read_index(INDEX_PATH)
            with open(CHUNKS_PATH) as f:
                self.chunks = json.load(f)
            tokenized = [c["text"].lower().split() for c in self.chunks]
            self.bm25 = BM25Okapi(tokenized)
            logger.info(f"Loaded index: {len(self.chunks)} chunks")
        except FileNotFoundError:
            logger.warning("No index found. Call /ingest first.")
            self.chunks = []

    def get_doc_count(self) -> int:
        return len(self.chunks) if self.chunks else 0

    def _vector_search(self, query: str, top_k: int = 20) -> List[int]:
        """FAISS semantic search, returns chunk indices."""
        if self.index is None:
            return []
        q_emb = self.embed_model.encode([query], show_progress_bar=False)
        q_emb = np.array(q_emb, dtype=np.float32)
        faiss.normalize_L2(q_emb)
        scores, indices = self.index.search(q_emb, min(top_k, len(self.chunks)))
        return indices[0].tolist()

    def _bm25_search(self, query: str, top_k: int = 20) -> List[int]:
        """BM25 keyword search, returns chunk indices."""
        if self.bm25 is None:
            return []
        tokenized_query = query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        return top_indices

    def retrieve(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        Full hybrid retrieval pipeline with reranking.
        Returns top_k chunks with scores attached.
        """
        if not self.chunks:
            return []

        # Step 1: Candidate retrieval
        vector_ids = self._vector_search(query, top_k=20)
        bm25_ids = self._bm25_search(query, top_k=20)

        # Step 2: Union + dedup
        candidate_ids = list(dict.fromkeys(vector_ids + bm25_ids))
        candidates = [self.chunks[i] for i in candidate_ids if i < len(self.chunks)]

        if not candidates:
            return []

        # Step 3: Cross-encoder reranking
        pairs = [(query, c["text"]) for c in candidates]
        rerank_scores = self.reranker.predict(pairs)

        # Step 4: Sort and return top-K
        scored = sorted(
            zip(candidates, rerank_scores),
            key=lambda x: x[1],
            reverse=True
        )

        results = []
        for chunk, score in scored[:top_k]:
            results.append({**chunk, "score": float(score)})

        return results
