"""
Ingestion Pipeline — Loads PDFs, chunks text, generates embeddings, stores in FAISS
Chunking Strategy:
  - Chunk size: 512 tokens (~400 words)
  - Overlap: 64 tokens (~50 words)
  - Rationale: Aviation manuals have dense paragraphs; 512 balances context
    richness with retrieval precision. Overlap preserves cross-boundary facts.
"""

import os
import re
import uuid
import json
import logging
from pathlib import Path
from typing import List, Dict

import fitz  # PyMuPDF
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

CHUNK_SIZE = 512       # characters (approx 100-130 tokens)
CHUNK_OVERLAP = 64     # characters
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
INDEX_PATH = "data/faiss.index"
CHUNKS_PATH = "data/chunks.json"
EMBED_DIM = 384


class IngestPipeline:
    def __init__(self):
        logger.info("Loading embedding model...")
        self.model = SentenceTransformer(EMBEDDING_MODEL)

    def load_pdf(self, pdf_path: str) -> List[Dict]:
        """Extract text page-by-page from PDF using PyMuPDF."""
        doc = fitz.open(pdf_path)
        pages = []
        source_name = Path(pdf_path).stem

        for page_num, page in enumerate(doc, start=1):
            text = page.get_text("text")
            # Clean: remove excessive whitespace, keep structure
            text = re.sub(r'\n{3,}', '\n\n', text)
            text = re.sub(r'[ \t]+', ' ', text)
            text = text.strip()
            if text:
                pages.append({
                    "source": source_name,
                    "page": page_num,
                    "text": text
                })

        logger.info(f"Loaded {len(pages)} pages from {pdf_path}")
        return pages

    def chunk_pages(self, pages: List[Dict]) -> List[Dict]:
        """
        Sliding window chunking with overlap.
        Works at the full-document level (not per-page) to avoid
        losing context at page boundaries.
        """
        chunks = []

        for page_data in pages:
            text = page_data["text"]
            source = page_data["source"]
            page = page_data["page"]

            start = 0
            while start < len(text):
                end = start + CHUNK_SIZE
                chunk_text = text[start:end]

                # Try to end at sentence boundary
                if end < len(text):
                    last_period = chunk_text.rfind('.')
                    if last_period > CHUNK_SIZE // 2:
                        chunk_text = chunk_text[:last_period + 1]

                chunk_text = chunk_text.strip()
                if len(chunk_text) > 50:  # skip tiny fragments
                    chunks.append({
                        "chunk_id": str(uuid.uuid4())[:8],
                        "source": source,
                        "page": page,
                        "text": chunk_text
                    })

                step = len(chunk_text) - CHUNK_OVERLAP
                start += max(step, CHUNK_SIZE // 4)

        logger.info(f"Generated {len(chunks)} chunks")
        return chunks

    def embed_chunks(self, chunks: List[Dict]) -> np.ndarray:
        """Generate embeddings for all chunks."""
        texts = [c["text"] for c in chunks]
        logger.info(f"Embedding {len(texts)} chunks...")
        embeddings = self.model.encode(texts, show_progress_bar=True, batch_size=32)
        return np.array(embeddings, dtype=np.float32)

    def build_faiss_index(self, embeddings: np.ndarray) -> faiss.Index:
        """Build FAISS index with cosine similarity (via L2 on normalized vectors)."""
        faiss.normalize_L2(embeddings)
        index = faiss.IndexFlatIP(EMBED_DIM)  # Inner product = cosine after normalization
        index.add(embeddings)
        logger.info(f"FAISS index built with {index.ntotal} vectors")
        return index

    def save(self, index: faiss.Index, chunks: List[Dict]):
        """Persist index and chunk metadata."""
        os.makedirs("data", exist_ok=True)
        faiss.write_index(index, INDEX_PATH)
        with open(CHUNKS_PATH, "w") as f:
            json.dump(chunks, f, indent=2)
        logger.info(f"Saved index to {INDEX_PATH} and chunks to {CHUNKS_PATH}")

    def run(self, pdf_paths: List[str]) -> List[Dict]:
        """Full pipeline: load → chunk → embed → index → save."""
        all_pages = []
        for path in pdf_paths:
            all_pages.extend(self.load_pdf(path))

        chunks = self.chunk_pages(all_pages)
        embeddings = self.embed_chunks(chunks)
        index = self.build_faiss_index(embeddings)
        self.save(index, chunks)
        return chunks


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python -m app.ingest <pdf1> [pdf2] ...")
        sys.exit(1)
    pipeline = IngestPipeline()
    pipeline.run(sys.argv[1:])
    print("Ingestion complete.")
