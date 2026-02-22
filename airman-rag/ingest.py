#!/usr/bin/env python3
"""
ingest.py — Standalone ingestion script
Usage: python ingest.py <pdf1> [pdf2] [pdf3]
Builds FAISS index and saves chunks metadata to data/
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.ingest import IngestPipeline

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python ingest.py <pdf_path> [pdf_path2 ...]")
        print("Example: python ingest.py data/ppl_manual.pdf data/atpl_sop.pdf")
        sys.exit(1)

    pdf_paths = sys.argv[1:]
    for path in pdf_paths:
        if not os.path.exists(path):
            print(f"ERROR: File not found: {path}")
            sys.exit(1)

    print(f"Ingesting {len(pdf_paths)} PDF(s)...")
    pipeline = IngestPipeline()
    chunks = pipeline.run(pdf_paths)

    print(f"\n✅ Ingestion Complete!")
    print(f"   Files processed: {len(pdf_paths)}")
    print(f"   Chunks indexed : {len(chunks)}")
    print(f"   Index saved to : data/faiss.index")
    print(f"   Metadata saved : data/chunks.json")
    print(f"\nNow start the API: uvicorn app.main:app --reload")
