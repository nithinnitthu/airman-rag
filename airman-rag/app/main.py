"""
AIRMAN RAG Chat - Main FastAPI Application
Aviation Document AI with strict hallucination prevention
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import uvicorn

from app.ingest import IngestPipeline
from app.retriever import HybridRetriever
from app.generator import AnswerGenerator

app = FastAPI(
    title="AIRMAN Aviation RAG Chat",
    description="Document-grounded aviation Q&A system",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global instances
retriever: Optional[HybridRetriever] = None
generator: Optional[AnswerGenerator] = None


class IngestRequest(BaseModel):
    pdf_paths: list[str]


class AskRequest(BaseModel):
    question: str
    debug: bool = False
    top_k: int = 5


class AskResponse(BaseModel):
    answer: str
    citations: list[dict]
    retrieved_chunks: Optional[list[dict]] = None
    refusal: bool = False


@app.get("/health")
def health():
    return {
        "status": "ok",
        "index_loaded": retriever is not None,
        "documents_indexed": retriever.get_doc_count() if retriever else 0
    }


@app.post("/ingest")
def ingest(request: IngestRequest):
    global retriever, generator
    pipeline = IngestPipeline()
    chunks = pipeline.run(request.pdf_paths)
    retriever = HybridRetriever(chunks)
    generator = AnswerGenerator()
    return {
        "status": "success",
        "chunks_indexed": len(chunks),
        "files": request.pdf_paths
    }


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest):
    if retriever is None or generator is None:
        raise HTTPException(status_code=400, detail="No documents ingested yet. Call POST /ingest first.")

    # Retrieve relevant chunks
    chunks = retriever.retrieve(request.question, top_k=request.top_k)

    if not chunks:
        return AskResponse(
            answer="This information is not available in the provided document(s).",
            citations=[],
            refusal=True
        )

    # Generate grounded answer
    result = generator.generate(request.question, chunks)

    response = AskResponse(
        answer=result["answer"],
        citations=result["citations"],
        refusal=result["refusal"]
    )

    if request.debug:
        response.retrieved_chunks = [
            {
                "chunk_id": c["chunk_id"],
                "source": c["source"],
                "page": c.get("page", "N/A"),
                "score": round(c["score"], 4),
                "text": c["text"][:300] + "..." if len(c["text"]) > 300 else c["text"]
            }
            for c in chunks[:3]
        ]

    return response


if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
