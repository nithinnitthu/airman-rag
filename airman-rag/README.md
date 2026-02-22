# ✈️ AIRMAN — Aviation RAG Chat System

**AI/ML Intern Technical Assignment — Document-Driven RAG**  
Built for [AIRMAN](https://www.theairman.org/) | Aviation AI that answers **strictly** from provided documents.

---

## 🏗️ Architecture Overview

```
User Question
     │
     ▼
┌─────────────────────────────────────────────────────────┐
│                    FastAPI /ask                          │
│                                                         │
│  ┌─────────────────────────────────────────────────┐   │
│  │           Hybrid Retriever (Level 2)             │   │
│  │                                                  │   │
│  │  BM25 Keyword Search ──┐                        │   │
│  │                         ├─→ Candidate Pool       │   │
│  │  FAISS Vector Search ──┘                        │   │
│  │                                                  │   │
│  │  Cross-Encoder Reranker → Top-K Chunks          │   │
│  └─────────────────────────────────────────────────┘   │
│                     │                                   │
│                     ▼                                   │
│  ┌─────────────────────────────────────────────────┐   │
│  │           Answer Generator (OpenAI)              │   │
│  │                                                  │   │
│  │  Grounded Prompt → Answer + Citations            │   │
│  │  Hallucination Check → Refusal if unsupported   │   │
│  └─────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────┘
     │
     ▼
Answer + Citations + (Debug: Retrieved Chunks)
```

---

## 📁 Project Structure

```
airman-rag/
├── app/
│   ├── __init__.py
│   ├── main.py          # FastAPI app + endpoints
│   ├── ingest.py        # PDF loading + chunking + FAISS indexing
│   ├── retriever.py     # Hybrid BM25 + Vector + Reranker (Level 2)
│   └── generator.py     # Grounded LLM answer generation
├── tests/
│   └── test_api.py      # API tests
├── data/                # Auto-created: FAISS index + chunks.json
├── aviation_docs/       # Put your aviation PDFs here
├── ingest.py            # Standalone ingestion script
├── evaluate.py          # 50-question evaluation runner
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── .env.example
└── README.md
```

---

## ⚡ Quick Start

### 1. Clone & Setup

```bash
git clone <your-repo-url>
cd airman-rag

python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure Environment

```bash
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY
```

### 3. Add Aviation PDFs

```bash
mkdir -p aviation_docs
# Copy your PPL/CPL/ATPL PDFs into aviation_docs/
```

### 4. Ingest Documents

```bash
# Option A: Standalone script
python ingest.py aviation_docs/ppl_manual.pdf aviation_docs/atpl_sop.pdf

# Option B: Via API (after starting server)
curl -X POST http://localhost:8000/ingest \
  -H "Content-Type: application/json" \
  -d '{"pdf_paths": ["aviation_docs/ppl_manual.pdf"]}'
```

### 5. Start the API

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### 6. Ask Questions

```bash
# Simple question
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the Dry Adiabatic Lapse Rate?", "debug": false}'

# With debug (shows top 3 retrieved chunks)
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the DALR?", "debug": true}'
```

---

## 🐳 Docker Setup

```bash
# Build and run
docker-compose up --build

# Ingest docs inside container
docker exec airman-rag python ingest.py aviation_docs/your_manual.pdf
```

---

## 📡 API Reference

### `GET /health`
```json
{
  "status": "ok",
  "index_loaded": true,
  "documents_indexed": 1247
}
```

### `POST /ingest`
```json
// Request
{"pdf_paths": ["path/to/doc.pdf", "path/to/doc2.pdf"]}

// Response
{"status": "success", "chunks_indexed": 1247, "files": [...]}
```

### `POST /ask`
```json
// Request
{
  "question": "What is the primary objective of Air Traffic Services?",
  "debug": true,
  "top_k": 5
}

// Response (answered)
{
  "answer": "The primary objective of Air Traffic Services is to prevent collisions between aircraft.",
  "citations": [
    {
      "document": "atpl_manual",
      "page": 42,
      "chunk_id": "a3f9",
      "snippet": "Air Traffic Services are established with the primary objective..."
    }
  ],
  "retrieved_chunks": [...],  // Only when debug=true
  "refusal": false
}

// Response (refused — off-topic or not in docs)
{
  "answer": "This information is not available in the provided document(s).",
  "citations": [],
  "refusal": true
}
```

---

## 📊 Chunking Strategy

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Chunk size | 512 chars (~100-130 tokens) | Aviation manuals are dense; balances context richness with retrieval precision |
| Overlap | 64 chars | Prevents information loss at chunk boundaries |
| Boundary detection | Sentence-aware | Chunks end at sentence boundaries where possible |
| Min chunk size | 50 chars | Filters out page headers, footers, page numbers |

---

## 🔬 Level 2: Hybrid Retrieval

The retrieval pipeline combines three stages:

```
Query
  │
  ├── BM25 Keyword Search → 20 candidates (exact term matching)
  │
  ├── FAISS Vector Search → 20 candidates (semantic similarity)
  │
  ├── Union + Deduplication → ~30-35 unique candidates
  │
  └── Cross-Encoder Reranker → Top-5 final chunks
```

**Why hybrid beats single-method:**
- BM25 catches exact aviation terms: "RVSM", "ILS", "QNH", "DALR"
- Vector search catches paraphrases: "airspeed indicator" vs "ASI"
- Cross-encoder reranks by deep query-document relevance

**Models used:**
- Embeddings: `all-MiniLM-L6-v2` (fast, 384-dim)
- Reranker: `cross-encoder/ms-marco-MiniLM-L-6-v2`

---

## 📋 Running Evaluation

```bash
# Start the API first, then:
python evaluate.py

# Output:
# - evaluation_results.json (raw results per question)
# - report.md (formatted evaluation report)
```

The evaluation covers 50 questions:
- **20 Simple Factual** — definitions, direct lookups
- **20 Applied/Scenario** — operational, procedural
- **10 Higher-Order Reasoning** — multi-step, trade-offs
- Includes **4 off-topic trap questions** to test refusal behavior

---

## 🧪 Tests

```bash
pytest tests/ -v
```

---

## ⚠️ Hallucination Control

The system enforces strict grounding:

1. **System prompt** explicitly forbids using external knowledge
2. **Temperature = 0.0** for deterministic, conservative answers
3. **Refusal phrase** is exact: `"This information is not available in the provided document(s)."`
4. **Off-topic detection**: Questions about geography, art, history → auto-refused
5. **Citation requirement**: Every answer must trace to a specific document + page

---

## 🔧 Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `OPENAI_API_KEY` | ✅ Yes | — | Your OpenAI API key |
| `OPENAI_MODEL` | No | `gpt-4o-mini` | LLM model to use |
| `API_BASE` | No | `http://localhost:8000` | Base URL for evaluate.py |
| `LOG_LEVEL` | No | `INFO` | Logging verbosity |

---

## 📦 Tech Stack

| Component | Library |
|-----------|---------|
| API Framework | FastAPI |
| PDF Extraction | PyMuPDF (fitz) |
| Embeddings | sentence-transformers (all-MiniLM-L6-v2) |
| Vector Index | FAISS (faiss-cpu) |
| Keyword Search | rank-bm25 |
| Reranker | cross-encoder/ms-marco-MiniLM-L-6-v2 |
| LLM | OpenAI GPT-4o-mini |
| Containerization | Docker + docker-compose |

---

## 📽️ Demo Video Outline (5–8 min)

1. **System Architecture** (1 min) — explain the hybrid RAG pipeline
2. **Ingestion** (1 min) — run `ingest.py`, show chunks created
3. **Live Q&A** (2 min) — ask 5 questions with debug mode on
4. **Refusal Demo** (1 min) — show off-topic questions being refused
5. **Evaluation** (1–2 min) — run `evaluate.py`, show metrics
6. **Level 2 Comparison** (1 min) — explain BM25 + vector + reranker improvement

---

*Submitted by: [Your Name] | AIRMAN AI/ML Intern Assignment*
