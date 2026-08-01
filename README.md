
# OmniRAG — Production-Grade Adaptive RAG System

> Hybrid Search · Graph RAG · Agentic RAG · Multimodal Ingestion · Multi-Document Scoping · Zero API Cost

OmniRAG is a production-ready Retrieval-Augmented Generation system that
**dynamically routes each query to the optimal retrieval strategy** using an
LLM router, then applies cross-encoder reranking, context compression, and
automatic evaluation — all running free on HuggingFace models.

<img width="100%" alt="OmniRAG chat interface answering questions with strategy routing" src="https://github.com/user-attachments/assets/56ff6e32-73c4-430a-909e-60154b5fd453" />

---

## Architecture

```
User query
    │
    ├── Cache lookup (LRU in-memory, SHA-256 key, 1hr TTL)
    │       └── HIT → return instantly
    │
    ├── Agentic router (Phi-3-mini, HF API)
    │       └── hybrid | graph | agentic
    │
    ├── Multi-query expansion (5 variants, Phi-3-mini)
    │
    ├── Fallback chain (auto-degrades on failure or empty result)
    │       ├── Agentic RAG  (LangGraph + Llama-3.1-8B)
    │       ├── Graph RAG    (GLiNER NER + Kuzu Cypher)
    │       ├── Hybrid search (Qdrant dense + BM25s sparse → RRF)
    │       └── Direct LLM   (last resort, no context)
    │
    ├── Cross-encoder reranker (BAAI/bge-reranker-large, local)
    ├── Context compression  (LLMLingua-2, local)
    ├── Answer generation    (Llama-3.1-8B-Instruct, HF API)
    ├── RAGAS evaluation     (faithfulness · relevancy · precision)
    └── Observability trace  (all stages → Postgres)
```

Every retrieval strategy also honours an optional **document-scope filter**, so
a query can be restricted to a single source in a multi-document knowledge base.

## Stack

| Component | Technology | Runs |
|---|---|---|
| Embeddings | BAAI/bge-large-en-v1.5 | Local CPU/MPS |
| Reranker | BAAI/bge-reranker-large | Local CPU/MPS |
| NER | GLiNER medium-v2 | Local CPU/MPS |
| Compression | LLMLingua-2 | Local CPU/MPS |
| Image captioning | BLIP-2 (Salesforce) | Local CPU/MPS |
| Router LLM | Phi-3-mini-4k-instruct | HF Inference API |
| Generator LLM | Llama-3.1-8B-Instruct | HF Inference API |
| Vector store | Qdrant (self-hosted) | Docker |
| Sparse search | BM25s | In-process |
| Graph DB | Kuzu (embedded) | In-process |
| Persistence | Postgres | Local or Docker |
| API | FastAPI + asyncpg | Python |
| Frontend | Streamlit + Plotly | Python |

**Total API cost: ₹0** — only generative LLM calls hit HF Inference API.

---

## Demo

### Adaptive routing on factual questions

The router selects a strategy per query and every answer is grounded in a cited
source.

<img width="100%" alt="Factual questions routed to hybrid retrieval with grounded answers" src="https://github.com/user-attachments/assets/da621f64-a07a-4099-95a7-d65b60eed5be" />

### Multi-document knowledge base with source scoping

The sidebar **Document** filter restricts retrieval to a single source, so a
large document cannot drown out a small one.

<img width="100%" alt="Document filter listing indexed sources for per-query scoping" src="https://github.com/user-attachments/assets/edad8b07-c785-4272-a100-96b759de9837" />

### Observability dashboard

Every query is traced end-to-end — per-stage latency, cache hit rate, fallback
rate, rerank score, and routing breakdown.

<img width="100%" alt="Pipeline observability tab showing per-stage latency and aggregate metrics" src="https://github.com/user-attachments/assets/127e6d05-265a-4f82-87d3-4f22aeff8c35" />

<img width="100%" alt="Quality tab showing routing breakdown and score trends" src="https://github.com/user-attachments/assets/96bafb36-5e17-4647-80b5-69ba0ad25911" />

---

## Quick start

### 1. Prerequisites

```bash
# System packages (Ubuntu/Debian)
apt-get install -y tesseract-ocr poppler-utils libmagic1

# macOS
brew install tesseract poppler libmagic
```

### 2. Clone and install

```bash
git clone https://github.com/yourname/omnirag
cd omnirag
pip install -r requirements.txt
```

### 3. Configure

```bash
cp .env.example .env
# Edit .env and set HF_TOKEN=hf_your_token_here
# The token needs "Make calls to Inference Providers" permission.
# Get a free token at https://huggingface.co/settings/tokens
```

### 4. Start infrastructure

```bash
docker-compose up -d          # starts Qdrant on port 6333
curl http://localhost:6333/collections   # verify it's up
```

Postgres can run in Docker or as a local install; set the connection details in
`.env`. (This project uses a local Postgres, so `docker-compose` only manages
Qdrant.)

### 5. Index documents

```bash
python -c "
from ingestion.indexer import Indexer
idx = Indexer()
idx.index_directory('./sample_docs')
"
```

### 6. Start the API

```bash
uvicorn api.main:app --port 8010
# API docs at http://localhost:8010/docs
```

### 7. Start the dashboard

```bash
streamlit run frontend/app.py
# Dashboard at http://localhost:8501
```

---

## Supported document formats

PDF, DOCX, XLSX, CSV, PPTX, TXT, MD, HTML, JSON, JSONL,
XML, RST, EPUB, EML, and source code files (.py .js .ts .java .go).

Images inside PDFs and DOCX are automatically captioned by BLIP-2.
Tables are extracted as structured markdown and never split across chunk boundaries.

---

## API reference

### POST /query

```json
{
  "query": "What were the key revenue drivers in Q3?",
  "strategy": "auto",
  "element_type_filter": null,
  "source_filter": null
}
```

`source_filter` restricts retrieval to a single indexed document (by filename);
omit it or set `null` to search the whole knowledge base.

Response includes: `answer`, `strategy`, `strategy_reason`, `sources`,
`chunks_retrieved`, `ragas_scores`, `latency_ms`, `cache_hit`,
`fallback_triggered`, `compression_ratio`.

### POST /ingest/file

Upload any supported file for indexing. Multipart form upload.

### GET /documents

Lists the distinct source documents currently in the index (powers the UI
document-scope filter).

### GET /health

Returns Qdrant status, models loaded, cache hit rate.

### GET /observability/dashboard

Aggregate metrics: avg latency per stage, cache hit rate, fallback rate, RAGAS averages.

### GET /observability/traces?limit=50

Recent query traces with full per-stage timing.

### GET /observability/strategy-breakdown

Per-strategy counts, avg faithfulness, avg latency.

### GET /eval/stats

RAGAS averages across all queries.

---

## Evaluation

OmniRAG was evaluated against two independent source documents — an environmental
report and an SEC financial bulletin — to test retrieval quality and the adaptive
router across heterogeneous content. See [`BENCHMARK.md`](BENCHMARK.md) for the
full question-by-question results, and [`TEST_QUESTIONS.md`](TEST_QUESTIONS.md)
for the reproducible demo script.

Key findings: factual, multi-fact, and relationship questions answer correctly;
the generator never fabricated answers on unanswerable questions; and two fixes
made during evaluation (reranker top-k tuning and a cross-document scope filter)
are documented with their reasoning. The one known limitation — layout-aware
extraction of figures and multi-column pages — is described in `BENCHMARK.md`.

---

## Running tests

```bash
pytest tests/ -v
```

No API calls are made during tests — all LLM and retriever calls are mocked.

---

## Project structure

```
omnirag/
├── config/          # settings.py — single source of truth
├── db/              # Postgres pool + schema bootstrap
├── ingestion/       # document parsing, chunking, embedding, indexing
├── retrieval/       # dense, sparse, hybrid, graph, reranker, multi-query
├── cache/           # LRU in-memory cache (swap backend to Redis for prod)
├── resilience/      # fallback chain with auto-degradation
├── observability/   # per-query structured traces to Postgres
├── agents/          # router, agentic RAG graph, context compressor
├── generation/      # Llama-3.1-8B generator with citation injection
├── evaluation/      # RAGAS pipeline + Postgres logger
├── api/             # FastAPI endpoints + Pydantic schemas
├── frontend/        # Streamlit dashboard (3 tabs)
└── tests/           # pytest unit tests, all mocked
```

---
## 🎥 Demo

[Watch Demo](https://github.com/user-attachments/assets/dc4425f2-ac5c-492a-b4d8-96a5cfac36de)

## License

MIT
