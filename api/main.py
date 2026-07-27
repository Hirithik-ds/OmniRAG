import time
import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from agents.agentic_rag import AgenticRAGGraph
from agents.compressor import ContextCompressor
from agents.router import AgenticRouter, RetrievalStrategy
from api.schemas import (
    HealthResponse, IngestResponse, QueryRequest, QueryResponse,
)
from cache.embedding_cache import cache
from config.settings import settings
from db.connection import bootstrap_schema, close_async_pool, get_async_pool
from evaluation.ragas_eval import RAGASEvaluator
from ingestion.indexer import Indexer
from observability.tracer import QueryTrace, tracer
from resilience.fallback_chain import FallbackChain
from retrieval.multi_query import MultiQueryExpander
from retrieval.reranker import Reranker

logger = logging.getLogger("omnirag.api")

SUPPORTED_EXTENSIONS = {
    ".pdf", ".txt", ".md", ".docx", ".xlsx", ".csv",
    ".pptx", ".json", ".jsonl", ".html", ".htm", ".eml",
    ".epub", ".py", ".js", ".ts", ".java", ".go",
    ".rst", ".xml", ".yaml", ".yml",
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # ── Startup ───────────────────────────────────────────────────────────────
    bootstrap_schema()       # create Postgres tables if not exist
    await get_async_pool()   # warm up async connection pool
    logger.info("OmniRAG API started.")
    yield
    # ── Shutdown ──────────────────────────────────────────────────────────────
    await close_async_pool()
    logger.info("OmniRAG API stopped.")


app = FastAPI(
    title="OmniRAG API",
    description=(
        "Production-grade adaptive RAG — "
        "Hybrid Search · Graph RAG · Agentic RAG · "
        "Multimodal Ingestion · Postgres · HuggingFace"
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Initialise all components once at startup ─────────────────────────────────
router_llm  = AgenticRouter()
rag_agent   = AgenticRAGGraph()
expander    = MultiQueryExpander()
reranker    = Reranker()
compressor  = ContextCompressor()
evaluator   = RAGASEvaluator()
indexer     = Indexer()
fallback    = FallbackChain()


# ═════════════════════════════════════════════════════════════════════════════
# QUERY ENDPOINT — 8-stage pipeline
# ═════════════════════════════════════════════════════════════════════════════

@app.post("/query", response_model=QueryResponse)
async def query(req: QueryRequest):
    total_start = time.time()
    trace = QueryTrace(
        query=req.query,
        strategy_requested=req.strategy or "auto",
    )

    # ── Stage 0: Cache check ──────────────────────────────────────────────────
    cache_key = f"{req.query}::{req.strategy}::{req.element_type_filter}::{req.source_filter}"
    cached_response = cache.get("response", cache_key)
    if cached_response:
        trace.cache_hit = True
        trace.cache_hit_type = "response"
        trace.latency_total_ms = round((time.time() - total_start) * 1000, 1)
        tracer.persist(trace)
        return QueryResponse(**cached_response)

    # ── Stage 1: Route ────────────────────────────────────────────────────────
    with tracer.trace_stage(trace, "route"):
        if req.strategy == "auto":
            routing  = router_llm.route_with_reason(req.query)
            strategy = routing["strategy"]
            reason   = routing["reason"]
        else:
            try:
                strategy = RetrievalStrategy(req.strategy)
            except ValueError:
                strategy = RetrievalStrategy.HYBRID
            reason = "Manually specified"
    trace.strategy_used = strategy.value

    # ── Stage 2: Multi-query expansion ───────────────────────────────────────
    with tracer.trace_stage(trace, "expand"):
        queries = expander.expand(req.query)
    trace.num_queries_expanded = len(queries)

    # ── Stage 3: Retrieve with fallback chain ─────────────────────────────────
    with tracer.trace_stage(trace, "retrieve"):
        fallback_result = fallback.retrieve_with_fallback(
            query=req.query,
            preferred_strategy=strategy.value,
            queries=queries,
            source_filter=req.source_filter,
        )
    trace.chunks_retrieved    = len(fallback_result.chunks)
    trace.fallback_triggered  = fallback_result.fallback_triggered
    trace.fallback_errors     = fallback_result.error_chain
    trace.strategy_used       = fallback_result.strategy_used

    # ── Stage 4: Rerank ───────────────────────────────────────────────────────
    with tracer.trace_stage(trace, "rerank"):
        reranked = reranker.rerank(req.query, fallback_result.chunks)
    tracer.compute_retrieval_quality(trace, reranked)

    # ── Stage 5: Compress ─────────────────────────────────────────────────────
    with tracer.trace_stage(trace, "compress"):
        raw_context = " ".join(c["text"] for c in reranked)
        trace.context_tokens_before = len(raw_context.split())
        compressed_context = compressor.compress(req.query, reranked)
        trace.context_tokens_after = len(compressed_context.split())
        if trace.context_tokens_before > 0:
            trace.compression_ratio = round(
                trace.context_tokens_after / trace.context_tokens_before, 3
            )

    # ── Stage 6: Generate ─────────────────────────────────────────────────────
    with tracer.trace_stage(trace, "generate"):
        if not reranked:
            # Direct LLM fallback — no context available
            from langchain_huggingface import HuggingFaceEndpoint
            llm = HuggingFaceEndpoint(
                repo_id=settings.GENERATOR_MODEL,
                task="text-generation",
                huggingfacehub_api_token=settings.HF_TOKEN,
                max_new_tokens=512,
                temperature=0.1,
            )
            answer  = llm.invoke(
                f"Answer this question from your general knowledge "
                f"(note: no document context available): {req.query}"
            )
            sources = []
        elif strategy == RetrievalStrategy.AGENTIC:
            result  = rag_agent.run(req.query, strategy.value, queries)
            answer  = result["answer"]
            sources = result.get("sources", [])
        else:
            from generation.generator import Generator
            gen    = Generator()
            answer = gen.generate(req.query, compressed_context)
            sources = list({
                c.get("source", "") for c in reranked if c.get("source")
            })

    trace.answer_length = len(answer.split())
    trace.sources = sources

    # ── Stage 7: Evaluate (RAGAS) ─────────────────────────────────────────────
    contexts     = [c["text"] for c in reranked[:3]]
    ragas_scores = evaluator.evaluate_response(
        query=req.query,
        answer=answer,
        contexts=contexts,
    )
    trace.ragas_faithfulness = ragas_scores.get("faithfulness", 0.0)
    trace.ragas_relevancy    = ragas_scores.get("answer_relevancy", 0.0)
    trace.ragas_precision    = ragas_scores.get("context_precision", 0.0)

    # ── Stage 8: Persist trace to Postgres ────────────────────────────────────
    trace.latency_total_ms = round((time.time() - total_start) * 1000, 1)
    tracer.persist(trace)

    response_data = dict(
        answer=answer,
        strategy=trace.strategy_used,
        strategy_reason=reason,
        sources=sources,
        chunks_retrieved=trace.chunks_after_rerank,
        ragas_scores=ragas_scores,
        latency_ms=trace.latency_total_ms,
        cache_hit=False,
        fallback_triggered=trace.fallback_triggered,
        compression_ratio=trace.compression_ratio,
        num_queries_expanded=trace.num_queries_expanded,
    )

    # Cache the full response for future identical queries
    cache.set("response", cache_key, response_data)

    return QueryResponse(**response_data)


# ═════════════════════════════════════════════════════════════════════════════
# INGESTION ENDPOINT
# ═════════════════════════════════════════════════════════════════════════════

@app.post("/ingest/file", response_model=IngestResponse)
async def ingest_file(file: UploadFile = File(...)):
    ext = Path(file.filename).suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unsupported format: {ext}. "
                f"Supported: {sorted(SUPPORTED_EXTENSIONS)}"
            ),
        )
    content = await file.read()
    tmp_path = f"/tmp/{file.filename}"
    with open(tmp_path, "wb") as f:
        f.write(content)
    indexer.index_file(tmp_path)
    return IngestResponse(
        status="indexed",
        filename=file.filename,
        format=ext,
    )


# ── Health ────────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse)
async def health():
    try:
        from qdrant_client import QdrantClient
        QdrantClient(
            host=settings.QDRANT_HOST, port=settings.QDRANT_PORT
        ).get_collections()
        qdrant_ok = True
    except Exception:
        qdrant_ok = False

    return HealthResponse(
        status="ok",
        qdrant=qdrant_ok,
        models_loaded=True,
        cache_hit_rate=cache.hit_rate(),
    )


# ── Observability endpoints ───────────────────────────────────────────────────

@app.get("/observability/dashboard")
async def obs_dashboard():
    return {**tracer.get_dashboard_data(), "cache": cache.info()}


@app.get("/observability/traces")
async def obs_traces(limit: int = 50):
    return tracer.get_recent_traces(limit=limit)


@app.get("/documents")
async def list_documents():
    """
    Return the distinct source filenames currently in the vector store, so
    the UI can offer a "scope to one document" dropdown. Uses Qdrant scroll
    to collect unique 'source' payload values.
    """
    try:
        from qdrant_client import QdrantClient
        client = QdrantClient(host=settings.QDRANT_HOST, port=settings.QDRANT_PORT)
        sources = set()
        next_page = None
        while True:
            points, next_page = client.scroll(
                collection_name=settings.QDRANT_COLLECTION,
                limit=256,
                offset=next_page,
                with_payload=["source"],
                with_vectors=False,
            )
            for p in points:
                s = (p.payload or {}).get("source")
                if s:
                    sources.add(s)
            if next_page is None:
                break
        return {"documents": sorted(sources)}
    except Exception as e:
        logger.error(f"/documents failed: {e}")
        return {"documents": []}


@app.get("/observability/strategy-breakdown")
async def strategy_breakdown():
    return tracer.get_strategy_breakdown()


@app.get("/observability/latency-trend")
async def latency_trend(limit: int = 50):
    return tracer.get_latency_trend(limit=limit)


# ── Evaluation endpoints ──────────────────────────────────────────────────────

@app.get("/eval/stats")
async def eval_stats():
    return evaluator.logger.get_averages()


@app.get("/eval/history")
async def eval_history(limit: int = 100):
    return evaluator.logger.get_all(limit=limit)