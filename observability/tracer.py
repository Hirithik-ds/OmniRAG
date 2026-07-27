import json
import logging
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from db.connection import get_sync_conn

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("omnirag")


@dataclass
class QueryTrace:
    """
    Full structured trace for one query end-to-end.
    Every field is populated by a specific pipeline stage in api/main.py.
    The entire object is persisted to Postgres in one INSERT at the end.
    """
    query: str
    strategy_requested: str = ""
    strategy_used: str = ""
    fallback_triggered: bool = False
    fallback_errors: List[str] = field(default_factory=list)
    cache_hit: bool = False
    cache_hit_type: str = ""
    num_queries_expanded: int = 1
    chunks_retrieved: int = 0
    chunks_after_rerank: int = 0
    top_rerank_score: float = 0.0
    avg_rerank_score: float = 0.0
    context_tokens_before: int = 0
    context_tokens_after: int = 0
    compression_ratio: float = 1.0
    answer_length: int = 0
    sources: List[str] = field(default_factory=list)
    latency_route_ms: float = 0.0
    latency_expand_ms: float = 0.0
    latency_retrieve_ms: float = 0.0
    latency_rerank_ms: float = 0.0
    latency_compress_ms: float = 0.0
    latency_generate_ms: float = 0.0
    latency_total_ms: float = 0.0
    ragas_faithfulness: float = 0.0
    ragas_relevancy: float = 0.0
    ragas_precision: float = 0.0
    error: Optional[str] = None


class ObservabilityTracer:
    """
    Structured per-step tracer for every query.

    Usage in api/main.py:
        trace = QueryTrace(query=req.query)
        with tracer.trace_stage(trace, "retrieve"):
            chunks = retriever.retrieve(query)
        # latency_retrieve_ms is now populated on the trace object

    After all stages:
        tracer.persist(trace)  # one INSERT to Postgres

    The Streamlit dashboard reads from Postgres via get_dashboard_data()
    and get_recent_traces() — no direct coupling to the pipeline.
    """

    @contextmanager
    def trace_stage(self, trace: QueryTrace, stage: str):
        """
        Context manager that times any pipeline stage.
        On exception: records error on trace, re-raises.
        On success: sets latency_{stage}_ms on trace.
        """
        start = time.time()
        try:
            yield
        except Exception as e:
            trace.error = f"{stage}: {type(e).__name__}: {str(e)[:200]}"
            logger.error(f"Stage '{stage}' failed: {e}", exc_info=True)
            raise
        finally:
            elapsed = round((time.time() - start) * 1000, 1)
            setattr(trace, f"latency_{stage}_ms", elapsed)
            logger.info(f"Stage '{stage}' completed in {elapsed}ms")

    def compute_retrieval_quality(
        self, trace: QueryTrace, reranked_chunks: List[Dict]
    ):
        """Compute proxy retrieval quality metrics from reranked results."""
        if not reranked_chunks:
            return
        scores = [c.get("rerank_score", 0.0) for c in reranked_chunks]
        trace.chunks_after_rerank = len(reranked_chunks)
        trace.top_rerank_score = round(max(scores), 4)
        trace.avg_rerank_score = round(sum(scores) / len(scores), 4)

    def persist(self, trace: QueryTrace):
        """Write completed trace to Postgres query_traces table."""
        conn = get_sync_conn()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO query_traces (
                        query, strategy_requested, strategy_used,
                        fallback_triggered, fallback_errors,
                        cache_hit, cache_hit_type,
                        num_queries_expanded,
                        chunks_retrieved, chunks_after_rerank,
                        top_rerank_score, avg_rerank_score,
                        context_tokens_before, context_tokens_after,
                        compression_ratio, answer_length, sources,
                        latency_route_ms, latency_expand_ms,
                        latency_retrieve_ms, latency_rerank_ms,
                        latency_compress_ms, latency_generate_ms,
                        latency_total_ms,
                        ragas_faithfulness, ragas_relevancy,
                        ragas_precision, error
                    ) VALUES (
                        %s,%s,%s,%s,%s,%s,%s,%s,%s,%s,
                        %s,%s,%s,%s,%s,%s,%s,%s,%s,%s,
                        %s,%s,%s,%s,%s,%s,%s,%s
                    )
                    """,
                    (
                        trace.query,
                        trace.strategy_requested,
                        trace.strategy_used,
                        trace.fallback_triggered,
                        json.dumps(trace.fallback_errors),
                        trace.cache_hit,
                        trace.cache_hit_type,
                        trace.num_queries_expanded,
                        trace.chunks_retrieved,
                        trace.chunks_after_rerank,
                        trace.top_rerank_score,
                        trace.avg_rerank_score,
                        trace.context_tokens_before,
                        trace.context_tokens_after,
                        trace.compression_ratio,
                        trace.answer_length,
                        json.dumps(trace.sources),
                        trace.latency_route_ms,
                        trace.latency_expand_ms,
                        trace.latency_retrieve_ms,
                        trace.latency_rerank_ms,
                        trace.latency_compress_ms,
                        trace.latency_generate_ms,
                        trace.latency_total_ms,
                        trace.ragas_faithfulness,
                        trace.ragas_relevancy,
                        trace.ragas_precision,
                        trace.error,
                    ),
                )
            conn.commit()
            logger.info(
                f"Trace persisted | strategy={trace.strategy_used} | "
                f"fallback={trace.fallback_triggered} | "
                f"cache={trace.cache_hit} | "
                f"total={trace.latency_total_ms:.0f}ms"
            )
        finally:
            conn.close()

    def get_dashboard_data(self) -> Dict:
        """Aggregate metrics for the Streamlit observability dashboard."""
        conn = get_sync_conn()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT
                        ROUND(AVG(latency_total_ms)::numeric,   1) AS avg_total_ms,
                        ROUND(AVG(latency_route_ms)::numeric,   1) AS avg_route_ms,
                        ROUND(AVG(latency_expand_ms)::numeric,  1) AS avg_expand_ms,
                        ROUND(AVG(latency_retrieve_ms)::numeric,1) AS avg_retrieve_ms,
                        ROUND(AVG(latency_rerank_ms)::numeric,  1) AS avg_rerank_ms,
                        ROUND(AVG(latency_compress_ms)::numeric,1) AS avg_compress_ms,
                        ROUND(AVG(latency_generate_ms)::numeric,1) AS avg_generate_ms,
                        ROUND(AVG(ragas_faithfulness)::numeric, 3) AS avg_faithfulness,
                        ROUND(AVG(ragas_relevancy)::numeric,    3) AS avg_relevancy,
                        ROUND(AVG(ragas_precision)::numeric,    3) AS avg_precision,
                        ROUND(AVG(CASE WHEN cache_hit
                              THEN 1.0 ELSE 0.0 END)::numeric * 100, 1)
                                                                   AS cache_hit_rate,
                        ROUND(AVG(CASE WHEN fallback_triggered
                              THEN 1.0 ELSE 0.0 END)::numeric * 100, 1)
                                                                   AS fallback_rate,
                        COUNT(*)                                   AS total_queries,
                        ROUND(AVG(top_rerank_score)::numeric,   3) AS avg_top_rerank_score,
                        ROUND(AVG(compression_ratio)::numeric,  3) AS avg_compression_ratio
                    FROM query_traces
                    """
                )
                row = dict(cur.fetchone())
                return {k: (float(v) if v is not None else 0) for k, v in row.items()}
        finally:
            conn.close()

    def get_recent_traces(self, limit: int = 100) -> List[Dict]:
        conn = get_sync_conn()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT * FROM query_traces ORDER BY timestamp DESC LIMIT %s",
                    (limit,),
                )
                rows = cur.fetchall()
                result = []
                for row in rows:
                    d = dict(row)
                    if isinstance(d.get("fallback_errors"), str):
                        d["fallback_errors"] = json.loads(d["fallback_errors"])
                    if isinstance(d.get("sources"), str):
                        d["sources"] = json.loads(d["sources"])
                    result.append(d)
                return result
        finally:
            conn.close()

    def get_strategy_breakdown(self) -> List[Dict]:
        conn = get_sync_conn()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT
                        strategy_used,
                        COUNT(*)                                     AS count,
                        ROUND(AVG(ragas_faithfulness)::numeric, 3)   AS avg_faithfulness,
                        ROUND(AVG(latency_total_ms)::numeric,   1)   AS avg_latency_ms,
                        ROUND(AVG(CASE WHEN cache_hit
                              THEN 1.0 ELSE 0.0 END)::numeric * 100, 1)
                                                                     AS cache_hit_rate
                    FROM query_traces
                    GROUP BY strategy_used
                    ORDER BY count DESC
                    """
                )
                return [dict(r) for r in cur.fetchall()]
        finally:
            conn.close()

    def get_latency_trend(self, limit: int = 50) -> List[Dict]:
        """Per-stage latency for last N queries — powers Streamlit stacked bar."""
        conn = get_sync_conn()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT
                        timestamp,
                        latency_route_ms,
                        latency_retrieve_ms,
                        latency_rerank_ms,
                        latency_compress_ms,
                        latency_generate_ms,
                        latency_total_ms,
                        strategy_used,
                        cache_hit
                    FROM query_traces
                    ORDER BY timestamp DESC
                    LIMIT %s
                    """,
                    (limit,),
                )
                return [dict(r) for r in cur.fetchall()]
        finally:
            conn.close()


# Singleton — imported by api/main.py
tracer = ObservabilityTracer()
