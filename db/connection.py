import asyncpg
import psycopg2
from psycopg2.extras import RealDictCursor # Normally psycopg2 output will be in tuple , in order to have irt in dict format we use RealDictCursor
from config.settings import settings

_async_pool: asyncpg.Pool | None = None


# ── Async pool (used by FastAPI endpoints) ────────────────────────────────────

async def get_async_pool() -> asyncpg.Pool:
    """
    Returns the shared asyncpg connection pool.
    Called once at FastAPI startup via lifespan.
    All async endpoints use this pool — never create their own connections.
    """
    global _async_pool
    if _async_pool is None:
        _async_pool = await asyncpg.create_pool(
            dsn=settings.POSTGRES_DSN,
            min_size=2,
            max_size=10,
            command_timeout=30,
        )
    return _async_pool


async def close_async_pool():
    global _async_pool
    if _async_pool:
        await _async_pool.close()
        _async_pool = None


# ── Sync connection (used by logger + tracer) ─────────────────────────────────

def get_sync_conn():
    """
    Returns a plain psycopg2 connection with RealDictCursor.
    Used by EvalLogger and ObservabilityTracer which run in background
    threads, not inside async endpoints.
    Caller is responsible for closing the connection.
    """
    return psycopg2.connect(
        host=settings.POSTGRES_HOST,
        port=settings.POSTGRES_PORT,
        user=settings.POSTGRES_USER,
        password=settings.POSTGRES_PASSWORD,
        dbname=settings.POSTGRES_DB,
        cursor_factory=RealDictCursor,
    )


# ── Schema bootstrap (run once at startup) ────────────────────────────────────

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS eval_logs (
    id                  BIGSERIAL PRIMARY KEY,
    timestamp           TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    query               TEXT,
    answer              TEXT,
    strategy            TEXT,
    faithfulness        FLOAT,
    answer_relevancy    FLOAT,
    context_precision   FLOAT,
    latency_ms          FLOAT
);

CREATE TABLE IF NOT EXISTS query_traces (
    id                      BIGSERIAL PRIMARY KEY,
    timestamp               TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    query                   TEXT,
    strategy_requested      TEXT,
    strategy_used           TEXT,
    fallback_triggered      BOOLEAN DEFAULT FALSE,
    fallback_errors         JSONB DEFAULT '[]',
    cache_hit               BOOLEAN DEFAULT FALSE,
    cache_hit_type          TEXT,
    num_queries_expanded    INT DEFAULT 1,
    chunks_retrieved        INT DEFAULT 0,
    chunks_after_rerank     INT DEFAULT 0,
    top_rerank_score        FLOAT DEFAULT 0,
    avg_rerank_score        FLOAT DEFAULT 0,
    context_tokens_before   INT DEFAULT 0,
    context_tokens_after    INT DEFAULT 0,
    compression_ratio       FLOAT DEFAULT 1,
    answer_length           INT DEFAULT 0,
    sources                 JSONB DEFAULT '[]',
    latency_route_ms        FLOAT DEFAULT 0,
    latency_expand_ms       FLOAT DEFAULT 0,
    latency_retrieve_ms     FLOAT DEFAULT 0,
    latency_rerank_ms       FLOAT DEFAULT 0,
    latency_compress_ms     FLOAT DEFAULT 0,
    latency_generate_ms     FLOAT DEFAULT 0,
    latency_total_ms        FLOAT DEFAULT 0,
    ragas_faithfulness      FLOAT DEFAULT 0,
    ragas_relevancy         FLOAT DEFAULT 0,
    ragas_precision         FLOAT DEFAULT 0,
    error                   TEXT
);

CREATE INDEX IF NOT EXISTS idx_eval_logs_timestamp    ON eval_logs    (timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_traces_timestamp       ON query_traces (timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_traces_strategy        ON query_traces (strategy_used);
CREATE INDEX IF NOT EXISTS idx_traces_cache_hit       ON query_traces (cache_hit);
CREATE INDEX IF NOT EXISTS idx_traces_fallback        ON query_traces (fallback_triggered);
"""


def bootstrap_schema():
    """
    Creates all tables and indexes if they do not exist.
    Safe to call on every startup — idempotent.
    """
    conn = get_sync_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(SCHEMA_SQL)
        conn.commit()
        print("Postgres schema ready.")
    finally:
        conn.close()
