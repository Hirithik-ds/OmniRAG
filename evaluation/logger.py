import logging
from typing import Dict, List

from db.connection import get_sync_conn

logger = logging.getLogger("omnirag.eval_logger")


class EvalLogger:
    """
    Writes RAGAS evaluation scores to Postgres eval_logs table.
    Uses psycopg2 sync connection — safe to call from background threads.

    Each method opens and closes its own connection because:
      - Eval logging is infrequent (once per query)
      - Connection pool overhead is not worth it here
      - Simpler error isolation — a failed log doesn't break the pipeline
    """

    def log(
        self,
        query: str,
        answer: str,
        scores: Dict,
        strategy: str = "",
        latency_ms: float = 0.0,
    ):
        conn = get_sync_conn()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO eval_logs
                        (query, answer, strategy,
                         faithfulness, answer_relevancy, context_precision,
                         latency_ms)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    """,
                    (
                        query,
                        answer,
                        strategy,
                        scores.get("faithfulness", 0.0),
                        scores.get("answer_relevancy", 0.0),
                        scores.get("context_precision", 0.0),
                        latency_ms,
                    ),
                )
            conn.commit()
        except Exception as e:
            logger.error(f"EvalLogger.log failed: {e}")
        finally:
            conn.close()

    def get_all(self, limit: int = 200) -> List[Dict]:
        conn = get_sync_conn()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT id, timestamp, query, answer, strategy,
                           faithfulness, answer_relevancy, context_precision,
                           latency_ms
                    FROM eval_logs
                    ORDER BY timestamp DESC
                    LIMIT %s
                    """,
                    (limit,),
                )
                return [dict(r) for r in cur.fetchall()]
        finally:
            conn.close()

    def get_averages(self) -> Dict:
        conn = get_sync_conn()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT
                        ROUND(AVG(faithfulness)::numeric,      3) AS avg_faithfulness,
                        ROUND(AVG(answer_relevancy)::numeric,  3) AS avg_answer_relevancy,
                        ROUND(AVG(context_precision)::numeric, 3) AS avg_context_precision,
                        ROUND(AVG(latency_ms)::numeric,        1) AS avg_latency_ms,
                        COUNT(*)                                   AS total_queries
                    FROM eval_logs
                    """
                )
                row = dict(cur.fetchone())
                return {k: (float(v) if v is not None else 0) for k, v in row.items()}
        finally:
            conn.close()
