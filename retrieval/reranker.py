import logging
from typing import Dict, List

from sentence_transformers import CrossEncoder
from config.settings import settings

logger = logging.getLogger("omnirag.reranker")


class Reranker:
    """
    Cross-encoder reranker using BAAI/bge-reranker-large.
    Runs fully locally on CPU — no API calls, no rate limits.

    Why cross-encoders beat bi-encoders for reranking:
      Bi-encoder (embedding): encodes query and passage SEPARATELY,
        then compares with cosine similarity. Fast but loses fine-grained
        interaction signal.
      Cross-encoder: sees BOTH query and passage together in one forward pass,
        learning deep interactions between them. 15-30% better precision
        but too slow to run on the entire corpus (use only on top-20).

    Pipeline position: receives top-20 from RRF fusion, returns top-5.
    This two-stage approach (fast retrieval + accurate reranking) is
    the standard production pattern at companies like Cohere and Pinecone.

    CPU inference time: ~200ms for 20 candidates on a modern laptop.
    """

    def __init__(self):
        logger.info(f"Loading reranker: {settings.RERANKER_MODEL}")
        self.model = CrossEncoder(settings.RERANKER_MODEL, max_length=512)

    def rerank(
        self,
        query: str,
        candidates: List[Dict],
        top_n: int = None,
    ) -> List[Dict]:
        top_n = top_n or settings.TOP_K_FINAL

        if not candidates:
            return []

        pairs = [(query, c["text"]) for c in candidates]
        scores = self.model.predict(pairs)

        for i, doc in enumerate(candidates):
            doc["rerank_score"] = float(scores[i])

        reranked = sorted(
            candidates, key=lambda x: x["rerank_score"], reverse=True
        )
        return reranked[:top_n]
