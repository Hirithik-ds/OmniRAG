import logging
import numpy as np
from typing import List
from sentence_transformers import SentenceTransformer
from cache.embedding_cache import cache
from config.settings import settings

logger = logging.getLogger("omnirag.embedder")


class LocalEmbedder:
    """
    BAAI/bge-large-en-v1.5 — top MTEB retrieval model.
    Runs entirely locally on CPU. No API calls, no rate limits, no cost.

    Two embedding modes:
      embed_query()     — adds BGE query prefix + checks LRU cache
      embed_documents() — no prefix, no cache (documents embed once at ingestion)

    The BGE prefix trick: BGE models are trained with a query prefix that
    shifts the embedding space slightly toward retrieval intent.
    Documents are NOT prefixed — only queries are. This asymmetry is
    intentional and is documented by BAAI.
    """

    def __init__(self):
        logger.info(f"Loading embedding model: {settings.EMBED_MODEL}")
        self.model = SentenceTransformer(settings.EMBED_MODEL)
        self.dimension = self.model.get_sentence_embedding_dimension()
        logger.info(f"Embedding dimension: {self.dimension}")

    def embed_query(self, query: str) -> np.ndarray:
        """
        Embed a single query string with BGE prefix.
        Checks the LRU cache first — same query never re-embeds.
        """
        cached = cache.get("embedding", query)
        if cached is not None:
            return np.array(cached)

        prefixed = (
            f"Represent this sentence for searching relevant passages: {query}"
        )
        vector = self.model.encode(
            [prefixed], normalize_embeddings=True
        )[0]

        cache.set("embedding", query, vector.tolist())
        return vector

    def embed_documents(self, texts: List[str]) -> np.ndarray:
        """
        Embed a batch of document texts without the query prefix.
        Not cached — documents are only embedded once during ingestion.
        normalize_embeddings=True makes cosine similarity equivalent to dot product.
        """
        return self.model.encode(
            texts,
            batch_size=32,
            show_progress_bar=len(texts) > 50,
            normalize_embeddings=True,
        )
