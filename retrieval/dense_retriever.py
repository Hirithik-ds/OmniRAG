import logging
from typing import Dict, List, Optional

from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue

from config.settings import settings
from ingestion.embedder import LocalEmbedder

logger = logging.getLogger("omnirag.dense_retriever")


class DenseRetriever:
    """
    Qdrant HNSW cosine similarity search.

    Supports optional payload filters:
      element_type_filter="table"       → only table chunks
      source_filter="reada10k.pdf"      → only chunks from that document
      both None                          → search everything (default)

    source_filter is what lets the UI scope a query to a single uploaded
    document so a large doc can't drown out a small one in the results.
    """

    def __init__(self, embedder: LocalEmbedder):
        self.embedder = embedder
        self.client = QdrantClient(
            host=settings.QDRANT_HOST,
            port=settings.QDRANT_PORT,
        )

    def retrieve(
        self,
        query: str,
        top_k: int = None,
        element_type_filter: Optional[str] = None,
        source_filter: Optional[str] = None,
    ) -> List[Dict]:
        top_k = top_k or settings.TOP_K_RETRIEVAL
        query_vec = self.embedder.embed_query(query)

        conditions = []
        if element_type_filter:
            conditions.append(
                FieldCondition(
                    key="element_type",
                    match=MatchValue(value=element_type_filter),
                )
            )
        if source_filter:
            conditions.append(
                FieldCondition(
                    key="source",
                    match=MatchValue(value=source_filter),
                )
            )
        qdrant_filter = Filter(must=conditions) if conditions else None

        # Qdrant renamed .search() -> .query_points() in recent versions.
        results = self.client.query_points(
            collection_name=settings.QDRANT_COLLECTION,
            query=query_vec.tolist(),
            query_filter=qdrant_filter,
            limit=top_k,
            with_payload=True,
        ).points

        return [
            {
                "chunk_id":       r.payload["chunk_id"],
                "text":           r.payload["text"],
                "source":         r.payload.get("source", ""),
                "score":          r.score,
                "element_type":   r.payload.get("element_type", "text"),
                "page_number":    r.payload.get("page_number"),
                "section_h1":     r.payload.get("section_h1", ""),
                "section_h2":     r.payload.get("section_h2", ""),
                "table_markdown": r.payload.get("table_markdown"),
                "image_caption":  r.payload.get("image_caption"),
                "slide_title":    r.payload.get("slide_title", ""),
                "retrieval_type": "dense",
            }
            for r in results
        ]