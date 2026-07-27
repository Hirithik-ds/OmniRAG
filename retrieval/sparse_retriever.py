import pickle
import logging
from typing import Dict, List, Optional

import bm25s

from config.settings import settings

logger = logging.getLogger("omnirag.sparse_retriever")


class SparseRetriever:
    """
    BM25s keyword (sparse) retrieval.

    BM25s has no native payload filter, so source scoping is done by
    over-fetching and then filtering the results by their 'source' field
    in Python. We fetch a larger pool when a source_filter is set so that
    enough matching chunks survive the filter.
    """

    def __init__(self):
        logger.info("Loading BM25s index from disk...")
        self.index = bm25s.BM25.load(settings.BM25_INDEX_PATH)
        with open(f"{settings.BM25_INDEX_PATH}/meta.pkl", "rb") as f:
            self.meta: List[dict] = pickle.load(f)
        logger.info(f"BM25s index loaded: {len(self.meta)} documents")

    def retrieve(
        self,
        query: str,
        top_k: int = None,
        source_filter: Optional[str] = None,
    ) -> List[Dict]:
        top_k = top_k or settings.TOP_K_RETRIEVAL

        # Over-fetch when filtering so enough matches survive.
        fetch_k = top_k * 5 if source_filter else top_k
        fetch_k = min(fetch_k, len(self.meta))

        tokenized_query = bm25s.tokenize([query])
        results, scores = self.index.retrieve(tokenized_query, k=fetch_k)

        output = []
        for idx, score in zip(results[0], scores[0]):
            meta = self.meta[idx]
            src = meta.get("source", "")
            if source_filter and src != source_filter:
                continue
            output.append({
                "chunk_id":       meta["chunk_id"],
                "text":           meta["text"],
                "source":         src,
                "score":          float(score),
                "element_type":   "text",
                "page_number":    None,
                "section_h1":     "",
                "section_h2":     "",
                "table_markdown": None,
                "image_caption":  None,
                "retrieval_type": "sparse",
            })
            if len(output) >= top_k:
                break
        return output