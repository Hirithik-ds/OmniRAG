import logging
from typing import Dict, List, Optional

from ingestion.embedder import LocalEmbedder
from retrieval.dense_retriever import DenseRetriever
from retrieval.sparse_retriever import SparseRetriever

logger = logging.getLogger("omnirag.hybrid_search")


class HybridSearcher:
    """
    Reciprocal Rank Fusion (RRF) combines dense and sparse ranked lists.

    RRF formula: score(d) = sum(1 / (k + rank_i(d))) for each list i
    k=60 is the standard constant.

    source_filter is passed to BOTH the dense and sparse retrievers so a
    query can be scoped to a single document across the whole hybrid path.
    """

    def __init__(self, embedder: LocalEmbedder, k: int = 60):
        self.dense = DenseRetriever(embedder)
        self.sparse = SparseRetriever()
        self.k = k

    def search(
        self,
        query: str,
        top_k: int = 20,
        element_type_filter: Optional[str] = None,
        source_filter: Optional[str] = None,
    ) -> List[Dict]:
        dense_results = self.dense.retrieve(
            query,
            top_k=top_k,
            element_type_filter=element_type_filter,
            source_filter=source_filter,
        )
        sparse_results = self.sparse.retrieve(
            query, top_k=top_k, source_filter=source_filter
        )
        return self._rrf_fuse(dense_results, sparse_results, top_k)

    def _rrf_fuse(
        self,
        dense: List[Dict],
        sparse: List[Dict],
        top_k: int,
    ) -> List[Dict]:
        rrf_scores: Dict[str, float] = {}
        chunk_data: Dict[str, Dict] = {}

        for rank, doc in enumerate(dense):
            cid = doc["chunk_id"]
            rrf_scores[cid] = rrf_scores.get(cid, 0) + 1 / (self.k + rank + 1)
            chunk_data[cid] = doc

        for rank, doc in enumerate(sparse):
            cid = doc["chunk_id"]
            rrf_scores[cid] = rrf_scores.get(cid, 0) + 1 / (self.k + rank + 1)
            if cid not in chunk_data:
                chunk_data[cid] = doc

        ranked = sorted(
            rrf_scores.items(), key=lambda x: x[1], reverse=True
        )[:top_k]

        results = []
        for cid, rrf_score in ranked:
            doc = chunk_data[cid].copy()
            doc["rrf_score"] = round(rrf_score, 6)
            doc["retrieval_type"] = "hybrid"
            results.append(doc)

        return results