import logging
from typing import Callable, Dict, List, Optional, Tuple

logger = logging.getLogger("omnirag.fallback_chain")


class FallbackResult:
    def __init__(
        self,
        chunks: List[Dict],
        strategy_used: str,
        fallback_triggered: bool,
        error_chain: List[str],
    ):
        self.chunks = chunks
        self.strategy_used = strategy_used
        self.fallback_triggered = fallback_triggered
        self.error_chain = error_chain


class FallbackChain:
    """
    Ordered chain of retrieval strategies with automatic failover.

    Degradation order (preferred strategy goes first):
      agentic  → graph → hybrid → direct_llm (no context)

    source_filter is threaded into every retriever so a scoped query
    ("only search reada10k.pdf") is honoured no matter which strategy
    in the chain ends up serving the request.
    """

    def __init__(self):
        from retrieval.hybrid_search import HybridSearcher
        from retrieval.graph_retriever import GraphRetriever
        from ingestion.embedder import LocalEmbedder

        embedder = LocalEmbedder()
        self._hybrid = HybridSearcher(embedder)
        self._graph = GraphRetriever()

    def retrieve_with_fallback(
        self,
        query: str,
        preferred_strategy: str,
        queries: List[str],
        top_k: int = 20,
        source_filter: Optional[str] = None,
    ) -> FallbackResult:
        error_chain: List[str] = []
        chain = self._build_chain(preferred_strategy, source_filter)

        for strategy_name, retriever_fn in chain:
            try:
                chunks = retriever_fn(query, queries, top_k)

                if chunks:
                    triggered = strategy_name != preferred_strategy
                    if triggered:
                        logger.warning(
                            f"Fallback triggered: {preferred_strategy} → "
                            f"{strategy_name} | errors: {error_chain}"
                        )
                    else:
                        logger.info(
                            f"Strategy '{strategy_name}' succeeded "
                            f"({len(chunks)} chunks)"
                        )
                    return FallbackResult(
                        chunks=chunks,
                        strategy_used=strategy_name,
                        fallback_triggered=triggered,
                        error_chain=error_chain,
                    )
                else:
                    msg = f"{strategy_name}: returned 0 chunks"
                    error_chain.append(msg)
                    logger.warning(msg)

            except Exception as e:
                msg = f"{strategy_name}: {type(e).__name__}: {str(e)[:120]}"
                error_chain.append(msg)
                logger.error(msg)

        logger.error(
            f"All retrievers failed. Using direct LLM. errors: {error_chain}"
        )
        return FallbackResult(
            chunks=[],
            strategy_used="direct_llm",
            fallback_triggered=True,
            error_chain=error_chain,
        )

    def _build_chain(
        self, preferred: str, source_filter: Optional[str] = None
    ) -> List[Tuple[str, Callable]]:
        def hybrid_fn(query, queries, top_k):
            results = []
            for q in queries[:3]:
                results += self._hybrid.search(
                    q, top_k=top_k // 2, source_filter=source_filter
                )
            return self._deduplicate(results)

        def graph_fn(query, queries, top_k):
            results = []
            for q in queries[:2]:
                results += self._graph.retrieve(
                    q, top_k=top_k // 2, source_filter=source_filter
                )
            return self._deduplicate(results)

        strategy_map = {
            "hybrid":  ("hybrid", hybrid_fn),
            "graph":   ("graph",  graph_fn),
            "agentic": ("hybrid", hybrid_fn),
        }

        preferred_entry = strategy_map.get(preferred, strategy_map["hybrid"])
        fallbacks = [
            v for k, v in strategy_map.items()
            if k != preferred and v[0] != preferred_entry[0]
        ]

        return [preferred_entry] + fallbacks

    def _deduplicate(self, chunks: List[Dict]) -> List[Dict]:
        seen = set()
        out = []
        for c in chunks:
            cid = c.get("chunk_id", c.get("text", "")[:50])
            if cid not in seen:
                seen.add(cid)
                out.append(c)
        return out