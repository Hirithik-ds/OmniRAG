import logging
from enum import Enum

from langchain_huggingface import HuggingFaceEndpoint
from config.settings import settings

logger = logging.getLogger("omnirag.router")


class RetrievalStrategy(str, Enum):
    HYBRID  = "hybrid"
    GRAPH   = "graph"
    AGENTIC = "agentic"


ROUTER_PROMPT = """You are a query router for a RAG system. Classify the query into exactly one category.

Categories:
- hybrid: Factual lookup, keyword search, finding specific IDs/numbers/names, straightforward questions
- graph: Questions about relationships between people/organizations/entities, collaboration, connections
- agentic: Multi-step reasoning, comparison across multiple documents, analytical questions requiring synthesis

Respond with ONLY one word: hybrid, graph, or agentic. No punctuation, no explanation.

Query: {query}
Category:"""

STRATEGY_REASONS = {
    RetrievalStrategy.HYBRID:  "Factual lookup — hybrid search gives best precision",
    RetrievalStrategy.GRAPH:   "Relationship query — graph traversal finds entity connections",
    RetrievalStrategy.AGENTIC: "Multi-step reasoning — agent will retrieve iteratively",
}


class AgenticRouter:
    """
    Uses Phi-3-mini (fast, small, MIT license) via HF Inference API
    to classify query intent and select the optimal retrieval strategy.

    Why Phi-3-mini instead of Llama:
      - 3.8B parameters vs 8B — 2x faster response
      - MIT license — zero restrictions
      - max_new_tokens=5 — we only need one word
      - temperature=0.0 — deterministic routing, no randomness

    Fallback: any exception or unrecognised output → HYBRID (safe default).
    HYBRID is chosen as default because it handles the broadest range of
    queries and never returns empty results on a populated index.
    """

    def __init__(self):
        self.llm = HuggingFaceEndpoint(
            repo_id=settings.ROUTER_MODEL,
            task="text-generation",
            huggingfacehub_api_token=settings.HF_TOKEN,
            max_new_tokens=5,
            temperature=0.0,
        )

    def route(self, query: str) -> RetrievalStrategy:
        prompt = ROUTER_PROMPT.format(query=query)
        try:
            response = self.llm.invoke(prompt).strip().lower()
            for strategy in RetrievalStrategy:
                if strategy.value in response:
                    logger.info(f"Routed '{query[:60]}...' → {strategy.value}")
                    return strategy
        except Exception as e:
            logger.warning(f"Router failed ({e}), defaulting to hybrid")
        return RetrievalStrategy.HYBRID #### if nothing works the LLM returns some hallucinated string instead of a valid strategy, just return hybrid as default

    def route_with_reason(self, query: str) -> dict:
        strategy = self.route(query)
        return {
            "strategy": strategy,
            "reason": STRATEGY_REASONS[strategy],
        }
