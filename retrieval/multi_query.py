import logging
from typing import List

from langchain_huggingface import HuggingFaceEndpoint
from config.settings import settings

logger = logging.getLogger("omnirag.multi_query")

MULTI_QUERY_PROMPT = """You are an AI assistant helping improve document retrieval.
Given the user's question, generate {n} different versions of it.
Each version should approach the information need from a different angle —
use synonyms, rephrase the structure, or ask about related sub-topics.
Output ONLY the {n} questions, one per line, no numbering, no extra text.

Original question: {query}

{n} alternative versions:"""


class MultiQueryExpander:
    """
    Generates N query variants using Phi-3-mini via HF Inference API.
    Running multiple queries against the retriever and merging results
    significantly improves recall — especially for ambiguous queries.

    Why Phi-3-mini here (not Llama):
      - Phi-3-mini is much faster for this simple classification task
      - Conserves Llama API quota for final answer generation
      - temperature=0.7 adds controlled variation between variants

    On failure (API timeout, bad output): the original query is always
    included, so retrieval still works even if expansion fails entirely.
    """

    def __init__(self):
        self.llm = HuggingFaceEndpoint(
            repo_id=settings.ROUTER_MODEL,
            task="text-generation",
            huggingfacehub_api_token=settings.HF_TOKEN,
            max_new_tokens=256,
            temperature=0.7,
        )

    def expand(self, query: str, n: int = None) -> List[str]:
        n = n or settings.NUM_QUERY_VARIANTS
        prompt = MULTI_QUERY_PROMPT.format(query=query, n=n)

        try:
            response = self.llm.invoke(prompt)
            variants = [
                q.strip()
                for q in response.strip().split("\n")
                if q.strip() and q.strip() != query
            ]
            variants = variants[:n]
        except Exception as e:
            logger.warning(f"Multi-query expansion failed: {e}")
            variants = []

        # Always include original query first
        all_queries = [query] + variants
        # Deduplicate while preserving order
        seen = set()
        unique = []
        for q in all_queries:
            if q not in seen:
                seen.add(q)
                unique.append(q)

        logger.info(f"Expanded query into {len(unique)} variants")
        return unique
