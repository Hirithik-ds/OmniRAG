import logging
import operator
from typing import Annotated, List, TypedDict

from langgraph.graph import END, StateGraph
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.messages import HumanMessage, SystemMessage

from config.settings import settings

logger = logging.getLogger("omnirag.agentic_rag")


class RAGState(TypedDict):
    query: str
    strategy: str
    queries: List[str]
    raw_chunks: Annotated[List, operator.add]   # accumulates across retries
    reranked_chunks: List
    compressed_context: str
    answer: str
    sources: List[str]
    iterations: int
    insufficient: bool


class AgenticRAGGraph:
    """
    LangGraph stateful agent with self-correction loop.

    Graph edges:
      retrieve → rerank → compress → generate → check_quality
      check_quality → [retry: retrieve | done: END]

    Self-correction: if the answer contains phrases like "I don't know"
    or "not in the context", check_quality sets insufficient=True and
    the graph loops back to retrieve (up to 3 times).

    On retry, a different set of query variants is used — the expander
    already generated 5 variants, so retry uses variants[1:] instead of
    the original. This changes the retrieval result set.

    Max iterations = 3 prevents infinite loops on genuinely unanswerable
    questions.

    Note: This graph handles the AGENTIC strategy routing path.
    HYBRID and GRAPH strategies use the simpler pipeline in api/main.py
    directly — they don't need the retry loop.
    """

    def __init__(self):
        from retrieval.hybrid_search import HybridSearcher
        from retrieval.graph_retriever import GraphRetriever
        from retrieval.reranker import Reranker
        from agents.compressor import ContextCompressor
        from ingestion.embedder import LocalEmbedder

        embedder = LocalEmbedder()
        self.hybrid = HybridSearcher(embedder)
        self.graph_ret = GraphRetriever()
        self.reranker = Reranker()
        self.compressor = ContextCompressor()

        # HuggingFace now serves Llama via the "conversational" task
        # (chat format), not "text-generation". We wrap the endpoint
        # in ChatHuggingFace so it sends chat-formatted requests.
        endpoint = HuggingFaceEndpoint(
            repo_id=settings.GENERATOR_MODEL,
            task="conversational",
            huggingfacehub_api_token=settings.HF_TOKEN,
            max_new_tokens=512,
            temperature=0.1,
        )
        self.llm = ChatHuggingFace(llm=endpoint)

        self.graph = self._build_graph()

    def _build_graph(self):
        g = StateGraph(RAGState)
        g.add_node("retrieve",      self._retrieve)
        g.add_node("rerank",        self._rerank)
        g.add_node("compress",      self._compress)
        g.add_node("generate",      self._generate)
        g.add_node("check_quality", self._check_quality)

        g.set_entry_point("retrieve")
        g.add_edge("retrieve",      "rerank")
        g.add_edge("rerank",        "compress")
        g.add_edge("compress",      "generate")
        g.add_edge("generate",      "check_quality")
        g.add_conditional_edges(
            "check_quality",
            self._should_retry,
            {"retry": "retrieve", "done": END},
        )
        return g.compile()

    def _retrieve(self, state: RAGState) -> dict:
        query = state["query"]
        strategy = state.get("strategy", "hybrid")
        queries = state.get("queries", [query])
        iterations = state.get("iterations", 0)

        # On retry, skip the first query variant (already tried it)
        active_queries = queries[iterations:] if iterations > 0 else queries
        chunks = []

        for q in active_queries[:3]:
            if strategy == "hybrid":
                chunks += self.hybrid.search(q, top_k=10)
            elif strategy == "graph":
                chunks += self.graph_ret.retrieve(q, top_k=10)
            else:
                # agentic — try both
                chunks += self.hybrid.search(q, top_k=6)
                chunks += self.graph_ret.retrieve(q, top_k=4)

        return {
            "raw_chunks": chunks,
            "iterations": iterations + 1,
        }

    def _rerank(self, state: RAGState) -> dict:
        reranked = self.reranker.rerank(
            state["query"], state["raw_chunks"]
        )
        sources = list({
            c.get("source", "") for c in reranked if c.get("source")
        })
        return {"reranked_chunks": reranked, "sources": sources}

    def _compress(self, state: RAGState) -> dict:
        compressed = self.compressor.compress(
            state["query"], state["reranked_chunks"]
        )
        return {"compressed_context": compressed}

    def _generate(self, state: RAGState) -> dict:
        system_prompt = (
            "You are a helpful assistant. Answer the question using ONLY "
            "the provided context. If the context doesn't contain the answer, "
            "say so clearly. Cite sources inline using [source_name] format."
        )
        user_prompt = (
            f"Context:\n{state['compressed_context']}\n\n"
            f"Question: {state['query']}"
        )
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ]
        response = self.llm.invoke(messages)
        # ChatHuggingFace returns a message object — extract .content
        answer = response.content
        return {"answer": answer}

    def _check_quality(self, state: RAGState) -> dict:
        answer = state.get("answer", "")
        insufficient_phrases = [
            "i don't know", "not in the context", "cannot find",
            "no information", "context doesn't", "not mentioned",
        ]
        insufficient = any(p in answer.lower() for p in insufficient_phrases)
        return {"insufficient": insufficient}

    def _should_retry(self, state: RAGState) -> str:
        if state.get("insufficient") and state.get("iterations", 0) < 3:
            logger.info(
                f"Self-correction triggered (iteration {state['iterations']})"
            )
            return "retry"
        return "done"

    def run(
        self,
        query: str,
        strategy: str = "hybrid",
        queries: List[str] = None,
    ) -> dict:
        initial = RAGState(
            query=query,
            strategy=strategy,
            queries=queries or [query],
            raw_chunks=[],
            reranked_chunks=[],
            compressed_context="",
            answer="",
            sources=[],
            iterations=0,
            insufficient=False,
        )
        final = self.graph.invoke(initial)
        return {
            "answer":           final["answer"],
            "sources":          final["sources"],
            "strategy":         strategy,
            "chunks_retrieved": len(final["reranked_chunks"]),
            "iterations":       final["iterations"],
            "reranked_chunks":  final["reranked_chunks"],
        }