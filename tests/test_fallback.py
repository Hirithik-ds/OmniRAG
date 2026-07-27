import pytest
from unittest.mock import MagicMock
from resilience.fallback_chain import FallbackChain, FallbackResult


def _make_chain(
    hybrid_chunks=None,
    graph_chunks=None,
    hybrid_raises=False,
    graph_raises=False,
) -> FallbackChain:
    """
    Helper: build a FallbackChain with mocked retrievers.
    hybrid_chunks / graph_chunks control what each retriever returns.
    *_raises controls whether the retriever raises an exception.
    """
    chain = FallbackChain.__new__(FallbackChain)
    chain._hybrid = MagicMock()
    chain._graph  = MagicMock()

    if hybrid_raises:
        chain._hybrid.search.side_effect = Exception("Qdrant unavailable")
    else:
        chain._hybrid.search.return_value = hybrid_chunks or []

    if graph_raises:
        chain._graph.retrieve.side_effect = Exception("Kuzu unavailable")
    else:
        chain._graph.retrieve.return_value = graph_chunks or []

    return chain


# ── Primary strategy succeeds ─────────────────────────────────────────────────

def test_hybrid_primary_succeeds():
    chunks = [{"chunk_id": "a", "text": "Revenue grew 18%", "source": "r.pdf"}]
    chain  = _make_chain(hybrid_chunks=chunks)
    result = chain.retrieve_with_fallback(
        query="revenue", preferred_strategy="hybrid", queries=["revenue"]
    )
    assert result.strategy_used      == "hybrid"
    assert result.fallback_triggered is False
    assert len(result.chunks)        == 1
    assert len(result.error_chain)   == 0


def test_graph_primary_succeeds():
    chunks = [{"chunk_id": "b", "text": "Alice and Bob collaborated", "source": "t.pdf"}]
    chain  = _make_chain(graph_chunks=chunks)
    result = chain.retrieve_with_fallback(
        query="Alice Bob", preferred_strategy="graph", queries=["Alice Bob"]
    )
    assert result.strategy_used      == "graph"
    assert result.fallback_triggered is False
    assert len(result.chunks)        == 1


# ── Fallback triggered on empty results ──────────────────────────────────────

def test_fallback_on_empty_results():
    """Empty result from preferred strategy should trigger fallback."""
    fallback_chunks = [{"chunk_id": "x", "text": "from hybrid", "source": "a.pdf"}]
    chain = _make_chain(
        hybrid_chunks=fallback_chunks,
        graph_chunks=[],     # graph returns empty
    )
    result = chain.retrieve_with_fallback(
        query="test", preferred_strategy="graph", queries=["test"]
    )
    assert result.fallback_triggered is True
    assert len(result.chunks) > 0
    assert len(result.error_chain) >= 1
    assert "graph" in result.error_chain[0]


# ── Fallback triggered on exception ──────────────────────────────────────────

def test_fallback_on_exception():
    """Exception in preferred strategy should trigger fallback."""
    fallback_chunks = [{"chunk_id": "y", "text": "fallback result", "source": "b.pdf"}]
    chain = _make_chain(
        hybrid_chunks=fallback_chunks,
        graph_raises=True,  # graph raises exception
    )
    result = chain.retrieve_with_fallback(
        query="test", preferred_strategy="graph", queries=["test"]
    )
    assert result.fallback_triggered is True
    assert len(result.chunks) > 0
    assert any("Exception" in e or "unavailable" in e.lower()
               for e in result.error_chain)


# ── All fail → direct LLM ─────────────────────────────────────────────────────

def test_all_retrievers_fail_returns_direct_llm():
    chain = _make_chain(
        hybrid_chunks=[],
        graph_chunks=[],
    )
    result = chain.retrieve_with_fallback(
        query="impossible query",
        preferred_strategy="agentic",
        queries=["impossible query"],
    )
    assert result.strategy_used      == "direct_llm"
    assert result.chunks             == []
    assert result.fallback_triggered is True
    assert len(result.error_chain)   >= 2   # at least two strategies tried


def test_all_retrievers_raise_returns_direct_llm():
    chain = _make_chain(hybrid_raises=True, graph_raises=True)
    result = chain.retrieve_with_fallback(
        query="test", preferred_strategy="hybrid", queries=["test"]
    )
    assert result.strategy_used == "direct_llm"
    assert result.chunks        == []


# ── Deduplication ─────────────────────────────────────────────────────────────

def test_deduplication_removes_duplicate_chunk_ids():
    chunks = [
        {"chunk_id": "x", "text": "a"},
        {"chunk_id": "x", "text": "a"},   # duplicate
        {"chunk_id": "y", "text": "b"},
    ]
    chain  = FallbackChain.__new__(FallbackChain)
    result = chain._deduplicate(chunks)
    assert len(result) == 2
    ids = [c["chunk_id"] for c in result]
    assert ids.count("x") == 1


def test_deduplication_preserves_order():
    chunks = [
        {"chunk_id": "c", "text": "third"},
        {"chunk_id": "a", "text": "first"},
        {"chunk_id": "b", "text": "second"},
        {"chunk_id": "a", "text": "first again"},  # duplicate
    ]
    chain  = FallbackChain.__new__(FallbackChain)
    result = chain._deduplicate(chunks)
    assert len(result) == 3
    assert result[0]["chunk_id"] == "c"
    assert result[1]["chunk_id"] == "a"
    assert result[2]["chunk_id"] == "b"


def test_deduplication_empty_input():
    chain  = FallbackChain.__new__(FallbackChain)
    result = chain._deduplicate([])
    assert result == []


# ── FallbackResult dataclass ──────────────────────────────────────────────────

def test_fallback_result_fields():
    r = FallbackResult(
        chunks=[{"chunk_id": "a"}],
        strategy_used="hybrid",
        fallback_triggered=False,
        error_chain=[],
    )
    assert r.strategy_used      == "hybrid"
    assert r.fallback_triggered is False
    assert len(r.chunks)        == 1
    assert r.error_chain        == []
