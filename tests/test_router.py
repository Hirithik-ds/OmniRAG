import pytest
from unittest.mock import MagicMock, patch
from agents.router import AgenticRouter, RetrievalStrategy


def _make_router(llm_response: str) -> AgenticRouter:
    """Helper: create router with mocked LLM returning a fixed string."""
    router = AgenticRouter.__new__(AgenticRouter)
    router.llm = MagicMock()
    router.llm.invoke.return_value = llm_response
    return router


@pytest.mark.parametrize("llm_output,expected", [
    ("hybrid",  RetrievalStrategy.HYBRID),
    ("HYBRID",  RetrievalStrategy.HYBRID),
    ("graph",   RetrievalStrategy.GRAPH),
    ("agentic", RetrievalStrategy.AGENTIC),
    ("  hybrid  ", RetrievalStrategy.HYBRID),  # whitespace handled
])
def test_route_parses_llm_output(llm_output, expected):
    router = _make_router(llm_output)
    assert router.route("any query") == expected


@pytest.mark.parametrize("query,expected", [
    ("Find invoice INV-2024-8821",              RetrievalStrategy.HYBRID),
    ("What is the refund policy?",              RetrievalStrategy.HYBRID),
    ("Who did Alice collaborate with?",         RetrievalStrategy.GRAPH),
    ("What projects connected Bob and Carol?",  RetrievalStrategy.GRAPH),
    ("Compare Q1 and Q3 risk factors",          RetrievalStrategy.AGENTIC),
    ("Summarise all product changes in 2024",   RetrievalStrategy.AGENTIC),
])
def test_route_expected_strategy(query, expected):
    """Router returns the strategy its LLM mock is set to."""
    router = _make_router(expected.value)
    assert router.route(query) == expected


def test_route_falls_back_on_exception():
    router = AgenticRouter.__new__(AgenticRouter)
    router.llm = MagicMock()
    router.llm.invoke.side_effect = Exception("API timeout")
    result = router.route("any query")
    assert result == RetrievalStrategy.HYBRID


def test_route_falls_back_on_unrecognised_output():
    router = _make_router("unknown_strategy_xyz")
    result = router.route("any query")
    assert result == RetrievalStrategy.HYBRID


def test_route_with_reason_returns_keys():
    router = _make_router("hybrid")
    result = router.route_with_reason("What is the revenue?")
    assert "strategy"  in result
    assert "reason"    in result
    assert isinstance(result["strategy"], RetrievalStrategy)
    assert isinstance(result["reason"], str)
    assert len(result["reason"]) > 10


def test_route_with_reason_graph():
    router = _make_router("graph")
    result = router.route_with_reason("Who worked with Alice?")
    assert result["strategy"] == RetrievalStrategy.GRAPH
    assert "graph" in result["reason"].lower() or "relationship" in result["reason"].lower()


def test_route_with_reason_agentic():
    router = _make_router("agentic")
    result = router.route_with_reason("Compare all documents")
    assert result["strategy"] == RetrievalStrategy.AGENTIC


def test_retrieval_strategy_enum_values():
    assert RetrievalStrategy.HYBRID.value  == "hybrid"
    assert RetrievalStrategy.GRAPH.value   == "graph"
    assert RetrievalStrategy.AGENTIC.value == "agentic"


def test_retrieval_strategy_is_string():
    """RetrievalStrategy(str, Enum) should be usable as a plain string."""
    assert RetrievalStrategy.HYBRID == "hybrid"
