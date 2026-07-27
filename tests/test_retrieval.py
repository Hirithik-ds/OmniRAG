import pytest
from unittest.mock import MagicMock, patch


# ── HybridSearcher — RRF fusion ───────────────────────────────────────────────

def test_rrf_fuse_deduplicates():
    """A chunk appearing in both lists should appear only once in output."""
    from retrieval.hybrid_search import HybridSearcher

    dense_results = [
        {"chunk_id": "doc_0", "text": "Revenue grew 18%", "score": 0.91, "source": "a.pdf",
         "element_type": "text", "page_number": None, "section_h1": "", "section_h2": "",
         "table_markdown": None, "image_caption": None, "retrieval_type": "dense"},
        {"chunk_id": "doc_1", "text": "Q3 results",       "score": 0.85, "source": "a.pdf",
         "element_type": "text", "page_number": None, "section_h1": "", "section_h2": "",
         "table_markdown": None, "image_caption": None, "retrieval_type": "dense"},
    ]
    sparse_results = [
        {"chunk_id": "doc_1", "text": "Q3 results",   "score": 12.4, "source": "a.pdf",
         "element_type": "text", "page_number": None, "section_h1": "", "section_h2": "",
         "table_markdown": None, "image_caption": None, "retrieval_type": "sparse"},
        {"chunk_id": "doc_2", "text": "Annual report", "score":  8.1, "source": "b.pdf",
         "element_type": "text", "page_number": None, "section_h1": "", "section_h2": "",
         "table_markdown": None, "image_caption": None, "retrieval_type": "sparse"},
    ]

    searcher = HybridSearcher.__new__(HybridSearcher)
    fused    = searcher._rrf_fuse(dense_results, sparse_results, top_k=10)

    ids = [r["chunk_id"] for r in fused]
    assert len(ids) == len(set(ids)), "Duplicate chunk_ids found after RRF"


def test_rrf_fuse_top_k_limits_output():
    from retrieval.hybrid_search import HybridSearcher

    dense = [
        {"chunk_id": f"d{i}", "text": f"text {i}", "score": float(i),
         "source": "", "element_type": "text", "page_number": None,
         "section_h1": "", "section_h2": "", "table_markdown": None,
         "image_caption": None, "retrieval_type": "dense"}
        for i in range(10)
    ]
    sparse = [
        {"chunk_id": f"s{i}", "text": f"sparse {i}", "score": float(i),
         "source": "", "element_type": "text", "page_number": None,
         "section_h1": "", "section_h2": "", "table_markdown": None,
         "image_caption": None, "retrieval_type": "sparse"}
        for i in range(10)
    ]
    searcher = HybridSearcher.__new__(HybridSearcher)
    fused    = searcher._rrf_fuse(dense, sparse, top_k=5)
    assert len(fused) <= 5


def test_rrf_fuse_returns_hybrid_type():
    from retrieval.hybrid_search import HybridSearcher

    dense = [{"chunk_id": "a", "text": "x", "score": 0.9,
              "source": "", "element_type": "text", "page_number": None,
              "section_h1": "", "section_h2": "", "table_markdown": None,
              "image_caption": None, "retrieval_type": "dense"}]
    searcher = HybridSearcher.__new__(HybridSearcher)
    fused    = searcher._rrf_fuse(dense, [], top_k=10)
    assert fused[0]["retrieval_type"] == "hybrid"


def test_rrf_chunk_in_both_lists_scores_higher():
    """Chunk appearing in both lists should rank above chunk in one list only."""
    from retrieval.hybrid_search import HybridSearcher

    # doc_shared appears in both; doc_only_dense appears only in dense
    dense = [
        {"chunk_id": "doc_shared",    "text": "shared",     "score": 0.9,
         "source": "", "element_type": "text", "page_number": None,
         "section_h1": "", "section_h2": "", "table_markdown": None,
         "image_caption": None, "retrieval_type": "dense"},
        {"chunk_id": "doc_only_dense","text": "dense only", "score": 0.95,
         "source": "", "element_type": "text", "page_number": None,
         "section_h1": "", "section_h2": "", "table_markdown": None,
         "image_caption": None, "retrieval_type": "dense"},
    ]
    sparse = [
        {"chunk_id": "doc_shared", "text": "shared", "score": 15.0,
         "source": "", "element_type": "text", "page_number": None,
         "section_h1": "", "section_h2": "", "table_markdown": None,
         "image_caption": None, "retrieval_type": "sparse"},
    ]
    searcher = HybridSearcher.__new__(HybridSearcher)
    fused    = searcher._rrf_fuse(dense, sparse, top_k=10)
    ids = [r["chunk_id"] for r in fused]
    assert ids.index("doc_shared") < ids.index("doc_only_dense")


# ── Reranker ──────────────────────────────────────────────────────────────────

def test_reranker_returns_sorted_descending():
    from retrieval.reranker import Reranker

    reranker   = Reranker()
    candidates = [
        {"chunk_id": "a", "text": "Quarterly revenue increased significantly in Q3 2024",
         "source": "x", "element_type": "text"},
        {"chunk_id": "b", "text": "The weather was nice yesterday in Chennai",
         "source": "x", "element_type": "text"},
        {"chunk_id": "c", "text": "Q3 2024 financial results show 18% growth in revenue",
         "source": "x", "element_type": "text"},
    ]
    reranked = reranker.rerank(
        "What were Q3 revenue results?", candidates, top_n=3
    )
    assert len(reranked) <= 3
    scores = [r["rerank_score"] for r in reranked]
    assert scores == sorted(scores, reverse=True), "Reranker output not sorted"


def test_reranker_respects_top_n():
    from retrieval.reranker import Reranker

    reranker   = Reranker()
    candidates = [
        {"chunk_id": str(i), "text": f"document {i} about revenue",
         "source": "x", "element_type": "text"}
        for i in range(10)
    ]
    reranked = reranker.rerank("revenue", candidates, top_n=3)
    assert len(reranked) == 3


def test_reranker_empty_input():
    from retrieval.reranker import Reranker
    reranker = Reranker()
    result   = reranker.rerank("query", [], top_n=5)
    assert result == []


def test_reranker_adds_score_field():
    from retrieval.reranker import Reranker

    reranker   = Reranker()
    candidates = [
        {"chunk_id": "a", "text": "some relevant text about the topic",
         "source": "x", "element_type": "text"},
    ]
    reranked = reranker.rerank("topic", candidates, top_n=1)
    assert "rerank_score" in reranked[0]
    assert isinstance(reranked[0]["rerank_score"], float)


# ── DenseRetriever — payload fields ──────────────────────────────────────────

def test_dense_retriever_payload_fields_present():
    """
    All new payload fields (element_type, page_number, section headings,
    table_markdown, image_caption) must be present in retrieval results.
    """
    from retrieval.dense_retriever import DenseRetriever
    from ingestion.embedder import LocalEmbedder

    mock_result         = MagicMock()
    mock_result.payload = {
        "chunk_id":       "doc_0",
        "text":           "Revenue table for Q3",
        "source":         "report.pdf",
        "element_type":   "table",
        "page_number":    3,
        "section_h1":     "Financial Results",
        "section_h2":     "Q3 Summary",
        "table_markdown": "| Revenue | 18% |",
        "image_caption":  None,
        "slide_title":    "",
    }
    mock_result.score = 0.92

    embedder                = MagicMock(spec=LocalEmbedder)
    embedder.embed_query.return_value = [0.1] * 1024

    retriever           = DenseRetriever.__new__(DenseRetriever)
    retriever.embedder  = embedder
    retriever.client    = MagicMock()
    retriever.client.search.return_value = [mock_result]

    results = retriever.retrieve("revenue table", top_k=1)

    assert results[0]["element_type"]   == "table"
    assert results[0]["page_number"]    == 3
    assert results[0]["section_h1"]     == "Financial Results"
    assert results[0]["table_markdown"] == "| Revenue | 18% |"
    assert results[0]["retrieval_type"] == "dense"


def test_dense_retriever_element_type_filter_passed_to_qdrant():
    """element_type_filter should result in a Qdrant filter being set."""
    from retrieval.dense_retriever import DenseRetriever
    from ingestion.embedder import LocalEmbedder

    embedder                 = MagicMock(spec=LocalEmbedder)
    embedder.embed_query.return_value = [0.0] * 1024

    retriever                = DenseRetriever.__new__(DenseRetriever)
    retriever.embedder       = embedder
    retriever.client         = MagicMock()
    retriever.client.search.return_value = []

    retriever.retrieve("find tables", top_k=5, element_type_filter="table")

    call_kwargs = retriever.client.search.call_args.kwargs
    assert call_kwargs["query_filter"] is not None
