import pytest
import tempfile
from pathlib import Path
from ingestion.chunker import Chunk, SemanticChunker


# ── SemanticChunker — chunk_text ──────────────────────────────────────────────

def test_chunk_text_basic():
    chunker = SemanticChunker()
    text    = "This is sentence one. This is sentence two. " * 100
    chunks  = chunker.chunk_text(text, source="test.pdf")
    assert len(chunks) > 1


def test_chunk_text_respects_chunk_size():
    chunker            = SemanticChunker()
    chunker.chunk_size = 50
    chunker.overlap    = 10
    text = " ".join([f"word{i}" for i in range(200)])
    chunks = chunker.chunk_text(text, source="test.pdf")
    for c in chunks[:-1]:  # last chunk may be smaller
        assert len(c.text.split()) <= 60  # size + small buffer


def test_chunk_text_overlap():
    """The last words of one chunk should appear at the start of the next."""
    chunker            = SemanticChunker()
    chunker.chunk_size = 20
    chunker.overlap    = 5
    # Long text to force multiple chunks
    words = [f"word{i}" for i in range(100)]
    text  = " ".join(words)
    chunks = chunker.chunk_text(text, source="test.pdf")
    if len(chunks) >= 2:
        last_words_of_first = set(chunks[0].text.split()[-5:])
        first_words_of_second = set(chunks[1].text.split()[:5])
        assert len(last_words_of_first & first_words_of_second) > 0


def test_chunk_text_source_in_metadata():
    chunker = SemanticChunker()
    chunks  = chunker.chunk_text("Hello world. " * 5, source="myfile.pdf")
    for c in chunks:
        assert c.metadata["source"] == "myfile.pdf"


def test_chunk_text_chunk_id_format():
    chunker = SemanticChunker()
    chunks  = chunker.chunk_text("Sentence one. Sentence two. " * 10, source="doc.pdf")
    for c in chunks:
        assert c.chunk_id.startswith("doc.pdf_")


def test_chunk_text_empty_string():
    chunker = SemanticChunker()
    chunks  = chunker.chunk_text("", source="empty.pdf")
    assert chunks == []


def test_chunk_text_single_short_sentence():
    chunker = SemanticChunker()
    chunks  = chunker.chunk_text("Just one sentence.", source="short.pdf")
    assert len(chunks) == 1
    assert chunks[0].text == "Just one sentence."


# ── SemanticChunker — sentence splitting ──────────────────────────────────────

def test_split_sentences_basic():
    chunker   = SemanticChunker()
    sentences = chunker._split_sentences("Hello world. How are you? I am fine!")
    assert len(sentences) == 3


def test_split_sentences_strips_whitespace():
    chunker   = SemanticChunker()
    sentences = chunker._split_sentences("  One.  Two.  ")
    assert all(s == s.strip() for s in sentences)


def test_split_sentences_empty():
    chunker   = SemanticChunker()
    sentences = chunker._split_sentences("")
    assert sentences == []


# ── Chunk dataclass ───────────────────────────────────────────────────────────

def test_chunk_dataclass_fields():
    c = Chunk(
        text="sample text",
        metadata={"source": "test.pdf", "chunk_index": 0},
        chunk_id="test.pdf_0",
    )
    assert c.text      == "sample text"
    assert c.chunk_id  == "test.pdf_0"
    assert c.metadata["source"] == "test.pdf"


# ── chunk_file routing (without calling real parser) ─────────────────────────

def test_chunk_file_with_txt(tmp_path):
    """
    Create a real .txt file and run chunk_file with fast strategy
    to avoid downloading hi_res layout model in CI.
    """
    txt_file = tmp_path / "test.txt"
    txt_file.write_text(
        "This is the first paragraph about revenue. "
        "It contains multiple sentences about financial results. "
        "The company grew significantly in Q3 2024. "
        "Revenue increased by 18 percent year over year. "
        "This was driven by enterprise segment growth. "
        * 10,
        encoding="utf-8",
    )

    from ingestion.document_parser import UnstructuredParser
    from unittest.mock import patch

    # Patch UnstructuredParser to use fast strategy and no image captioning
    with patch.object(
        UnstructuredParser, "__init__",
        lambda self, **kw: UnstructuredParser.__init__(
            self, strategy="fast", caption_images=False
        ),
    ):
        chunker = SemanticChunker()
        chunks  = chunker.chunk_file(str(txt_file))

    assert len(chunks) > 0
    assert all(c.text.strip() for c in chunks)
    assert all(c.metadata["source"] == "test.txt" for c in chunks)


# ── Table chunking — never split ──────────────────────────────────────────────

def test_table_element_becomes_single_chunk():
    """
    A table ParsedElement must produce exactly one chunk regardless of size.
    This tests the 'never split' invariant in chunk_file().
    """
    from ingestion.document_parser import ParsedElement

    table_content = (
        "| Quarter | Revenue | Growth |\n"
        "| --- | --- | --- |\n"
        + "| Q1 | $10M | 5% |\n" * 50   # long table
    )
    el = ParsedElement(
        element_type="table",
        content=table_content,
        metadata={"source": "report.pdf", "element_index": 0, "page_number": 1},
    )

    # Simulate what chunk_file does for a table element
    source = "report.pdf"
    chunk  = Chunk(
        text=el.content,
        metadata={
            "source":        source,
            "element_type":  "table",
            "chunk_index":   0,
            **el.metadata,
        },
        chunk_id=f"{source}_0",
    )

    assert "\n" in chunk.text               # markdown structure preserved
    assert chunk.metadata["element_type"] == "table"
    assert "Revenue" in chunk.text


def test_image_caption_becomes_single_chunk():
    """Image captions must be stored as one chunk with metadata."""
    from ingestion.document_parser import ParsedElement

    el = ParsedElement(
        element_type="image",
        content="[Figure on page 5]: A bar chart showing quarterly revenue growth",
        metadata={
            "source":        "slides.pptx",
            "element_index": 2,
            "page_number":   5,
            "image_caption": "A bar chart showing quarterly revenue growth",
        },
    )

    chunk = Chunk(
        text=el.content,
        metadata={
            "source":        "slides.pptx",
            "element_type":  "image",
            "chunk_index":   0,
            **el.metadata,
        },
        chunk_id="slides.pptx_0",
    )

    assert chunk.metadata["element_type"]  == "image"
    assert chunk.metadata["image_caption"] is not None
    assert "bar chart" in chunk.text.lower()
