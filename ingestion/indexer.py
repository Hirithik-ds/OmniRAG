import logging
import pickle
from pathlib import Path
from typing import List

import bm25s
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct, OptimizersConfigDiff,
)

from config.settings import settings
from ingestion.chunker import Chunk, SemanticChunker
from ingestion.embedder import LocalEmbedder
from ingestion.graph_builder import GraphBuilder

logger = logging.getLogger("omnirag.indexer")


class Indexer:
    """
    Triple-write indexer: every chunk goes into three stores simultaneously.

      1. Qdrant  — 1024-dim dense vectors for cosine similarity search
      2. BM25s   — sparse keyword index for exact-match retrieval
      3. Kuzu    — knowledge graph of entities and their relationships

    Qdrant payload carries element_type, page_number, section headings,
    table_markdown, and image_caption so retrieval can filter by content type.

    One call to index_chunks() writes all three. Calling it twice with the
    same chunks will upsert (Qdrant) and re-index (BM25s + Kuzu) — safe but
    slightly wasteful. For production use, track indexed chunk_ids.
    """

    def __init__(self):
        self.embedder = LocalEmbedder()
        self.graph = GraphBuilder()
        self.qdrant = QdrantClient(
            host=settings.QDRANT_HOST,
            port=settings.QDRANT_PORT,
        )
        self._init_qdrant_collection()
        self.bm25_corpus: List[str] = []
        self.bm25_meta: List[dict] = []

    def _init_qdrant_collection(self):
        existing = [
            c.name for c in self.qdrant.get_collections().collections
        ]
        if settings.QDRANT_COLLECTION not in existing:
            self.qdrant.create_collection(
                collection_name=settings.QDRANT_COLLECTION,
                vectors_config=VectorParams(
                    size=self.embedder.dimension,
                    distance=Distance.COSINE,
                ),
                optimizers_config=OptimizersConfigDiff(memmap_threshold=20000),
            )
            logger.info(f"Created Qdrant collection: {settings.QDRANT_COLLECTION}")

    def index_chunks(self, chunks: List[Chunk]):
        if not chunks:
            logger.warning("index_chunks called with empty list")
            return

        logger.info(f"Indexing {len(chunks)} chunks...")
        texts = [c.text for c in chunks]
        embeddings = self.embedder.embed_documents(texts)

        # ── 1. Qdrant upsert ──────────────────────────────────────────────────
        points = [
            PointStruct(
                id=abs(hash(c.chunk_id)) % (2 ** 63),
                vector=embeddings[i].tolist(),
                payload={
                    "chunk_id":       c.chunk_id,
                    "text":           c.text,
                    "source":         c.metadata.get("source", ""),
                    "chunk_index":    c.metadata.get("chunk_index", 0),
                    "element_type":   c.metadata.get("element_type", "text"),
                    "page_number":    c.metadata.get("page_number"),
                    "section_h1":     c.metadata.get("section_h1", ""),
                    "section_h2":     c.metadata.get("section_h2", ""),
                    "table_markdown": c.metadata.get("table_markdown"),
                    "image_caption":  c.metadata.get("image_caption"),
                    "slide_title":    c.metadata.get("slide_title", ""),
                },
            )
            for i, c in enumerate(chunks)
        ]
        self.qdrant.upsert(
            collection_name=settings.QDRANT_COLLECTION,
            points=points,
        )
        logger.info(f"Qdrant: {len(points)} vectors written")

        # ── 2. BM25s index ────────────────────────────────────────────────────
        for chunk in chunks:
            self.bm25_corpus.append(chunk.text)
            self.bm25_meta.append({
                "chunk_id": chunk.chunk_id,
                "text":     chunk.text,
                "source":   chunk.metadata.get("source", ""),
            })

        tokenized = bm25s.tokenize(self.bm25_corpus)
        bm25_index = bm25s.BM25()
        bm25_index.index(tokenized)
        self._save_bm25(bm25_index)
        logger.info("BM25s: index saved")

        # ── 3. Kuzu knowledge graph ───────────────────────────────────────────
        self.graph.build_from_chunks(chunks)
        logger.info("Kuzu: graph built")

    def _save_bm25(self, index):
        Path(settings.BM25_INDEX_PATH).mkdir(parents=True, exist_ok=True)
        index.save(settings.BM25_INDEX_PATH)
        with open(f"{settings.BM25_INDEX_PATH}/meta.pkl", "wb") as f:
            pickle.dump(self.bm25_meta, f)

    def index_file(self, file_path: str):
        """Index any supported file — format auto-detected by extension."""
        chunker = SemanticChunker()
        chunks = chunker.chunk_file(file_path)
        self.index_chunks(chunks)
        logger.info(f"Indexed: {file_path} ({len(chunks)} chunks)")

    def index_pdf(self, pdf_path: str):
        """Kept for backwards compatibility — calls index_file."""
        self.index_file(pdf_path)

    def index_directory(self, dir_path: str, extensions: list = None):
        """Index all supported files in a directory tree recursively."""
        supported = {
            ".pdf", ".txt", ".md", ".docx", ".xlsx", ".csv",
            ".pptx", ".json", ".jsonl", ".html", ".htm", ".eml",
            ".epub", ".py", ".js", ".ts", ".java", ".go",
            ".rst", ".xml", ".yaml", ".yml",
        }
        exts = set(extensions) if extensions else supported

        for f in Path(dir_path).rglob("*"):
            if f.suffix.lower() in exts and f.is_file():
                try:
                    self.index_file(str(f))
                except Exception as e:
                    logger.error(f"Failed to index {f.name}: {e}")
