import logging
from typing import Dict, List, Optional

from gliner import GLiNER
from config.settings import settings
from ingestion.graph_builder import GraphBuilder

logger = logging.getLogger("omnirag.graph_retriever")

ENTITY_TYPES = [
    "person", "organization", "location",
    "product", "technology", "concept",
]


class GraphRetriever:
    """
    Graph-based retrieval via Kuzu knowledge graph.

    Flow:
      1. Extract entities from query using GLiNER
      2. For each entity, query Kuzu for related document chunks
      3. Chunks matching multiple entities get a score boost (+0.5 per extra entity)
      4. Return ranked chunks

    Why graph retrieval wins over vector search for relationship queries:
      Query: "What projects did Alice and Bob collaborate on?"
      Dense search: embeds 'Alice and Bob collaborate' — may not find chunks
                    that mention each separately
      Graph search: finds all chunks mentioning 'alice', finds all chunks
                    mentioning 'bob', intersects → chunks mentioning both
                    score 2.0 vs chunks mentioning one scoring 1.0

    threshold=0.4 is lower than ingestion (0.5) — we want higher recall
    at query time, preferring false positives over false negatives.
    """

    def __init__(self):
        logger.info(f"Loading NER model for graph retrieval: {settings.NER_MODEL}")
        self.ner = GLiNER.from_pretrained(settings.NER_MODEL)
        self.graph = GraphBuilder()

    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        source_filter: Optional[str] = None,
    ) -> List[Dict]:
        entities = self.ner.predict_entities(
            query, ENTITY_TYPES, threshold=0.4
        )

        if not entities:
            logger.debug("Graph retriever: no entities found in query")
            return []

        all_chunks: Dict[str, Dict] = {}

        for ent in entities:
            name = ent["text"].strip().lower()
            chunks = self.graph.query_related_chunks(name)

            for c in chunks:
                # Scope to a single document if requested. The graph store
                # returns a 'source' per chunk when available; if it's blank
                # we keep the chunk only when no filter is active.
                csource = c.get("source", "")
                if source_filter and csource and csource != source_filter:
                    continue

                cid = c["chunk_id"]
                if cid not in all_chunks:
                    all_chunks[cid] = {
                        "chunk_id":       cid,
                        "text":           c["text"],
                        "source":         csource,
                        "score":          1.0,
                        "element_type":   "text",
                        "page_number":    None,
                        "section_h1":     "",
                        "section_h2":     "",
                        "table_markdown": None,
                        "image_caption":  None,
                        "matched_entities": [name],
                        "retrieval_type": "graph",
                    }
                else:
                    all_chunks[cid]["score"] += 0.5
                    all_chunks[cid]["matched_entities"].append(name)

        ranked = sorted(
            all_chunks.values(), key=lambda x: x["score"], reverse=True
        )
        return ranked[:top_k]