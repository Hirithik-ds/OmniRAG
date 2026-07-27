import logging
from typing import List
import kuzu
from gliner import GLiNER
from config.settings import settings
from ingestion.chunker import Chunk

logger = logging.getLogger("omnirag.graph_builder")

ENTITY_TYPES = [
    "person", "organization", "location", "date",
    "product", "concept", "technology",
]


class GraphBuilder:
    """
    Extracts named entities from chunks using GLiNER (zero-shot NER),
    then writes them into a Kuzu embedded knowledge graph.

    Graph schema:
      Node: Entity  (name, type)
      Node: Document (chunk_id, source, text)
      Edge: MENTIONED_IN  (Entity → Document)
      Edge: CO_OCCURS_WITH (Entity → Entity, chunk_id)

    The CO_OCCURS_WITH edge is the key insight: two entities appearing
    in the same chunk are likely related, even if there's no explicit
    relationship sentence. This enables relationship queries.

    Kuzu is Cypher-compatible, MIT licensed, embedded like SQLite —
    no server process required.
    """

    def __init__(self):
        logger.info(f"Loading NER model: {settings.NER_MODEL}")
        self.ner = GLiNER.from_pretrained(settings.NER_MODEL)
        self.db = kuzu.Database(settings.KUZU_DB_PATH)
        self.conn = kuzu.Connection(self.db)
        self._init_schema()

    def _init_schema(self):     ########
        self.conn.execute(""" 
            CREATE NODE TABLE IF NOT EXISTS Entity (
                name STRING,
                type STRING,
                PRIMARY KEY (name)
            )
        """)################ Entity node for eg : Entities. -->("google", "organization",...)
        self.conn.execute("""
            CREATE NODE TABLE IF NOT EXISTS Document (
                chunk_id STRING,
                source   STRING,
                text     STRING,
                PRIMARY KEY (chunk_id)
            )
        """) ############# Document node for eg : Document -->("chunk_123", "source.pdf", "text of the chunk...")
        self.conn.execute("""
            CREATE REL TABLE IF NOT EXISTS MENTIONED_IN (
                FROM Entity TO Document
            )
        """) ################ Relationship edge for eg : MENTIONED_IN -->("google" -[:MENTIONED_IN]-> "chunk_123")
        self.conn.execute("""
            CREATE REL TABLE IF NOT EXISTS CO_OCCURS_WITH (
                FROM Entity TO Entity,
                chunk_id STRING
            )
        """) ################ Relationship edge for eg : CO_OCCURS_WITH -->("google" -[:CO_OCCURS_WITH {chunk_id: "chunk_123"}]-> "ai")

    def build_from_chunks(self, chunks: List[Chunk]):
        for chunk in chunks:
            self._process_chunk(chunk)
        logger.info(f"Graph built from {len(chunks)} chunks")

    def _process_chunk(self, chunk: Chunk):
        # Insert document node
        # Chunk → Entities → Graph Nodes + Relationships
        self.conn.execute(
            "MERGE (d:Document {chunk_id: $cid}) "
            "SET d.source = $src, d.text = $txt",
            {
                "cid": chunk.chunk_id,
                "src": chunk.metadata.get("source", ""),
                "txt": chunk.text[:500],
            },
        )

        entities = self.ner.predict_entities(
            chunk.text, ENTITY_TYPES, threshold=0.5
        )
        entity_names: List[str] = []

        for ent in entities:
            name = ent["text"].strip().lower()
            etype = ent["label"]

            if len(name) < 2:
                continue

            # Upsert entity node
            self.conn.execute(
                "MERGE (e:Entity {name: $name}) SET e.type = $type",
                {"name": name, "type": etype},
            )

            # Link entity → document
            self.conn.execute(
                "MATCH (e:Entity {name: $ename}), (d:Document {chunk_id: $cid}) "
                "MERGE (e)-[:MENTIONED_IN]->(d)",
                {"ename": name, "cid": chunk.chunk_id},
            )
            entity_names.append(name)

        # Co-occurrence edges between entities in the same chunk
        #eg: entity_names = ["google", "tensorflow", "python"]
        #    (google, tensorflow)
         #    (google, python)
          #  (tensorflow, python)

          ## avoid matching same entities : (google, tensorflow) so that e1 != e2:
        for i, e1 in enumerate(entity_names):
            for e2 in entity_names[i + 1:]:
                if e1 != e2:
                    self.conn.execute(
                        "MATCH (a:Entity {name: $e1}), (b:Entity {name: $e2}) "
                        "MERGE (a)-[:CO_OCCURS_WITH {chunk_id: $cid}]->(b)",
                        {"e1": e1, "e2": e2, "cid": chunk.chunk_id},
                    )

    def query_related_chunks(self, entity_name: str) -> List[dict]:
        """Return document chunks containing this entity (partial match)."""
        result = self.conn.execute(
            "MATCH (e:Entity)-[:MENTIONED_IN]->(d:Document) "
            "WHERE e.name CONTAINS $name "
            "RETURN d.chunk_id, d.text LIMIT 20",
            {"name": entity_name.lower()},
        )
        chunks = []
        while result.has_next():
            row = result.get_next()
            chunks.append({"chunk_id": row[0], "text": row[1]})
        return chunks
