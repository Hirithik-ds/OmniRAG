import re
import logging
from pathlib import Path
from typing import List
from dataclasses import dataclass
from config.settings import settings

logger = logging.getLogger("omnirag.chunker")


@dataclass
class Chunk:
    text: str
    metadata: dict
    chunk_id: str


class SemanticChunker:
    """
    Converts parsed document elements into fixed-size overlapping chunks.

    Routing rules per element type:
      table  — never split; kept as one chunk regardless of size
      image  — never split; caption stored as one chunk
      code   — never split; entire block as one chunk
      list   — consecutive ListItems grouped into one chunk first
      title  — stored as standalone chunk (also updates section metadata)
      text   — sliding window with CHUNK_SIZE words and CHUNK_OVERLAP overlap
    """

    def __init__(self):
        self.chunk_size = settings.CHUNK_SIZE
        self.overlap = settings.CHUNK_OVERLAP

    def chunk_file(self, file_path: str) -> List[Chunk]:
        """
        Entry point for all file types.
        Routes through UnstructuredParser → typed ParsedElements → Chunks.
        """
        from ingestion.document_parser import UnstructuredParser

        parser = UnstructuredParser(strategy="hi_res", caption_images=True)
        elements = parser.parse(file_path)
        source = Path(file_path).name
        chunks: List[Chunk] = []

        i = 0
        while i < len(elements):
            el = elements[i]

            # ── Group consecutive list items ──────────────────────────────────
            if el.element_type == "list":
                group = [el.content]
                j = i + 1
                while j < len(elements) and elements[j].element_type == "list":
                    group.append(elements[j].content)
                    j += 1
                combined = "\n".join(group)
                chunks.append(Chunk(
                    text=combined,
                    metadata={
                        "source": source,
                        "element_type": "list",
                        "chunk_index": len(chunks),
                        **el.metadata,
                    },
                    chunk_id=f"{source}_{len(chunks)}",
                ))
                i = j
                continue
            
            
#The Mental Model: The Conveyor Belt & The Scout
#Imagine you are standing at a conveyor belt of items. Your job is to pack these items into boxes (chunks).

#The Main Worker (i): You step through the items one by one.

#The "List" Trigger: Suddenly, you come across a list item (like a bullet point). You don't want to pack just one bullet point into a box; you want the whole bulleted list together!

#Deploying the Scout (j): You freeze your position (i). You send a mini-scout (j) to run ahead down the conveyor belt to see how many more list items are coming right after this one.

#The Gathering: The scout j runs ahead, grabbing list item after list item, piling them into a group. The moment the scout hits something that is not a list item (or reaches the end of the belt), the scout stops.

#The Teleport (i = j): You take that whole bundle, pack it into a single Chunk, and then—instead of walking through those list items one by one—you instantly teleport your position (i) straight to where the scout (j) stopped.     

      

            # ── Never split: tables, images, code ────────────────────────────
            if el.element_type in ("table", "image", "code"):
                if el.content.strip():
                    chunks.append(Chunk(
                        text=el.content,
                        metadata={
                            "source": source,
                            "element_type": el.element_type,
                            "chunk_index": len(chunks),
                            **el.metadata,
                        },
                        chunk_id=f"{source}_{len(chunks)}",
                    ))
                i += 1
                continue

            # ── Titles as standalone chunks ───────────────────────────────────
            if el.element_type == "title":
                if el.content.strip():
                    chunks.append(Chunk(
                        text=el.content,
                        metadata={
                            "source": source,
                            "element_type": "title",
                            "chunk_index": len(chunks),
                            **el.metadata,
                        },
                        chunk_id=f"{source}_{len(chunks)}",
                    ))
                i += 1
                continue

            # ── Normal text — sliding window ──────────────────────────────────
            sub_chunks = self.chunk_text(el.content, source=source)
            for sc in sub_chunks:
                sc.metadata.update({
                    "element_type": el.element_type,
                    "section_h1": el.metadata.get("section_h1", ""),
                    "section_h2": el.metadata.get("section_h2", ""),
                    "page_number": el.metadata.get("page_number"),
                    "slide_title": el.metadata.get("slide_title", ""),
                })
                sc.chunk_id = f"{source}_{len(chunks)}"
                chunks.append(sc)
            i += 1

        logger.info(
            f"chunk_file: {source} → "
            f"{len(elements)} elements → {len(chunks)} chunks"
        )
        return chunks

    def chunk_text(self, text: str, source: str = "unknown") -> List[Chunk]:
        """
        Sliding window chunker for plain text.
        Splits on sentence boundaries, keeps CHUNK_OVERLAP words of context.
        """
        sentences = self._split_sentences(text)
        chunks: List[Chunk] = []
        current_tok: List[str] = []
        current_len = 0
        chunk_idx = 0

        for sentence in sentences:
            words = sentence.split()
            sent_len = len(words)

            if current_len + sent_len > self.chunk_size and current_tok:
                chunks.append(Chunk(
                    text=" ".join(current_tok),
                    metadata={"source": source, "chunk_index": chunk_idx},
                    chunk_id=f"{source}_{chunk_idx}",
                ))
                chunk_idx += 1
                current_tok = current_tok[-self.overlap:] + words
                current_len = len(current_tok)
            else:
                current_tok.extend(words)
                current_len += sent_len

        if current_tok:
            chunks.append(Chunk(
                text=" ".join(current_tok),
                metadata={"source": source, "chunk_index": chunk_idx},
                chunk_id=f"{source}_{chunk_idx}",
            ))

        return chunks

    def _split_sentences(self, text: str) -> List[str]:
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        return [s.strip() for s in sentences if s.strip()]
