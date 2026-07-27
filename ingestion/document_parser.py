import logging
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass, field

logger = logging.getLogger("omnirag.parser")


@dataclass
class ParsedElement:
    """
    A typed document element. Every format — PDF, DOCX, XLSX, PPTX, HTML —
    normalises into ParsedElements before chunking. This unified interface
    means the chunker never needs to know the original file format.

    element_type values:
      text    — narrative paragraph, sentence, heading body
      table   — markdown-formatted table (never split across chunks)
      image   — BLIP-2 generated caption prefixed with [Figure on page N]
      title   — section heading (stored as metadata + standalone chunk)
      list    — bullet/numbered list item (grouped before chunking)
      code    — code block (never split across chunks)
    """
    element_type: str
    content: str
    metadata: Dict = field(default_factory=dict)
    
#ParsedElement(
 #   element_type="text",
  #  content="IBM Research works on AI",
  #  metadata={
   #     page:2,
   #     section:"Introduction"
   # }
#)
   


class UnstructuredParser:
    """
    Single parser for ALL document formats using unstructured[all-docs].

    unstructured handles:
      - Layout detection (finds tables, images, titles automatically)
      - HTML table → structured data with text_as_html
      - Image extraction from PDFs and DOCX
      - OCR for scanned documents (requires tesseract installed)
      - 20+ file formats with one unified partition API

    We add BLIP-2 on top for image semantic captioning.
    unstructured runs OCR on images; we add semantic understanding.

    strategy choices:
      hi_res  — uses layout model, finds tables + images, ~2-5s/page
      fast    — text extraction only, ~0.1s/page (use during development)
      auto    — unstructured picks based on file type
    """

    def __init__(
        self,
        strategy: str = "hi_res",
        caption_images: bool = True,
        image_output_dir: str = "/tmp/omnirag_images",
    ):
        self.strategy = strategy
        self.caption_images = caption_images
        self.image_output_dir = image_output_dir
        self._blip_processor = None
        self._blip_model = None

        Path(image_output_dir).mkdir(parents=True, exist_ok=True)

    # ── Public API ────────────────────────────────────────────────────────────

    def parse(self, file_path: str) -> List[ParsedElement]:
        """
        Parse any supported file. Returns typed ParsedElements
        with content always as a clean string ready for embedding.
        """
        path = Path(file_path)
        ext = path.suffix.lower()
        raw_elements = self._partition(file_path, ext)
        return self._convert_elements(raw_elements, source=path.name)

    # ── Partition — routes to correct unstructured function ───────────────────
    @staticmethod
    def _safe_str(el) -> str:
        """
        Safely convert an unstructured element to a string.
        Some elements have a broken __str__ that returns None,
        which crashes str(el).strip(). This guards against that.
        """
        try:
            text = str(el)
        except (TypeError, AttributeError):
            text = None
        if text is None:
            # fall back to the .text attribute if str() failed
            text = getattr(el, "text", "") or ""
        return text
        
    def _partition(self, file_path: str, ext: str) -> list:
        common = dict(include_metadata=True, include_page_breaks=True)
        pdf_kwargs = dict(
            **common,
            strategy=self.strategy,
            infer_table_structure=True,
            extract_images_in_pdf=True,
            extract_image_block_output_dir=self.image_output_dir,
            extract_image_block_types=["Image", "Table"],
        )

        try:
            if ext == ".pdf":
                from unstructured.partition.pdf import partition_pdf
                return partition_pdf(filename=file_path, **pdf_kwargs)

            elif ext in (".docx", ".doc"):
                from unstructured.partition.docx import partition_docx
                return partition_docx(
                    filename=file_path, infer_table_structure=True, **common
                )

            elif ext in (".xlsx", ".xls"):
                from unstructured.partition.xlsx import partition_xlsx
                return partition_xlsx(filename=file_path, **common)

            elif ext == ".csv":
                from unstructured.partition.csv import partition_csv
                return partition_csv(filename=file_path, **common)

            elif ext in (".pptx", ".ppt"):
                from unstructured.partition.pptx import partition_pptx
                return partition_pptx(filename=file_path, **common)

            elif ext in (".html", ".htm"):
                from unstructured.partition.html import partition_html
                return partition_html(filename=file_path, **common)

            elif ext == ".md":
                from unstructured.partition.md import partition_md
                return partition_md(filename=file_path, **common)

            elif ext == ".rst":
                from unstructured.partition.rst import partition_rst
                return partition_rst(filename=file_path, **common)

            elif ext == ".txt":
                from unstructured.partition.text import partition_text
                return partition_text(filename=file_path, **common)

            elif ext == ".eml":
                from unstructured.partition.email import partition_email
                return partition_email(filename=file_path, **common)

            elif ext == ".msg":
                from unstructured.partition.msg import partition_msg
                return partition_msg(filename=file_path, **common)

            elif ext in (".json", ".jsonl"):
                from unstructured.partition.json import partition_json
                return partition_json(filename=file_path, **common)

            elif ext == ".xml":
                from unstructured.partition.xml import partition_xml
                return partition_xml(filename=file_path, **common)

            elif ext == ".epub":
                from unstructured.partition.epub import partition_epub
                return partition_epub(filename=file_path, **common)

            elif ext in (".png", ".jpg", ".jpeg", ".tiff", ".bmp"):
                from unstructured.partition.image import partition_image
                return partition_image(
                    filename=file_path, strategy=self.strategy, **common
                )

            elif ext in (
                ".py", ".js", ".ts", ".java", ".go", ".cpp", ".c",
                ".rs", ".rb", ".sh", ".yaml", ".yml", ".toml", ".ini",
            ):
                from unstructured.partition.text import partition_text
                return partition_text(filename=file_path, **common)

            else:
                from unstructured.partition.auto import partition
                return partition(filename=file_path, **common)

        except Exception as e:
            logger.error(f"Partition failed for {file_path}: {e}")
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read()
            from unstructured.documents.elements import NarrativeText
            return [NarrativeText(text)]

    # ── Convert unstructured elements → ParsedElement ─────────────────────────

    def _convert_elements(self, raw_elements: list, source: str) -> List[ParsedElement]:
        elements: List[ParsedElement] = []
        current_h1 = ""
        current_h2 = ""
        pending_caption = ""

        for idx, el in enumerate(raw_elements):
            el_class = type(el).__name__
            el_text = self._safe_str(el).strip()

            meta = {
                "source": source,
                "element_index": idx,
                "element_class": el_class,
                "page_number": getattr(el.metadata, "page_number", None),
                "section_h1": current_h1,
                "section_h2": current_h2,
                "filename": getattr(el.metadata, "filename", source),
            }

            if el_class == "Title":
                if not current_h1:
                    current_h1 = el_text
                else:
                    current_h2 = el_text
                elements.append(ParsedElement("title", el_text, meta))

            elif el_class == "Table":
                html = getattr(el.metadata, "text_as_html", None)
                md = self._html_to_markdown(html) if html else el_text
                meta["table_markdown"] = md
                elements.append(ParsedElement("table", md, meta))

            elif el_class == "Image":
                img_path = getattr(el.metadata, "image_path", None)
                caption = ""

                if pending_caption:
                    caption = pending_caption
                    pending_caption = ""

                if self.caption_images and img_path:
                    blip_cap = self._caption_image(img_path)
                    if blip_cap:
                        caption = f"{caption}. {blip_cap}" if caption else blip_cap

                if not caption and el_text:
                    caption = el_text

                if caption:
                    page = meta.get("page_number", "?")
                    meta["image_path"] = img_path
                    meta["image_caption"] = caption
                    elements.append(ParsedElement(
                        "image",
                        f"[Figure on page {page}]: {caption}",
                        meta,
                    ))

            elif el_class == "FigureCaption":
                pending_caption = el_text

            elif el_class == "ListItem":
                if el_text:
                    elements.append(ParsedElement("list", f"• {el_text}", meta))

            elif el_class == "CodeSnippet":
                if el_text:
                    elements.append(ParsedElement("code", el_text, meta))

            elif el_class == "PageBreak":
                current_h2 = ""

            elif el_class in ("Header", "Footer"):
                pass  # structural noise — skip content, don't add to elements

            elif el_class in (
                "NarrativeText", "Text", "Address",
                "EmailAddress", "Formula",
            ):
                if el_text:
                    elements.append(ParsedElement("text", el_text, meta))

            else:
                if el_text:
                    elements.append(ParsedElement("text", el_text, meta))

        return elements

    # ── BLIP-2 image captioner ────────────────────────────────────────────────

    def _load_blip(self):
        if self._blip_model is None:
            from transformers import BlipProcessor, BlipForConditionalGeneration
            import torch
            logger.info("Loading BLIP-2 captioner (first run: ~900 MB download)...")
            model_id = "Salesforce/blip-image-captioning-base"
            self._blip_processor = BlipProcessor.from_pretrained(model_id)
            self._blip_model = BlipForConditionalGeneration.from_pretrained(
                model_id, torch_dtype=torch.float32
            )
            logger.info("BLIP-2 loaded.")

    def _caption_image(self, image_path: str) -> str:
        try:
            from PIL import Image
            import torch
            self._load_blip()
            img = Image.open(image_path).convert("RGB")
            inputs = self._blip_processor(img, return_tensors="pt")
            with torch.no_grad():
                out = self._blip_model.generate(**inputs, max_new_tokens=80)
            return self._blip_processor.decode(
                out[0], skip_special_tokens=True
            ).strip()
        except Exception as e:
            logger.warning(f"Image captioning failed for {image_path}: {e}")
            return ""

    # ── Table HTML → Markdown ─────────────────────────────────────────────────

    def _html_to_markdown(self, html: str) -> str:
        try:
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(html, "html.parser")
            rows = []
            for tr in soup.find_all("tr"):
                cells = [
                    td.get_text(" ", strip=True)
                    for td in tr.find_all(["td", "th"])
                ]
                rows.append(cells)

            if not rows:
                return html

            header = "| " + " | ".join(rows[0]) + " |"
            separator = "| " + " | ".join(["---"] * len(rows[0])) + " |"
            body_rows = []
            for row in rows[1:]:
                # pad short rows
                while len(row) < len(rows[0]):
                    row.append("")
                body_rows.append("| " + " | ".join(row) + " |")

            return "\n".join([header, separator] + body_rows)
        except Exception:
            return html





