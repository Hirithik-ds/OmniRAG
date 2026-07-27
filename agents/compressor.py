import logging
from typing import Dict, List

from config.settings import settings

logger = logging.getLogger("omnirag.compressor")


class ContextCompressor:
    """
    LLMLingua-2 token-level context compression.

    Why compress context:
      - Removes redundant sentences while preserving key information
      - Reduces token count by 3-5x → faster generation, lower cost
      - Removes noise that confuses the generator
      - Stays within context window limits for long documents

    COMPRESSION_RATIO=0.4 means: keep 40% of tokens, discard 60%.
    LLMLingua-2 uses a trained model (not random removal) so the
    retained 40% contains the most information-dense content.

    Runs fully locally on CPU — no API calls.
    Fallback: simple truncation to 3000 characters if LLMLingua fails.
    """

    def __init__(self):
        logger.info("Loading LLMLingua-2 compressor...")
        try:
            from llmlingua import PromptCompressor
            self.compressor = PromptCompressor(
                model_name="microsoft/llmlingua-2-xlm-roberta-large-meetingbank",
                use_llmlingua2=True,
                device_map="cpu",
            )
            self.ratio = settings.COMPRESSION_RATIO
            logger.info("LLMLingua-2 loaded.")
        except Exception as e:
            logger.warning(f"LLMLingua-2 load failed ({e}). Will use truncation fallback.")
            self.compressor = None
            self.ratio = settings.COMPRESSION_RATIO

    def compress(self, query: str, chunks: List[Dict]) -> str:
        if not chunks:
            return ""

        context = "\n\n---\n\n".join([
            f"[Source: {c.get('source', 'unknown')}]\n{c['text']}"
            for c in chunks
        ])

        if self.compressor is None:
            return context[:3000]

        try:
            result = self.compressor.compress_prompt(
                context,
                instruction=f"Answer the question: {query}",
                question=query,
                rate=self.ratio,
                force_tokens=["\n", ".", "?", "!"],
            )
            compressed = result["compressed_prompt"]
            original_len = len(context.split())
            compressed_len = len(compressed.split())
            logger.info(
                f"Compressed context: {original_len} → {compressed_len} tokens "
                f"({compressed_len/max(original_len,1):.1%} kept)"
            )
            return compressed
        except Exception as e:
            logger.warning(f"LLMLingua compression failed ({e}), using truncation")
            return context[:3000]
