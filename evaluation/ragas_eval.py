import logging
from typing import Dict, List

from config.settings import settings
from evaluation.logger import EvalLogger

logger = logging.getLogger("omnirag.ragas_eval")


class RAGASEvaluator:
    """
    Runs RAGAS evaluation on every query using HF models as judge.
    Scores are persisted to Postgres via EvalLogger.

    Metrics computed:
      faithfulness      — is the answer supported by the context?
      answer_relevancy  — does the answer address the question?
      context_precision — are the retrieved chunks actually relevant?

    RAGAS uses an LLM-as-judge approach — it calls Llama-3.1-8B via
    HF Inference API to assess each metric. This costs API quota but
    gives a principled, reproducible quality signal.

    On failure (API timeout, parse error): returns zeros rather than
    crashing — evaluation is non-critical to the user-facing pipeline.
    """

    def __init__(self):
        self.logger = EvalLogger()
        self._llm = None
        self._embeddings = None

    def _get_llm(self):
        if self._llm is None:
            from langchain_huggingface import HuggingFaceEndpoint
            self._llm = HuggingFaceEndpoint(
                repo_id=settings.GENERATOR_MODEL,
                task="text-generation",
                huggingfacehub_api_token=settings.HF_TOKEN,
                max_new_tokens=256,
            )
        return self._llm

    def _get_embeddings(self):
        if self._embeddings is None:
            from langchain_huggingface import HuggingFaceEmbeddings
            self._embeddings = HuggingFaceEmbeddings(
                model_name=settings.EMBED_MODEL
            )
        return self._embeddings

    def evaluate_response(
        self,
        query: str,
        answer: str,
        contexts: List[str],
        ground_truth: str = "",
    ) -> Dict:
        try:
            from datasets import Dataset
            from ragas import evaluate
            from ragas.metrics import (
                faithfulness,
                answer_relevancy,
                context_precision,
            )

            data = {
                "question":    [query],
                "answer":      [answer],
                "contexts":    [contexts],
                "ground_truth": [ground_truth or answer],
            }
            dataset = Dataset.from_dict(data)

            result = evaluate(
                dataset,
                metrics=[faithfulness, answer_relevancy, context_precision],
                llm=self._get_llm(),
                embeddings=self._get_embeddings(),
            )

            scores = {
                "faithfulness":      round(float(result["faithfulness"]), 3), #Faithfulness — catches hallucination. Does the answer only contain claims that are actually backed by the retrieved context? (A high-faithfulness, low-relevancy answer might be accurate but off-topic.)
                "answer_relevancy":  round(float(result["answer_relevancy"]), 3), #Answer relevancy — catches non-answers or tangential answers. Does the answer actually address what was asked?
                "context_precision": round(float(result["context_precision"]), 3),#Context precision — catches retrieval quality issues. Of the chunks retrieved, how many were actually useful/relevant to answering the question? (This is a retrieval-side metric, not a generation-side one — it evaluates the compressor/retriever's input quality, separate from how the LLM used that input.)
            }

        except Exception as e:
            logger.warning(f"RAGAS evaluation failed: {e}")
            scores = {
                "faithfulness": 0.0,
                "answer_relevancy": 0.0,
                "context_precision": 0.0,
            }

        # Always log — even zero scores are useful for debugging
        self.logger.log(query=query, answer=answer, scores=scores)
        return scores
