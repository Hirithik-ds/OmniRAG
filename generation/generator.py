import logging
from typing import Iterator, List

from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.messages import HumanMessage, SystemMessage
from config.settings import settings

logger = logging.getLogger("omnirag.generator")

GENERATION_PROMPT = """You are a precise, helpful assistant. Answer the question using ONLY the provided context.
Rules:
- Be concise and factual
- Cite sources inline using [filename] format, e.g. [report.pdf]
- If context is insufficient, state clearly what is missing
- Do not add information not present in the context

Context:
{context}

Question: {query}

Answer:"""


class Generator:
    def __init__(self):
        endpoint = HuggingFaceEndpoint(
            repo_id=settings.GENERATOR_MODEL,
            task="conversational",              # ← changed
            huggingfacehub_api_token=settings.HF_TOKEN,
            max_new_tokens=512,
            temperature=0.1,
        )
        self.llm = ChatHuggingFace(llm=endpoint)  # ← wrap in chat

    def generate(self, query: str, context: str) -> str:
        messages = [
            SystemMessage(content=(
                "You are a helpful assistant. Answer the question using ONLY "
                "the provided context. If the context is insufficient, say so."
            )),
            HumanMessage(content=f"Context:\n{context}\n\nQuestion: {query}"),
        ]
        response = self.llm.invoke(messages)
        return response.content
    def stream(self, query: str, context: str) -> Iterator[str]:
        """Streaming generation — yields tokens for Streamlit st.write_stream."""
        prompt = GENERATION_PROMPT.format(context=context, query=query)
        for chunk in self.llm.stream(prompt):
            yield chunk
