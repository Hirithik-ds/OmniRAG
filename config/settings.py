import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()


@dataclass
class Settings:
    # ── HuggingFace ───────────────────────────────────────────────────────────
    HF_TOKEN: str = os.getenv("HF_TOKEN", "")

    # ── Local models (run on CPU, no API key needed) ──────────────────────────
    EMBED_MODEL: str = "BAAI/bge-large-en-v1.5"
    RERANKER_MODEL: str = "BAAI/bge-reranker-large"
    NER_MODEL: str = "urchade/gliner_medium-v2.1"

    # ── HF Inference API models (generative, use token) ──────────────────────
    ROUTER_MODEL: str = "microsoft/Phi-3-mini-4k-instruct"
    GENERATOR_MODEL: str = "meta-llama/Llama-3.1-8B-Instruct"

    # ── Qdrant ────────────────────────────────────────────────────────────────
    QDRANT_HOST: str = "localhost"
    QDRANT_PORT: int = 6333
    QDRANT_COLLECTION: str = "omnirag_docs"

    # ── Kuzu embedded graph DB ────────────────────────────────────────────────
    KUZU_DB_PATH: str = "./data/kuzu_graph"

    # ── BM25s sparse index ────────────────────────────────────────────────────
    BM25_INDEX_PATH: str = "./data/bm25_index"

    # ── Chunking ──────────────────────────────────────────────────────────────
    CHUNK_SIZE: int = 512       # max words per chunk
    CHUNK_OVERLAP: int = 64     # overlap words between chunks

    # ── Retrieval ─────────────────────────────────────────────────────────────
    TOP_K_RETRIEVAL: int = 20   # candidates before reranking
    TOP_K_FINAL: int = 8       # chunks passed to generator
    RERANK_TOP_N: int = 20

    # ── Multi-query expansion ─────────────────────────────────────────────────
    NUM_QUERY_VARIANTS: int = 5

    # ── LLMLingua compression ratio (keep 40% of tokens) ─────────────────────
    COMPRESSION_RATIO: float = 0.5

    # ── Postgres ──────────────────────────────────────────────────────────────
    POSTGRES_HOST: str = os.getenv("POSTGRES_HOST", "localhost")
    POSTGRES_PORT: int = int(os.getenv("POSTGRES_PORT", "5432"))
    POSTGRES_USER: str = os.getenv("POSTGRES_USER", "omnirag")
    POSTGRES_PASSWORD: str = os.getenv("POSTGRES_PASSWORD", "omnirag")
    POSTGRES_DB: str = os.getenv("POSTGRES_DB", "omnirag")

    @property
    def POSTGRES_DSN(self) -> str:
        return (
            f"postgresql://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}"
            f"@{self.POSTGRES_HOST}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"
        )

    @property
    def ASYNC_POSTGRES_DSN(self) -> str:
        return (
            f"postgresql+asyncpg://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}"
            f"@{self.POSTGRES_HOST}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"
        )


settings = Settings()




#class A:
#    def square(self):
#       return 5 * 5

#a = A()


#print(a.square())    # 25

#with @property


#class A:
 #   @property
 #   def square(self):
  #      return 5 * 5

#a = A()

#print(a.square)  #25