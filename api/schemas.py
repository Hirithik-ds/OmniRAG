from typing import Dict, List, Optional
from pydantic import BaseModel


class QueryRequest(BaseModel):
    query: str
    strategy: Optional[str] = "auto"           # auto | hybrid | graph | agentic
    element_type_filter: Optional[str] = None  # text | table | image | None
    source_filter: Optional[str] = None        # filename to scope to, or None


class QueryResponse(BaseModel):
    answer: str
    strategy: str
    strategy_reason: str
    sources: List[str]
    chunks_retrieved: int
    ragas_scores: Dict
    latency_ms: float
    cache_hit: bool = False
    fallback_triggered: bool = False
    compression_ratio: float = 1.0
    num_queries_expanded: int = 1


class IngestResponse(BaseModel):
    status: str
    filename: str
    format: str


class HealthResponse(BaseModel):
    status: str
    qdrant: bool
    models_loaded: bool
    cache_hit_rate: float = 0.0