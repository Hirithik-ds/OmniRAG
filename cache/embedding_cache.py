import hashlib
import time
from typing import Any, Dict, Optional
from collections import OrderedDict


class InMemoryCache:
    """
    LRU cache for two expensive operation types:
      namespace='embedding'  — query embedding vectors (np arrays as lists)
      namespace='response'   — full RAG pipeline responses

    Uses OrderedDict for O(1) LRU eviction.
    Thread-safe for single-process FastAPI.

    Production upgrade path: replace the OrderedDict backend
    with a Redis client — the public interface (get/set/delete) stays identical.
    """

    def __init__(self, max_size: int = 500, ttl_seconds: int = 3600):
        self.max_size = max_size
        self.ttl = ttl_seconds
        self._cache: OrderedDict[str, Dict[str, Any]] = OrderedDict()
        self.stats: Dict[str, int] = {"hits": 0, "misses": 0, "evictions": 0}

    def _make_key(self, namespace: str, value: str) -> str:   # Value is the user question and the namespace is the either embeddings or response 
        """SHA-256 hash ensures identical strings always map to identical keys."""
        raw = f"{namespace}:{value.strip().lower()}"
        return hashlib.sha256(raw.encode()).hexdigest()[:24]

    def get(self, namespace: str, value: str) -> Optional[Any]:
        key = self._make_key(namespace, value)
        entry = self._cache.get(key)

        if entry is None:
            self.stats["misses"] += 1
            return None

        if time.time() - entry["ts"] > self.ttl:
            del self._cache[key]
            self.stats["misses"] += 1
            return None

        # Move to end (most recently used)
        self._cache.move_to_end(key)
        self.stats["hits"] += 1
        return entry["value"]

    def set(self, namespace: str, value: str, data: Any):
        key = self._make_key(namespace, value)

        if key in self._cache:
            self._cache.move_to_end(key)

        self._cache[key] = {"value": data, "ts": time.time()}

        if len(self._cache) > self.max_size:
            self._cache.popitem(last=False)  # evict oldest (LRU)
            self.stats["evictions"] += 1

    def delete(self, namespace: str, value: str):
        key = self._make_key(namespace, value)
        self._cache.pop(key, None)

    def hit_rate(self) -> float:
        total = self.stats["hits"] + self.stats["misses"]
        return round(self.stats["hits"] / total, 3) if total else 0.0

    def info(self) -> Dict:
        return {
            "size": len(self._cache),
            "max_size": self.max_size,
            "hit_rate": self.hit_rate(),
            **self.stats,
        }


# Singleton — imported by embedder.py and api/main.py
cache = InMemoryCache(max_size=500, ttl_seconds=3600)
