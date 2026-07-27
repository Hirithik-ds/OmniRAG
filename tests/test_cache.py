import time
import pytest
from cache.embedding_cache import InMemoryCache


def test_cache_set_and_get():
    c = InMemoryCache(max_size=10, ttl_seconds=60)
    c.set("emb", "hello world", [0.1, 0.2, 0.3])
    result = c.get("emb", "hello world")
    assert result == [0.1, 0.2, 0.3]


def test_cache_hit_increments_stat():
    c = InMemoryCache(max_size=10, ttl_seconds=60)
    c.set("emb", "query", [1.0])
    c.get("emb", "query")
    assert c.stats["hits"] == 1
    assert c.stats["misses"] == 0


def test_cache_miss_increments_stat():
    c = InMemoryCache(max_size=10, ttl_seconds=60)
    result = c.get("emb", "not stored")
    assert result is None
    assert c.stats["misses"] == 1
    assert c.stats["hits"] == 0


def test_cache_ttl_expiry():
    c = InMemoryCache(max_size=10, ttl_seconds=1)
    c.set("emb", "expiring key", [1, 2, 3])
    assert c.get("emb", "expiring key") == [1, 2, 3]
    time.sleep(1.1)
    assert c.get("emb", "expiring key") is None


def test_cache_lru_eviction():
    c = InMemoryCache(max_size=3, ttl_seconds=60)
    c.set("ns", "a", 1)
    c.set("ns", "b", 2)
    c.set("ns", "c", 3)
    # Adding a 4th item should evict the least recently used ("a")
    c.set("ns", "d", 4)
    assert c.get("ns", "a") is None   # evicted
    assert c.get("ns", "d") == 4      # new item present
    assert c.stats["evictions"] == 1


def test_cache_lru_access_updates_order():
    """Accessing an item should protect it from eviction."""
    c = InMemoryCache(max_size=3, ttl_seconds=60)
    c.set("ns", "a", 1)
    c.set("ns", "b", 2)
    c.set("ns", "c", 3)
    # Access "a" to make it recently used
    c.get("ns", "a")
    # Now add "d" — should evict "b" (oldest not recently used)
    c.set("ns", "d", 4)
    assert c.get("ns", "a") is not None   # still present
    assert c.get("ns", "b") is None       # evicted


def test_cache_namespaces_isolated():
    c = InMemoryCache(max_size=10, ttl_seconds=60)
    c.set("embedding", "query", [0.1])
    c.set("response",  "query", {"answer": "yes"})
    assert c.get("embedding", "query") == [0.1]
    assert c.get("response",  "query") == {"answer": "yes"}


def test_hit_rate_calculation():
    c = InMemoryCache(max_size=10, ttl_seconds=60)
    c.set("ns", "x", 1)
    c.get("ns", "x")   # hit
    c.get("ns", "y")   # miss
    assert c.hit_rate() == 0.5


def test_hit_rate_no_queries():
    c = InMemoryCache(max_size=10, ttl_seconds=60)
    assert c.hit_rate() == 0.0


def test_cache_info_keys():
    c = InMemoryCache(max_size=100, ttl_seconds=60)
    info = c.info()
    assert "size" in info
    assert "max_size" in info
    assert "hit_rate" in info
    assert "hits" in info
    assert "misses" in info
    assert "evictions" in info


def test_cache_delete():
    c = InMemoryCache(max_size=10, ttl_seconds=60)
    c.set("ns", "key", "value")
    assert c.get("ns", "key") == "value"
    c.delete("ns", "key")
    assert c.get("ns", "key") is None


def test_cache_key_is_case_insensitive():
    """Key normalisation strips and lowercases the value."""
    c = InMemoryCache(max_size=10, ttl_seconds=60)
    c.set("ns", "Hello World", 42)
    assert c.get("ns", "hello world") == 42
    assert c.get("ns", "  HELLO WORLD  ") == 42
