"""
Redis Cache — avoids redundant API calls for repeated queries.

When a user asks the same question twice (or a similar one), we
skip the entire retrieval + LLM pipeline and return the cached answer.

HOW IT WORKS:
1. Hash the query into a cache key
2. Check Redis — if hit, return cached answer instantly
3. If miss, run the full pipeline, cache the result, return it

WHY Redis instead of a Python dict?
- Survives server restarts (persistence)
- Shared across multiple API workers (scalability)
- Built-in TTL (auto-expire old entries)
- In interviews: shows you think about production concerns

TTL = 3600 seconds (1 hour). After that, cached answers expire
so users get fresh results if documents were updated.
"""

import json
import hashlib
from typing import Optional, Dict, Any
import redis
from app.config import settings


class RedisCache:
    """Cache layer using Redis for query results."""

    def __init__(self):
        self.client = redis.Redis(
            host=settings.REDIS_HOST,
            port=settings.REDIS_PORT,
            decode_responses=True,  # return strings, not bytes
        )
        self.ttl = settings.CACHE_TTL

    def _make_key(self, query: str) -> str:
        """
        Create a cache key from the query.
        We hash it because queries can be long and Redis keys
        should be short for performance.
        """
        query_hash = hashlib.sha256(query.lower().strip().encode()).hexdigest()
        return f"rag:query:{query_hash}"

    def get(self, query: str) -> Optional[Dict[str, Any]]:
        """
        Look up a cached result for this query.
        Returns None on cache miss.
        """
        try:
            key = self._make_key(query)
            cached = self.client.get(key)
            if cached:
                return json.loads(cached)
            return None
        except redis.ConnectionError:
            # If Redis is down, just skip the cache
            return None

    def set(self, query: str, result: Dict[str, Any]):
        """
        Cache a query result with TTL.
        The result dict should contain 'answer' and optionally 'sources'.
        """
        try:
            key = self._make_key(query)
            self.client.setex(
                name=key,
                time=self.ttl,
                value=json.dumps(result),
            )
        except redis.ConnectionError:
            # If Redis is down, just skip caching
            pass

    def invalidate(self, query: str):
        """Remove a specific query from cache."""
        try:
            key = self._make_key(query)
            self.client.delete(key)
        except redis.ConnectionError:
            pass

    def clear_all(self):
        """Flush all cached queries (use after re-indexing docs)."""
        try:
            # Delete only our keys, not everything in Redis
            cursor = 0
            while True:
                cursor, keys = self.client.scan(
                    cursor=cursor, match="rag:query:*", count=100
                )
                if keys:
                    self.client.delete(*keys)
                if cursor == 0:
                    break
        except redis.ConnectionError:
            pass

    def is_connected(self) -> bool:
        """Check if Redis is reachable."""
        try:
            self.client.ping()
            return True
        except redis.ConnectionError:
            return False