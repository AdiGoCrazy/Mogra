"""In-memory LRU caching layer for local embeddings and query intent payloads."""

import hashlib
import json
import logging
from typing import Any, Optional
from collections import OrderedDict

logger = logging.getLogger(__name__)

class SimpleLRUCache:
    """Thread-safe, lightweight LRU Cache implementation."""

    def __init__(self, capacity: int = 256) -> None:
        """Initialize cache capacity.

        Args:
            capacity: Maximum number of entries before evicting least recently used.
        """
        self.capacity = capacity
        self.cache: OrderedDict[str, Any] = OrderedDict()

    def _hash_key(self, key_data: Any) -> str:
        """Generate MD5 hash string for lookup keys.

        Args:
            key_data: String or serializable dict/list key data.

        Returns:
            32-character MD5 hash string.
        """
        if isinstance(key_data, (dict, list)):
            key_str = json.dumps(key_data, sort_keys=True)
        else:
            key_str = str(key_data)
        return hashlib.md5(key_str.encode("utf-8")).hexdigest()

    def get(self, key_data: Any) -> Optional[Any]:
        """Retrieve cached entry if present.

        Args:
            key_data: Lookup key.

        Returns:
            Cached value or None.
        """
        hashed = self._hash_key(key_data)
        if hashed in self.cache:
            self.cache.move_to_end(hashed)
            logger.debug(f"Cache HIT for key hash '{hashed[:8]}'")
            return self.cache[hashed]
        return None

    def put(self, key_data: Any, value: Any) -> None:
        """Insert or update entry in cache.

        Args:
            key_data: Lookup key.
            value: Value payload to cache.
        """
        hashed = self._hash_key(key_data)
        if hashed in self.cache:
            self.cache.move_to_end(hashed)
        self.cache[hashed] = value
        if len(self.cache) > self.capacity:
            self.cache.popitem(last=False)
            logger.debug("Cache capacity reached; evicted oldest entry.")

    def clear(self) -> None:
        """Clear all cached items."""
        self.cache.clear()

embedding_cache = SimpleLRUCache(capacity=512)
intent_cache = SimpleLRUCache(capacity=256)
