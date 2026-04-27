"""Small in-process TTL/LRU cache helpers."""

from __future__ import annotations

import time
from collections import OrderedDict
from typing import Generic, TypeVar


K = TypeVar("K")
V = TypeVar("V")


class TTLLRUCache(Generic[K, V]):
    """Tiny dependency-free cache for expensive deterministic helpers."""

    def __init__(self, maxsize: int = 128, ttl_seconds: int = 300) -> None:
        self.maxsize = max(0, int(maxsize))
        self.ttl_seconds = max(0, int(ttl_seconds))
        self._items: "OrderedDict[K, tuple[float, V]]" = OrderedDict()

    def get(self, key: K) -> V | None:
        if self.maxsize <= 0 or self.ttl_seconds <= 0:
            return None

        item = self._items.get(key)
        if item is None:
            return None

        expires_at, value = item
        if expires_at <= time.monotonic():
            self._items.pop(key, None)
            return None

        self._items.move_to_end(key)
        return value

    def set(self, key: K, value: V) -> None:
        if self.maxsize <= 0 or self.ttl_seconds <= 0:
            return

        self._items[key] = (time.monotonic() + self.ttl_seconds, value)
        self._items.move_to_end(key)
        while len(self._items) > self.maxsize:
            self._items.popitem(last=False)

    def clear(self) -> None:
        self._items.clear()

    def __len__(self) -> int:
        return len(self._items)
