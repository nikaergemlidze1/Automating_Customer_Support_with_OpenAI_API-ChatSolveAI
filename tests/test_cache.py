"""Tests for the small TTL/LRU cache helper."""

from __future__ import annotations

import time

from pipeline.cache import TTLLRUCache


def test_cache_returns_recent_value():
    cache = TTLLRUCache[str, int](maxsize=2, ttl_seconds=60)
    cache.set("a", 1)
    assert cache.get("a") == 1


def test_cache_evicts_oldest_lru_entry():
    cache = TTLLRUCache[str, int](maxsize=2, ttl_seconds=60)
    cache.set("a", 1)
    cache.set("b", 2)
    assert cache.get("a") == 1
    cache.set("c", 3)

    assert cache.get("a") == 1
    assert cache.get("b") is None
    assert cache.get("c") == 3


def test_cache_expires_values():
    cache = TTLLRUCache[str, int](maxsize=2, ttl_seconds=1)
    cache.set("a", 1)
    time.sleep(1.01)
    assert cache.get("a") is None


def test_cache_can_be_disabled():
    cache = TTLLRUCache[str, int](maxsize=0, ttl_seconds=60)
    cache.set("a", 1)
    assert cache.get("a") is None
