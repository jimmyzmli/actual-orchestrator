"""
Redis-based cache for receipt processing pipeline.

Caches OCR text and parsed JSON data keyed by file content hash,
so re-processing the same PDF skips expensive LLM calls.
"""

import hashlib
import json
import os
from typing import Optional

import redis

_client: Optional[redis.Redis] = None

CACHE_PREFIX = "receipt:"


def _get_client() -> redis.Redis:
    """Lazy-init a Redis connection (reused across calls)."""
    global _client
    if _client is None:
        url = os.environ.get("REDIS_URL", "redis://localhost:6379/0")
        _client = redis.from_url(url, decode_responses=True)
    return _client


def _file_hash(file_path: str) -> str:
    """SHA-256 of the file contents – stable cache key."""
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


# ── public helpers ────────────────────────────────────────────

def get_ocr(file_path: str) -> Optional[str]:
    """Return cached OCR text for *file_path*, or None."""
    key = f"{CACHE_PREFIX}{_file_hash(file_path)}:ocr"
    return _get_client().get(key)


def set_ocr(file_path: str, text: str) -> None:
    """Store OCR text (no expiry – persist forever)."""
    key = f"{CACHE_PREFIX}{_file_hash(file_path)}:ocr"
    _get_client().set(key, text)


def get_parsed(file_path: str) -> Optional[dict]:
    """Return cached parsed receipt dict, or None."""
    key = f"{CACHE_PREFIX}{_file_hash(file_path)}:parsed"
    raw = _get_client().get(key)
    if raw is not None:
        return json.loads(raw)
    return None


def set_parsed(file_path: str, data: dict) -> None:
    """Store parsed receipt dict as JSON."""
    key = f"{CACHE_PREFIX}{_file_hash(file_path)}:parsed"
    _get_client().set(key, json.dumps(data))
