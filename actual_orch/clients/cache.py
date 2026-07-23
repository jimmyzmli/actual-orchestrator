"""
Redis-based cache for receipt processing pipeline.

Caches OCR text and parsed JSON data keyed by file content hash,
so re-processing the same PDF skips expensive LLM calls.
"""

import hashlib
import json
import os
from typing import Optional

try:
    import redis
except ImportError:
    redis = None

_client: Optional['redis.Redis'] = None

CACHE_PREFIX = "receipt:"

def _init_cache():
    global _client
    if _client is not None:
        return
        
    url = os.environ.get("REDIS_URL", "")
    if not url or not redis:
        raise ValueError("REDIS_URL environment variable is not set or redis python package is missing.")
        
    try:
        _client = redis.from_url(url, decode_responses=True)
        # test connection
        _client.ping()
    except Exception as e:
        raise ConnectionError(f"Failed to connect to Redis at {url}: {e}")


def _file_hash(file_path: str) -> str:
    """SHA-256 of the file contents – stable cache key."""
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


# ── public helpers ────────────────────────────────────────────

def get_ocr(file_path: str, prefix: str = "receipt_") -> Optional[str]:
    """Return cached OCR text for *file_path*, or None."""
    _init_cache()
    h = _file_hash(file_path)
    return _client.get(f"{prefix}{h}:ocr")


def set_ocr(file_path: str, text: str, prefix: str = "receipt_") -> None:
    """Store OCR text (no expiry – persist forever)."""
    _init_cache()
    h = _file_hash(file_path)
    _client.set(f"{prefix}{h}:ocr", text)


def get_parsed(file_path: str, prefix: str = "receipt_") -> Optional[dict]:
    """Return cached parsed dict, or None."""
    _init_cache()
    h = _file_hash(file_path)
    raw = _client.get(f"{prefix}{h}:parsed")
    if raw is not None:
        return json.loads(raw)
    return None


def set_parsed(file_path: str, data: dict, prefix: str = "receipt_") -> None:
    """Store parsed dict as JSON formatted to be human readable."""
    _init_cache()
    h = _file_hash(file_path)
    _client.set(f"{prefix}{h}:parsed", json.dumps(data))


def get_meta(file_path: str, prefix: str = "receipt_") -> Optional[dict]:
    """Return cached metadata dict, or None."""
    _init_cache()
    h = _file_hash(file_path)
    raw = _client.get(f"{prefix}{h}:meta")
    if raw is not None:
        return json.loads(raw)
    return None


def set_meta(file_path: str, meta: dict, prefix: str = "receipt_") -> None:
    """Store metadata dict as JSON formatted to be human readable."""
    _init_cache()
    h = _file_hash(file_path)
    _client.set(f"{prefix}{h}:meta", json.dumps(meta))
