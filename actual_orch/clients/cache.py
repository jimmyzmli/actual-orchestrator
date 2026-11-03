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

import sqlite3

class _SQLiteCacheFallback:
    def __init__(self, db_path=None):
        if not db_path:
            db_path = os.getenv("SQLITE_CACHE_PATH", os.path.join(os.getcwd(), "data", "cache.sqlite"))
        self.db_path = os.path.abspath(db_path)
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("CREATE TABLE IF NOT EXISTS cache (key TEXT PRIMARY KEY, val TEXT)")

    def get(self, key):
        try:
            with sqlite3.connect(self.db_path) as conn:
                cur = conn.cursor()
                cur.execute("SELECT val FROM cache WHERE key = ?", (key,))
                row = cur.fetchone()
                return row[0] if row else None
        except Exception:
            return None

    def set(self, key, val):
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("INSERT OR REPLACE INTO cache (key, val) VALUES (?, ?)", (key, val))
        except Exception:
            pass

    def delete(self, *keys):
        if not keys:
            return 0
        try:
            with sqlite3.connect(self.db_path) as conn:
                q = f"DELETE FROM cache WHERE key IN ({','.join(['?']*len(keys))})"
                cur = conn.execute(q, keys)
                return cur.rowcount
        except Exception:
            return 0

    def keys(self, pattern="*"):
        try:
            sql_pattern = pattern.replace("*", "%").replace("?", "_")
            with sqlite3.connect(self.db_path) as conn:
                cur = conn.cursor()
                cur.execute("SELECT key FROM cache WHERE key LIKE ?", (sql_pattern,))
                return [r[0] for r in cur.fetchall()]
        except Exception:
            return []


_client: Optional[any] = None

CACHE_PREFIX = "receipt:"

def _init_cache():
    global _client
    if _client is not None:
        return
        
    url = os.environ.get("REDIS_URL", "")
    if url and redis:
        try:
            c = redis.from_url(url, decode_responses=True)
            c.ping()
            _client = c
            return
        except Exception as e:
            print(f"[cache][warn] Redis at {url} unreachable ({e}), falling back to SQLite cache.")

    _client = _SQLiteCacheFallback()


def _file_hash(file_path: str) -> str:
    """SHA-256 of the file contents – stable cache key.
    If the file does not exist, hashes the file_path string itself."""
    h = hashlib.sha256()
    try:
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
    except FileNotFoundError:
        h.update(file_path.encode("utf-8"))
    return h.hexdigest()


def _get_identifier(file_path: str) -> str:
    """Return the cache key identifier. For emails, use the sanitized msg_id. Otherwise, hash the file."""
    if "email_source" in file_path:
        basename = os.path.basename(file_path)
        if basename.endswith(".html.png"):
            return basename[:-9]
        if basename.endswith(".html"):
            return basename[:-5]
        return basename
    return _file_hash(file_path)

# ── public helpers ────────────────────────────────────────────

def get_cache_entry(file_path: str, cache_type: str, prefix: str = "receipt_") -> Optional[dict]:
    """Return the cached JSON envelope for the given file and cache type, or None."""
    _init_cache()
    h = _get_identifier(file_path)
    raw = _client.get(f"{prefix}{h}:{cache_type}")
    if raw is None:
        return None
    try:
        data = json.loads(raw)
        if not isinstance(data, dict) or "data" not in data or "meta" not in data:
            raise ValueError(f"Legacy or invalid cache format detected for key {prefix}{h}:{cache_type}. Cache should have been cleared.")
        return data
    except json.JSONDecodeError:
        raise ValueError(f"Legacy or invalid cache format detected for key {prefix}{h}:{cache_type}. Cache should have been cleared.")

def set_cache_entry(file_path: str, cache_type: str, data: any, meta: dict, prefix: str = "receipt_") -> None:
    """Store the data and meta in a JSON envelope."""
    _init_cache()
    h = _get_identifier(file_path)
    envelope = {
        "meta": meta,
        "data": data
    }
    _client.set(f"{prefix}{h}:{cache_type}", json.dumps(envelope))

def get_ocr(file_path: str, prefix: str = "receipt_") -> Optional[str]:
    """Return cached OCR text."""
    entry = get_cache_entry(file_path, "ocr", prefix)
    return entry["data"] if entry else None

def set_ocr(file_path: str, text: str, meta: dict, prefix: str = "receipt_") -> None:
    """Store OCR text in an envelope."""
    set_cache_entry(file_path, "ocr", text, meta, prefix)

def get_parsed(file_path: str, prefix: str = "receipt_") -> Optional[dict]:
    """Return cached parsed dict."""
    entry = get_cache_entry(file_path, "parsed", prefix)
    return entry["data"] if entry else None

def set_parsed(file_path: str, data: dict, meta: dict, prefix: str = "receipt_") -> None:
    """Store parsed dict in an envelope."""
    set_cache_entry(file_path, "parsed", data, meta, prefix)

def get_classify(file_path: str, prefix: str = "receipt_") -> Optional[str]:
    """Return cached classification text (e.g. RECEIPT or NOT_RECEIPT)."""
    entry = get_cache_entry(file_path, "classify", prefix)
    return entry["data"] if entry else None

def set_classify(file_path: str, classification: str, meta: dict, prefix: str = "receipt_") -> None:
    """Store classification text in an envelope."""
    set_cache_entry(file_path, "classify", classification, meta, prefix)
