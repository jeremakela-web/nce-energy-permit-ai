"""
B2B API key management — INACTIVE by default.
Activate: set API_KEYS_ENABLED=true in environment.

Provides:
  init_api_keys_db() — create api_keys table in shared SQLite
  create_api_key()   — generate + store a new Bearer key
  verify_api_key()   — validate key and return customer metadata
  list_api_keys()    — admin listing (no raw key values)
  revoke_api_key()   — deactivate a key by key_id
"""
import hashlib
import logging
import os
import secrets
import sqlite3
import time
from typing import Optional

log = logging.getLogger(__name__)

API_KEYS_ENABLED: bool = os.getenv("API_KEYS_ENABLED", "false").lower() == "true"

# Same DB file as stripe_payments — single persistent-disk file for all business data
_PAYMENTS_DB = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "permit_ai", "embeddings", "payments.db")
)


def init_api_keys_db() -> None:
    """Create api_keys table. Safe to call multiple times (NOOP when disabled)."""
    if not API_KEYS_ENABLED:
        return
    os.makedirs(os.path.dirname(_PAYMENTS_DB), exist_ok=True)
    with sqlite3.connect(_PAYMENTS_DB) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS api_keys (
                key_id       TEXT PRIMARY KEY,
                key_hash     TEXT UNIQUE NOT NULL,
                company_name TEXT,
                email        TEXT,
                logo_url     TEXT,
                footer_name  TEXT,
                active       INTEGER DEFAULT 1,
                created_at   REAL DEFAULT (strftime('%s','now')),
                last_used    REAL
            )
        """)
        conn.commit()
    log.info("[api_keys] DB initialized at %s", _PAYMENTS_DB)


def create_api_key(
    company_name: str,
    email:        str,
    logo_url:     str = "",
    footer_name:  str = "",
) -> dict:
    """
    Generate a new API key. Returns {"key": "nce_...", "key_id": "..."}.
    The raw key is shown once — only its SHA-256 hash is stored.
    """
    if not API_KEYS_ENABLED:
        raise RuntimeError("API keys disabled (API_KEYS_ENABLED=false)")
    raw_key  = "nce_" + secrets.token_urlsafe(32)
    key_hash = hashlib.sha256(raw_key.encode()).hexdigest()
    key_id   = secrets.token_hex(8)
    with sqlite3.connect(_PAYMENTS_DB) as conn:
        conn.execute(
            """INSERT INTO api_keys
               (key_id, key_hash, company_name, email, logo_url, footer_name)
               VALUES (?,?,?,?,?,?)""",
            (key_id, key_hash, company_name, email, logo_url, footer_name),
        )
        conn.commit()
    log.info("[api_keys] Created key %s for %s (%s)", key_id, company_name, email)
    return {"key": raw_key, "key_id": key_id}


def verify_api_key(raw_key: str) -> Optional[dict]:
    """
    Validate a Bearer token. Returns customer metadata dict or None if invalid/inactive.
    Also updates last_used timestamp on success.

    When API_KEYS_ENABLED=false returns a dev mock so the B2B route is testable locally.
    """
    if not API_KEYS_ENABLED:
        return {"company_name": "dev", "email": "", "logo_url": "", "footer_name": "", "key_id": "dev"}
    if not raw_key:
        return None
    key_hash = hashlib.sha256(raw_key.encode()).hexdigest()
    with sqlite3.connect(_PAYMENTS_DB) as conn:
        row = conn.execute(
            """SELECT key_id, company_name, email, logo_url, footer_name, active
               FROM api_keys WHERE key_hash=?""",
            (key_hash,),
        ).fetchone()
        if not row or not row[5]:
            return None
        conn.execute(
            "UPDATE api_keys SET last_used=? WHERE key_hash=?",
            (time.time(), key_hash),
        )
        conn.commit()
    return {
        "key_id":       row[0],
        "company_name": row[1],
        "email":        row[2],
        "logo_url":     row[3],
        "footer_name":  row[4],
    }


def list_api_keys() -> list[dict]:
    """Return all keys — no raw key values, metadata only."""
    if not API_KEYS_ENABLED:
        return []
    with sqlite3.connect(_PAYMENTS_DB) as conn:
        rows = conn.execute(
            """SELECT key_id, company_name, email, logo_url, footer_name,
                      active, created_at, last_used
               FROM api_keys ORDER BY created_at DESC"""
        ).fetchall()
    return [
        {
            "key_id":       r[0],
            "company_name": r[1],
            "email":        r[2],
            "logo_url":     r[3],
            "footer_name":  r[4],
            "active":       bool(r[5]),
            "created_at":   r[6],
            "last_used":    r[7],
        }
        for r in rows
    ]


def revoke_api_key(key_id: str) -> bool:
    """Deactivate a key by key_id. Returns True if the key was found."""
    if not API_KEYS_ENABLED:
        return False
    with sqlite3.connect(_PAYMENTS_DB) as conn:
        cur = conn.execute(
            "UPDATE api_keys SET active=0 WHERE key_id=?", (key_id,)
        )
        conn.commit()
    return cur.rowcount > 0
