"""
Stripe payment integration — INACTIVE by default.
Activate: set PAYMENT_ENABLED=true in environment.

Provides:
  init_db()                 — create payments table in SQLite
  create_checkout_session() — Stripe Checkout (one-time or subscription)
  handle_webhook()          — validate Stripe webhook, update DB
  get_payment_status()      — check if a session is paid
"""
import logging
import os
import sqlite3
import time

log = logging.getLogger(__name__)

PAYMENT_ENABLED: bool = os.getenv("PAYMENT_ENABLED", "false").lower() == "true"

# Persistent disk path matches ChromaDB location — survives Render redeploys
_PAYMENTS_DB = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "permit_ai", "embeddings", "payments.db")
)

_STRIPE_SECRET_KEY      = os.getenv("STRIPE_SECRET_KEY", "")
_STRIPE_WEBHOOK_SECRET  = os.getenv("STRIPE_WEBHOOK_SECRET", "")
_STRIPE_SUCCESS_URL     = os.getenv("STRIPE_SUCCESS_URL", "https://ncenergy.fi/success")
_STRIPE_CANCEL_URL      = os.getenv("STRIPE_CANCEL_URL",  "https://ncenergy.fi/cancel")
_STRIPE_PRICE_ID_SINGLE = os.getenv("STRIPE_PRICE_ID_SINGLE", "")   # one-time report
_STRIPE_PRICE_ID_SUB    = os.getenv("STRIPE_PRICE_ID_SUB",    "")   # monthly subscription


def init_db() -> None:
    """Create payments table. Safe to call multiple times (NOOP when disabled)."""
    if not PAYMENT_ENABLED:
        return
    os.makedirs(os.path.dirname(_PAYMENTS_DB), exist_ok=True)
    with sqlite3.connect(_PAYMENTS_DB) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS payments (
                session_id     TEXT PRIMARY KEY,
                customer_email TEXT,
                amount_total   INTEGER,
                currency       TEXT,
                mode           TEXT,
                status         TEXT DEFAULT 'pending',
                created_at     REAL DEFAULT (strftime('%s','now')),
                paid_at        REAL
            )
        """)
        conn.commit()
    log.info("[payments] DB initialized at %s", _PAYMENTS_DB)


def create_checkout_session(
    customer_email: str,
    mode: str = "payment",          # "payment" | "subscription"
    success_url: str | None = None,
    cancel_url:  str | None = None,
) -> dict:
    """
    Create a Stripe Checkout Session.
    Returns {"url": ..., "session_id": ...}.
    Raises RuntimeError if PAYMENT_ENABLED is false.
    """
    if not PAYMENT_ENABLED:
        raise RuntimeError("Payment system is disabled (PAYMENT_ENABLED=false)")
    import stripe  # deferred — not imported at module level

    stripe.api_key = _STRIPE_SECRET_KEY
    price_id = _STRIPE_PRICE_ID_SUB if mode == "subscription" else _STRIPE_PRICE_ID_SINGLE

    session = stripe.checkout.Session.create(
        payment_method_types=["card"],
        mode=mode,
        customer_email=customer_email or None,
        line_items=[{"price": price_id, "quantity": 1}],
        success_url=(success_url or _STRIPE_SUCCESS_URL) + "?session_id={CHECKOUT_SESSION_ID}",
        cancel_url=cancel_url or _STRIPE_CANCEL_URL,
    )
    with sqlite3.connect(_PAYMENTS_DB) as conn:
        conn.execute(
            "INSERT OR IGNORE INTO payments (session_id, customer_email, mode) VALUES (?,?,?)",
            (session.id, customer_email, mode),
        )
        conn.commit()
    return {"url": session.url, "session_id": session.id}


def handle_webhook(payload: bytes, sig_header: str) -> dict:
    """
    Validate and process a Stripe webhook event.
    Returns {"status": "ok", "event_type": ...}.
    Raises ValueError on invalid signature.
    """
    if not PAYMENT_ENABLED:
        return {"status": "disabled"}
    import stripe

    stripe.api_key = _STRIPE_SECRET_KEY
    try:
        event = stripe.Webhook.construct_event(payload, sig_header, _STRIPE_WEBHOOK_SECRET)
    except stripe.error.SignatureVerificationError as exc:
        raise ValueError(f"Invalid Stripe signature: {exc}") from exc

    if event["type"] in ("checkout.session.completed", "invoice.paid"):
        sess = event["data"]["object"]
        session_id = sess.get("id") or sess.get("subscription", "")
        with sqlite3.connect(_PAYMENTS_DB) as conn:
            conn.execute(
                "UPDATE payments SET status='paid', paid_at=? WHERE session_id=?",
                (time.time(), session_id),
            )
            conn.commit()
        log.info("[payments] Marked paid: %s", session_id)

    return {"status": "ok", "event_type": event["type"]}


def get_payment_status(session_id: str) -> str:
    """Return 'paid', 'pending', 'not_found', 'disabled', or 'error'."""
    if not PAYMENT_ENABLED:
        return "disabled"
    try:
        with sqlite3.connect(_PAYMENTS_DB) as conn:
            row = conn.execute(
                "SELECT status FROM payments WHERE session_id=?", (session_id,)
            ).fetchone()
        return row[0] if row else "not_found"
    except Exception:
        return "error"
