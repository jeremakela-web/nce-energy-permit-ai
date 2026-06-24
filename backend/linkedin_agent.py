"""
LinkedIn posting agent for NCE Permit AI.
Generates post drafts via Claude, stores them pending human approval.
No auto-publishing — Jere approves and posts manually, then marks published.
"""
import os
import secrets
import sqlite3
from datetime import datetime

import anthropic

client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

# Persistent disk — survives Render redeploys
POST_DB = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__), "..", "permit_ai", "embeddings", "post_queue.db"
    )
)

BRAND_VOICE = """
You are writing LinkedIn posts for NCE Permit AI (ncenergy.fi).

Brand voice:
- Professional but not corporate. Direct and confident.
- Never hype. Let facts speak.
- Always grounded in real numbers (8 countries, 20 project types, 6891 RAG chunks).
- Tagline when relevant: "The document is the interface. The engine is the product."
- Audience: energy developers, BESS operators, consultants, infrastructure investors, regulators.
- Language: English (default) or Finnish if specified.
- Length: 150-250 words. Short punchy sentences. 3-5 hashtags at end.
- Never use: exclamation marks, emojis unless explicitly requested, generic AI hype phrases.
- Always end with: ncenergy.fi

Mandatory hashtags pool (pick 3-5 relevant ones):
#EnergyPermitting #RegulatoryAI #BESS #WindEnergy #SMR #EnergyTransition
#CleanEnergy #RegulatoryIntelligence #NCEPermitAI
"""

POST_TYPES = {
    "product_update":     "Share a product milestone or new feature.",
    "market_insight":     "Share regulatory or market insight relevant to energy developers.",
    "customer_story":     "Share a use case or customer success (anonymized if needed).",
    "thought_leadership": "Share a strategic perspective on energy permitting or AI.",
    "country_launch":     "Announce expansion to a new country or project type.",
    "funding_news":       "Share funding or partnership news professionally.",
}


def init_post_db() -> None:
    """Create post_queue table. Safe to call multiple times."""
    os.makedirs(os.path.dirname(POST_DB), exist_ok=True)
    with sqlite3.connect(POST_DB) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS post_queue (
                post_id      TEXT PRIMARY KEY,
                post_type    TEXT,
                topic        TEXT,
                draft_text   TEXT,
                status       TEXT DEFAULT 'pending',
                created_at   TEXT,
                approved_at  TEXT,
                published_at TEXT,
                linkedin_url TEXT
            )
        """)
        conn.commit()


def generate_post_draft(
    post_type: str,
    topic: str,
    extra_context: str = "",
    language: str = "en",
) -> dict:
    """Generate a LinkedIn post draft via Claude. Returns {post_id, draft_text}."""
    lang_instruction = "Write in Finnish." if language == "fi" else "Write in English."
    type_instruction = POST_TYPES.get(post_type, "Write a general LinkedIn update.")

    prompt = f"""{BRAND_VOICE}

Post type: {post_type}
Instruction: {type_instruction}
{lang_instruction}

Topic / key message:
{topic}

Additional context:
{extra_context}

Write the LinkedIn post now. Output only the post text, nothing else."""

    message = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=600,
        messages=[{"role": "user", "content": prompt}],
    )
    draft_text = message.content[0].text.strip()
    post_id = "post_" + secrets.token_urlsafe(8)

    with sqlite3.connect(POST_DB) as conn:
        conn.execute(
            "INSERT INTO post_queue (post_id, post_type, topic, draft_text, created_at) VALUES (?,?,?,?,?)",
            (post_id, post_type, topic, draft_text, datetime.utcnow().isoformat()),
        )
        conn.commit()

    return {"post_id": post_id, "draft_text": draft_text}


def get_pending_posts() -> list:
    """Return all posts with status='pending', newest first."""
    with sqlite3.connect(POST_DB) as conn:
        rows = conn.execute(
            """SELECT post_id, post_type, topic, draft_text, created_at
               FROM post_queue WHERE status='pending' ORDER BY created_at DESC"""
        ).fetchall()
    return [
        {"post_id": r[0], "post_type": r[1], "topic": r[2], "draft_text": r[3], "created_at": r[4]}
        for r in rows
    ]


def approve_post(post_id: str, edited_text: str | None = None) -> dict:
    """Approve a post, optionally replacing draft_text with edited_text."""
    with sqlite3.connect(POST_DB) as conn:
        if edited_text:
            conn.execute(
                "UPDATE post_queue SET status='approved', approved_at=?, draft_text=? WHERE post_id=?",
                (datetime.utcnow().isoformat(), edited_text, post_id),
            )
        else:
            conn.execute(
                "UPDATE post_queue SET status='approved', approved_at=? WHERE post_id=?",
                (datetime.utcnow().isoformat(), post_id),
            )
        conn.commit()
    return {"post_id": post_id, "status": "approved"}


def reject_post(post_id: str) -> dict:
    """Mark a post as rejected."""
    with sqlite3.connect(POST_DB) as conn:
        conn.execute("UPDATE post_queue SET status='rejected' WHERE post_id=?", (post_id,))
        conn.commit()
    return {"post_id": post_id, "status": "rejected"}


def mark_published(post_id: str, linkedin_url: str | None = None) -> dict:
    """Mark post as published after manual LinkedIn posting."""
    with sqlite3.connect(POST_DB) as conn:
        conn.execute(
            "UPDATE post_queue SET status='published', published_at=?, linkedin_url=? WHERE post_id=?",
            (datetime.utcnow().isoformat(), linkedin_url, post_id),
        )
        conn.commit()
    return {"post_id": post_id, "status": "published", "linkedin_url": linkedin_url}
