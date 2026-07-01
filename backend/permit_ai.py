"""
RAG-pohjainen lupaprosessi-AI.
Käyttää ChromaDB-vektorivarastoa (~/bess_tool/permit_ai/embeddings)
ja Claude-mallia vastausten generointiin.
"""

import os
from collections import deque
from functools import lru_cache

import chromadb
import anthropic
from sentence_transformers import SentenceTransformer

_DB_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "permit_ai", "embeddings"))
_COLLECTION = "permit_docs"                      # v1 fallback; switched to v2 at runtime
_MODEL_ID = "claude-sonnet-4-6"
_EMBED_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"   # multilingual; switched to v2 at runtime

_COLLECTION_V2 = "permit_docs_v2"
_EMBED_MODEL_V2 = "paraphrase-multilingual-mpnet-base-v2"

SYSTEM_PROMPT = (
    "Olet Nordic Clean Energy (NCE) Energy Permit AI -asiantuntija. "
    "Avustat energia-alan hankkeiden lupaprosesseissa Suomessa. "
    "Tiimi: Jere Mäkelä (kehitys & AI), Jyrki Rintanen (energia & BESS), "
    "Alexander Ignatjev (rahoitus & kansainvälinen). "
    "Yhtiö: Kansallisvaranto Oy / Nordic Clean Energy. "
    "Kolme tuotealuetta: "
    "1. Tuulivoima YVA — massamarkkina, FCG/Ramboll/Sweco ovat nykyiset kilpailijat, markkinahinta 150–400k€/hanke "
    "2. BESS ympäristölupa — 30–80k€/hanke "
    "3. SMR/ydinvoima pre-licensing — STUK YVL -dokumentit, 1–3,5M€/hanke "
    "Viranomaiset: Lupa- ja valvontavirasto (aloitti 1.1.2026), STUK (ydinvoima), AVI (vesiluvat), ELY-keskus. "
    "Strategia: Aloita tuulivoimasta (standardoitu prosessi, nopea skaalaus), lisää SMR moduulina. "
    "AI tekee 70% dokumentaatiosta, NCE veloittaa 50–60% markkinahinnasta, kate ~80%. "
    "Käyttäytyminen: Vastaa suomeksi. Ole konkreettinen — anna aina seuraava toimenpide."
)


@lru_cache(maxsize=1)
def _get_embed_model() -> SentenceTransformer:
    return SentenceTransformer(_EMBED_MODEL)


@lru_cache(maxsize=1)
def _get_collection() -> chromadb.Collection:
    client = chromadb.PersistentClient(path=_DB_DIR)
    return client.get_or_create_collection(_COLLECTION, metadata={"hnsw:space": "cosine"})


def query_permit_ai(question: str, n_results: int = 5) -> dict:
    """
    Hae relevantit dokumenttichunkit vektorivarastosta ja generoi vastaus Claudella.
    Palauttaa {'answer': str, 'sources': list[str]}.
    """
    embed_model = _get_embed_model()
    col = _get_collection()

    embedding = embed_model.encode([question]).tolist()
    results = col.query(query_embeddings=embedding, n_results=n_results)

    docs = results["documents"][0]
    ids  = results["ids"][0]
    context = "\n\n---\n\n".join(docs)

    # Lähdetiedostot ilman chunk-indeksiä
    sources = sorted({
        "_".join(id_.split("_")[:-1])   # "sjv2024_fingrid.pdf_3" → "sjv2024_fingrid.pdf"
        for id_ in ids
    })

    claude = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    resp = claude.messages.create(
        model=_MODEL_ID,
        max_tokens=1200,
        system=SYSTEM_PROMPT,
        messages=[{
            "role": "user",
            "content": f"Konteksti dokumenteista:\n{context}\n\nKysymys: {question}",
        }],
    )

    return {
        "answer":  resp.content[0].text,
        "sources": sources,
    }


def activate_v2() -> None:
    """Switch retrieval to permit_docs_v2 + mpnet (called after re-index completes)."""
    global _COLLECTION, _EMBED_MODEL
    _COLLECTION = _COLLECTION_V2
    _EMBED_MODEL = _EMBED_MODEL_V2
    _get_embed_model.cache_clear()
    _get_collection.cache_clear()


# ── Interactive chat (COMMIT 3) ───────────────────────────────────────────────

_CHAT_SESSIONS: dict[str, deque] = {}
_CHAT_MAX_MSGS = 20  # 10 turns × 2

_CHAT_SYSTEM = (
    "Olet NCE Energy Permit AI -asiantuntija. Avustat energia-alan hankkeiden lupaprosesseissa "
    "Suomessa ja muissa Pohjoismaissa sekä Puolassa. "
    "Käytä annettua RAG-dokumenttikontekstia vastaustesi tukena. "
    "Ole konkreettinen — anna aina selkeä seuraava toimenpide. "
    "Vastaa käyttäjän kirjoittamalla kielellä."
)


@lru_cache(maxsize=1)
def _get_embed_v2() -> SentenceTransformer:
    return SentenceTransformer(_EMBED_MODEL_V2)


@lru_cache(maxsize=1)
def _get_col_v2() -> chromadb.Collection:
    client = chromadb.PersistentClient(path=_DB_DIR)
    return client.get_or_create_collection(_COLLECTION_V2)


def query_permit_ai_chat(
    question: str,
    session_id: str,
    n_results: int = 6,
    hanketyyppi: str = "",
    country: str = "FI",
) -> dict:
    """
    Stateful RAG chat with per-session conversation history.
    Returns {'answer': str, 'sources': list[str]}.
    """
    from source_policy import is_chunk_relevant  # permit_ai/ is on sys.path via main.py

    model = _get_embed_v2()
    col   = _get_col_v2()

    # RAG: over-fetch then filter by hanketyyppi
    embedding        = model.encode([question]).tolist()
    allowed_countries = list({country, "EU"})
    candidates = col.query(
        query_embeddings=embedding,
        n_results=min(n_results * 5, 50),
        where={"country": {"$in": allowed_countries}},
    )
    docs  = candidates["documents"][0]
    metas = candidates["metadatas"][0]
    ids   = candidates["ids"][0]

    if hanketyyppi:
        filtered = [
            (d, m, i) for d, m, i in zip(docs, metas, ids)
            if is_chunk_relevant(m or {}, hanketyyppi)
        ]
        if not filtered:
            filtered = list(zip(docs, metas, ids))
    else:
        filtered = list(zip(docs, metas, ids))

    filtered = filtered[:n_results]
    if filtered:
        f_docs, f_metas, f_ids = zip(*filtered)
    else:
        f_docs, f_metas, f_ids = [], [], []

    context = "\n\n---\n\n".join(f_docs) if f_docs else "(Ei RAG-kontekstia.)"
    sources = sorted({(m or {}).get("source", i) for m, i in zip(f_metas, f_ids)})

    # Session history
    if session_id not in _CHAT_SESSIONS:
        _CHAT_SESSIONS[session_id] = deque(maxlen=_CHAT_MAX_MSGS)
    history = _CHAT_SESSIONS[session_id]

    # Build message list: history + new turn (RAG context injected into user message)
    messages = list(history) + [{
        "role": "user",
        "content": f"RAG-konteksti:\n{context}\n\n---\nKysymys: {question}",
    }]

    claude = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    resp   = claude.messages.create(
        model=_MODEL_ID,
        max_tokens=1200,
        system=_CHAT_SYSTEM,
        messages=messages,
    )
    answer = resp.content[0].text

    # Store bare question/answer (no RAG blob) for next turn context
    history.append({"role": "user",      "content": question})
    history.append({"role": "assistant", "content": answer})

    return {"answer": answer, "sources": sources}
