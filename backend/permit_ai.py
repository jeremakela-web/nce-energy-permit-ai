"""
RAG-pohjainen lupaprosessi-AI.
Käyttää ChromaDB-vektorivarastoa (~/bess_tool/permit_ai/embeddings)
ja Claude-mallia vastausten generointiin.
"""

import os
from functools import lru_cache

import chromadb
import anthropic
from sentence_transformers import SentenceTransformer

_DB_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "permit_ai", "embeddings"))
_COLLECTION = "permit_docs"                      # v1 fallback; switched to v2 at runtime
_MODEL_ID = "claude-sonnet-4-6"
_EMBED_MODEL = "all-MiniLM-L6-v2"               # v1 fallback; switched to v2 at runtime

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
    return client.get_or_create_collection(_COLLECTION)


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
