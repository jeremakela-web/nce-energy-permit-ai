import chromadb, os, anthropic
from sentence_transformers import SentenceTransformer

DB_DIR = os.path.expanduser("~/bess_tool/permit_ai/embeddings")
model = SentenceTransformer("paraphrase-multilingual-mpnet-base-v2")
client = chromadb.PersistentClient(path=DB_DIR)
col = client.get_or_create_collection("permit_docs_v2")
claude = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

SYSTEM = """Olet Nordic Clean Energy (NCE) Energy Permit AI -asiantuntija. Avustat energia-alan hankkeiden lupaprosesseissa Suomessa. Tiimi: Jere Mäkelä (kehitys & AI), Jyrki Rintanen (energia & BESS), Alexander Ignatjev (rahoitus & kansainvälinen). Yhtiö: Kansallisvaranto Oy / Nordic Clean Energy. Kolme tuotealuetta: 1. Tuulivoima YVA — massamarkkina, FCG/Ramboll/Sweco ovat nykyiset kilpailijat, markkinahinta 150–400k€/hanke 2. BESS ympäristölupa — 30–80k€/hanke 3. SMR/ydinvoima pre-licensing — STUK YVL -dokumentit, 1–3,5M€/hanke Viranomaiset: Lupa- ja valvontavirasto (aloitti 1.1.2026), STUK (ydinvoima), AVI (vesiluvat), ELY-keskus. Strategia: Aloita tuulivoimasta (standardoitu prosessi, nopea skaalaus), lisää SMR moduulina. AI tekee 70% dokumentaatiosta, NCE veloittaa 50–60% markkinahinnasta, kate ~80%. Käyttäytyminen: Vastaa suomeksi. Ole konkreettinen — anna aina seuraava toimenpide."""

def ask(question: str) -> str:
    emb = model.encode([question]).tolist()
    results = col.query(query_embeddings=emb, n_results=5)
    context = "\n\n---\n\n".join(results["documents"][0])
    resp = claude.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1000,
        system=SYSTEM,
        messages=[{"role": "user", "content": f"Konteksti dokumenteista:\n{context}\n\nKysymys: {question}"}]
    )
    return resp.content[0].text

if __name__ == "__main__":
    q = input("Kysymys: ")
    print("\n" + ask(q))
