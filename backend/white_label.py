"""
White-label PDF branding — activated per B2B API key.
No separate env flag: white-label activates automatically when
verify_api_key() returns a non-empty logo_url or footer_name.

Provides:
  NCE_LOGO_PATH             — default NCE logo path (fallback)
  get_customer_logo_path()  — download + cache a customer logo URL
"""
import hashlib
import logging
import os
from typing import Optional

log = logging.getLogger(__name__)

# Default NCE logo used when no customer logo is provided
NCE_LOGO_PATH = os.path.join(
    os.path.dirname(__file__), "static", "nce_energy_logo.png"
)

# Logo cache lives on the persistent disk so cached files survive redeploys
_LOGO_CACHE_DIR = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__), "..", "permit_ai", "embeddings", "_logo_cache"
    )
)


def get_customer_logo_path(logo_url: str) -> Optional[str]:
    """
    Download logo_url and cache it locally (keyed by SHA-256 of the URL).
    Converts any image format to PNG via Pillow.
    Returns the local path on success, None on any failure.
    """
    if not logo_url:
        return None
    try:
        import io
        import requests
        from PIL import Image

        url_hash = hashlib.sha256(logo_url.encode()).hexdigest()[:16]
        os.makedirs(_LOGO_CACHE_DIR, exist_ok=True)

        # Return cached file if it already exists
        cached_path = os.path.join(_LOGO_CACHE_DIR, f"{url_hash}.png")
        if os.path.exists(cached_path):
            return cached_path

        resp = requests.get(logo_url, timeout=10)
        resp.raise_for_status()
        img = Image.open(io.BytesIO(resp.content)).convert("RGBA")
        img.save(cached_path, "PNG")
        log.info("[white_label] Logo cached: %s → %s", logo_url, cached_path)
        return cached_path
    except Exception as exc:
        log.warning("[white_label] Logo download failed (%s): %s", logo_url, exc)
        return None
