from __future__ import annotations

import re
import unicodedata


def normalize_legal_text(raw_text: str | None) -> str | None:
    """Derive retrieval-friendly text without replacing the source raw_text."""
    if raw_text is None:
        return None

    normalized = unicodedata.normalize("NFC", raw_text)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized or None
