from __future__ import annotations

import re
import unicodedata


_SUPERSCRIPT_DIGITS = {
    "⁰": "_0",
    "¹": "_1",
    "²": "_2",
    "³": "_3",
    "⁴": "_4",
    "⁵": "_5",
    "⁶": "_6",
    "⁷": "_7",
    "⁸": "_8",
    "⁹": "_9",
}

_PREFIX_MAP = {
    "titlu": "titlu",
    "capitol": "capitol",
    "sectiune": "sectiune",
    "sectiunea": "sectiune",
    "articol": "art",
    "article": "art",
    "alineat": "alin",
    "paragraph": "alin",
    "litera": "lit",
    "letter": "lit",
    "punct": "pct",
    "point": "pct",
}


def _ascii_lower(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text)
    without_marks = "".join(ch for ch in normalized if not unicodedata.combining(ch))
    return without_marks.lower()


def _replace_superscript_digits(text: str) -> str:
    for superscript, replacement in _SUPERSCRIPT_DIGITS.items():
        text = text.replace(superscript, replacement)
    return text


def _slugify(text: str) -> str:
    text = _ascii_lower(_replace_superscript_digits(text))
    text = re.sub(r"[^a-z0-9]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text


def normalize_number(text: str) -> str:
    """
    Normalize Romanian legal numbering.

    Examples:
        "41^1" -> "41_1"
        "41¹"  -> "41_1"
        "(1)"  -> "1"
        "K)"   -> "k"
    """
    if not text:
        return ""

    normalized = _replace_superscript_digits(str(text).strip())
    normalized = normalized.replace("^", "_")
    normalized = _ascii_lower(normalized)
    normalized = re.sub(r"^(?:art\.?|articolul|alin\.?|lit\.?|pct\.?)\s*", "", normalized)
    normalized = re.sub(r"[^a-z0-9_]", "", normalized)
    normalized = re.sub(r"_+", "_", normalized).strip("_")
    return normalized


def make_law_id(law_title: str) -> str:
    """
    Build a deterministic Romanian law id from a known act title.

    Examples:
        "Codul muncii" -> "ro.codul_muncii"
        "Legea nr. 53/2003" -> "ro.lege_53_2003"
    """
    title = _ascii_lower(str(law_title).strip())
    compact = re.sub(r"\s+", " ", title)

    numbered_patterns = [
        (r"\b(?:legea|lege)\s+(?:nr\.?\s*)?(\d+)\s*/\s*(\d+)\b", "lege"),
        (r"\b(?:oug|o\.u\.g\.|ordonanta de urgenta)\s+(?:nr\.?\s*)?(\d+)\s*/\s*(\d+)\b", "oug"),
        (r"\b(?:og|o\.g\.|ordonanta guvernului)\s+(?:nr\.?\s*)?(\d+)\s*/\s*(\d+)\b", "og"),
        (r"\b(?:hg|h\.g\.|hotararea guvernului)\s+(?:nr\.?\s*)?(\d+)\s*/\s*(\d+)\b", "hg"),
    ]
    for pattern, prefix in numbered_patterns:
        match = re.search(pattern, compact)
        if match:
            return f"ro.{prefix}_{match.group(1)}_{match.group(2)}"

    if re.fullmatch(r"constitutia(?: romaniei)?", compact):
        return "ro.constitutia"

    slug = _slugify(compact)
    if slug.startswith("codul_"):
        return f"ro.{slug}"
    if slug.startswith("legea_"):
        slug = "lege_" + slug.removeprefix("legea_")
    return f"ro.{slug}"


def make_article_segment(article_number: str) -> str:
    return f"art_{normalize_number(article_number)}"


def make_paragraph_segment(paragraph_number: str) -> str:
    return f"alin_{normalize_number(paragraph_number)}"


def make_letter_segment(letter_number: str) -> str:
    return f"lit_{normalize_number(letter_number)}"


def make_point_segment(point_number: str) -> str:
    return f"pct_{normalize_number(point_number)}"


def make_unit_segment(level_type: str, level_value: str) -> str:
    level = _ascii_lower(level_type)
    prefix = _PREFIX_MAP.get(level, level)

    if prefix == "art":
        return make_article_segment(level_value)
    if prefix == "alin":
        return make_paragraph_segment(level_value)
    if prefix == "lit":
        return make_letter_segment(level_value)
    if prefix == "pct":
        return make_point_segment(level_value)
    return f"{prefix}_{normalize_number(level_value)}"


def make_unit_id(corpus_id: str, hierarchy_path: list) -> str:
    """
    Generate a deterministic legal unit id.

    Existing parser callers pass tuple paths such as
    [("articol", "41"), ("alineat", "1")], which remain supported.
    """
    parts = [corpus_id]
    parts.extend(make_unit_segment(level_type, level_val) for level_type, level_val in hierarchy_path)
    return ".".join(part for part in parts if part)


def make_parent_unit_id(law_id: str, hierarchy_path: list) -> str | None:
    if not hierarchy_path:
        return None
    return make_unit_id(law_id, hierarchy_path[:-1])


def make_canonical_id(law_id: str, hierarchy_path: list) -> str:
    base = law_id.removeprefix("ro.")
    segments = [make_unit_segment(level_type, level_val) for level_type, level_val in hierarchy_path]
    return ":".join([base, *segments]) if segments else base
