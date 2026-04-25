from __future__ import annotations

import re
from datetime import date
from typing import Any, Mapping

from pydantic import BaseModel

from ingestion.contracts import ParserActMetadata, ParsedLegalUnit
from ingestion.legal_domains import get_registered_legal_domain
from ingestion.legal_ids import (
    make_canonical_id,
    make_law_id,
    make_parent_unit_id,
    make_unit_id,
)
from ingestion.normalizer import normalize_legal_text


CITABLE_LEVELS = {"articol", "alineat", "litera", "punct"}


def parsed_legal_unit_to_legal_unit_dict(
    unit: ParsedLegalUnit,
    *,
    include_parser_warnings: bool = True,
) -> dict[str, Any]:
    """Export the canonical ingestion LegalUnit dict for JSON/import consumers."""
    return unit.to_legal_unit_dict(include_parser_warnings=include_parser_warnings)


def build_parsed_legal_unit(
    *,
    law_title: str,
    raw_text: str,
    hierarchy_path: list[tuple[str, str]] | None = None,
    law_id: str | None = None,
    source_id: str | None = None,
    source_url: str | None = None,
    status: str | None = None,
    act_type: str | None = None,
    act_number: str | None = None,
    publication_date: date | None = None,
    effective_date: date | None = None,
    version_start: date | None = None,
    version_end: date | None = None,
    legal_concepts: list[str] | None = None,
    parser_warnings: list[str] | None = None,
) -> ParsedLegalUnit:
    path = hierarchy_path or []
    resolved_law_id = law_id or make_law_id(law_title)
    resolved_domain = get_registered_legal_domain(resolved_law_id) or "unknown"
    warnings = list(parser_warnings or [])

    if resolved_domain == "unknown":
        warnings.append("legal_domain_unknown")
    if source_id is None:
        warnings.append("source_id_unknown")
    if source_url is None:
        warnings.append("source_url_unknown")
    if status is None or status == "unknown":
        warnings.append("status_unknown")
    if publication_date is None:
        warnings.append("publication_date_unknown")
    if effective_date is None:
        warnings.append("effective_date_unknown")
    if version_start is None:
        warnings.append("version_start_unknown")
    if version_end is None:
        warnings.append("version_end_unknown")

    parsed_act_type, parsed_act_number = _parse_act_identity(law_title)
    resolved_act_type = act_type if act_type is not None else parsed_act_type
    resolved_act_number = act_number if act_number is not None else parsed_act_number

    return ParsedLegalUnit(
        id=make_unit_id(resolved_law_id, path),
        canonical_id=make_canonical_id(resolved_law_id, path),
        source_id=source_id,
        law_id=resolved_law_id,
        law_title=law_title,
        act_type=resolved_act_type,
        act_number=resolved_act_number,
        publication_date=publication_date,
        effective_date=effective_date,
        version_start=version_start,
        version_end=version_end,
        status=status or "unknown",
        hierarchy_path=render_hierarchy_path(path),
        article_number=_level_value(path, "articol"),
        paragraph_number=_level_value(path, "alineat"),
        letter_number=_level_value(path, "litera"),
        point_number=_level_value(path, "punct"),
        raw_text=raw_text,
        normalized_text=normalize_legal_text(raw_text),
        legal_domain=resolved_domain,
        legal_concepts=legal_concepts or [],
        source_url=source_url,
        parent_id=make_parent_unit_id(resolved_law_id, path),
        children_ids=[],
        outgoing_reference_ids=[],
        incoming_reference_ids=[],
        parser_warnings=warnings,
    )


def build_legal_unit_dict(
    *,
    law_title: str,
    raw_text: str,
    hierarchy_path: list[tuple[str, str]] | None = None,
    law_id: str | None = None,
    source_id: str | None = None,
    source_url: str | None = None,
    status: str | None = None,
    parser_warnings: list[str] | None = None,
) -> dict[str, Any]:
    parsed = build_parsed_legal_unit(
        law_title=law_title,
        raw_text=raw_text,
        hierarchy_path=hierarchy_path,
        law_id=law_id,
        source_id=source_id,
        source_url=source_url,
        status=status,
        parser_warnings=parser_warnings,
    )
    return parsed_legal_unit_to_legal_unit_dict(parsed)


def legacy_unit_to_parsed_legal_unit(
    legacy_unit: Mapping[str, Any],
    act_metadata: ParserActMetadata | BaseModel | Mapping[str, Any],
) -> ParsedLegalUnit:
    metadata = _metadata_dict(act_metadata)
    law_id = metadata.get("law_id") or legacy_unit.get("corpus_id")
    law_title = metadata.get("law_title") or law_id or "unknown"
    source_url = metadata.get("source_url")
    source_id = metadata.get("source_id")
    status = metadata.get("status")
    raw_text = str(legacy_unit.get("raw_text") or legacy_unit.get("text") or "")
    path = citable_path_from_unit_id(str(legacy_unit.get("id", "")), law_id)

    return build_parsed_legal_unit(
        law_title=law_title,
        law_id=law_id,
        raw_text=raw_text,
        hierarchy_path=path,
        source_id=source_id,
        source_url=source_url,
        status=status,
        act_type=metadata.get("act_type"),
        act_number=metadata.get("act_number"),
        publication_date=metadata.get("publication_date"),
        effective_date=metadata.get("effective_date"),
        version_start=metadata.get("version_start"),
        version_end=metadata.get("version_end"),
        parser_warnings=["legacy_unit_converted"],
    )


def legacy_unit_to_legal_unit_dict(
    legacy_unit: Mapping[str, Any],
    act_metadata: ParserActMetadata | BaseModel | Mapping[str, Any],
) -> dict[str, Any]:
    parsed = legacy_unit_to_parsed_legal_unit(legacy_unit, act_metadata)
    return parsed_legal_unit_to_legal_unit_dict(parsed)


def render_hierarchy_path(hierarchy_path: list[tuple[str, str]]) -> list[str]:
    rendered: list[str] = []
    for level_type, value in hierarchy_path:
        level = level_type.lower()
        if level == "articol":
            rendered.append(f"Art. {_strip_article_label(value)}")
        elif level == "alineat":
            rendered.append(f"Alin. ({_strip_wrapping_punctuation(value)})")
        elif level == "litera":
            rendered.append(f"Lit. {_strip_wrapping_punctuation(value).lower()})")
        elif level == "punct":
            rendered.append(f"Pct. {_strip_wrapping_punctuation(value)}")
        else:
            rendered.append(str(value))
    return rendered


def citable_path_from_unit_id(unit_id: str, law_id: str | None = None) -> list[tuple[str, str]]:
    if law_id and unit_id.startswith(f"{law_id}."):
        raw_segments = unit_id.removeprefix(f"{law_id}.").split(".")
    else:
        raw_segments = unit_id.split(".")[2:]

    path: list[tuple[str, str]] = []
    for segment in raw_segments:
        if segment.startswith("art_"):
            path.append(("articol", segment.removeprefix("art_")))
        elif segment.startswith("alin_"):
            path.append(("alineat", segment.removeprefix("alin_")))
        elif segment.startswith("lit_"):
            path.append(("litera", segment.removeprefix("lit_")))
        elif segment.startswith("pct_"):
            path.append(("punct", segment.removeprefix("pct_")))
    return path


def _metadata_dict(metadata: ParserActMetadata | BaseModel | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(metadata, BaseModel):
        return metadata.model_dump()
    return dict(metadata)


def _level_value(hierarchy_path: list[tuple[str, str]], level_type: str) -> str | None:
    for current_type, value in reversed(hierarchy_path):
        if current_type.lower() == level_type:
            if level_type == "articol":
                return _strip_article_label(value)
            return _strip_wrapping_punctuation(value).lower() if level_type == "litera" else _strip_wrapping_punctuation(value)
    return None


def _strip_article_label(value: str) -> str:
    text = str(value).strip()
    text = re.sub(r"^(?:art\.?|articolul)\s*", "", text, flags=re.IGNORECASE)
    return text.strip()


def _strip_wrapping_punctuation(value: str) -> str:
    text = str(value).strip()
    text = re.sub(r"^(?:alin\.?|lit\.?|pct\.?)\s*", "", text, flags=re.IGNORECASE)
    return text.strip().strip("() .")


def _parse_act_identity(law_title: str) -> tuple[str | None, str | None]:
    text = str(law_title).strip()
    match = re.search(r"\b(legea|lege)\s+(?:nr\.?\s*)?(\d+\s*/\s*\d+)\b", text, re.IGNORECASE)
    if match:
        return "lege", re.sub(r"\s+", "", match.group(2))
    if re.search(r"\bcodul\b", text, re.IGNORECASE):
        return "cod", None
    return None, None
