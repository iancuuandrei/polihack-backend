from __future__ import annotations

import math
import re
import unicodedata
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from ..schemas.retrieval import (
    RawExactCitation,
    RawRetrievalRequest,
    RawRetrievalResponse,
    RetrievalCandidate,
)
from .retrieval_scoring import ScoreBreakdown, reciprocal_rank_fusion, weighted_retrieval_score


DEFAULT_METHOD_LIMIT_MULTIPLIER = 3
FTS_METHOD = "fts"
EXACT_METHOD = "exact_citation"
DENSE_METHOD = "dense_optional"


class RawRetrievalStore(Protocol):
    async def exact_citation_lookup(
        self,
        citations: Sequence[dict[str, Any]],
        *,
        filters: Mapping[str, Any],
        limit: int,
    ) -> list[dict[str, Any]]:
        ...

    async def lexical_search(
        self,
        question: str,
        *,
        filters: Mapping[str, Any],
        limit: int,
    ) -> list[dict[str, Any]]:
        ...

    async def dense_search(
        self,
        query_embedding: Sequence[float],
        *,
        filters: Mapping[str, Any],
        limit: int,
    ) -> list[dict[str, Any]]:
        ...


class PostgresRawRetrievalStore:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session
        self._unit_columns: set[str] | None = None

    async def rollback_after_error(self) -> None:
        await self.session.rollback()

    async def exact_citation_lookup(
        self,
        citations: Sequence[dict[str, Any]],
        *,
        filters: Mapping[str, Any],
        limit: int,
    ) -> list[dict[str, Any]]:
        if not citations:
            return []
        available_columns = await self._get_unit_columns()
        select_columns = _unit_select_columns_sql(available_columns, alias="u")
        rows: list[dict[str, Any]] = []
        seen: set[str] = set()
        for citation in citations:
            clauses, params = _sql_filters(
                filters,
                alias="u",
                available_columns=available_columns,
            )
            law_id = _text_value(citation.get("law_id"))
            article_number = _text_value(citation.get("article_number"))
            if not law_id or not article_number:
                continue
            clauses.extend(
                [
                    _sql_text_equals(available_columns, "law_id", "law_id", alias="u"),
                    _sql_text_equals(
                        available_columns,
                        "article_number",
                        "article_number",
                        alias="u",
                    ),
                ]
            )
            params.update({"law_id": law_id, "article_number": article_number})
            for field_name in ("paragraph_number", "letter_number", "point_number"):
                value = _text_value(citation.get(field_name))
                if value:
                    clauses.append(
                        _sql_text_equals(
                            available_columns,
                            field_name,
                            field_name,
                            alias="u",
                        )
                    )
                    params[field_name] = value
            params["limit"] = max(1, limit - len(rows))
            order_by = _unit_order_by_sql(available_columns, alias="u")
            async with self.session.begin_nested():
                result = await self.session.execute(
                    text(
                        f"""
                        SELECT {select_columns}
                        FROM legal_units u
                        WHERE {' AND '.join(clauses)}
                        ORDER BY {order_by}
                        LIMIT :limit
                        """
                    ),
                    params,
                )
            for row in result.mappings().all():
                unit = _unit_from_row(row)
                if unit["id"] not in seen:
                    rows.append(unit)
                    seen.add(unit["id"])
            if len(rows) >= limit:
                break
        return rows

    async def lexical_search(
        self,
        question: str,
        *,
        filters: Mapping[str, Any],
        limit: int,
    ) -> list[dict[str, Any]]:
        available_columns = await self._get_unit_columns()
        clauses, params = _sql_filters(
            filters,
            alias="u",
            available_columns=available_columns,
        )
        select_columns = _unit_select_columns_sql(available_columns, alias="u")
        search_document = _unit_search_text_sql(available_columns, alias="u")
        params.update({"question": question, "limit": limit})
        try:
            async with self.session.begin_nested():
                result = await self.session.execute(
                    text(
                        f"""
                        WITH q AS (
                            SELECT websearch_to_tsquery('simple', :question) AS query
                        )
                        SELECT
                            {select_columns},
                            ts_rank_cd(
                                to_tsvector('simple', {search_document}),
                                q.query
                            ) AS bm25_score
                        FROM legal_units u, q
                        WHERE q.query @@ to_tsvector('simple', {search_document})
                          AND {' AND '.join(clauses)}
                        ORDER BY bm25_score DESC, u.id
                        LIMIT :limit
                        """
                    ),
                    params,
                )
            return [_unit_from_row(row) for row in result.mappings().all()]
        except Exception:
            return await self._lexical_ilike_fallback(
                question,
                filters=filters,
                limit=limit,
                available_columns=available_columns,
            )

    async def _lexical_ilike_fallback(
        self,
        question: str,
        *,
        filters: Mapping[str, Any],
        limit: int,
        available_columns: set[str] | None = None,
    ) -> list[dict[str, Any]]:
        terms = _query_terms(question)[:8]
        if not terms:
            return []
        if available_columns is None:
            available_columns = await self._get_unit_columns()
        clauses, params = _sql_filters(
            filters,
            alias="u",
            available_columns=available_columns,
        )
        select_columns = _unit_select_columns_sql(available_columns, alias="u")
        search_document = _unit_search_text_sql(available_columns, alias="u")
        if search_document == "''":
            return []
        match_clauses: list[str] = []
        score_parts: list[str] = []
        for index, term in enumerate(terms):
            param_name = f"term_{index}"
            params[param_name] = f"%{term}%"
            condition = f"{search_document} ILIKE :{param_name}"
            match_clauses.append(condition)
            score_parts.append(f"CASE WHEN {condition} THEN 1.0 ELSE 0.0 END")
        params["limit"] = limit
        result = await self.session.execute(
            text(
                f"""
                SELECT
                    {select_columns},
                    (({' + '.join(score_parts)}) / {float(len(terms))}) AS bm25_score
                FROM legal_units u
                WHERE ({' OR '.join(match_clauses)})
                  AND {' AND '.join(clauses)}
                ORDER BY bm25_score DESC, u.id
                LIMIT :limit
                """
            ),
            params,
        )
        return [_unit_from_row(row) for row in result.mappings().all()]

    async def dense_search(
        self,
        query_embedding: Sequence[float],
        *,
        filters: Mapping[str, Any],
        limit: int,
    ) -> list[dict[str, Any]]:
        available_columns = await self._get_unit_columns()
        clauses, params = _sql_filters(
            filters,
            alias="u",
            available_columns=available_columns,
        )
        select_columns = _unit_select_columns_sql(available_columns, alias="u")
        params.update(
            {
                "embedding": _vector_literal(query_embedding),
                "limit": limit,
            }
        )
        async with self.session.begin_nested():
            result = await self.session.execute(
                text(
                    f"""
                    SELECT
                        {select_columns},
                        greatest(0.0, least(1.0, 1.0 - (e.embedding <=> :embedding::vector))) AS dense_score
                    FROM legal_embeddings e
                    JOIN legal_units u ON u.id = e.legal_unit_id
                    WHERE {' AND '.join(clauses)}
                    ORDER BY e.embedding <=> :embedding::vector, u.id
                    LIMIT :limit
                    """
                ),
                params,
            )
        return [_unit_from_row(row) for row in result.mappings().all()]

    async def _get_unit_columns(self) -> set[str]:
        if self._unit_columns is None:
            async with self.session.begin_nested():
                result = await self.session.execute(
                    text(
                        """
                        SELECT a.attname AS column_name
                        FROM pg_attribute a
                        WHERE a.attrelid = 'legal_units'::regclass
                          AND a.attnum > 0
                          AND NOT a.attisdropped
                        """
                    )
                )
            self._unit_columns = {
                str(row["column_name"]) for row in result.mappings().all()
            }
        return self._unit_columns


class EmptyRawRetrievalStore:
    async def exact_citation_lookup(
        self,
        citations: Sequence[dict[str, Any]],
        *,
        filters: Mapping[str, Any],
        limit: int,
    ) -> list[dict[str, Any]]:
        return []

    async def lexical_search(
        self,
        question: str,
        *,
        filters: Mapping[str, Any],
        limit: int,
    ) -> list[dict[str, Any]]:
        return []

    async def dense_search(
        self,
        query_embedding: Sequence[float],
        *,
        filters: Mapping[str, Any],
        limit: int,
    ) -> list[dict[str, Any]]:
        return []


@dataclass
class _CandidateAccumulator:
    unit: dict[str, Any]
    bm25: float = 0.0
    dense: float = 0.0
    exact_citation_boost: float = 0.0
    methods: set[str] = field(default_factory=set)
    rrf: float = 0.0


class RawRetriever:
    def __init__(
        self,
        store: RawRetrievalStore,
        *,
        initial_warnings: Sequence[str] | None = None,
    ) -> None:
        self.store = store
        self.initial_warnings = list(initial_warnings or [])

    async def retrieve(self, request: RawRetrievalRequest) -> RawRetrievalResponse:
        top_k = request.top_k
        method_limit = max(top_k, top_k * DEFAULT_METHOD_LIMIT_MULTIPLIER)
        filters = _normalized_filters(request)
        warnings = list(self.initial_warnings)
        retrieval_methods = [EXACT_METHOD, FTS_METHOD, DENSE_METHOD]
        rankings: dict[str, list[str]] = {}
        candidates: dict[str, _CandidateAccumulator] = {}

        citation_filters = _citation_filters(request)
        exact_rows = await self._call_store(
            self.store.exact_citation_lookup(
                citation_filters,
                filters=filters,
                limit=method_limit,
            ),
            warnings=warnings,
            warning_code="exact_citation_lookup_failed",
            debug=request.debug,
        )
        rankings[EXACT_METHOD] = [row["id"] for row in exact_rows]
        for row in exact_rows:
            accumulator = candidates.setdefault(
                row["id"],
                _CandidateAccumulator(unit=row),
            )
            accumulator.exact_citation_boost = 1.0
            accumulator.methods.add(EXACT_METHOD)

        fts_rows = await self._call_store(
            self.store.lexical_search(request.question, filters=filters, limit=method_limit),
            warnings=warnings,
            warning_code="fts_retrieval_failed",
            debug=request.debug,
        )
        max_bm25 = max([_float(row.get("bm25_score")) for row in fts_rows] or [0.0])
        rankings[FTS_METHOD] = [row["id"] for row in fts_rows]
        for row in fts_rows:
            accumulator = candidates.setdefault(
                row["id"],
                _CandidateAccumulator(unit=row),
            )
            raw_score = _float(row.get("bm25_score"))
            accumulator.bm25 = max(accumulator.bm25, _normalize_score(raw_score, max_bm25))
            accumulator.methods.add(FTS_METHOD)

        dense_rows: list[dict[str, Any]] = []
        if request.query_embedding:
            dense_rows = await self._call_store(
                self.store.dense_search(
                    request.query_embedding,
                    filters=filters,
                    limit=method_limit,
                ),
                warnings=warnings,
                warning_code="dense_retrieval_failed_or_unavailable",
                debug=request.debug,
            )
        else:
            warnings.append("dense_retrieval_skipped_no_query_embedding")
        rankings[DENSE_METHOD] = [row["id"] for row in dense_rows]
        for row in dense_rows:
            accumulator = candidates.setdefault(
                row["id"],
                _CandidateAccumulator(unit=row),
            )
            accumulator.dense = max(accumulator.dense, _clamp01(_float(row.get("dense_score"))))
            accumulator.methods.add(DENSE_METHOD)

        domain_rank = [
            unit_id
            for unit_id, candidate in sorted(candidates.items())
            if _domain_match(filters.get("legal_domain"), candidate.unit.get("legal_domain")) > 0.0
        ]
        if domain_rank:
            rankings["domain_filtered_search"] = domain_rank
        rrf_scores = reciprocal_rank_fusion(rankings)
        for unit_id, score in rrf_scores.items():
            if unit_id in candidates:
                candidates[unit_id].rrf = score

        dense_available = bool(dense_rows)
        ranked_candidates = [
            self._to_candidate(
                accumulator,
                filters=filters,
                question=request.question,
                dense_available=dense_available,
            )
            for accumulator in candidates.values()
        ]
        ranked_candidates.sort(
            key=lambda candidate: (
                candidate.retrieval_score,
                candidate.score_breakdown.get("exact_citation_boost", 0.0),
                candidate.score_breakdown.get("rrf", 0.0),
            ),
            reverse=True,
        )
        ranked_candidates = ranked_candidates[:top_k]
        for rank, candidate in enumerate(ranked_candidates, start=1):
            candidate.rank = rank

        debug_payload = None
        if request.debug:
            debug_payload = {
                "candidate_count_before_top_k": len(candidates),
                "candidate_count": len(ranked_candidates),
                "top_k": top_k,
                "filters": filters,
                "dense_available": dense_available,
                "rankings": rankings,
                "rrf_scores": rrf_scores,
                "warnings": warnings,
            }

        return RawRetrievalResponse(
            candidates=ranked_candidates,
            retrieval_methods=retrieval_methods,
            warnings=warnings,
            debug=debug_payload,
        )

    async def _call_store(
        self,
        store_call,
        *,
        warnings: list[str],
        warning_code: str,
        debug: bool,
    ) -> list[dict[str, Any]]:
        try:
            return await store_call
        except Exception as exc:
            warnings.append(_debug_warning(warning_code, exc) if debug else warning_code)
            rollback = getattr(self.store, "rollback_after_error", None)
            if rollback is not None:
                try:
                    await rollback()
                except Exception:
                    pass
            return []

    def _to_candidate(
        self,
        accumulator: _CandidateAccumulator,
        *,
        filters: Mapping[str, Any],
        question: str,
        dense_available: bool,
    ) -> RetrievalCandidate:
        unit = accumulator.unit
        breakdown = ScoreBreakdown(
            bm25=accumulator.bm25,
            dense=accumulator.dense,
            rrf=accumulator.rrf,
            domain_match=_domain_match(filters.get("legal_domain"), unit.get("legal_domain")),
            metadata_validity=_metadata_validity(unit),
            exact_citation_boost=accumulator.exact_citation_boost,
        )
        score = weighted_retrieval_score(breakdown, dense_available=dense_available)
        return RetrievalCandidate(
            unit_id=unit["id"],
            unit=unit,
            rank=0,
            retrieval_score=round(_clamp01(score), 6),
            score_breakdown={
                "bm25": round(breakdown.bm25, 6),
                "dense": round(breakdown.dense, 6),
                "domain_match": round(breakdown.domain_match, 6),
                "metadata_validity": round(breakdown.metadata_validity, 6),
                "exact_citation_boost": round(breakdown.exact_citation_boost, 6),
                "rrf": round(breakdown.rrf, 6),
            },
            matched_terms=_matched_terms(question, unit),
            why_retrieved=_why_retrieved(accumulator.methods),
        )


def _normalized_filters(request: RawRetrievalRequest) -> dict[str, Any]:
    filters = dict(request.retrieval_filters or {})
    filters.update(request.filters or {})
    filters.pop("exact_citation_filters", None)
    if filters.get("legal_domain"):
        filters["legal_domain"] = _canonical_domain(filters["legal_domain"])
    return filters


def _citation_filters(request: RawRetrievalRequest) -> list[dict[str, Any]]:
    filters: list[dict[str, Any]] = []
    for raw in request.retrieval_filters.get("exact_citation_filters", []) or []:
        if isinstance(raw, dict):
            filters.append(_canonical_citation_filter(raw))
    for citation in request.exact_citations:
        payload = (
            citation.model_dump(exclude_none=True)
            if isinstance(citation, RawExactCitation)
            else dict(citation)
        )
        filters.append(_canonical_citation_filter(payload))
    deduped: list[dict[str, Any]] = []
    seen: set[tuple[tuple[str, str], ...]] = set()
    for item in filters:
        key = tuple(sorted((name, str(value)) for name, value in item.items() if value))
        if key and key not in seen:
            deduped.append(item)
            seen.add(key)
    return deduped


def _debug_warning(warning_code: str, exc: Exception) -> str:
    return f"{warning_code}:{type(exc).__name__}:{_sanitize_exception_message(exc)}"


def _sanitize_exception_message(exc: Exception) -> str:
    message = next((line.strip() for line in str(exc).splitlines() if line.strip()), "")
    if not message:
        message = repr(exc)
    for pattern, replacement in _WARNING_REDACTIONS:
        message = pattern.sub(replacement, message)
    if len(message) > _MAX_WARNING_MESSAGE_LENGTH:
        message = message[: _MAX_WARNING_MESSAGE_LENGTH - 3].rstrip() + "..."
    return message


def _canonical_citation_filter(payload: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "law_id": _text_value(payload.get("law_id") or payload.get("law_id_hint")),
        "article_number": _text_value(payload.get("article_number") or payload.get("article")),
        "paragraph_number": _text_value(payload.get("paragraph_number") or payload.get("paragraph")),
        "letter_number": _text_value(payload.get("letter_number") or payload.get("letter")),
        "point_number": _text_value(payload.get("point_number") or payload.get("point")),
    }


def _sql_filters(
    filters: Mapping[str, Any],
    *,
    alias: str,
    available_columns: set[str] | None = None,
) -> tuple[list[str], dict[str, Any]]:
    clauses = ["1 = 1"]
    params: dict[str, Any] = {}
    legal_domain = _text_value(filters.get("legal_domain"))
    if legal_domain:
        if _has_column(available_columns, "legal_domain"):
            clauses.append(f"lower({alias}.legal_domain::text) = lower(:legal_domain)")
            params["legal_domain"] = legal_domain
        else:
            clauses.append("1 = 0")
    status = _text_value(filters.get("status"))
    if status:
        if _has_column(available_columns, "status"):
            clauses.append(f"{alias}.status::text = :status")
            params["status"] = status
        else:
            clauses.append("1 = 0")
    law_id = _text_value(filters.get("law_id"))
    if law_id:
        if _has_column(available_columns, "law_id"):
            clauses.append(f"{alias}.law_id::text = :filter_law_id")
            params["filter_law_id"] = law_id
        else:
            clauses.append("1 = 0")
    return clauses, params


def _has_column(available_columns: set[str] | None, column: str) -> bool:
    return available_columns is None or column in available_columns


def _sql_text_equals(
    available_columns: set[str],
    column: str,
    param_name: str,
    *,
    alias: str,
) -> str:
    if column not in available_columns:
        return "1 = 0"
    return f"{alias}.{column}::text = :{param_name}"


def _unit_select_columns_sql(available_columns: set[str], *, alias: str) -> str:
    select_parts = []
    for column in _UNIT_COLUMNS:
        if column in available_columns:
            select_parts.append(f"{alias}.{column} AS {column}")
        else:
            select_parts.append(f"NULL AS {column}")
    return ", ".join(select_parts)


def _unit_search_text_sql(available_columns: set[str], *, alias: str) -> str:
    text_parts = [
        f"coalesce({alias}.{column}::text, '')"
        for column in ("raw_text", "normalized_text")
        if column in available_columns
    ]
    if not text_parts:
        return "''"
    return " || ' ' || ".join(text_parts)


def _unit_order_by_sql(available_columns: set[str], *, alias: str) -> str:
    order_parts = []
    if "status" in available_columns:
        order_parts.append(
            f"""
            CASE WHEN {alias}.status::text = 'active' THEN 0
                 WHEN {alias}.status::text = 'unknown' THEN 1
                 ELSE 2 END
            """
        )
    order_parts.append(f"{alias}.id")
    return ", ".join(order_parts)


def _unit_from_row(row: Mapping[str, Any]) -> dict[str, Any]:
    unit = {column: row.get(column) for column in _UNIT_COLUMNS}
    unit["legal_concepts"] = unit.get("legal_concepts") or []
    unit["hierarchy_path"] = unit.get("hierarchy_path") or []
    if "bm25_score" in row:
        unit["bm25_score"] = _float(row.get("bm25_score"))
    if "dense_score" in row:
        unit["dense_score"] = _float(row.get("dense_score"))
    return unit


def _query_terms(question: str) -> list[str]:
    terms = []
    for raw in _normalize_text(question).replace("-", " ").split():
        token = raw.strip(".,;:!?()[]{}\"'")
        if len(token) >= 3 and token not in _STOP_WORDS and token not in terms:
            terms.append(token)
    return terms


def _matched_terms(question: str, unit: Mapping[str, Any]) -> list[str]:
    haystack = _normalize_text(
        f"{unit.get('raw_text') or ''} {unit.get('normalized_text') or ''}"
    )
    matches = [term for term in _query_terms(question) if term in haystack]
    normalized_question = _normalize_text(question)
    if "act aditional" in normalized_question and "act" in matches and "aditional" in matches:
        matches.append("act adițional")
    return matches[:10]


def _domain_match(filter_domain: Any, unit_domain: Any) -> float:
    if not filter_domain:
        return 0.5
    return 1.0 if _normalize_text(str(filter_domain)) == _normalize_text(str(unit_domain or "")) else 0.0


def _metadata_validity(unit: Mapping[str, Any]) -> float:
    status = _text_value(unit.get("status")) or "unknown"
    if status in {"active", "unknown"}:
        return 1.0
    if status == "historical":
        return 0.7
    if status == "repealed":
        return 0.4
    return 0.7


def _why_retrieved(methods: set[str]) -> str:
    labels = []
    if EXACT_METHOD in methods:
        labels.append("exact citation")
    if FTS_METHOD in methods:
        labels.append("lexical match")
    if DENSE_METHOD in methods:
        labels.append("dense similarity")
    return " + ".join(labels) if labels else "domain filtered search"


def _normalize_score(value: float, maximum: float) -> float:
    if maximum <= 0.0:
        return 0.0
    return _clamp01(value / maximum)


def _clamp01(value: float) -> float:
    if math.isnan(value):
        return 0.0
    return max(0.0, min(1.0, value))


def _float(value: Any) -> float:
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _text_value(value: Any) -> str | None:
    if value is None:
        return None
    text_value = str(value).strip()
    return text_value or None


def _normalize_text(value: str) -> str:
    value = (
        value.replace("Äƒ", "a")
        .replace("È™", "s")
        .replace("È›", "t")
        .replace("Å£", "t")
        .replace("ÅŸ", "s")
    )
    normalized = unicodedata.normalize("NFKD", value.lower())
    return "".join(character for character in normalized if not unicodedata.combining(character))


def _canonical_domain(value: Any) -> str:
    return _normalize_text(str(value)).replace(" ", "_")


def _vector_literal(vector: Sequence[float]) -> str:
    return "[" + ",".join(str(float(value)) for value in vector) + "]"


_UNIT_COLUMNS = (
    "id",
    "canonical_id",
    "source_id",
    "law_id",
    "law_title",
    "act_type",
    "act_number",
    "publication_date",
    "effective_date",
    "version_start",
    "version_end",
    "status",
    "hierarchy_path",
    "article_number",
    "paragraph_number",
    "letter_number",
    "point_number",
    "raw_text",
    "normalized_text",
    "legal_domain",
    "legal_concepts",
    "source_url",
    "parent_id",
    "parser_warnings",
    "created_at",
    "updated_at",
)
_MAX_WARNING_MESSAGE_LENGTH = 240
_WARNING_REDACTIONS = (
    (
        re.compile(r"(?i)\b(postgres(?:ql)?(?:\+asyncpg)?://)([^:@/\s]+):([^@/\s]+)@"),
        r"\1\2:***@",
    ),
    (re.compile(r"(?i)(\bpassword=)([^&\s;]+)"), r"\1***"),
    (re.compile(r"(?i)\bDATABASE_URL=\S+"), "<redacted_database_url>"),
    (
        re.compile(
            r"(?is)((?:['\"])?(?:raw_text|normalized_text|embedding_text)(?:['\"])?\s*[:=]\s*)(['\"])(.*?)(\2)"
        ),
        lambda match: f"{match.group(1)}{match.group(2)}<redacted>{match.group(4)}",
    ),
    (re.compile(r"(?i)\b(raw_text|normalized_text|embedding_text)\b"), "<text_field>"),
    (
        re.compile(r"(?is)((?:['\"])?embedding(?:['\"])?\s*[:=]\s*)\[[^\]]*\]"),
        r"\1[<redacted>]",
    ),
    (
        re.compile(
            r"\[(?:\s*-?\d+(?:\.\d+)?(?:e[+-]?\d+)?\s*,){2,}\s*-?\d+(?:\.\d+)?(?:e[+-]?\d+)?\s*\]",
            re.IGNORECASE,
        ),
        "[<redacted_vector>]",
    ),
)
_STOP_WORDS = {
    "care",
    "este",
    "sunt",
    "fara",
    "fără",
    "poate",
    "prin",
    "din",
    "pentru",
    "angajatorul",
    "imi",
    "îmi",
    "sa",
    "să",
}
