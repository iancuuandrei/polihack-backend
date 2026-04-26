from __future__ import annotations

import time
from typing import Any

from fastapi import APIRouter
from sqlalchemy import text

from ..config import settings
from ..db import session_context
from ..schemas import QueryRequest

router = APIRouter(tags=["health"])

DEMO_QUERY = "Poate angajatorul să-mi scadă salariul fără act adițional?"
REQUIRED_DEMO_UNIT_IDS = [
    "ro.codul_muncii.art_41.alin_1",
    "ro.codul_muncii.art_41.alin_3",
]
REQUIRED_GROUNDEDNESS_SCORE = 0.75
DB_TABLES = ("legal_units", "legal_edges", "legal_embeddings")


@router.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok", "service": "lexai-api"}


@router.get("/health/config")
async def config_health() -> dict[str, Any]:
    return {
        "status": "ok",
        "service": "lexai-api",
        "app_env": settings.app_env,
        "database_url_configured": bool((settings.database_url or "").strip()),
        "raw_retrieval_base_url_configured": bool(
            (settings.raw_retrieval_base_url or "").strip()
        ),
        "admin_ingest_secret_configured": bool(
            (settings.admin_ingest_secret or "").strip()
        ),
        "cors_origins": settings.cors_origins,
        "cors_origins_count": len(settings.cors_origins),
        "llm_query_decomposer_enabled": settings.enable_llm_query_decomposer,
        "llm_query_decomposer_base_url_configured": bool(
            (settings.llm_query_decomposer_base_url or "").strip()
        ),
        "llm_query_decomposer_api_key_configured": bool(
            (settings.llm_query_decomposer_api_key or "").strip()
        ),
    }


@router.get("/health/db")
async def db_health() -> dict:
    started = time.perf_counter()
    database_url_configured = bool((settings.database_url or "").strip())
    payload: dict[str, Any] = {
        "status": "degraded",
        "service": "postgres",
        "database_url_configured": database_url_configured,
        "db_reachable": False,
        "latency_ms": 0.0,
        "select_1": None,
        "server_version": None,
        "extensions": [],
        "tables": {
            table_name: {"exists": False, "row_count": None}
            for table_name in DB_TABLES
        },
        "warnings": [],
    }
    if not database_url_configured:
        payload["warnings"].append("database_url_not_configured")
        payload["latency_ms"] = _latency_ms(started)
        return payload

    try:
        async with session_context() as session:
            select_one = (await session.execute(text("SELECT 1"))).scalar_one()
            server_version = (
                await session.execute(text("SHOW server_version"))
            ).scalar_one()
            extensions = (
                await session.execute(
                    text("SELECT extname FROM pg_extension ORDER BY extname")
                )
            ).scalars().all()
            payload["db_reachable"] = True
            payload["select_1"] = select_one
            payload["server_version"] = server_version
            payload["extensions"] = list(extensions)
            for table_name in DB_TABLES:
                exists = await _table_exists(session, table_name)
                table_payload = {"exists": exists, "row_count": None}
                if exists:
                    table_payload["row_count"] = await _table_count(
                        session,
                        table_name,
                    )
                else:
                    payload["warnings"].append(f"table_missing:{table_name}")
                payload["tables"][table_name] = table_payload
    except RuntimeError:
        payload["warnings"].append("database_url_not_configured")
    except Exception as exc:
        payload["warnings"].append(f"database_unavailable:{type(exc).__name__}")

    payload["latency_ms"] = _latency_ms(started)
    payload["status"] = "ok" if payload["db_reachable"] else "degraded"
    return payload


@router.post("/health/retrieval-demo")
async def retrieval_demo_health() -> dict[str, Any]:
    from . import query as query_route

    warnings: list[str] = []
    candidate_unit_ids: list[str] = []
    retrieval_methods: list[str] = []
    raw_retrieval_warnings: list[str] = []
    try:
        request = QueryRequest(question=DEMO_QUERY, debug=True)
        orchestrator = query_route.orchestrator
        query_plan = orchestrator.query_understanding.build_plan(request)
        query_frame = orchestrator.query_frame_builder.build(
            question=request.question,
            plan=query_plan,
        )
        raw_retrieval = await orchestrator._retrieve_raw(
            query_plan=query_plan,
            query_frame=query_frame,
            query_embedding=None,
            debug=True,
        )
        candidate_unit_ids = [
            candidate.unit_id for candidate in raw_retrieval.candidates
        ]
        retrieval_methods = raw_retrieval.retrieval_methods
        raw_retrieval_warnings = raw_retrieval.warnings
    except Exception as exc:
        warnings.append(f"retrieval_demo_failed:{type(exc).__name__}")

    missing_unit_ids = _missing_required(candidate_unit_ids)
    checks = {
        "required_art_41_units_present": not missing_unit_ids,
        "candidate_count_positive": bool(candidate_unit_ids),
    }
    status = "ok" if all(checks.values()) and not warnings else "degraded"
    return {
        "status": status,
        "question": DEMO_QUERY,
        "required_unit_ids": REQUIRED_DEMO_UNIT_IDS,
        "candidate_count": len(candidate_unit_ids),
        "candidate_unit_ids_first_10": candidate_unit_ids[:10],
        "found_required_unit_ids": _found_required(candidate_unit_ids),
        "missing_required_unit_ids": missing_unit_ids,
        "retrieval_methods": retrieval_methods,
        "checks": checks,
        "warnings": _dedupe(raw_retrieval_warnings + warnings),
    }


@router.post("/health/query-demo")
async def query_demo_health() -> dict[str, Any]:
    try:
        response = await _run_demo_query_and_store(debug=True)
    except Exception as exc:
        return _demo_error_payload("query_demo_failed", exc)

    citation_unit_ids = [citation.legal_unit_id for citation in response.citations]
    verified_citation_unit_ids = [
        citation.legal_unit_id
        for citation in response.citations
        if citation.verified
    ]
    missing_citations = _missing_required(citation_unit_ids)
    checks = {
        "verifier_passed": response.verifier.verifier_passed,
        "required_citations_present": not missing_citations,
        "citations_checked_at_least_2": response.verifier.citations_checked >= 2,
        "groundedness_score_at_least_0_75": (
            response.verifier.groundedness_score >= REQUIRED_GROUNDEDNESS_SCORE
        ),
    }
    return {
        "status": "ok" if all(checks.values()) else "degraded",
        "query_id": response.query_id,
        "question": response.question,
        "required_unit_ids": REQUIRED_DEMO_UNIT_IDS,
        "citation_unit_ids": citation_unit_ids,
        "verified_citation_unit_ids": verified_citation_unit_ids,
        "missing_required_citation_unit_ids": missing_citations,
        "verifier": {
            "verifier_passed": response.verifier.verifier_passed,
            "citations_checked": response.verifier.citations_checked,
            "groundedness_score": response.verifier.groundedness_score,
            "claims_total": response.verifier.claims_total,
            "claims_supported": response.verifier.claims_supported,
            "claims_unsupported": response.verifier.claims_unsupported,
        },
        "checks": checks,
        "warnings": response.warnings,
    }


@router.post("/health/query-graph-demo")
async def query_graph_demo_health() -> dict[str, Any]:
    from . import query as query_route

    try:
        response = await _run_demo_query_and_store(debug=True)
        graph_response = query_route.query_response_store.get_graph(
            response.query_id,
        )
    except Exception as exc:
        return _demo_error_payload("query_graph_demo_failed", exc)

    if graph_response is None:
        return {
            "status": "degraded",
            "question": DEMO_QUERY,
            "required_unit_ids": REQUIRED_DEMO_UNIT_IDS,
            "checks": {
                "graph_response_available": False,
                "required_cited_units_present": False,
                "highlighted_node_ids_non_empty": False,
                "highlighted_edge_ids_non_empty_if_cited_edges_exist": False,
            },
            "warnings": ["query_graph_not_found_after_demo_query"],
        }

    backend_produces_cited_edges = any(
        edge.type in {"cited_in_answer", "supports_claim"}
        for edge in graph_response.graph.edges
    )
    missing_cited_units = _missing_required(graph_response.cited_unit_ids)
    highlighted_edge_check = (
        bool(graph_response.highlighted_edge_ids)
        if backend_produces_cited_edges
        else True
    )
    checks = {
        "graph_response_available": True,
        "required_cited_units_present": not missing_cited_units,
        "highlighted_node_ids_non_empty": bool(graph_response.highlighted_node_ids),
        "highlighted_edge_ids_non_empty_if_cited_edges_exist": highlighted_edge_check,
    }

    return {
        "status": "ok" if all(checks.values()) else "degraded",
        "query_id": graph_response.query_id,
        "question": graph_response.question,
        "required_unit_ids": REQUIRED_DEMO_UNIT_IDS,
        "cited_unit_ids": graph_response.cited_unit_ids,
        "missing_required_cited_unit_ids": missing_cited_units,
        "highlighted_node_ids": graph_response.highlighted_node_ids,
        "highlighted_edge_ids": graph_response.highlighted_edge_ids,
        "backend_produces_cited_edges": backend_produces_cited_edges,
        "graph_counts": {
            "nodes": len(graph_response.graph.nodes),
            "edges": len(graph_response.graph.edges),
        },
        "verifier_summary": graph_response.verifier_summary,
        "checks": checks,
        "warnings": response.warnings,
    }


async def _table_exists(session: Any, table_name: str) -> bool:
    result = await session.execute(
        text("SELECT to_regclass(:table_name)"),
        {"table_name": f"public.{table_name}"},
    )
    return result.scalar_one() is not None


async def _table_count(session: Any, table_name: str) -> int:
    result = await session.execute(text(f"SELECT COUNT(*) FROM {table_name}"))
    return int(result.scalar_one() or 0)


def _latency_ms(started: float) -> float:
    return round((time.perf_counter() - started) * 1000, 2)


def _missing_required(unit_ids: list[str]) -> list[str]:
    present = set(unit_ids)
    return [unit_id for unit_id in REQUIRED_DEMO_UNIT_IDS if unit_id not in present]


def _found_required(unit_ids: list[str]) -> list[str]:
    present = set(unit_ids)
    return [unit_id for unit_id in REQUIRED_DEMO_UNIT_IDS if unit_id in present]


async def _run_demo_query_and_store(*, debug: bool):
    from . import query as query_route

    request = QueryRequest(question=DEMO_QUERY, debug=debug)
    response = await query_route.orchestrator.run(request)
    query_route.query_response_store.save(response)
    return response


def _demo_error_payload(error_code: str, exc: Exception) -> dict[str, Any]:
    return {
        "status": "degraded",
        "question": DEMO_QUERY,
        "required_unit_ids": REQUIRED_DEMO_UNIT_IDS,
        "checks": {},
        "warnings": [f"{error_code}:{type(exc).__name__}"],
    }


def _dedupe(values: list[str]) -> list[str]:
    deduped: list[str] = []
    for value in values:
        if value not in deduped:
            deduped.append(value)
    return deduped
