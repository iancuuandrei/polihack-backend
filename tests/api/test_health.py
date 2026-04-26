import os
import subprocess
import sys
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi.testclient import TestClient

from apps.api.app.main import app
from apps.api.app.routes import corpus as corpus_module
from apps.api.app.routes import health as health_module
import apps.api.app.routes.query as query_route
from apps.api.app.services.query_orchestrator import QueryOrchestrator
from tests.helpers.live_like_demo import LiveLikeRawRetriever

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class FakeScalarResult:
    def __init__(self, scalar=None, scalar_list=None, rows=None):
        self._scalar = scalar
        self._scalar_list = scalar_list or []
        self._rows = rows or []

    def scalar_one(self):
        return self._scalar

    def scalars(self):
        return self

    def all(self):
        if self._rows:
            return self._rows
        return self._scalar_list


class FakeSession:
    async def execute(self, statement, params=None):
        query = str(statement)
        if query == "SELECT 1":
            return FakeScalarResult(scalar=1)
        if query == "SHOW server_version":
            return FakeScalarResult(scalar="16.0")
        if query == "SELECT extname FROM pg_extension ORDER BY extname":
            return FakeScalarResult(scalar_list=["plpgsql", "vector"])
        if query == "SELECT to_regclass(:table_name)":
            table_name = (params or {}).get("table_name", "")
            if table_name in {
                "public.legal_units",
                "public.legal_edges",
                "public.legal_embeddings",
            }:
                return FakeScalarResult(scalar=table_name)
            return FakeScalarResult(scalar=None)
        if query == "SELECT COUNT(*) FROM legal_units":
            return FakeScalarResult(scalar=8)
        if query == "SELECT COUNT(*) FROM legal_edges":
            return FakeScalarResult(scalar=3)
        if query == "SELECT COUNT(*) FROM legal_embeddings":
            return FakeScalarResult(scalar=2)
        if query == "SELECT COUNT(*) FROM legal_units WHERE status = 'active'":
            return FakeScalarResult(scalar=7)
        if (
            query
            == "SELECT COUNT(*) FROM legal_units WHERE status = 'active' AND legal_domain = 'munca'"
        ):
            return FakeScalarResult(scalar=6)
        if (
            query
            == "SELECT COUNT(*) FROM legal_units WHERE law_id = 'ro.codul_muncii' OR lower(law_title) LIKE '%codul muncii%'"
        ):
            return FakeScalarResult(scalar=5)
        if (
            query
            == "SELECT COUNT(*) FROM legal_units WHERE raw_text IS NULL OR btrim(raw_text) = ''"
        ):
            return FakeScalarResult(scalar=1)
        if query == "SELECT id, raw_text FROM legal_units":
            return FakeScalarResult(
                rows=[
                    ("ro.codul_muncii.art_41.alin_1", "Contract de muncă"),
                    ("ro.codul_muncii.art_41.alin_3", "pÄrÈilor"),
                ],
            )
        raise AssertionError(f"Unexpected query: {query}")


def test_app_imports_without_database_url(tmp_path):
    env = os.environ.copy()
    env.pop("DATABASE_URL", None)
    env["PYTHONPATH"] = str(PROJECT_ROOT)

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "from apps.api.app.main import app; print(app.title)",
        ],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "LexAI API" in result.stdout


def test_api_health_does_not_require_database():
    with TestClient(app) as client:
        response = client.get("/api/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "service": "lexai-api"}


def test_legacy_health_does_not_require_database():
    with TestClient(app) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "service": "lexai-api"}


def test_api_health_db_returns_degraded_without_database_url(monkeypatch):
    import apps.api.app.db.session as db_module

    monkeypatch.setattr(health_module.settings, "database_url", None)
    monkeypatch.setattr(db_module.settings, "database_url", None)
    monkeypatch.setattr(db_module, "_engine", None)
    monkeypatch.setattr(db_module, "_sessionmaker", None)

    with TestClient(app) as client:
        response = client.get("/api/health/db")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "degraded"
    assert payload["service"] == "postgres"
    assert payload["database_url_configured"] is False
    assert payload["db_reachable"] is False
    assert "database_url_not_configured" in payload["warnings"]


def test_api_health_db_uses_separate_database_check(monkeypatch):
    @asynccontextmanager
    async def fake_session_context():
        yield FakeSession()

    monkeypatch.setattr(health_module.settings, "database_url", "postgresql://configured")
    monkeypatch.setattr(health_module, "session_context", fake_session_context)

    with TestClient(app) as client:
        response = client.get("/api/health/db")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["service"] == "postgres"
    assert payload["database_url_configured"] is True
    assert payload["db_reachable"] is True
    assert payload["select_1"] == 1
    assert payload["server_version"] == "16.0"
    assert payload["extensions"] == ["plpgsql", "vector"]
    assert payload["tables"]["legal_units"] == {"exists": True, "row_count": 8}
    assert payload["tables"]["legal_edges"] == {"exists": True, "row_count": 3}
    assert payload["tables"]["legal_embeddings"] == {"exists": True, "row_count": 2}
    assert payload["warnings"] == []


def test_api_health_config_redacts_database_url(monkeypatch):
    secret_url = "postgresql://user:password@secret-host:5432/lexai"
    monkeypatch.setattr(health_module.settings, "database_url", secret_url)

    with TestClient(app) as client:
        response = client.get("/api/health/config")

    payload = response.json()
    assert response.status_code == 200
    assert payload["database_url_configured"] is True
    assert secret_url not in str(payload)
    assert "password" not in str(payload)
    assert "secret-host" not in str(payload)


def test_api_corpus_stats_returns_counts_and_mojibake_warning(monkeypatch):
    @asynccontextmanager
    async def fake_session_context():
        yield FakeSession()

    monkeypatch.setattr(corpus_module.settings, "database_url", "postgresql://configured")
    monkeypatch.setattr(corpus_module, "session_context", fake_session_context)

    with TestClient(app) as client:
        response = client.get("/api/corpus/stats")

    payload = response.json()
    assert response.status_code == 200
    assert payload["status"] == "ok"
    assert payload["database_url_configured"] is True
    assert payload["db_reachable"] is True
    assert payload["counts"]["total_legal_units"] == 8
    assert payload["counts"]["active_units"] == 7
    assert payload["counts"]["munca_active_units"] == 6
    assert payload["counts"]["codul_muncii_units"] == 5
    assert payload["counts"]["legal_edges"] == 3
    assert payload["counts"]["empty_raw_text"] == 1
    assert payload["counts"]["mojibake_raw_text"] == 1
    assert payload["mojibake_sample_unit_ids"] == ["ro.codul_muncii.art_41.alin_3"]
    assert "raw_text_contains_romanian_mojibake" in payload["warnings"]


def test_api_health_retrieval_demo_reports_required_art_41_units():
    original_orchestrator = query_route.orchestrator
    query_route.orchestrator = QueryOrchestrator(
        raw_retriever_client=LiveLikeRawRetriever(),
    )
    try:
        with TestClient(app) as client:
            response = client.post("/api/health/retrieval-demo")
    finally:
        query_route.orchestrator = original_orchestrator

    payload = response.json()
    assert response.status_code == 200
    assert payload["status"] == "ok"
    assert payload["checks"]["required_art_41_units_present"] is True
    assert payload["missing_required_unit_ids"] == []
    assert "ro.codul_muncii.art_41.alin_1" in payload["found_required_unit_ids"]
    assert "ro.codul_muncii.art_41.alin_3" in payload["found_required_unit_ids"]


def test_api_health_query_demo_reports_verifier_and_required_citations():
    original_orchestrator = query_route.orchestrator
    query_route.orchestrator = QueryOrchestrator(
        raw_retriever_client=LiveLikeRawRetriever(),
    )
    try:
        with TestClient(app) as client:
            response = client.post("/api/health/query-demo")
    finally:
        query_route.orchestrator = original_orchestrator

    payload = response.json()
    assert response.status_code == 200
    assert payload["status"] == "ok"
    assert payload["checks"]["verifier_passed"] is True
    assert payload["checks"]["required_citations_present"] is True
    assert payload["checks"]["citations_checked_at_least_2"] is True
    assert payload["checks"]["groundedness_score_at_least_0_75"] is True
    assert "ro.codul_muncii.art_41.alin_1" in payload["citation_unit_ids"]
    assert "ro.codul_muncii.art_41.alin_3" in payload["citation_unit_ids"]


def test_api_health_query_graph_demo_reports_highlights():
    original_orchestrator = query_route.orchestrator
    query_route.orchestrator = QueryOrchestrator(
        raw_retriever_client=LiveLikeRawRetriever(),
    )
    query_route.query_response_store.clear()
    try:
        with TestClient(app) as client:
            response = client.post("/api/health/query-graph-demo")
    finally:
        query_route.query_response_store.clear()
        query_route.orchestrator = original_orchestrator

    payload = response.json()
    assert response.status_code == 200
    assert payload["status"] == "ok"
    assert payload["checks"]["required_cited_units_present"] is True
    assert payload["checks"]["highlighted_node_ids_non_empty"] is True
    assert payload["checks"]["highlighted_edge_ids_non_empty_if_cited_edges_exist"] is True
    assert "ro.codul_muncii.art_41.alin_1" in payload["cited_unit_ids"]
    assert "ro.codul_muncii.art_41.alin_3" in payload["cited_unit_ids"]
