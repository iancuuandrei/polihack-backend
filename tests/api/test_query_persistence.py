import pytest
from fastapi.testclient import TestClient

import apps.api.app.routes.query as query_route
from apps.api.app.main import app
from apps.api.app.services.evidence_pack_compiler import EvidencePackCompiler
from apps.api.app.services.graph_expansion_policy import GraphExpansionPolicy
from apps.api.app.services.query_orchestrator import QueryOrchestrator
from tests.helpers.fixture_handoff03 import FixtureGraphClient, FixtureRawRetriever


DEMO_QUERY = (
    "Poate angajatorul s\u0103-mi scad\u0103 salariul "
    "f\u0103r\u0103 act adi\u021bional?"
)
DEMO_PAYLOAD = {
    "question": DEMO_QUERY,
    "jurisdiction": "RO",
    "date": "current",
    "mode": "strict_citations",
}


@pytest.fixture(autouse=True)
def clear_query_response_store():
    query_route.query_response_store.clear()
    yield
    query_route.query_response_store.clear()


@pytest.fixture
def client_with_demo_orchestrator(monkeypatch):
    monkeypatch.setattr(query_route, "orchestrator", demo_orchestrator())
    with TestClient(app) as client:
        yield client


def demo_orchestrator() -> QueryOrchestrator:
    return QueryOrchestrator(
        raw_retriever_client=FixtureRawRetriever(),
        graph_expansion_policy=GraphExpansionPolicy(
            neighbors_client=FixtureGraphClient(),
        ),
        evidence_pack_compiler=EvidencePackCompiler(
            target_evidence_units=4,
            max_evidence_units=4,
        ),
    )


def post_demo_query(client: TestClient, *, debug: bool = True):
    return client.post("/api/query", json={**DEMO_PAYLOAD, "debug": debug})


def test_post_saves_response(client_with_demo_orchestrator):
    post_response = post_demo_query(client_with_demo_orchestrator, debug=True)

    assert post_response.status_code == 200
    post_payload = post_response.json()
    get_response = client_with_demo_orchestrator.get(
        f"/api/query/{post_payload['query_id']}"
    )

    assert get_response.status_code == 200
    get_payload = get_response.json()
    assert get_payload["query_id"] == post_payload["query_id"]
    assert get_payload["answer"] == post_payload["answer"]
    assert len(get_payload["citations"]) == len(post_payload["citations"])


def test_get_graph_works(client_with_demo_orchestrator):
    post_response = post_demo_query(client_with_demo_orchestrator, debug=True)
    query_id = post_response.json()["query_id"]

    graph_response = client_with_demo_orchestrator.get(f"/api/query/{query_id}/graph")

    assert graph_response.status_code == 200
    payload = graph_response.json()
    assert payload["query_id"] == query_id
    assert payload["question"] == DEMO_QUERY
    assert payload["graph"]["nodes"]
    assert payload["graph"]["edges"]
    assert payload["cited_unit_ids"]
    assert "highlighted_node_ids" in payload
    assert "highlighted_edge_ids" in payload
    assert "verifier_summary" in payload
    assert payload["verifier_summary"]["citations_checked"] > 0
    assert payload["verifier_summary"]["verifier_passed"] is True


def test_demo_graph_highlights_cited_units(client_with_demo_orchestrator):
    post_response = post_demo_query(client_with_demo_orchestrator, debug=True)
    query_id = post_response.json()["query_id"]

    graph_payload = client_with_demo_orchestrator.get(
        f"/api/query/{query_id}/graph"
    ).json()

    assert {
        "ro.codul_muncii.art_41.alin_1",
        "ro.codul_muncii.art_41.alin_3",
    }.issubset(set(graph_payload["cited_unit_ids"]))
    assert graph_payload["highlighted_node_ids"]

    graph_edges_by_id = {
        edge["id"]: edge["type"] for edge in graph_payload["graph"]["edges"]
    }
    expected_highlighted_edge_ids = {
        edge_id
        for edge_id, edge_type in graph_edges_by_id.items()
        if edge_type in {"cited_in_answer", "supports_claim"}
    }
    assert expected_highlighted_edge_ids.issubset(
        set(graph_payload["highlighted_edge_ids"])
    )


def test_unknown_query_returns_404():
    with TestClient(app) as client:
        query_response = client.get("/api/query/does-not-exist")
        graph_response = client.get("/api/query/does-not-exist/graph")

    assert query_response.status_code == 404
    assert graph_response.status_code == 404
    assert query_response.json()["error_code"] == "query_not_found"
    assert graph_response.json()["error_code"] == "query_not_found"


def test_debug_false_still_persisted(client_with_demo_orchestrator):
    post_response = post_demo_query(client_with_demo_orchestrator, debug=False)

    assert post_response.status_code == 200
    post_payload = post_response.json()
    get_response = client_with_demo_orchestrator.get(
        f"/api/query/{post_payload['query_id']}"
    )

    assert get_response.status_code == 200
    get_payload = get_response.json()
    assert get_payload["debug"] is None
    assert get_payload["query_id"] == post_payload["query_id"]
    assert get_payload["answer"] == post_payload["answer"]
    assert get_payload["citations"]
    assert get_payload["graph"]["nodes"]
