import pytest

from apps.api.app.schemas import QueryRequest
from apps.api.app.services.evidence_pack_compiler import EvidencePackCompiler
from apps.api.app.services.graph_expansion_policy import GraphExpansionPolicy
from apps.api.app.services.query_orchestrator import QueryOrchestrator
from tests.helpers.fixture_handoff03 import (
    DEMO_QUERY_WITH_DIACRITICS,
    FixtureGraphClient,
    FixtureRawRetriever,
)


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


@pytest.mark.anyio
async def test_query_orchestrator_generates_grounded_draft_with_debug():
    response = await demo_orchestrator().run(
        QueryRequest(
            question=DEMO_QUERY_WITH_DIACRITICS,
            jurisdiction="RO",
            date="current",
            mode="strict_citations",
            debug=True,
        )
    )

    evidence_ids = {unit.id for unit in response.evidence_units}
    citation_ids = {citation.legal_unit_id for citation in response.citations}
    assert response.answer.short_answer
    assert response.answer.confidence == 0.0
    assert response.answer.refusal_reason is None
    assert response.evidence_units
    assert response.citations
    assert citation_ids <= evidence_ids
    assert "ro.codul_muncii.art_41.alin_4" in citation_ids
    assert "ro.codul_muncii.art_41.alin_3" in citation_ids
    for citation in response.citations:
        evidence = next(
            unit
            for unit in response.evidence_units
            if unit.id == citation.legal_unit_id
        )
        assert citation.quote in evidence.raw_text
        assert citation.verified is True
    assert response.verifier.citations_checked == len(response.citations)
    assert response.verifier.claims_total > 0
    assert response.verifier.verifier_passed is True
    assert response.verifier.repair_applied is False
    warnings = " ".join(response.warnings + response.verifier.warnings)
    assert "answer_refused" not in warnings
    assert "answer_repaired" not in warnings
    assert "answer_tempered" not in warnings
    assert "CitationVerifier has not run yet" not in warnings
    assert "generation_unverified_citation_verifier_not_run" not in warnings
    assert "nu a fost verificat final de CitationVerifier" not in response.answer.short_answer
    assert "CitationVerifier V1" in response.answer.short_answer
    assert response.debug.generation["generation_mode"] == "deterministic_extractive_v1"
    assert response.debug.generation["evidence_unit_count_used"] == len(response.citations)
    assert set(response.debug.generation["citation_unit_ids"]) == citation_ids
    assert response.debug.verifier["claim_extraction"]["claims_total"] > 0
    assert response.debug.answer_repair["repair_action"] == "none"
    assert response.debug.answer_repair["warnings_added"] == []


@pytest.mark.anyio
async def test_query_orchestrator_omits_generation_debug_when_debug_false():
    response = await demo_orchestrator().run(
        QueryRequest(
            question=DEMO_QUERY_WITH_DIACRITICS,
            jurisdiction="RO",
            date="current",
            mode="strict_citations",
            debug=False,
        )
    )

    assert response.debug is None
    assert response.citations
    assert response.answer.confidence == 0.0
    assert response.verifier.citations_checked == len(response.citations)
    assert response.verifier.claims_total > 0
    assert response.verifier.repair_applied is False
