from apps.api.app.schemas import AnswerPayload, Citation, EvidenceUnit
from apps.api.app.services.citation_verifier import (
    CITATION_MISSING_FOR_CLAIM,
    CITATION_SNIPPET_NOT_FOUND,
    CITATION_UNIT_MISSING,
    CITATION_VERIFIER_FAILED,
    NO_LEGAL_CLAIMS_DETECTED,
    UNSUPPORTED_LEGAL_CLAIMS_DETECTED,
    VERIFIER_INSUFFICIENT_EVIDENCE,
    CitationVerifier,
)


def evidence_unit(
    unit_id: str = "ro.codul_muncii.art_41",
    raw_text: str = "Contractul individual de munca poate fi modificat numai prin acordul partilor.",
) -> EvidenceUnit:
    return EvidenceUnit(
        id=unit_id,
        law_id="ro.codul_muncii",
        law_title="Codul muncii",
        status="active",
        hierarchy_path=["Codul muncii", "Art. 41"],
        article_number="41",
        raw_text=raw_text,
        normalized_text=None,
        legal_domain="munca",
        legal_concepts=[],
        source_url=None,
        evidence_id=f"evidence:{unit_id}",
        excerpt=raw_text,
        rank=1,
        relevance_score=0.9,
        retrieval_method="test",
        retrieval_score=0.9,
        rerank_score=0.9,
        support_role="direct_basis",
    )


def citation(
    quote: str = "Contractul individual de munca poate fi modificat numai prin acordul partilor.",
    unit_id: str = "ro.codul_muncii.art_41",
) -> Citation:
    return Citation(
        citation_id="citation:1",
        evidence_id=f"evidence:{unit_id}",
        legal_unit_id=unit_id,
        label="Codul muncii, art. 41",
        quote=quote,
        verified=False,
    )


def answer(short_answer: str) -> AnswerPayload:
    return AnswerPayload(
        short_answer=short_answer,
        detailed_answer=None,
        confidence=0.0,
        not_legal_advice=True,
    )


def test_empty_evidence_returns_insufficient_evidence_status():
    result = CitationVerifier().verify(
        answer=answer("Nu pot formula un raspuns juridic sustinut."),
        citations=[],
        evidence_units=[],
        debug=True,
    )

    assert result.verifier.verifier_passed is False
    assert result.verifier.groundedness_score == 0.0
    assert result.verifier.citations_checked == 0
    assert VERIFIER_INSUFFICIENT_EVIDENCE in result.verifier.warnings


def test_no_legal_claims_detected_fails_closed():
    unit = evidence_unit()
    result = CitationVerifier().verify(
        answer=answer("Acesta este un draft generat peste EvidencePack."),
        citations=[],
        evidence_units=[unit],
        debug=True,
    )

    assert result.verifier.verifier_passed is False
    assert result.verifier.claims_total == 0
    assert NO_LEGAL_CLAIMS_DETECTED in result.verifier.warnings


def test_missing_citation_unit_id_is_reported():
    unit = evidence_unit()
    result = CitationVerifier().verify(
        answer=answer("Contractul individual de munca poate fi modificat numai prin acordul partilor."),
        citations=[citation(unit_id="ro.codul_muncii.art_missing")],
        evidence_units=[unit],
        debug=True,
    )

    assert result.verifier.verifier_passed is False
    assert CITATION_UNIT_MISSING in result.verifier.warnings
    assert CITATION_VERIFIER_FAILED in result.verifier.warnings
    assert result.debug["failed_citations"] == ["citation:1"]


def test_citation_snippet_not_in_raw_text_blocks_pass():
    unit = evidence_unit()
    result = CitationVerifier().verify(
        answer=answer("Contractul individual de munca poate fi modificat numai prin acordul partilor."),
        citations=[citation(quote="Text legal inventat care nu exista in raw text.")],
        evidence_units=[unit],
        debug=True,
    )

    assert result.verifier.verifier_passed is False
    assert CITATION_SNIPPET_NOT_FOUND in result.verifier.warnings
    assert result.debug["citation_checks"][0]["confidence"] == 0.4


def test_supported_claim_scores_supported_or_strongly_supported():
    unit = evidence_unit()
    result = CitationVerifier().verify(
        answer=answer("Contractul individual de munca poate fi modificat numai prin acordul partilor."),
        citations=[citation()],
        evidence_units=[unit],
        debug=True,
    )

    claim = result.verifier.claim_results[0]
    assert result.verifier.verifier_passed is True
    assert result.verifier.groundedness_score == 1.0
    assert result.verifier.claims_supported == 1
    assert claim.status in {"supported", "strongly_supported"}
    assert claim.support_score >= 0.60
    assert claim.supporting_unit_ids == [unit.id]
    assert "lexical_semantic_similarity_fallback" in claim.score_breakdown


def test_unsupported_claim_fails_with_claim_warning():
    unit = evidence_unit()
    result = CitationVerifier().verify(
        answer=answer("Angajatorul poate concedia salariatul fara preaviz."),
        citations=[citation()],
        evidence_units=[unit],
        debug=True,
    )

    claim = result.verifier.claim_results[0]
    assert result.verifier.verifier_passed is False
    assert claim.status == "unsupported"
    assert claim.support_score < 0.45
    assert CITATION_MISSING_FOR_CLAIM in claim.warnings
    assert UNSUPPORTED_LEGAL_CLAIMS_DETECTED in result.verifier.warnings
