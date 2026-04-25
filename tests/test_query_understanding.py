from apps.api.app.schemas import QueryRequest
from apps.api.app.services.query_understanding import (
    QueryUnderstanding,
    detect_query_types,
    normalize_ro_text,
)


def build_plan(question: str, date: str = "current"):
    request = QueryRequest(
        question=question,
        jurisdiction="RO",
        date=date,
        mode="strict_citations",
        debug=True,
    )
    return QueryUnderstanding().build_plan(request)


def test_normalize_ro_text_lowercases_and_collapses_spaces():
    normalized = normalize_ro_text("  Poate   ANGAJATORUL să-mi   scadă salariul?  ")

    assert normalized == "poate angajatorul să-mi scadă salariul?"


def test_detect_query_types_for_demo_question():
    query_types = detect_query_types(
        normalize_ro_text("Poate angajatorul să-mi scadă salariul fără act adițional?")
    )

    assert "prohibition" in query_types
    assert "obligation" in query_types


def test_query_plan_builds_retrieval_filters_for_current_labor_question():
    plan = build_plan("Poate angajatorul să-mi scadă salariul fără act adițional?")

    assert plan.legal_domain == "muncă"
    assert plan.retrieval_filters["legal_domain"] == "muncă"
    assert plan.retrieval_filters["status"] == "active"
    assert plan.retrieval_filters["date_context"] == "current"


def test_query_plan_keeps_exact_citations_empty_in_phase_2():
    plan = build_plan("Ce spune art. 41 alin. 1 despre contractul de muncă?")

    assert plan.exact_citations == []
    assert "exact_citation_detection_pending" in plan.ambiguity_flags


def test_query_plan_marks_explicit_year_as_temporal_ambiguity():
    plan = build_plan("În 2020 putea angajatorul să modifice salariul?")

    assert plan.temporal_context == "current"
    assert "ambiguous_temporal_context" in plan.ambiguity_flags


def test_query_plan_expansion_policy_is_declarative_only():
    plan = build_plan("Poate angajatorul să-mi scadă salariul fără act adițional?")

    assert plan.expansion_policy["max_depth"] == 2
    assert plan.expansion_policy["max_expanded_nodes"] == 80
    assert plan.expansion_policy["include_references"] is True
    assert plan.expansion_policy["include_exceptions"] is True
