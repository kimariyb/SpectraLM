"""Tests for formula-constrained candidate filtering and summaries."""

from __future__ import annotations

from src.evaluation.constrained import (
    filter_generated_candidates,
    resolve_ranked_candidate,
    summarize_constrained_predictions,
)


def test_filter_candidates_enforces_domain_and_formula() -> None:
    result = filter_generated_candidates(
        ["CCO", "OCC", "CCN", "C.C", "not_smiles"],
        molecular_formula="C2H6O",
    )

    assert result.raw_count == 5
    assert result.unique_count == 2
    assert result.domain_valid_candidates == ("CCO", "CCN")
    assert result.formula_valid_candidates == ("CCO",)
    assert result.formula_constraint_applicable is True
    assert result.formula_constraint_failed is False


def test_filter_candidates_returns_empty_on_hard_constraint_failure() -> None:
    result = filter_generated_candidates(
        ["CCN"],
        molecular_formula="C2H6O",
    )

    assert result.formula_valid_candidates == ()
    assert result.selectable_candidates == ()
    assert result.formula_constraint_failed is True


def test_filter_candidates_without_formula_uses_domain_only() -> None:
    result = filter_generated_candidates(["CCO"], molecular_formula=None)

    assert result.formula_constraint_applicable is False
    assert result.formula_constraint_failed is False
    assert result.selectable_candidates == ("CCO",)


def test_filter_candidates_rejects_isotopes_and_disconnected_structures() -> None:
    result = filter_generated_candidates(
        ["[13CH3]CO", "C[NH3+].[Cl-]", "CCO"],
        molecular_formula="C2H6O",
    )

    assert result.domain_valid_candidates == ("CCO",)


def test_resolve_ranked_candidate_rejects_out_of_set_output() -> None:
    selection = resolve_ranked_candidate(("CCO", "COC"), "CCN")

    assert selection.prediction == "CCO"
    assert selection.ranking_failed is True


def test_resolve_ranked_candidate_accepts_canonical_equivalent() -> None:
    selection = resolve_ranked_candidate(("CCO", "COC"), "OCC")

    assert selection.prediction == "CCO"
    assert selection.ranking_failed is False


def test_resolve_ranked_candidate_accepts_json_smiles_response() -> None:
    selection = resolve_ranked_candidate(("CCO", "COC"), '{"smiles":"OCC"}')

    assert selection.prediction == "CCO"
    assert selection.ranking_failed is False


def test_constrained_summary_counts_failures_in_denominator() -> None:
    rows = [
        {
            "formula_constraint_applicable": True,
            "formula_constraint_failed": False,
            "raw_candidate_count": 32,
            "unique_candidate_count": 10,
            "domain_valid_candidate_count": 8,
            "formula_valid_candidate_count": 2,
            "candidate_oracle_exact": True,
            "candidate_oracle_connectivity": True,
            "ranking_attempted": True,
            "ranking_failed": False,
            "exact_match": True,
            "connectivity_exact_match": True,
        },
        {
            "formula_constraint_applicable": True,
            "formula_constraint_failed": True,
            "raw_candidate_count": 32,
            "unique_candidate_count": 5,
            "domain_valid_candidate_count": 3,
            "formula_valid_candidate_count": 0,
            "candidate_oracle_exact": False,
            "candidate_oracle_connectivity": False,
            "ranking_attempted": False,
            "ranking_failed": False,
            "exact_match": False,
            "connectivity_exact_match": False,
        },
    ]

    summary = summarize_constrained_predictions(rows)

    assert summary["formula_constraint_coverage"] == 0.5
    assert summary["formula_constraint_failure_rate"] == 0.5
    assert summary["candidate_oracle_exact_at_32"] == 0.5
    assert summary["ranked_top1_exact_match"] == 0.5
    assert summary["ranking_failure_rate"] == 0.0
