"""Tests for one-sample constrained structure inference."""

from __future__ import annotations

from src.training.constrained_inference import constrain_and_rank_sample


def _sample() -> dict[str, object]:
    return {
        "id": "ethanol",
        "canonical_smiles": "CCO",
        "molecular_formula": "C2H6O",
    }


def test_constrained_inference_filters_then_ranks() -> None:
    observed: list[tuple[str, ...]] = []

    def ranker(candidates: tuple[str, ...]) -> str:
        observed.append(candidates)
        return "CCO"

    row = constrain_and_rank_sample(
        _sample(),
        ["COC", "CCN", "OCC"],
        ranker=ranker,
    )

    assert observed == [("COC", "CCO")]
    assert row["prediction"] == "CCO"
    assert row["candidate_oracle_connectivity"] is True
    assert row["formula_valid_candidate_count"] == 2
    assert row["ranking_attempted"] is True
    assert row["ranking_failed"] is False
    assert row["connectivity_exact_match"] is True


def test_constrained_inference_returns_empty_on_formula_failure() -> None:
    row = constrain_and_rank_sample(
        _sample(),
        ["CCN"],
        ranker=lambda candidates: candidates[0],
    )

    assert row["prediction"] == ""
    assert row["formula_constraint_failed"] is True
    assert row["ranking_attempted"] is False
    assert row["exact_match"] is False


def test_constrained_inference_skips_ranker_for_single_candidate() -> None:
    row = constrain_and_rank_sample(
        _sample(),
        ["OCC"],
        ranker=lambda candidates: (_ for _ in ()).throw(AssertionError()),
    )

    assert row["prediction"] == "CCO"
    assert row["ranking_attempted"] is False
    assert row["ranking_failed"] is False
