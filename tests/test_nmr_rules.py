"""Tests for the one-dimensional NMR rule library."""

from __future__ import annotations

import pytest

from src.nmr_rules.formula import calculate_dbe, parse_formula
from src.nmr_rules.engine import analyze_sample
from src.nmr_rules.models import RuleAnalysis, RuleEvidence
from src.nmr_rules.validator import validate_candidate


def test_parse_formula_and_calculate_dbe() -> None:
    """Formula constraints should follow the neutral closed-shell DBE rule."""
    assert parse_formula("C8H10O") == {"C": 8, "H": 10, "O": 1}
    assert calculate_dbe("C8H10O") == pytest.approx(4.0)
    assert calculate_dbe("C2H5Cl") == pytest.approx(0.0)


def test_dbe_covers_silicon_phosphorus_sulfur_and_halogen_formulae() -> None:
    """Generalized common-valence DBE should cover every allowed element."""
    assert calculate_dbe("Si2H6") == pytest.approx(0.0)
    assert calculate_dbe("C2H7P") == pytest.approx(0.0)
    assert calculate_dbe("C2H6S") == pytest.approx(0.0)
    assert calculate_dbe("CF3Cl") == pytest.approx(0.0)
    assert parse_formula("C2H5BrI") == {
        "C": 2,
        "H": 5,
        "Br": 1,
        "I": 1,
    }


def test_formula_constraints_skip_missing_and_reject_unsupported_formulae() -> None:
    """Missing formula is valid, while salts and fractional DBE are not inferred."""
    assert calculate_dbe(None) is None
    assert calculate_dbe("") is None
    with pytest.raises(ValueError, match="single neutral molecular formula"):
        parse_formula("C2H5O.Na")
    with pytest.raises(ValueError, match="non-negative integer or half-integer"):
        calculate_dbe("CH5")


def test_rule_analysis_serializes_stable_evidence() -> None:
    """Rule evidence should be auditable and JSON-compatible."""
    evidence = RuleEvidence(
        rule_id="FORMULA_DBE_001",
        category="formula",
        conclusion="Molecular formula C8H10O gives DBE = 4.",
        confidence=1.0,
        strength="hard",
        human_tip="Account for the DBE before assigning fragments.",
    )
    analysis = RuleAnalysis(
        library_name="nmr_1d",
        molecular_formula="C8H10O",
        dbe=4.0,
        evidence=(evidence,),
        warnings=(),
    )

    assert analysis.to_dict() == {
        "library_name": "nmr_1d",
        "molecular_formula": "C8H10O",
        "dbe": 4.0,
        "evidence": [
            {
                "rule_id": "FORMULA_DBE_001",
                "category": "formula",
                "conclusion": "Molecular formula C8H10O gives DBE = 4.",
                "confidence": 1.0,
                "strength": "hard",
                "human_tip": "Account for the DBE before assigning fragments.",
                "metadata": {},
            }
        ],
        "warnings": [],
    }


def test_rule_engine_detects_formula_and_ethyl_evidence(ethanol_sample) -> None:
    """Matching triplet/quartet J values should support an ethyl fragment."""
    analysis = analyze_sample(ethanol_sample, include_formula=True)
    by_id = {item.rule_id: item for item in analysis.evidence}

    assert analysis.library_name == "nmr_1d"
    assert analysis.molecular_formula == "C2H6O"
    assert analysis.dbe == pytest.approx(0.0)
    assert "FORMULA_DBE_001" in by_id
    assert "H1_FRAGMENT_ETHYL_001" in by_id
    assert by_id["H1_FRAGMENT_ETHYL_001"].strength == "strong"
    assert by_id["H1_FRAGMENT_ETHYL_001"].metadata["matching_j_hz"] == 7.0


def test_rule_engine_emits_soft_shift_region_evidence(ethanol_sample) -> None:
    """Shift-region assignments should be presented as non-exclusive evidence."""
    analysis = analyze_sample(ethanol_sample)
    ids = {item.rule_id for item in analysis.evidence}

    assert "H1_SHIFT_HETEROATOM_SP3" in ids
    assert "C13_SHIFT_HETEROATOM_SP3" in ids
    assert "C13_SIGNAL_COUNT_001" in ids


def test_rule_engine_skips_formula_rules_when_formula_is_unavailable(
    ethanol_sample,
) -> None:
    """Formula-free inference must not reconstruct formula from the label."""
    sample = dict(ethanol_sample)
    sample.pop("molecular_formula")

    analysis = analyze_sample(sample, include_formula=True)

    assert analysis.molecular_formula is None
    assert analysis.dbe is None
    assert all(item.category != "formula" for item in analysis.evidence)
    assert any(item.rule_id == "H1_FRAGMENT_ETHYL_001" for item in analysis.evidence)


def test_rule_engine_contains_no_solvent_peak_or_2d_assumptions(
    ethanol_sample,
) -> None:
    """The synthetic-data rule scope should remain strictly solvent-peak-free 1D."""
    analysis = analyze_sample(ethanol_sample)
    text = " ".join(
        [item.conclusion + " " + item.human_tip for item in analysis.evidence]
        + list(analysis.warnings)
    ).lower()

    forbidden = ["solvent peak", "cosy", "hsqc", "hmbc", "noesy", "dept"]
    assert all(term not in text for term in forbidden)


def test_candidate_validator_accepts_consistent_ethanol(ethanol_sample) -> None:
    """A matching structure should satisfy all applicable hard and motif checks."""
    result = validate_candidate("CCO", ethanol_sample, include_formula=True)

    assert result.valid_smiles is True
    assert result.applicable_checks >= 4
    assert result.satisfied_checks == result.applicable_checks
    assert result.consistency_rate == pytest.approx(1.0)
    assert result.contradictions == ()
    assert result.checks["formula_match"] is True
    assert result.checks["dbe_match"] is True
    assert result.checks["c13_signal_count_feasible"] is True
    assert result.checks["H1_FRAGMENT_ETHYL_001"] is True


def test_candidate_validator_reports_formula_contradiction(ethanol_sample) -> None:
    """A valid structure with the wrong formula should remain valid but conflict."""
    result = validate_candidate("CCCO", ethanol_sample, include_formula=True)

    assert result.valid_smiles is True
    assert result.checks["formula_match"] is False
    assert "formula_match" in result.contradictions
    assert result.consistency_rate < 1.0


def test_candidate_validator_compares_formula_by_element_counts(
    ethanol_sample,
) -> None:
    """Equivalent formula strings should not be rejected for element order."""
    sample = dict(ethanol_sample)
    sample["molecular_formula"] = "H6C2O"

    result = validate_candidate("CCO", sample, include_formula=True)

    assert result.checks["formula_match"] is True


def test_candidate_validator_supports_formula_free_samples(ethanol_sample) -> None:
    """Formula-free validation should retain spectrum-derived checks."""
    sample = dict(ethanol_sample)
    sample.pop("molecular_formula")

    result = validate_candidate("CCO", sample, include_formula=True)

    assert "formula_match" not in result.checks
    assert "dbe_match" not in result.checks
    assert result.checks["c13_signal_count_feasible"] is True
    assert result.checks["H1_FRAGMENT_ETHYL_001"] is True


def test_candidate_validator_handles_invalid_smiles(ethanol_sample) -> None:
    """Invalid candidates should produce a stable zero-consistency result."""
    result = validate_candidate("not_a_smiles", ethanol_sample)

    assert result.valid_smiles is False
    assert result.applicable_checks == 1
    assert result.satisfied_checks == 0
    assert result.consistency_rate == 0.0
    assert result.contradictions == ("valid_smiles",)
