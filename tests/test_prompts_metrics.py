"""Tests for prompt generation and structure metrics."""

import pytest

import src.evaluation.metrics as metrics_module
from src.evaluation.metrics import (
    classify_output_behavior,
    evaluate_candidate_ranking,
    evaluate_structure_prediction,
    tanimoto_similarity,
)
from src.evaluation.prompts import (
    STRUCTURE_PROMPTS,
    build_structure_prompt,
    select_structure_prompt,
)


def test_structure_prompt_collection_has_research_grade_constraints() -> None:
    """Every training template should define the same multimodal task."""
    assert len(STRUCTURE_PROMPTS) == 8
    assert len(set(STRUCTURE_PROMPTS)) == len(STRUCTURE_PROMPTS)

    for prompt in STRUCTURE_PROMPTS:
        normalized = " ".join(prompt.split()).lower()
        assert prompt.count("{peak_tables}") == 1
        assert "first image" in normalized
        assert "second image" in normalized
        assert "1h nmr" in normalized
        assert "13c nmr" in normalized
        assert "canonical smiles" in normalized
        assert "only" in normalized or "nothing else" in normalized


def test_inference_selects_current_prompt_by_explicit_index() -> None:
    """Inference should select a stable current template by explicit index."""
    assert select_structure_prompt(0) == STRUCTURE_PROMPTS[0]
    with pytest.raises(ValueError, match="prompt_template_index"):
        select_structure_prompt(len(STRUCTURE_PROMPTS))


def test_structure_prompt_can_omit_formula_for_ablation(ethanol_sample) -> None:
    """Formula-free ablations should not leak formula from labels."""
    prompt = build_structure_prompt(
        ethanol_sample,
        "Predict.\n\n{peak_tables}",
        include_formula=False,
    )
    assert "Molecular formula:" not in prompt


def test_structure_prompt_reads_formula_without_target_smiles(ethanol_sample) -> None:
    """Formula-conditioned prompts should use the explicit input field."""
    sample = dict(ethanol_sample)
    sample.pop("canonical_smiles")
    sample.pop("smiles")

    prompt = build_structure_prompt(
        sample,
        "Predict.\n\n{peak_tables}",
        include_formula=True,
    )

    assert "Molecular formula: C2H6O" in prompt


def test_structure_prompt_never_derives_formula_from_target(ethanol_sample) -> None:
    """A missing input formula must stay missing even when a label exists."""
    sample = dict(ethanol_sample)
    sample.pop("molecular_formula")

    prompt = build_structure_prompt(
        sample,
        "Predict.\n\n{peak_tables}",
        include_formula=True,
    )

    assert "Molecular formula:" not in prompt


def test_structure_prompt_can_include_compact_rule_context(ethanol_sample) -> None:
    """Rule-context experiments should add bounded auditable evidence."""
    prompt = build_structure_prompt(
        ethanol_sample,
        "Predict.\n\n{peak_tables}",
        include_formula=True,
        include_rule_context=True,
        max_rule_evidence=3,
    )

    assert "## Derived 1D NMR Constraints" in prompt
    assert "DBE = 0" in prompt
    assert "ethyl fragment" in prompt
    assert prompt.count("- [") == 3


def test_formula_free_rule_context_does_not_emit_dbe(ethanol_sample) -> None:
    """Formula-free rule evidence should remain useful without leaking DBE."""
    prompt = build_structure_prompt(
        ethanol_sample,
        "Predict.\n\n{peak_tables}",
        include_formula=False,
        include_rule_context=True,
    )

    assert "Molecular formula:" not in prompt
    assert "DBE" not in prompt
    assert "ethyl fragment" in prompt


def test_structure_evaluation_handles_invalid_smiles(ethanol_sample) -> None:
    """Evaluation should not fail on invalid model structures."""
    row = evaluate_structure_prediction(
        "Final canonical SMILES: not_a_smiles",
        ethanol_sample["canonical_smiles"],
    )
    assert row["predicted_smiles"] is None
    assert row["valid_smiles"] is False
    assert row["tanimoto"] == 0.0


def test_structure_evaluation_extracts_smiles_from_markdown_fence() -> None:
    """Zero-shot Markdown formatting should not be scored as invalid chemistry."""
    row = evaluate_structure_prediction(
        "```plaintext\nCCO\n```",
        "CCO",
    )

    assert row["predicted_smiles"] == "CCO"
    assert row["valid_smiles"] is True
    assert row["exact_match"] is True


def test_structure_evaluation_reports_formula_and_connectivity_matches() -> None:
    """Metrics should separate stereochemical exactness from connectivity."""
    row = evaluate_structure_prediction(
        "F[C@H](Cl)Br",
        "F[C@@H](Cl)Br",
    )

    assert row["exact_match"] is False
    assert row["connectivity_exact_match"] is True
    assert row["formula_match"] is True


def test_tanimoto_similarity_scores_exact_match() -> None:
    """Identical valid SMILES should score perfect Tanimoto similarity."""
    assert tanimoto_similarity("CCO", "CCO") == 1.0


def test_structure_summary_reports_direct_prediction_metrics() -> None:
    """Direct SMILES experiments should report structure-level outcomes."""
    evaluate_structure = getattr(
        metrics_module,
        "evaluate_structure_prediction",
        None,
    )
    summarize_structure = getattr(
        metrics_module,
        "summarize_structure_predictions",
        None,
    )

    assert callable(evaluate_structure)
    assert callable(summarize_structure)
    rows = [
        evaluate_structure("CCO", "CCO"),
        evaluate_structure("not_a_smiles", "CCN"),
    ]
    summary = summarize_structure(rows)

    assert summary["samples"] == 2
    assert summary["exact_match"] == 0.5
    assert summary["connectivity_exact_match"] == 0.5
    assert summary["formula_match_rate"] == 0.5
    assert summary["valid_smiles_rate"] == 0.5
    assert summary["mean_tanimoto"] == 0.5
    assert summary["median_tanimoto"] == 0.5
    assert summary["tanimoto_ge_0_5_rate"] == 0.5
    assert summary["molecular_formula_accuracy"] == 0.5
    assert summary["output_format_compliance_rate"] == 0.5
    assert summary["illegal_structure_rate"] == 0.5
    assert summary["non_smiles_output_rate"] == 0.0


def test_structure_metrics_can_include_rule_consistency(ethanol_sample) -> None:
    """Rule-enabled evaluation should report candidate contradictions."""
    exact = evaluate_structure_prediction(
        "CCO",
        "CCO",
        sample=ethanol_sample,
        include_formula=True,
    )
    wrong_formula = evaluate_structure_prediction(
        "CCCO",
        "CCO",
        sample=ethanol_sample,
        include_formula=True,
    )

    assert exact["rule_consistency_rate"] == 1.0
    assert exact["rule_contradiction_count"] == 0
    assert wrong_formula["rule_consistency_rate"] < 1.0
    assert wrong_formula["rule_contradiction_count"] >= 1

    summary = metrics_module.summarize_structure_predictions(
        [exact, wrong_formula]
    )
    assert summary["mean_rule_consistency_rate"] < 1.0
    assert summary["rule_contradiction_rate"] == 0.5


def test_structure_metrics_report_scaffold_match_and_coverage() -> None:
    """Murcko matching should exclude acyclic references from its denominator."""
    ring_match = evaluate_structure_prediction("Cc1ccccc1", "c1ccccc1")
    acyclic = evaluate_structure_prediction("CCN", "CCO")

    assert ring_match["exact_match"] is False
    assert ring_match["scaffold_evaluable"] is True
    assert ring_match["scaffold_match"] is True
    assert acyclic["scaffold_evaluable"] is False
    assert acyclic["scaffold_match"] is None

    summary = metrics_module.summarize_structure_predictions([ring_match, acyclic])
    assert summary["scaffold_coverage"] == 0.5
    assert summary["scaffold_match_rate"] == 1.0


def test_structure_metrics_report_functional_group_f1() -> None:
    """Functional-group similarity should compare controlled ontology sets."""
    exact = evaluate_structure_prediction("CCO", "CCO")
    mismatch = evaluate_structure_prediction("COC", "CCO")

    assert exact["predicted_functional_groups"] == ["alcohol"]
    assert exact["functional_group_f1"] == 1.0
    assert mismatch["predicted_functional_groups"] == ["ether"]
    assert mismatch["functional_group_f1"] == 0.0


def test_functional_group_spectral_consistency_uses_soft_1d_signatures(
    ethanol_sample,
) -> None:
    """Predicted functional groups should be checked against observable regions."""
    supported = evaluate_structure_prediction(
        "CCO",
        "CCO",
        sample=ethanol_sample,
    )
    unsupported = evaluate_structure_prediction(
        "CC(=O)C",
        "CCO",
        sample=ethanol_sample,
    )

    assert supported["functional_group_spectral_consistency"] == 1.0
    assert supported["spectral_functional_group_checks"]["alcohol"] is True
    assert unsupported["functional_group_spectral_consistency"] == 0.0
    assert unsupported["spectral_functional_group_checks"]["ketone"] is False


def test_output_behavior_states_are_disjoint() -> None:
    """Formatting, illegal structures, and non-SMILES text need distinct rates."""
    compliant = classify_output_behavior("CCO")
    illegal = classify_output_behavior("not_a_smiles")
    prose = classify_output_behavior("Final SMILES: CCO")
    fenced = classify_output_behavior("```smiles\nCCO\n```")

    assert compliant == {
        "output_format_compliant": True,
        "illegal_structure": False,
        "non_smiles_output": False,
    }
    assert illegal == {
        "output_format_compliant": False,
        "illegal_structure": True,
        "non_smiles_output": False,
    }
    assert prose["non_smiles_output"] is True
    assert fenced["non_smiles_output"] is True
    for result in (compliant, illegal, prose, fenced):
        assert sum(bool(value) for value in result.values()) == 1


def test_candidate_ranking_metrics_report_top1_and_mrr() -> None:
    """Candidate selection should expose Top-1 accuracy and reciprocal rank."""
    first = evaluate_candidate_ranking(["CCO", "COC", "CCN"], "CCO")
    second = evaluate_candidate_ranking(["COC", "CCO", "CCN"], "CCO")
    missing = evaluate_candidate_ranking(["COC", "CCN"], "CCO")

    assert first == {"candidate_top1_accuracy": 1.0, "candidate_mrr": 1.0}
    assert second == {"candidate_top1_accuracy": 0.0, "candidate_mrr": 0.5}
    assert missing == {"candidate_top1_accuracy": 0.0, "candidate_mrr": 0.0}
