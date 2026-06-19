"""Tests for prompt generation and structure metrics."""

from src.evaluation.metrics import evaluate_predictions, tanimoto_similarity
from src.evaluation.prompts import build_reasoning_target, build_structure_prompt


def test_reasoning_target_outputs_reasoning_before_structures(ethanol_sample) -> None:
    """Training target should put reasoning before SELFIES and SMILES."""
    target = build_reasoning_target(ethanol_sample)
    assert target.index("Spectral reasoning:") < target.index("Final SELFIES:")
    assert target.index("Final SELFIES:") < target.index("Final canonical SMILES:")


def test_structure_prompt_can_include_rule_contract(ethanol_sample) -> None:
    """Rule-enabled prompts should expose NMR reasoning hints."""
    prompt = build_structure_prompt(
        ethanol_sample,
        "Predict.\n\n{peak_tables}",
        include_rules=True,
    )
    assert "1H NMR Peak Table" in prompt
    assert "13C NMR Peak Table" in prompt
    assert "NMR rules to consider" in prompt


def test_structure_prompt_can_omit_formula_for_ablation(ethanol_sample) -> None:
    """Formula-free ablations should not leak formula from labels."""
    prompt = build_structure_prompt(
        ethanol_sample,
        "Predict.\n\n{peak_tables}",
        include_formula=False,
    )
    assert "Molecular formula:" not in prompt


def test_evaluate_predictions_handles_invalid_smiles(ethanol_sample) -> None:
    """Evaluation should not fail on invalid model structures."""
    report = evaluate_predictions(
        [
            (
                "Spectral reasoning: 1H and 13C NMR. "
                "Final canonical SMILES: not_a_smiles"
            )
        ],
        [ethanol_sample],
    )
    assert report["summary"]["samples"] == 1
    assert report["rows"][0]["predicted_smiles"] is None
    assert report["rows"][0]["tanimoto"] == 0.0
    assert report["summary"]["invalid_structure_rate"] == 1.0


def test_tanimoto_similarity_scores_exact_match() -> None:
    """Identical valid SMILES should score perfect Tanimoto similarity."""
    assert tanimoto_similarity("CCO", "CCO") == 1.0
