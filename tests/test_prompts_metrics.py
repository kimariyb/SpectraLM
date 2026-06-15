"""Tests for prompt generation and evaluation metrics."""

from spectralm.evaluation.metrics import evaluate_predictions
from spectralm.training.prompts import build_reasoning_target, build_structure_prompt


def test_reasoning_target_outputs_reasoning_before_structures(ethanol_sample) -> None:
    """Training target should put reasoning before SELFIES and SMILES."""
    target = build_reasoning_target(ethanol_sample)
    assert target.index("Spectral reasoning:") < target.index("Final SELFIES:")
    assert target.index("Final SELFIES:") < target.index("Final canonical SMILES:")


def test_structure_prompt_contains_image_table_rule_contract(ethanol_sample) -> None:
    """Structure prompt should include peak tables and NMR rule hints."""
    prompt = build_structure_prompt(ethanol_sample, "Predict.")
    assert "1H NMR peak table" in prompt
    assert "13C NMR peak table" in prompt
    assert "NMR rules to consider" in prompt


def test_evaluate_predictions_handles_invalid_smiles(ethanol_sample) -> None:
    """Evaluation should not fail on invalid model structures."""
    report = evaluate_predictions(["Spectral reasoning: 1H and 13C NMR. Final canonical SMILES: not_a_smiles"], [ethanol_sample])
    assert report["summary"]["samples"] == 1
    assert report["rows"][0]["predicted_smiles"] is None
    assert report["summary"]["invalid_structure_rate"] == 1.0

