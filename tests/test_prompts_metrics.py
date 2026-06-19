"""Tests for prompt generation and structure metrics."""

import src.evaluation.metrics as metrics_module
from src.evaluation.metrics import evaluate_structure_prediction, tanimoto_similarity
from src.evaluation.prompts import build_structure_prompt


def test_structure_prompt_can_omit_formula_for_ablation(ethanol_sample) -> None:
    """Formula-free ablations should not leak formula from labels."""
    prompt = build_structure_prompt(
        ethanol_sample,
        "Predict.\n\n{peak_tables}",
        include_formula=False,
    )
    assert "Molecular formula:" not in prompt


def test_structure_evaluation_handles_invalid_smiles(ethanol_sample) -> None:
    """Evaluation should not fail on invalid model structures."""
    row = evaluate_structure_prediction(
        "Final canonical SMILES: not_a_smiles",
        ethanol_sample["canonical_smiles"],
    )
    assert row["predicted_smiles"] is None
    assert row["valid_smiles"] is False
    assert row["tanimoto"] == 0.0


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
    assert summary["valid_smiles_rate"] == 0.5
    assert summary["mean_tanimoto"] == 0.5
