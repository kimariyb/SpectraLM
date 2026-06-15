"""Tests for prediction prompt modes and JSONL output."""

from __future__ import annotations

import json

from spectralm.inference.predict import (
    PredictionRecord,
    build_prediction_example,
    write_prediction_jsonl,
)


def test_build_prediction_example_supports_image_table_rule_mode(ethanol_sample) -> None:
    """Image-table-rule mode should include an image, peak tables, and rule hints."""
    example = build_prediction_example(ethanol_sample, mode="image_table_rule")
    content = example["messages"][0]["content"]
    text = content[1]["text"]
    assert content[0]["type"] == "image"
    assert "1H NMR peak table" in text
    assert "13C NMR peak table" in text
    assert "NMR rules to consider" in text


def test_build_prediction_example_supports_ablation_modes(ethanol_sample) -> None:
    """Ablation modes should remove the requested modalities from the prompt."""
    table_only = build_prediction_example(ethanol_sample, mode="table_only")
    table_content = table_only["messages"][0]["content"]
    assert all(part["type"] != "image" for part in table_content)
    assert "1H NMR peak table" in table_content[0]["text"]
    assert "NMR rules to consider" not in table_content[0]["text"]

    image_only = build_prediction_example(ethanol_sample, mode="image_only")
    image_content = image_only["messages"][0]["content"]
    assert image_content[0]["type"] == "image"
    assert "1H NMR peak table" not in image_content[1]["text"]
    assert "NMR rules to consider" not in image_content[1]["text"]


def test_write_prediction_jsonl_outputs_evaluation_fields(tmp_path, ethanol_sample) -> None:
    """Prediction JSONL should contain fields needed by the evaluator."""
    output = tmp_path / "predictions.jsonl"
    record = PredictionRecord(
        sample_id="ethanol",
        prediction="Spectral reasoning: 1H and 13C NMR. Final canonical SMILES: CCO",
        reference_smiles="CCO",
        mode="table_only",
    )
    write_prediction_jsonl(output, [record])
    row = json.loads(output.read_text(encoding="utf-8"))
    assert row["id"] == "ethanol"
    assert row["prediction"].startswith("Spectral reasoning:")
    assert row["reference_smiles"] == "CCO"
    assert row["mode"] == "table_only"
