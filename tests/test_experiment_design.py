"""Tests for the 50k NMR-to-structure experiment matrix."""

from __future__ import annotations

from pathlib import Path
import subprocess

import yaml


CONFIG_DIR = Path(__file__).parents[1] / "configs"

TRAIN_RUNS = {
    "experiments/train_scale_5k.yaml": ("clean_5k_train", 3407, True),
    "experiments/train_scale_10k.yaml": ("clean_10k_train", 3407, True),
    "experiments/train_scale_25k.yaml": ("clean_25k_train", 3407, True),
    "train_cuda_48g_jsonl.yaml": ("clean_50k_train", 3407, True),
    "experiments/train_main_50k_seed42.yaml": ("clean_50k_train", 42, True),
    "experiments/train_main_50k_seed2026.yaml": ("clean_50k_train", 2026, True),
    "train_cuda_48g_no_formula.yaml": ("clean_50k_train", 3407, False),
}

INFERENCE_RUNS = {
    "experiments/infer_zero_shot_50k.yaml": (None, True),
    "experiments/infer_scale_5k.yaml": (
        "outputs/experiments/scale-5k-formula-seed3407/best_model",
        True,
    ),
    "experiments/infer_scale_10k.yaml": (
        "outputs/experiments/scale-10k-formula-seed3407/best_model",
        True,
    ),
    "experiments/infer_scale_25k.yaml": (
        "outputs/experiments/scale-25k-formula-seed3407/best_model",
        True,
    ),
    "experiments/infer_main_50k_seed3407.yaml": (
        "outputs/experiments/main-50k-formula-seed3407/best_model",
        True,
    ),
    "experiments/infer_main_50k_seed42.yaml": (
        "outputs/experiments/main-50k-formula-seed42/best_model",
        True,
    ),
    "experiments/infer_main_50k_seed2026.yaml": (
        "outputs/experiments/main-50k-formula-seed2026/best_model",
        True,
    ),
    "experiments/infer_no_formula_50k.yaml": (
        "outputs/experiments/no-formula-50k-seed3407/best_model",
        False,
    ),
}


def _read_yaml(name: str) -> dict:
    path = CONFIG_DIR / name
    assert path.exists(), f"Missing experiment config: {path}"
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _assert_no_legacy_keys(config: dict) -> None:
    legacy = {"dataset_backend", "target_format", "mode", "render_cache_dir"}
    assert legacy.isdisjoint(config)


def test_training_matrix_uses_nested_splits_and_shared_validation() -> None:
    """All runs should implement the approved scaling and seed design."""
    output_dirs: set[str] = set()
    for name, (train_split, seed, include_formula) in TRAIN_RUNS.items():
        config = _read_yaml(name)
        _assert_no_legacy_keys(config)
        assert config["train_split_name"] == train_split
        assert config["eval_split_name"] == "clean_50k_val"
        assert config["max_eval_samples"] == 5000
        assert config["num_train_epochs"] == 2
        assert config["seed"] == seed
        assert config["include_formula"] is include_formula
        output_dirs.add(config["output_dir"])

    assert len(output_dirs) == len(TRAIN_RUNS)


def test_inference_matrix_uses_one_shared_scaffold_disjoint_test() -> None:
    """Zero-shot, seeds, and ablation must use exactly the same test IDs."""
    outputs: set[str] = set()
    for name, (adapter_path, include_formula) in INFERENCE_RUNS.items():
        config = _read_yaml(name)
        _assert_no_legacy_keys(config)
        assert config["split"] == "clean_50k_test"
        assert config["max_samples"] == 5000
        assert config.get("adapter_path") == adapter_path
        assert config["include_formula"] is include_formula
        outputs.add(config["output"])

    assert len(outputs) == len(INFERENCE_RUNS)


def test_experiment_runner_lists_all_named_runs() -> None:
    """One entrypoint should expose every training and inference run."""
    script = Path(__file__).parents[1] / "script" / "run_50k_experiment.sh"
    assert script.exists()

    result = subprocess.run(
        ["bash", str(script), "list"],
        check=True,
        capture_output=True,
        text=True,
    )

    for run_name in [
        "scale-5k",
        "scale-10k",
        "scale-25k",
        "main-3407",
        "main-42",
        "main-2026",
        "no-formula",
        "zero-shot",
    ]:
        assert run_name in result.stdout
