"""Tests for the current NMR-to-structure experiment matrix."""

from __future__ import annotations

import ast
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
        "outputs/experiments/structure/scale-5k-formula-seed3407/best_model",
        True,
    ),
    "experiments/infer_scale_10k.yaml": (
        "outputs/experiments/structure/scale-10k-formula-seed3407/best_model",
        True,
    ),
    "experiments/infer_scale_25k.yaml": (
        "outputs/experiments/structure/scale-25k-formula-seed3407/best_model",
        True,
    ),
    "experiments/infer_main_50k_seed3407.yaml": (
        "outputs/experiments/structure/main-50k-formula-seed3407/best_model",
        True,
    ),
    "experiments/infer_main_50k_seed42.yaml": (
        "outputs/experiments/structure/main-50k-formula-seed42/best_model",
        True,
    ),
    "experiments/infer_main_50k_seed2026.yaml": (
        "outputs/experiments/structure/main-50k-formula-seed2026/best_model",
        True,
    ),
    "experiments/infer_no_formula_50k.yaml": (
        "outputs/experiments/structure/no-formula-50k-seed3407/best_model",
        False,
    ),
}

RULE_TRAIN_RUNS = {
    "experiments/train_rules_50k.yaml": (True, "rules-50k-formula-seed3407"),
    "experiments/train_rules_no_formula_50k.yaml": (
        False,
        "rules-50k-no-formula-seed3407",
    ),
}

RULE_INFERENCE_RUNS = {
    "experiments/infer_rules_50k.yaml": (
        True,
        "rules-50k-formula-seed3407",
    ),
    "experiments/infer_rules_no_formula_50k.yaml": (
        False,
        "rules-50k-no-formula-seed3407",
    ),
}

MULTITASK_TRAIN_CONFIG = "experiments/train_multitask_50k.yaml"
MULTITASK_INFERENCE_CONFIG = "experiments/infer_multitask_50k.yaml"


def test_train_imports_unsloth_before_the_cuda_training_stack() -> None:
    """Unsloth must patch the training stack before Torch/Transformers imports."""
    train_path = Path(__file__).parents[1] / "src" / "training" / "train.py"
    tree = ast.parse(train_path.read_text(encoding="utf-8"))
    normal_imports = [
        node
        for node in tree.body
        if isinstance(node, (ast.Import, ast.ImportFrom))
        and not (isinstance(node, ast.ImportFrom) and node.module == "__future__")
    ]

    first_import = normal_imports[0]
    assert isinstance(first_import, ast.Import)
    assert [alias.name for alias in first_import.names] == ["unsloth"]


def _read_yaml(name: str) -> dict:
    path = CONFIG_DIR / name
    assert path.exists(), f"Missing experiment config: {path}"
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _assert_no_legacy_keys(config: dict) -> None:
    legacy = {
        "dataset_backend",
        "target_format",
        "mode",
        "render_cache_dir",
        "protocol_" + "version",
        "prompt_set_" + "version",
        "metric_" + "version",
        "library_" + "version",
    }
    assert legacy.isdisjoint(config)


def test_all_training_configs_set_positive_eval_accumulation_steps() -> None:
    """Every formal and smoke run should explicitly bound evaluation buffering."""
    training_configs = [
        path
        for path in CONFIG_DIR.rglob("*.yaml")
        if path.name.startswith("train_")
    ]
    assert training_configs
    for path in training_configs:
        config = yaml.safe_load(path.read_text(encoding="utf-8"))
        assert config["eval_accumulation_steps"] > 0, path


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
        assert config["image_size"] == [768, 432]
        assert config["per_device_train_batch_size"] == 16
        assert config["per_device_eval_batch_size"] > 0
        assert config["gradient_accumulation_steps"] > 0
        assert config["output_dir"].startswith("outputs/experiments/structure/")
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
        assert config["prompt_template_index"] == 0
        assert config["image_size"] == [768, 432]
        assert config["output"].startswith(
            "outputs/experiments/structure/predictions/"
        )
        outputs.add(config["output"])

    assert len(outputs) == len(INFERENCE_RUNS)


def test_rule_training_runs_are_isolated_from_baselines() -> None:
    """Rule-context training should keep data and optimization controls fixed."""
    for name, (include_formula, output_name) in RULE_TRAIN_RUNS.items():
        config = _read_yaml(name)
        baseline = _read_yaml(
            "train_cuda_48g_jsonl.yaml"
            if include_formula
            else "train_cuda_48g_no_formula.yaml"
        )
        _assert_no_legacy_keys(config)
        assert config["rule_context_enabled"] is True
        assert config["max_rule_evidence"] == 12
        assert config["include_formula"] is include_formula
        assert config["train_split_name"] == "clean_50k_train"
        assert config["eval_split_name"] == "clean_50k_val"
        assert config["max_eval_samples"] == 5000
        assert config["seed"] == 3407
        for key in (
            "num_train_epochs",
            "per_device_train_batch_size",
            "per_device_eval_batch_size",
            "eval_accumulation_steps",
            "gradient_accumulation_steps",
            "learning_rate",
            "weight_decay",
            "warmup_steps",
            "lr_scheduler_type",
            "optim",
            "eval_steps",
            "save_steps",
            "early_stopping_patience",
            "early_stopping_threshold",
        ):
            assert config[key] == baseline[key]
        assert config["output_dir"].endswith(output_name)
        assert config["output_dir"].startswith("outputs/experiments/rules/")


def test_rule_inference_runs_enable_candidate_validation() -> None:
    """Rule adapters should be evaluated on the shared 50k test subset."""
    for name, (include_formula, adapter_name) in RULE_INFERENCE_RUNS.items():
        config = _read_yaml(name)
        _assert_no_legacy_keys(config)
        assert config["rule_context_enabled"] is True
        assert config["rule_validation_enabled"] is True
        assert config["max_rule_evidence"] == 12
        assert config["include_formula"] is include_formula
        assert config["split"] == "clean_50k_test"
        assert config["max_samples"] == 5000
        assert config["adapter_path"].endswith(f"{adapter_name}/best_model")
        assert config["output"].startswith("outputs/experiments/rules/")


def test_multitask_run_uses_isolated_protocol_and_candidate_sidecars() -> None:
    """The approved four-task mixture should preserve the main-task majority."""
    train = _read_yaml(MULTITASK_TRAIN_CONFIG)
    _assert_no_legacy_keys(train)
    assert train["task_weights"] == {
        "structure_prediction": 0.60,
        "functional_group_recognition": 0.15,
        "candidate_ranking": 0.15,
        "spectral_region_classification": 0.10,
    }
    assert train["train_candidate_sidecar_path"].endswith(
        "candidate_sets_clean_50k_train.jsonl"
    )
    assert train["eval_candidate_sidecar_path"].endswith(
        "candidate_sets_clean_50k_val.jsonl"
    )
    assert train["output_dir"].startswith(
        "outputs/experiments/multitask/"
    )
    assert train["train_split_name"] == "clean_50k_train"
    assert train["eval_split_name"] == "clean_50k_val"

    inference = _read_yaml(MULTITASK_INFERENCE_CONFIG)
    _assert_no_legacy_keys(inference)
    assert inference["multitask_model"] is True
    assert inference["rule_validation_enabled"] is True
    assert inference["split"] == "clean_50k_test"
    assert inference["output"].startswith(
        "outputs/experiments/multitask/"
    )


def test_experiment_runner_lists_all_named_runs() -> None:
    """One entrypoint should expose every training and inference run."""
    script = Path(__file__).parents[1] / "script" / "run_experiment.sh"
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
        "rules-50k",
        "rules-no-formula",
        "multitask-50k",
        "zero-shot",
    ]:
        assert run_name in result.stdout


def test_removed_legacy_entrypoints_are_absent() -> None:
    """A clean rebuild should not expose migration or rescoring commands."""
    script_dir = Path(__file__).parents[1] / "script"
    removed = {
        "backfill_formula_jsonl.py",
        "filter_common_elements_jsonl.py",
        "rescore_predictions.py",
        "run_50k_experiment.sh",
    }
    assert all(not (script_dir / name).exists() for name in removed)


def test_active_research_files_do_not_use_numbered_release_labels() -> None:
    """The repository should describe one current research implementation."""
    root = Path(__file__).parents[1]
    scan_roots = ["configs", "script", "src", "rules", "docs", "README.md"]
    numbered_labels = tuple("_v" + str(index) for index in range(1, 4))
    forbidden_path = "protocol-" + "v" + "1"
    for relative in scan_roots:
        path = root / relative
        files = (
            [path]
            if path.is_file()
            else [item for item in path.rglob("*") if item.is_file()]
        )
        for item in files:
            if "__pycache__" in item.parts:
                continue
            text = item.read_text(encoding="utf-8")
            assert not any(label in text for label in numbered_labels), item
            assert forbidden_path not in text, item


def test_all_training_configs_enable_early_stopping() -> None:
    """Every smoke, baseline, ablation, rule, and multitask run should stop early."""
    training_configs = sorted(CONFIG_DIR.rglob("train*.yaml"))

    assert training_configs
    for path in training_configs:
        config = yaml.safe_load(path.read_text(encoding="utf-8"))
        assert config["early_stopping_patience"] == 3, path
        assert config["early_stopping_threshold"] == 0.001, path
        assert config["eval_steps"] == config["save_steps"], path
