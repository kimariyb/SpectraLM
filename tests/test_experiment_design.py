"""Contract tests for the single supported two-stage 10k experiment."""

from __future__ import annotations

import ast
from pathlib import Path
import subprocess

import yaml


ROOT = Path(__file__).parents[1]
CONFIG_DIR = ROOT / "configs"
ACTIVE_CONFIGS = {
    "train_smoke.yaml",
    "experiments/train_stage1_10k.yaml",
    "experiments/train_stage2_10k.yaml",
    "experiments/infer_stage2_10k.yaml",
    "experiments/infer_constrained_10k.yaml",
}


def _read_yaml(name: str) -> dict:
    path = CONFIG_DIR / name
    assert path.exists(), f"Missing experiment config: {path}"
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def test_train_imports_unsloth_before_the_cuda_training_stack() -> None:
    train_path = ROOT / "src" / "training" / "train.py"
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


def test_only_current_experiment_configs_remain() -> None:
    retained = {
        str(path.relative_to(CONFIG_DIR))
        for path in CONFIG_DIR.rglob("*.yaml")
    }
    assert retained == ACTIVE_CONFIGS


def test_two_stage_10k_configs_share_the_approved_protocol() -> None:
    stage1 = _read_yaml("experiments/train_stage1_10k.yaml")
    stage2 = _read_yaml("experiments/train_stage2_10k.yaml")
    assert stage1["train_split_name"] == stage2["train_split_name"] == "clean_10k_train"
    assert stage1["eval_split_name"] == stage2["eval_split_name"] == "clean_10k_val"
    assert stage1["max_eval_samples"] == stage2["max_eval_samples"] == 1000
    expected_modes = {
        "full": 0.50,
        "image_only": 0.25,
        "peak_table_only": 0.25,
    }
    assert stage1["input_mode_weights"] == expected_modes
    assert stage2["input_mode_weights"] == expected_modes
    assert stage1["eval_input_mode_weights"] == {"full": 1.0}
    assert stage2["eval_input_mode_weights"] == {"full": 1.0}
    assert stage1["task_weights"] == {
        "structure_prediction": 0.40,
        "functional_group_recognition": 0.20,
        "candidate_ranking": 0.30,
        "spectral_region_classification": 0.10,
    }
    assert stage2["task_weights"] == {"structure_prediction": 1.0}
    assert stage2["initial_adapter_path"] == f"{stage1['output_dir']}/best_model"
    assert stage1["target_stereochemistry"] == "remove"
    assert stage2["target_stereochemistry"] == "remove"
    assert stage1["num_train_epochs"] == 1
    assert stage2["num_train_epochs"] == 2
    assert stage1["learning_rate"] == 1.0e-4
    assert stage2["learning_rate"] == 5.0e-5


def test_training_configs_have_matched_cuda_and_early_stopping_controls() -> None:
    for name in (
        "train_smoke.yaml",
        "experiments/train_stage1_10k.yaml",
        "experiments/train_stage2_10k.yaml",
    ):
        config = _read_yaml(name)
        assert config["model_path"] == "/mnt/data/kimariyb/models/Qwen3.5-9B"
        assert config["image_size"] == [512, 288]
        assert config["eval_accumulation_steps"] > 0
        assert config["early_stopping_patience"] > 0
        assert config["early_stopping_threshold"] >= 0
        assert config["per_device_eval_batch_size"] <= 16
        assert config["include_formula"] is True
        assert config["rule_context_enabled"] is False


def test_inference_configs_use_stage2_and_shared_test() -> None:
    stage2 = _read_yaml("experiments/train_stage2_10k.yaml")
    for name in (
        "experiments/infer_stage2_10k.yaml",
        "experiments/infer_constrained_10k.yaml",
    ):
        config = _read_yaml(name)
        assert config["adapter_path"] == f"{stage2['output_dir']}/best_model"
        assert config["split"] == "clean_10k_test"
        assert config["max_samples"] == 5000
        assert config["input_mode"] == "full"
        assert config["include_formula"] is True


def test_constrained_inference_uses_approved_sampling() -> None:
    config = _read_yaml("experiments/infer_constrained_10k.yaml")
    assert config["num_candidates"] == 32
    assert config["candidate_temperature"] == 0.7
    assert config["candidate_top_p"] == 0.9
    assert config["ranking_temperature"] == 0.0
    assert config["output"].startswith(
        "outputs/experiments/structure/predictions/"
    )


def test_experiment_runner_exposes_only_current_workflow() -> None:
    completed = subprocess.run(
        ["bash", "script/run_experiment.sh", "list"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert "prepare split-10k" in completed.stdout
    assert "train stage1-10k" in completed.stdout
    assert "train stage2-10k" in completed.stdout
    assert "infer constrained-10k" in completed.stdout
    assert "formula-only" not in completed.stdout
    subprocess.run(
        ["bash", "-n", "script/run_experiment.sh"],
        cwd=ROOT,
        check=True,
    )
