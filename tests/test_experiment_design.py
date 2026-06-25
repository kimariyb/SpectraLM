"""Contract tests for the supported text-only 10k experiment."""

from __future__ import annotations

import ast
from pathlib import Path
import subprocess

import yaml


ROOT = Path(__file__).parents[1]
CONFIG_DIR = ROOT / "configs"
ACTIVE_CONFIGS = {
    "train_smoke.yaml",
    "experiments/train_stage1_formula_10k.yaml",
    "experiments/train_stage2_formula_10k.yaml",
    "experiments/train_stage1_no_formula_10k.yaml",
    "experiments/train_stage2_no_formula_10k.yaml",
    "experiments/infer_direct_formula_10k.yaml",
    "experiments/infer_candidates_formula_10k.yaml",
    "experiments/infer_direct_no_formula_10k.yaml",
    "experiments/infer_candidates_no_formula_10k.yaml",
}
LEGACY_VISUAL_KEYS = {
    "image_backend",
    "rendered_image_dir",
    "missing_image_policy",
    "image_size",
    "h_snr",
    "c_snr",
    "render_seed",
    "input_mode",
    "input_mode_weights",
    "eval_input_mode_weights",
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


def test_training_configs_share_text_model_and_no_visual_keys() -> None:
    for name in ACTIVE_CONFIGS:
        config = _read_yaml(name)
        assert config["model_path"] == "/mnt/data/kimariyb/models/Qwen3-8B"
        assert not (set(config) & LEGACY_VISUAL_KEYS)


def test_formula_and_no_formula_stages_are_separate() -> None:
    formula_stage1 = _read_yaml("experiments/train_stage1_formula_10k.yaml")
    formula_stage2 = _read_yaml("experiments/train_stage2_formula_10k.yaml")
    no_formula_stage1 = _read_yaml("experiments/train_stage1_no_formula_10k.yaml")
    no_formula_stage2 = _read_yaml("experiments/train_stage2_no_formula_10k.yaml")

    for config in (
        formula_stage1,
        formula_stage2,
        no_formula_stage1,
        no_formula_stage2,
    ):
        assert config["train_split_name"] == "clean_10k_train"
        assert config["eval_split_name"] == "clean_10k_val"
        assert config["max_eval_samples"] == 1000
        assert config["target_stereochemistry"] == "remove"
        assert config["rule_context_enabled"] is False
        assert config["eval_accumulation_steps"] > 0
        assert config["early_stopping_patience"] > 0
        assert config["early_stopping_threshold"] >= 0

    assert formula_stage1["include_formula"] is True
    assert formula_stage2["include_formula"] is True
    assert no_formula_stage1["include_formula"] is False
    assert no_formula_stage2["include_formula"] is False
    assert formula_stage2["initial_adapter_path"] == (
        f"{formula_stage1['output_dir']}/best_model"
    )
    assert no_formula_stage2["initial_adapter_path"] == (
        f"{no_formula_stage1['output_dir']}/best_model"
    )
    assert formula_stage1["task_weights"]["candidate_ranking"] == 0.30
    assert formula_stage2["task_weights"] == {"structure_prediction": 1.0}


def test_inference_configs_match_stage2_adapters_and_shared_test() -> None:
    pairs = [
        (
            "experiments/train_stage2_formula_10k.yaml",
            "experiments/infer_direct_formula_10k.yaml",
            True,
        ),
        (
            "experiments/train_stage2_formula_10k.yaml",
            "experiments/infer_candidates_formula_10k.yaml",
            True,
        ),
        (
            "experiments/train_stage2_no_formula_10k.yaml",
            "experiments/infer_direct_no_formula_10k.yaml",
            False,
        ),
        (
            "experiments/train_stage2_no_formula_10k.yaml",
            "experiments/infer_candidates_no_formula_10k.yaml",
            False,
        ),
    ]
    for train_name, infer_name, include_formula in pairs:
        stage2 = _read_yaml(train_name)
        config = _read_yaml(infer_name)
        assert config["adapter_path"] == f"{stage2['output_dir']}/best_model"
        assert config["split"] == "clean_10k_test"
        assert config["max_samples"] == 1000
        assert config["include_formula"] is include_formula


def test_candidate_inference_uses_approved_sampling() -> None:
    for name in (
        "experiments/infer_candidates_formula_10k.yaml",
        "experiments/infer_candidates_no_formula_10k.yaml",
    ):
        config = _read_yaml(name)
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
    assert "prepare candidates-formula-10k-train" in completed.stdout
    assert "train stage1-formula-10k" in completed.stdout
    assert "train stage2-no-formula-10k" in completed.stdout
    assert "infer candidates-formula-10k" in completed.stdout
    assert "image-only" not in completed.stdout
    assert "formula-only" not in completed.stdout
    subprocess.run(
        ["bash", "-n", "script/run_experiment.sh"],
        cwd=ROOT,
        check=True,
    )


def test_active_code_and_docs_have_no_legacy_visual_workflow_terms() -> None:
    paths = [
        ROOT / "src",
        ROOT / "script",
        ROOT / "configs",
        ROOT / "README.md",
        ROOT / "docs" / "experiments.md",
        ROOT / "docs" / "research_design.md",
    ]
    needles = [
        "FastVisionModel",
        "UnslothVisionDataCollator",
        "load_sample_images",
        "image_backend",
        "rendered_image_dir",
        "input_mode_weights",
    ]
    for path in paths:
        files = path.rglob("*.py") if path.is_dir() else [path]
        for file_path in files:
            if file_path == ROOT / "src" / "training" / "arguments.py":
                continue
            text = file_path.read_text(encoding="utf-8")
            for needle in needles:
                assert needle not in text, f"{needle} remains in {file_path}"
