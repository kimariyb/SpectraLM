"""Tests for pure SFT training argument construction."""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest


def _build_sft_kwargs(config: dict):
    """Load the requested pure argument builder with a clear test failure."""
    try:
        module = importlib.import_module("src.training.arguments")
    except ModuleNotFoundError:
        pytest.fail("src.training.arguments has not been implemented")
    return module.build_sft_kwargs(config)


def _training_arguments_module():
    """Load pure training helpers without importing the CUDA training stack."""
    return importlib.import_module("src.training.arguments")


def test_build_sft_kwargs_exposes_dataloader_and_eval_batch_settings() -> None:
    """CUDA input-pipeline settings should reach Hugging Face SFTConfig."""
    kwargs = _build_sft_kwargs(
        {
            "num_train_epochs": 1,
            "per_device_eval_batch_size": 32,
            "eval_accumulation_steps": 8,
            "dataloader_num_workers": 4,
            "dataloader_prefetch_factor": 2,
            "dataloader_persistent_workers": True,
            "dataloader_pin_memory": True,
        }
    )

    assert kwargs["per_device_eval_batch_size"] == 32
    assert kwargs["eval_accumulation_steps"] == 8
    assert kwargs["dataloader_num_workers"] == 4
    assert kwargs["dataloader_prefetch_factor"] == 2
    assert kwargs["dataloader_persistent_workers"] is True
    assert kwargs["dataloader_pin_memory"] is True


def test_build_sft_kwargs_defaults_eval_accumulation_steps_to_four() -> None:
    """Evaluation logits should be moved off-device at a bounded cadence."""
    kwargs = _build_sft_kwargs({"num_train_epochs": 1})

    assert kwargs["eval_accumulation_steps"] == 4


def test_build_sft_kwargs_rejects_invalid_eval_accumulation_steps() -> None:
    """Evaluation accumulation must use a positive number of steps."""
    with pytest.raises(ValueError, match="eval_accumulation_steps must be positive"):
        _build_sft_kwargs(
            {"num_train_epochs": 1, "eval_accumulation_steps": 0}
        )


def test_build_sft_kwargs_omits_prefetch_for_single_process_loading() -> None:
    """PyTorch rejects prefetch_factor when DataLoader has no workers."""
    kwargs = _build_sft_kwargs(
        {
            "num_train_epochs": 1,
            "dataloader_num_workers": 0,
            "dataloader_prefetch_factor": 2,
        }
    )

    assert kwargs["dataloader_num_workers"] == 0
    assert "dataloader_prefetch_factor" not in kwargs


def test_visual_collator_builder_is_removed() -> None:
    """The text workflow should not expose image collator settings."""
    assert not hasattr(_training_arguments_module(), "build_vision_collator_kwargs")


def test_legacy_visual_config_is_rejected() -> None:
    """Old image-workflow YAML fields should fail before model loading."""
    reject = getattr(_training_arguments_module(), "reject_legacy_visual_config")

    with pytest.raises(ValueError, match="image_backend"):
        reject({"image_backend": "pre_rendered"})

    reject({"include_formula": True})


def test_response_only_collator_kwargs_match_qwen_chat_boundaries() -> None:
    """Every training run should supervise only the assistant response."""
    builder = getattr(
        _training_arguments_module(),
        "build_response_only_collator_kwargs",
        None,
    )

    assert callable(builder)
    assert builder() == {
        "train_on_responses_only": True,
        "instruction_part": "<|im_start|>user\n",
        "response_part": "<|im_start|>assistant\n",
        "force_match": True,
        "last_response_only": True,
    }


def test_training_log_dir_is_isolated_under_each_output_dir() -> None:
    """Concurrent single-GPU runs must not overwrite each other's logs."""
    resolver = getattr(
        _training_arguments_module(),
        "training_log_dir",
        None,
    )

    assert callable(resolver)
    assert resolver({"output_dir": "outputs/experiments/run-a"}) == Path(
        "outputs/experiments/run-a/logs"
    )


def test_early_stopping_kwargs_use_configured_patience_and_threshold() -> None:
    """Early stopping should expose validated Transformers callback arguments."""
    builder = getattr(
        _training_arguments_module(),
        "build_early_stopping_kwargs",
        None,
    )

    assert callable(builder)
    assert builder(
        {
            "early_stopping_patience": 3,
            "early_stopping_threshold": 0.001,
        }
    ) == {
        "early_stopping_patience": 3,
        "early_stopping_threshold": 0.001,
    }


@pytest.mark.parametrize(
    ("config", "message"),
    [
        ({"early_stopping_patience": 0}, "early_stopping_patience must be positive"),
        (
            {"early_stopping_threshold": -0.001},
            "early_stopping_threshold must be non-negative",
        ),
    ],
)
def test_early_stopping_kwargs_reject_invalid_values(
    config: dict,
    message: str,
) -> None:
    """Invalid early-stopping controls should fail before model loading."""
    builder = getattr(
        _training_arguments_module(),
        "build_early_stopping_kwargs",
        None,
    )

    assert callable(builder)
    with pytest.raises(ValueError, match=message):
        builder(config)
