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
            "dataloader_num_workers": 4,
            "dataloader_prefetch_factor": 2,
            "dataloader_persistent_workers": True,
            "dataloader_pin_memory": True,
        }
    )

    assert kwargs["per_device_eval_batch_size"] == 32
    assert kwargs["dataloader_num_workers"] == 4
    assert kwargs["dataloader_prefetch_factor"] == 2
    assert kwargs["dataloader_persistent_workers"] is True
    assert kwargs["dataloader_pin_memory"] is True


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


def test_build_vision_collator_kwargs_uses_configured_image_size() -> None:
    """The configured render size should also control Unsloth collation."""
    builder = getattr(
        _training_arguments_module(),
        "build_vision_collator_kwargs",
        None,
    )

    assert callable(builder)
    assert builder({"image_size": [768, 432]}) == {"resize": (768, 432)}


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
