"""Pure construction of TRL SFT configuration keyword arguments."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def build_response_only_collator_kwargs() -> dict[str, Any]:
    """Return Qwen3-VL boundaries for assistant-only supervision.

    These markers match the active Qwen3-VL chat template. Keeping response
    masking mandatory prevents long NMR prompts and peak tables from
    dominating the language-model loss.
    """
    return {
        "train_on_responses_only": True,
        "instruction_part": "<|im_start|>user\n",
        "response_part": "<|im_start|>assistant\n",
        "force_match": True,
        "last_response_only": True,
    }


def build_vision_collator_kwargs(config: dict[str, Any]) -> dict[str, Any]:
    """Build image-resize arguments for ``UnslothVisionDataCollator``."""
    image_size = config.get("image_size")
    if image_size is None:
        return {}
    if not isinstance(image_size, (list, tuple)) or len(image_size) != 2:
        raise ValueError("image_size must contain exactly [width, height]")
    width, height = (int(value) for value in image_size)
    if width <= 0 or height <= 0:
        raise ValueError("image_size dimensions must be positive")
    return {"resize": (width, height)}


def training_log_dir(config: dict[str, Any]) -> Path:
    """Return the run-local training log directory."""
    return Path(config.get("output_dir", "outputs")) / "logs"


def build_early_stopping_kwargs(config: dict[str, Any]) -> dict[str, int | float]:
    """Build validated arguments for ``EarlyStoppingCallback``."""
    patience = int(config.get("early_stopping_patience", 3))
    threshold = float(config.get("early_stopping_threshold", 0.001))
    if patience <= 0:
        raise ValueError("early_stopping_patience must be positive")
    if threshold < 0:
        raise ValueError("early_stopping_threshold must be non-negative")
    return {
        "early_stopping_patience": patience,
        "early_stopping_threshold": threshold,
    }


def build_sft_kwargs(config: dict[str, Any]) -> dict[str, Any]:
    """Build validated keyword arguments for :class:`trl.SFTConfig`.

    Parameters
    ----------
    config
        Parsed training YAML mapping.

    Returns
    -------
    dict[str, Any]
        Keyword arguments accepted by ``SFTConfig``.
    """
    num_workers = int(config.get("dataloader_num_workers", 0))
    if num_workers < 0:
        raise ValueError("dataloader_num_workers must be non-negative")

    eval_batch_size = int(config.get("per_device_eval_batch_size", 8))
    if eval_batch_size <= 0:
        raise ValueError("per_device_eval_batch_size must be positive")

    eval_accumulation_steps = int(config.get("eval_accumulation_steps", 4))
    if eval_accumulation_steps <= 0:
        raise ValueError("eval_accumulation_steps must be positive")

    persistent_workers = bool(
        config.get("dataloader_persistent_workers", False)
    )
    if persistent_workers and num_workers == 0:
        raise ValueError(
            "dataloader_persistent_workers requires dataloader_num_workers > 0"
        )

    kwargs: dict[str, Any] = {
        "per_device_train_batch_size": config.get(
            "per_device_train_batch_size", 4
        ),
        "per_device_eval_batch_size": eval_batch_size,
        "eval_accumulation_steps": eval_accumulation_steps,
        "gradient_accumulation_steps": config.get(
            "gradient_accumulation_steps", 4
        ),
        "warmup_steps": config.get("warmup_steps", 5),
        "num_train_epochs": float(config["num_train_epochs"]),
        "learning_rate": float(config.get("learning_rate", 2e-4)),
        "logging_strategy": "steps",
        "logging_steps": config.get("logging_steps", 1),
        "logging_first_step": True,
        "eval_strategy": "steps",
        "eval_steps": config.get("eval_steps", 50),
        "save_strategy": "steps",
        "save_steps": config.get("save_steps", 50),
        "save_total_limit": int(config.get("save_total_limit", 5)),
        "metric_for_best_model": "eval_loss",
        "greater_is_better": False,
        "load_best_model_at_end": True,
        "optim": config.get("optim", "adamw_8bit"),
        "weight_decay": float(config.get("weight_decay", 0.001)),
        "lr_scheduler_type": config.get(
            "lr_scheduler_type", "cosine_with_restarts"
        ),
        "seed": config.get("seed", 3407),
        "output_dir": config.get("output_dir", "outputs"),
        "report_to": config.get("report_to", "none"),
        "remove_unused_columns": False,
        "dataset_text_field": "",
        "dataset_kwargs": {"skip_prepare_dataset": True},
        "max_length": config.get("max_seq_length", 8192),
        "bf16": bool(config.get("bf16", True)),
        "fp16": bool(config.get("fp16", False)),
        "dataloader_num_workers": num_workers,
        "dataloader_persistent_workers": persistent_workers,
        "dataloader_pin_memory": bool(
            config.get("dataloader_pin_memory", True)
        ),
    }

    if num_workers > 0:
        prefetch_factor = int(config.get("dataloader_prefetch_factor", 2))
        if prefetch_factor <= 0:
            raise ValueError("dataloader_prefetch_factor must be positive")
        kwargs["dataloader_prefetch_factor"] = prefetch_factor

    if config.get("max_steps") is not None:
        kwargs["max_steps"] = int(config["max_steps"])

    return kwargs
