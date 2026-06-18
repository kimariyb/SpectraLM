"""Configuration loading and training-log persistence for YAML-driven workflows."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments
from src.logger import TrainingLogger
import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML configuration file.

    Parameters
    ----------
    path
        Path to a YAML configuration file.

    Returns
    -------
    dict[str, Any]
        Parsed configuration values.

    Raises
    ------
    FileNotFoundError
        If the config file does not exist.
    TypeError
        If the YAML root is not a mapping.
    """
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise TypeError(
            f"Expected mapping config from {config_path}, "
            f"got {type(payload).__name__}"
        )
    return payload


class TrainingLoggerCallback(TrainerCallback):
    """HuggingFace Trainer callback for TrainingLogger.

    It records logs from Trainer.on_log and optionally saves JSON
    periodically during training.

    Parameters
    ----------
    logger
        TrainingLogger instance.
    save_every_n_logs
        If positive, save JSON every n log events.
    filename
        Optional fixed JSON filename. Useful for continuously overwriting
        the same log file during training.
    """

    def __init__(
        self,
        logger: TrainingLogger,
        save_every_n_logs: int = 0,
        filename: str = "training_log_live.json",
    ) -> None:
        self.logger = logger
        self.save_every_n_logs = int(save_every_n_logs)
        self.filename = filename
        self.num_logs = 0

    def on_log(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        logs: dict[str, Any] | None = None,
        **kwargs: Any,
    ):
        if not logs:
            return control

        entry = dict(logs)

        # HuggingFace logs sometimes do not include step.
        entry.setdefault("step", state.global_step)

        self.logger.log_entry(entry)
        self.num_logs += 1

        if self.save_every_n_logs > 0:
            if self.num_logs % self.save_every_n_logs == 0:
                self.logger.save_json(self.filename)

        return control

    def on_train_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: Any,
    ):
        self.logger.save_json(self.filename)
        self.logger.save_csv(prefix="training_log_live")
        return control