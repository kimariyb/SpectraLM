"""Configuration loading and training-log persistence for YAML-driven workflows."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml


# ---------------------------------------------------------------------------
# YAML config
# ---------------------------------------------------------------------------


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

class TrainingLogger:
    """Collect and persist training / evaluation metrics to JSON.

    Designed to work with :class:`trl.SFTTrainer` but usable standalone
    for any step-wise metric recording.

    Parameters
    ----------
    output_dir
        Directory where the log JSON will be written.
    config
        Optional config dict to embed in the log for reproducibility.
    """

    def __init__(
        self,
        output_dir: str | Path,
        config: dict[str, Any] | None = None,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.config = config or {}
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        self.train_steps: list[dict[str, Any]] = []
        self.eval_steps: list[dict[str, Any]] = []

    @classmethod
    def from_trainer(
        cls,
        trainer: Any,
        output_dir: str | Path,
        config: dict[str, Any] | None = None,
    ) -> TrainingLogger:
        """Build a logger by extracting the log history from a finished
        :class:`~trl.SFTTrainer`.

        Entries containing ``"loss"`` (without ``"eval_loss"``) go to
        :attr:`train_steps`; entries containing ``"eval_loss"`` go to
        :attr:`eval_steps`.

        Parameters
        ----------
        trainer
            An :class:`~trl.SFTTrainer` instance whose ``.state.log_history``
            has been populated during training.
        output_dir
            Directory for the output JSON log file.
        config
            Optional config dict to embed.

        Returns
        -------
        TrainingLogger
            Populated logger ready to :meth:`save`.
        """
        logger = cls(output_dir, config)
        logger._ingest_trainer_logs(trainer)
        return logger

    def _ingest_trainer_logs(self, trainer: Any) -> None:
        """Split ``trainer.state.log_history`` into train / eval steps."""
        if not (hasattr(trainer, "state") and trainer.state.log_history):
            return

        for entry in trainer.state.log_history:
            if "eval_loss" in entry:
                self.eval_steps.append(entry)
            elif "loss" in entry:
                self.train_steps.append(entry)

    def log_train(self, step: int, loss: float, **extra: Any) -> None:
        """Record a training step.

        Parameters
        ----------
        step
            Global training step number.
        loss
            Training loss value.
        **extra
            Additional scalars (``learning_rate``, ``grad_norm``, …).
        """
        entry: dict[str, Any] = {"step": step, "loss": loss}
        entry.update(extra)
        self.train_steps.append(entry)

    def log_eval(self, step: int, eval_loss: float, **extra: Any) -> None:
        """Record an evaluation step.

        Parameters
        ----------
        step
            Global step number at which eval ran.
        eval_loss
            Evaluation loss value.
        **extra
            Additional scalars.
        """
        entry: dict[str, Any] = {"step": step, "eval_loss": eval_loss}
        entry.update(extra)
        self.eval_steps.append(entry)

    def to_dict(self) -> dict[str, Any]:
        """Build the dictionary that will be persisted.

        Returns
        -------
        dict[str, Any]
            Log payload.
        """
        payload: dict[str, Any] = {
            "timestamp": self.timestamp,
            "train_log": self.train_steps,
            "eval_log": self.eval_steps,
            "final_train_loss": (
                self.train_steps[-1]["loss"] if self.train_steps else None
            ),
            "final_eval_loss": (
                self.eval_steps[-1]["eval_loss"] if self.eval_steps else None
            ),
        }
        if self.config:
            payload["config"] = {
                k: v for k, v in self.config.items() if k != "model_path"
            }
        return payload

    def save(self) -> Path:
        """Write the log to ``<output_dir>/training_log_<timestamp>.json``.

        Returns
        -------
        Path
            Path to the written JSON file.
        """
        self.output_dir.mkdir(parents=True, exist_ok=True)
        log_path = self.output_dir / f"training_log_{self.timestamp}.json"
        log_path.write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"Training log saved to {log_path}")
        return log_path
