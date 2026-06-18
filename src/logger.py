from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any



class TrainingLogger:
    """Collect and persist training / evaluation metrics.

    Designed for HuggingFace Trainer / TRL SFTTrainer / Unsloth training,
    but can also be used standalone.

    Parameters
    ----------
    output_dir
        Directory where log files will be written.
    config
        Optional config dict embedded in the log for reproducibility.
    run_name
        Optional run name. If not provided, timestamp is used.
    """

    def __init__(
        self,
        output_dir: str | Path,
        config: dict[str, Any] | None = None,
        run_name: str | None = None,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.config = config or {}
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_name = run_name or f"run_{self.timestamp}"

        self.train_steps: list[dict[str, Any]] = []
        self.eval_steps: list[dict[str, Any]] = []
        self.other_logs: list[dict[str, Any]] = []

    # ---------------------------------------------------------------------
    # Constructors
    # ---------------------------------------------------------------------

    @classmethod
    def from_trainer(
        cls,
        trainer: Any,
        output_dir: str | Path,
        config: dict[str, Any] | None = None,
        run_name: str | None = None,
    ) -> "TrainingLogger":
        """Build logger from trainer.state.log_history."""

        logger = cls(
            output_dir=output_dir,
            config=config,
            run_name=run_name,
        )
        logger.ingest_trainer_logs(trainer)
        return logger

    def ingest_trainer_logs(self, trainer: Any) -> None:
        """Extract logs from trainer.state.log_history."""

        if not hasattr(trainer, "state"):
            return

        log_history = getattr(trainer.state, "log_history", None)
        if not log_history:
            return

        for entry in log_history:
            self.log_entry(entry)

    # ---------------------------------------------------------------------
    # Logging methods
    # ---------------------------------------------------------------------

    def log_entry(self, entry: dict[str, Any]) -> None:
        """Route one trainer log entry to train/eval/other logs."""

        clean = self._json_safe_dict(entry)

        if "eval_loss" in clean:
            self.eval_steps.append(clean)
        elif "loss" in clean:
            self.train_steps.append(clean)
        else:
            self.other_logs.append(clean)

    def log_train(
        self,
        step: int,
        loss: float,
        epoch: float | None = None,
        **extra: Any,
    ) -> None:
        """Record one training step manually."""

        entry: dict[str, Any] = {
            "step": step,
            "loss": float(loss),
        }

        if epoch is not None:
            entry["epoch"] = float(epoch)

        entry.update(extra)
        self.train_steps.append(self._json_safe_dict(entry))

    def log_eval(
        self,
        step: int,
        eval_loss: float,
        epoch: float | None = None,
        **extra: Any,
    ) -> None:
        """Record one evaluation step manually."""

        entry: dict[str, Any] = {
            "step": step,
            "eval_loss": float(eval_loss),
        }

        if epoch is not None:
            entry["epoch"] = float(epoch)

        entry.update(extra)
        self.eval_steps.append(self._json_safe_dict(entry))

    # ---------------------------------------------------------------------
    # Summary
    # ---------------------------------------------------------------------

    @property
    def final_train_loss(self) -> float | None:
        if not self.train_steps:
            return None
        return self.train_steps[-1].get("loss")

    @property
    def final_eval_loss(self) -> float | None:
        if not self.eval_steps:
            return None
        return self.eval_steps[-1].get("eval_loss")

    @property
    def best_eval_loss(self) -> float | None:
        losses = [
            item.get("eval_loss")
            for item in self.eval_steps
            if item.get("eval_loss") is not None
        ]
        if not losses:
            return None
        return min(losses)

    @property
    def best_eval_step(self) -> int | None:
        if not self.eval_steps:
            return None

        valid = [
            item for item in self.eval_steps
            if item.get("eval_loss") is not None
        ]
        if not valid:
            return None

        best = min(valid, key=lambda x: x["eval_loss"])
        return best.get("step")

    def summary(self) -> dict[str, Any]:
        """Return compact training summary."""

        return {
            "num_train_logs": len(self.train_steps),
            "num_eval_logs": len(self.eval_steps),
            "final_train_loss": self.final_train_loss,
            "final_eval_loss": self.final_eval_loss,
            "best_eval_loss": self.best_eval_loss,
            "best_eval_step": self.best_eval_step,
        }

    # ---------------------------------------------------------------------
    # Serialization
    # ---------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Build JSON-serializable log payload."""

        payload: dict[str, Any] = {
            "run_name": self.run_name,
            "timestamp": self.timestamp,
            "summary": self.summary(),
            "train_log": self.train_steps,
            "eval_log": self.eval_steps,
            "other_log": self.other_logs,
        }

        if self.config:
            payload["config"] = self._filter_config(self.config)

        return payload

    def save_json(self, filename: str | None = None) -> Path:
        """Write log to JSON."""

        self.output_dir.mkdir(parents=True, exist_ok=True)

        if filename is None:
            filename = f"training_log_{self.timestamp}.json"

        log_path = self.output_dir / filename
        log_path.write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        return log_path

    def save_csv(self, prefix: str | None = None) -> tuple[Path | None, Path | None]:
        """Write train and eval logs to separate CSV files."""

        self.output_dir.mkdir(parents=True, exist_ok=True)

        prefix = prefix or f"training_log_{self.timestamp}"

        train_path = None
        eval_path = None

        if self.train_steps:
            train_path = self.output_dir / f"{prefix}_train.csv"
            self._write_csv(train_path, self.train_steps)

        if self.eval_steps:
            eval_path = self.output_dir / f"{prefix}_eval.csv"
            self._write_csv(eval_path, self.eval_steps)

        return train_path, eval_path

    def save_all(self) -> dict[str, Path | None]:
        """Save JSON and CSV logs."""

        json_path = self.save_json()
        train_csv, eval_csv = self.save_csv()

        return {
            "json": json_path,
            "train_csv": train_csv,
            "eval_csv": eval_csv,
        }

    # ---------------------------------------------------------------------
    # Internal helpers
    # ---------------------------------------------------------------------

    @staticmethod
    def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
        """Write list of dictionaries to CSV with union of all keys."""

        fieldnames: list[str] = sorted(
            {key for row in rows for key in row.keys()}
        )

        # Put common columns first.
        priority = ["step", "epoch", "loss", "eval_loss", "learning_rate", "grad_norm"]
        ordered = [key for key in priority if key in fieldnames]
        ordered += [key for key in fieldnames if key not in ordered]

        with path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=ordered)
            writer.writeheader()
            writer.writerows(rows)

    @staticmethod
    def _filter_config(config: dict[str, Any]) -> dict[str, Any]:
        """Remove fields that are not useful or may leak local paths."""

        excluded_keys = {
            "model_path",
            "tokenizer_path",
            "dataset_path",
            "cache_dir",
            "hf_cache_dir",
            "render_cache_dir",
        }

        return {
            key: TrainingLogger._json_safe(value)
            for key, value in config.items()
            if key not in excluded_keys
        }

    @staticmethod
    def _json_safe_dict(data: dict[str, Any]) -> dict[str, Any]:
        return {
            str(key): TrainingLogger._json_safe(value)
            for key, value in data.items()
        }

    @staticmethod
    def _json_safe(value: Any) -> Any:
        """Convert common non-JSON values to JSON-safe objects."""

        if value is None:
            return None

        if isinstance(value, (str, int, float, bool)):
            return value

        if isinstance(value, Path):
            return str(value)

        # numpy scalar
        if hasattr(value, "item"):
            try:
                return value.item()
            except Exception:
                pass

        # list / tuple
        if isinstance(value, (list, tuple)):
            return [TrainingLogger._json_safe(v) for v in value]

        # dict
        if isinstance(value, dict):
            return {
                str(k): TrainingLogger._json_safe(v)
                for k, v in value.items()
            }

        return str(value)