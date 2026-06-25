"""LoRA/QLoRA fine-tuning entrypoint for SpectraLM text experiments.

Usage::

    python -m src.training.train configs/train_smoke.yaml
"""

from __future__ import annotations
import unsloth
import sys
from pathlib import Path
from typing import Any

import torch
from peft import PeftModel
from torch.utils.data import Subset
from transformers import EarlyStoppingCallback
from trl import SFTConfig, SFTTrainer
from unsloth import FastLanguageModel

from src.config import TrainingLoggerCallback, load_config
from src.data.dataset import load_lazy_nmr_dataset
from src.data.tasks import normalize_task_weights
from src.logger import TrainingLogger
from src.training.arguments import (
    build_early_stopping_kwargs,
    build_sft_kwargs,
    training_log_dir,
)
from src.training.model_setup import setup_lora_model
from src.training.response_masking import (
    assistant_response_text,
    validate_response_only_batch,
)
from src.training.text_collator import TextResponseOnlyCollator


def _limit_dataset(dataset, max_samples: int | None):
    """Return a deterministic prefix subset when max_samples is configured."""
    if max_samples is None:
        return dataset
    n = min(int(max_samples), len(dataset))
    return Subset(dataset, range(n))


def main(config: dict[str, Any]) -> None:
    """Run a text-only LoRA fine-tuning job from a configuration dictionary."""
    seed = int(config.get("seed", 3407))
    rule_context_enabled = bool(config.get("rule_context_enabled", False))
    task_weights = normalize_task_weights(config.get("task_weights"))

    dataset_dir: str = config["dataset_dir"]
    train_split_name: str = config.get("train_split_name", "train")
    eval_split_name: str = config.get("eval_split_name", "validation")

    dataset_kwargs = {
        "include_formula": config.get("include_formula", True),
        "include_rule_context": rule_context_enabled,
        "max_rule_evidence": int(config.get("max_rule_evidence", 12)),
        "task_weights": task_weights,
        "seed": seed,
        "prompt_template_index": config.get("prompt_template_index"),
        "target_stereochemistry": config.get(
            "target_stereochemistry", "preserve"
        ),
    }
    train_ds = load_lazy_nmr_dataset(
        dataset_dir,
        split=train_split_name,
        candidate_sidecar_path=config.get("train_candidate_sidecar_path"),
        **dataset_kwargs,
    )
    eval_ds = load_lazy_nmr_dataset(
        dataset_dir,
        split=eval_split_name,
        candidate_sidecar_path=config.get("eval_candidate_sidecar_path"),
        **dataset_kwargs,
    )

    train_ds = _limit_dataset(train_ds, config.get("max_train_samples"))
    eval_ds = _limit_dataset(eval_ds, config.get("max_eval_samples"))

    print(f"Train samples: {len(train_ds)}  |  Eval samples: {len(eval_ds)}")

    if config.get("dry_run", False):
        first = train_ds[0]
        print("Dry run complete.")
        print(f"First sample keys: {list(first.keys())}")
        if "messages" in first:
            print(f"First sample roles: {[msg['role'] for msg in first['messages']]}")
            print(f"First user prompt:\n{first['messages'][1]['content']}")
            print(f"First target: {first['messages'][2]['content']}")
        return

    model, tokenizer = FastLanguageModel.from_pretrained(
        config["model_path"],
        max_seq_length=config.get("max_seq_length", 8192),
        load_in_4bit=config.get("load_in_4bit", True),
        use_gradient_checkpointing=config.get(
            "use_gradient_checkpointing", "unsloth"
        ),
        attn_implementation=config.get("attn_implementation", "sdpa"),
    )

    model = setup_lora_model(
        model,
        config,
        fast_language_model=FastLanguageModel,
        peft_model_class=PeftModel,
    )
    FastLanguageModel.for_training(model)

    sft_kwargs = build_sft_kwargs(config)
    early_stopping_kwargs = build_early_stopping_kwargs(config)
    data_collator = TextResponseOnlyCollator(
        tokenizer,
        max_length=config.get("max_seq_length", 8192),
    )

    preflight_sample = train_ds[0]
    masking_stats = validate_response_only_batch(
        data_collator([preflight_sample]),
        tokenizer,
        expected_response=assistant_response_text(preflight_sample),
    )
    print(
        "Response-only supervision verified: "
        f"{masking_stats['supervised_tokens']}/"
        f"{masking_stats['sequence_tokens']} tokens are supervised; "
        f"target={masking_stats['decoded_response']!r}"
    )

    training_logger = TrainingLogger(
        output_dir=training_log_dir(config),
        config={
            "model_name": Path(config.get("model_path")).name,
            "initial_adapter_path": config.get("initial_adapter_path"),
            "learning_rate": sft_kwargs["learning_rate"],
            "num_train_epochs": sft_kwargs["num_train_epochs"],
            "per_device_train_batch_size": sft_kwargs[
                "per_device_train_batch_size"
            ],
            "gradient_accumulation_steps": sft_kwargs[
                "gradient_accumulation_steps"
            ],
            "per_device_eval_batch_size": sft_kwargs[
                "per_device_eval_batch_size"
            ],
            "eval_accumulation_steps": sft_kwargs["eval_accumulation_steps"],
            "dataloader_num_workers": sft_kwargs["dataloader_num_workers"],
            "dataloader_prefetch_factor": sft_kwargs.get(
                "dataloader_prefetch_factor"
            ),
            "eval_steps": sft_kwargs["eval_steps"],
            **early_stopping_kwargs,
            "include_formula": config.get("include_formula", True),
            "target_stereochemistry": config.get(
                "target_stereochemistry", "preserve"
            ),
            "prompt_template_index": config.get("prompt_template_index"),
            "rule_context_enabled": rule_context_enabled,
            "max_rule_evidence": int(config.get("max_rule_evidence", 12)),
            "task_weights": task_weights,
            "train_candidate_sidecar_path": config.get(
                "train_candidate_sidecar_path"
            ),
            "eval_candidate_sidecar_path": config.get(
                "eval_candidate_sidecar_path"
            ),
            "train_on_responses_only": True,
            "train_split_name": train_split_name,
            "eval_split_name": eval_split_name,
        },
        run_name="nmr_text_sft",
    )

    logger_callback = TrainingLoggerCallback(
        logger=training_logger,
        save_every_n_logs=10,
        filename="training_log_live.json",
    )
    early_stopping_callback = EarlyStoppingCallback(**early_stopping_kwargs)

    print(
        "Early stopping: "
        f"patience={early_stopping_kwargs['early_stopping_patience']} evals, "
        f"threshold={early_stopping_kwargs['early_stopping_threshold']}"
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        data_collator=data_collator,
        callbacks=[logger_callback, early_stopping_callback],
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        args=SFTConfig(**sft_kwargs),
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )

    start_gpu_memory = 0.0
    max_memory = 0.0
    if torch.cuda.is_available():
        gpu_stats = torch.cuda.get_device_properties(0)
        start_gpu_memory = round(
            torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3
        )
        max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
        print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
        print(f"{start_gpu_memory} GB of memory reserved.")

    trainer_stats = trainer.train()

    best_model_dir = Path(config.get("output_dir", "outputs")) / "best_model"
    trainer.save_model(best_model_dir)
    tokenizer.save_pretrained(best_model_dir)

    print(f"Best model saved to: {best_model_dir}")
    print(f"{trainer_stats.metrics['train_runtime']} seconds used for training.")
    print(
        f"{round(trainer_stats.metrics['train_runtime'] / 60, 2)} "
        "minutes used for training."
    )

    if torch.cuda.is_available() and max_memory > 0:
        used_memory = round(
            torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3
        )
        used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
        used_percentage = round(used_memory / max_memory * 100, 3)
        lora_percentage = round(used_memory_for_lora / max_memory * 100, 3)
        print(f"Peak reserved memory = {used_memory} GB.")
        print(f"Peak reserved memory for training = {used_memory_for_lora} GB.")
        print(f"Peak reserved memory % of max memory = {used_percentage} %.")
        print(
            "Peak reserved memory for training % of max memory = "
            f"{lora_percentage} %."
        )


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m src.training.train <config.yaml>")
        sys.exit(1)
    main(load_config(sys.argv[1]))
