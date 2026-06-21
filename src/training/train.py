"""LoRA/QLoRA fine-tuning entrypoint for SpectraLM VLM experiments.

Usage::

    python -m src.training.train configs/train_cuda_48g_jsonl.yaml
"""

from __future__ import annotations

import sys
import torch
from typing import Any
from pathlib import Path
from torch.utils.data import Subset
from unsloth import FastVisionModel
from unsloth.trainer import UnslothVisionDataCollator
from src.config import TrainingLoggerCallback, load_config
from src.logger import TrainingLogger
from src.data.dataset import load_lazy_nmr_dataset
from src.data.tasks import normalize_task_weights
from src.training.arguments import (
    build_sft_kwargs,
    build_vision_collator_kwargs,
    training_log_dir,
)
from trl import SFTConfig, SFTTrainer


def _limit_dataset(dataset, max_samples: int | None):
    """Return a deterministic prefix subset when max_samples is configured."""
    if max_samples is None:
        return dataset
    n = min(int(max_samples), len(dataset))
    return Subset(dataset, range(n))


def main(config: dict[str, Any]) -> None:
    """Run a LoRA fine-tuning job from a configuration dictionary.

    Parameters
    ----------
    config
        Model, dataset, LoRA, and optimization settings for the active
        lazy-JSONL CUDA training path.
    """
    seed: int = config.get("seed", 3407)
    rule_context_enabled = bool(config.get("rule_context_enabled", False))
    task_weights = normalize_task_weights(config.get("task_weights"))

    # 1. Build datasets first so dry-run works without loading a VLM.
    dataset_dir: str = config["dataset_dir"]
    train_split_name: str = config.get("train_split_name", "train")
    eval_split_name: str = config.get("eval_split_name", "validation")

    dataset_kwargs = {
        "include_formula": config.get("include_formula", True),
        "include_rule_context": rule_context_enabled,
        "max_rule_evidence": int(config.get("max_rule_evidence", 12)),
        "task_weights": task_weights,
        "seed": seed,
        "h_snr": float(config.get("h_snr", 500.0)),
        "c_snr": float(config.get("c_snr", 300.0)),
        "render_seed": config.get("render_seed", seed),
        "image_size": config.get("image_size"),
        "image_backend": config.get("image_backend", "lazy_render"),
        "rendered_image_dir": config.get("rendered_image_dir"),
        "missing_image_policy": config.get("missing_image_policy", "error"),
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
        return

    # 2. Load model
    model, tokenizer = FastVisionModel.from_pretrained(
        config["model_path"],
        max_seq_length=config.get("max_seq_length", 8192),
        load_in_4bit=config.get("load_in_4bit", True),
        use_gradient_checkpointing=config.get("use_gradient_checkpointing", "unsloth"),
    )

    # 3. Apply LoRA adapters
    model = FastVisionModel.get_peft_model(
        model,
        finetune_vision_layers=True,
        finetune_language_layers=True,
        finetune_attention_modules=True,
        finetune_mlp_modules=True,
        r=config.get("lora_r", 16),
        lora_alpha=config.get("lora_alpha", 16),
        lora_dropout=float(config.get("lora_dropout", 0)),
        bias="none",
        random_state=seed,
        use_rslora=False,
        loftq_config=None,
    )

    # 4. Train
    FastVisionModel.for_training(model)

    sft_kwargs = build_sft_kwargs(config)
    vision_collator_kwargs = build_vision_collator_kwargs(config)

    training_logger = TrainingLogger(
        output_dir=training_log_dir(config),
        config={
            "model_name": Path(config.get("model_path")).name,
            "learning_rate": sft_kwargs['learning_rate'],
            "num_train_epochs": sft_kwargs['num_train_epochs'],
            "per_device_train_batch_size": sft_kwargs["per_device_train_batch_size"],
            "gradient_accumulation_steps": sft_kwargs['gradient_accumulation_steps'],
            "per_device_eval_batch_size": sft_kwargs["per_device_eval_batch_size"],
            "eval_accumulation_steps": sft_kwargs["eval_accumulation_steps"],
            "dataloader_num_workers": sft_kwargs["dataloader_num_workers"],
            "dataloader_prefetch_factor": sft_kwargs.get(
                "dataloader_prefetch_factor"
            ),
            "eval_steps": sft_kwargs['eval_steps'],
            "include_formula": config.get("include_formula", True),
            "rule_context_enabled": rule_context_enabled,
            "max_rule_evidence": int(config.get("max_rule_evidence", 12)),
            "task_weights": task_weights,
            "train_candidate_sidecar_path": config.get(
                "train_candidate_sidecar_path"
            ),
            "eval_candidate_sidecar_path": config.get(
                "eval_candidate_sidecar_path"
            ),
            "image_backend": config.get("image_backend", "lazy_render"),
            "image_size": config.get("image_size"),
            "collator_resize": vision_collator_kwargs.get("resize"),
            "rendered_image_dir": config.get("rendered_image_dir"),
            "train_split_name": train_split_name,
            "eval_split_name": eval_split_name,
        },
        run_name="nmr_vl_sft",
    )

    logger_callback = TrainingLoggerCallback(
        logger=training_logger,
        save_every_n_logs=10,
        filename="training_log_live.json",
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        data_collator=UnslothVisionDataCollator(
            model,
            tokenizer,
            **vision_collator_kwargs,
        ),
        callbacks=[logger_callback],
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        args=SFTConfig(**sft_kwargs),
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )

    # @title Show current memory stats
    gpu_stats = torch.cuda.get_device_properties(0)
    start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
    print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
    print(f"{start_gpu_memory} GB of memory reserved.")

    trainer_stats = trainer.train()
    
    best_model_dir = Path(config.get("output_dir", "outputs")) / "best_model"
    trainer.save_model(best_model_dir)
    tokenizer.save_pretrained(best_model_dir)

    print(f"Best model saved to: {best_model_dir}")
    
    # Show final memory and time stats
    used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
    used_percentage = round(used_memory / max_memory * 100, 3)
    lora_percentage = round(used_memory_for_lora / max_memory * 100, 3)
    print(f"{trainer_stats.metrics['train_runtime']} seconds used for training.")
    print(
        f"{round(trainer_stats.metrics['train_runtime']/60, 2)} minutes used for training."
    )
    print(f"Peak reserved memory = {used_memory} GB.")
    print(f"Peak reserved memory for training = {used_memory_for_lora} GB.")
    print(f"Peak reserved memory % of max memory = {used_percentage} %.")
    print(f"Peak reserved memory for training % of max memory = {lora_percentage} %.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m src.training.train <config.yaml>")
        sys.exit(1)
    main(load_config(sys.argv[1]))
