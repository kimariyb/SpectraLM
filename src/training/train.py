"""LoRA/QLoRA fine-tuning entrypoint for SpectraLM VLM experiments.

Usage::

    python -m src.training.train configs/train.yaml
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
from src.data.dataset import load_lazy_nmr_dataset, load_nmr_dataset
from trl import SFTConfig, SFTTrainer


def _limit_dataset(dataset, max_samples: int | None):
    """Return a deterministic prefix subset when max_samples is configured."""
    if max_samples is None:
        return dataset
    n = min(int(max_samples), len(dataset))
    if hasattr(dataset, "select"):
        return dataset.select(range(n))
    return Subset(dataset, range(n))


def main(config: dict[str, Any]) -> None:
    """Run a LoRA fine-tuning job from a configuration dictionary.

    Parameters
    ----------
    config
        Configuration with keys for model, dataset, LoRA, and training
        hyperparameters.  See ``configs/train.yaml`` for the full schema.
    """
    seed: int = config.get("seed", 3407)

    # 1. Build datasets first so dry-run works without loading a VLM.
    dataset_dir: str = config["dataset_dir"]
    train_size: float = float(config.get("train_size", 0.8))
    eval_split: float = float(config.get("eval_split", 0.1))
    train_split_name: str = config.get("train_split_name", "train")
    eval_split_name: str = config.get("eval_split_name", "validation")

    dataset_backend = config.get("dataset_backend", "hf")
    if dataset_backend == "lazy_jsonl":
        train_ds = load_lazy_nmr_dataset(
            dataset_dir,
            split=train_split_name,
            target_format=config.get("target_format", "smiles"),
            include_formula=config.get("include_formula", True),
            seed=seed,
            h_snr=float(config.get("h_snr", 500.0)),
            c_snr=float(config.get("c_snr", 300.0)),
            render_seed=config.get("render_seed", seed),
            image_size=config.get("image_size"),
            image_backend=config.get("image_backend", "lazy_render"),
            rendered_image_dir=config.get("rendered_image_dir"),
            missing_image_policy=config.get("missing_image_policy", "error"),
        )
        eval_ds = load_lazy_nmr_dataset(
            dataset_dir,
            split=eval_split_name,
            target_format=config.get("target_format", "smiles"),
            include_formula=config.get("include_formula", True),
            seed=seed,
            h_snr=float(config.get("h_snr", 500.0)),
            c_snr=float(config.get("c_snr", 300.0)),
            render_seed=config.get("render_seed", seed),
            image_size=config.get("image_size"),
            image_backend=config.get("image_backend", "lazy_render"),
            rendered_image_dir=config.get("rendered_image_dir"),
            missing_image_policy=config.get("missing_image_policy", "error"),
        )
    else:
        full_ds = load_nmr_dataset(
            dataset_dir,
            split=train_split_name,
            train_size=train_size,
            render_cache_dir=config.get("train_cache_dir"),
            render_cache_version=config.get("cache_version", "1"),
            target_format=config.get("target_format", "smiles"),
            include_formula=config.get("include_formula", True),
            seed=seed,
        )

        # Split a fraction for periodic evaluation
        split_ds = full_ds.train_test_split(
            test_size=eval_split, seed=seed
        )
        train_ds = split_ds["train"].shuffle(seed=seed)
        eval_ds = split_ds["test"]

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

    sft_kwargs: dict[str, Any] = {
        "per_device_train_batch_size": config.get("per_device_train_batch_size", 4),
        "gradient_accumulation_steps": config.get("gradient_accumulation_steps", 4),
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
        "lr_scheduler_type": config.get("lr_scheduler_type", "cosine_with_restarts"),
        "seed": seed,
        "output_dir": config.get("output_dir", "outputs"),
        "report_to": config.get("report_to", "none"),
        
        # Vision fine-tuning requirements
        "remove_unused_columns": False,
        "dataset_text_field": "",
        "dataset_kwargs": {"skip_prepare_dataset": True},
        "max_length": config.get("max_seq_length", 8192),
        "bf16": bool(config.get("bf16", True)),
        "fp16": bool(config.get("fp16", False)),
    }

    if config.get("max_steps") is not None:
        sft_kwargs["max_steps"] = int(config["max_steps"])

    training_logger = TrainingLogger(
        output_dir="outputs/logs",
        config={
            "model_name": Path(config.get("model_path")).name,
            "learning_rate": sft_kwargs['learning_rate'],
            "num_train_epochs": sft_kwargs['num_train_epochs'],
            "per_device_train_batch_size": sft_kwargs["per_device_train_batch_size"],
            "gradient_accumulation_steps": sft_kwargs['gradient_accumulation_steps'],
            "eval_steps": sft_kwargs['eval_steps'],
            "target_format": config.get("target_format", "smiles"),
            "include_formula": config.get("include_formula", True),
            "dataset_backend": dataset_backend,
            "image_backend": config.get("image_backend", "lazy_render"),
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
        data_collator=UnslothVisionDataCollator(model, tokenizer),
        callbacks=[logger_callback],
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        args=SFTConfig(**sft_kwargs),
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
