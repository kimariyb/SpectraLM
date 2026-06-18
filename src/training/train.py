"""LoRA/QLoRA fine-tuning entrypoint for SpectraLM VLM experiments.

Usage::

    python -m src.training.train configs/train.yaml
"""

from __future__ import annotations

import sys
import torch
from typing import Any

from src.config import TrainingLoggerCallback, load_config
from src.logger import TrainingLogger
from src.data.dataset import load_nmr_dataset
from unsloth import FastVisionModel
from unsloth.trainer import UnslothVisionDataCollator
from trl import SFTConfig, SFTTrainer



def main(config: dict[str, Any]) -> None:
    """Run a LoRA fine-tuning job from a configuration dictionary.

    Parameters
    ----------
    config
        Configuration with keys for model, dataset, LoRA, and training
        hyperparameters.  See ``configs/train.yaml`` for the full schema.
    """
    seed: int = config.get("seed", 3407)

    # 1. Load model
    model, tokenizer = FastVisionModel.from_pretrained(
        config["model_path"],
        max_seq_length=config.get("max_seq_length", 8192),
        load_in_4bit=config.get("load_in_4bit", True),
        use_gradient_checkpointing="unsloth",
    )

    # 2. Apply LoRA adapters
    model = FastVisionModel.get_peft_model(
        model,
        finetune_vision_layers=True,
        finetune_language_layers=True,
        finetune_attention_modules=True,
        finetune_mlp_modules=True,
        r=config.get("lora_r", 16),
        lora_alpha=config.get("lora_alpha", 16),
        lora_dropout=0,
        bias="none",
        random_state=seed,
        use_rslora=False,
        loftq_config=None,
    )

    # 3. Build datasets
    dataset_dir: str = config["dataset_dir"]
    train_size: float = float(config.get("train_size", 0.8))
    eval_split: float = float(config.get("eval_split", 0.1))

    full_ds = load_nmr_dataset(
        dataset_dir,
        split="train",
        train_size=train_size,
        render_cache_dir=config.get("train_cache_dir"),
        render_cache_version=config.get("cache_version", "1"),
        seed=seed,
    )

    # Split a fraction for periodic evaluation
    split_ds = full_ds.train_test_split(
        test_size=eval_split, seed=seed
    )
    train_ds = split_ds["train"].shuffle(seed=seed)
    eval_ds = split_ds["test"]

    print(f"Train samples: {len(train_ds)}  |  Eval samples: {len(eval_ds)}")

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
        "save_total_limit": 5,
        
        "metric_for_best_model": "eval_loss", 
        "greater_is_better": False,          
        "load_best_model_at_end": True,       

        "optim": "adamw_8bit",
        "weight_decay": float(config.get("weight_decay", 0.001)),
        "lr_scheduler_type": "cosine_with_restarts",
        "seed": seed,
        "output_dir": config.get("output_dir", "outputs"),
        "report_to": "none",
        
        # Vision fine-tuning requirements
        "remove_unused_columns": False,
        "dataset_text_field": "",
        "dataset_kwargs": {"skip_prepare_dataset": True},
        "max_length": config.get("max_seq_length", 8192),
        "bf16": True,
    }

    training_logger = TrainingLogger(
        output_dir="outputs/logs",
        config={
            "model_name": "Qwen2.5-VL",
            "learning_rate": 2e-4,
            "num_train_epochs": 3,
            "per_device_train_batch_size": 1,
            "gradient_accumulation_steps": 8,
            "eval_steps": 100,
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
