"""LoRA/QLoRA fine-tuning entrypoint for SpectraLM VLM experiments.

Usage::

    python -m src.training.train configs/train.yaml
"""

from __future__ import annotations

import sys
from typing import Any

from src.config import load_config
from src.data.dataset import NmrReasoningDataset
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
    # ------------------------------------------------------------------
    # 1. Load model
    # ------------------------------------------------------------------
    model, tokenizer = FastVisionModel.from_pretrained(
        config["model_path"],
        max_seq_length=config.get("max_seq_length", 8192),
        load_in_4bit=config.get("load_in_4bit", True),
        use_gradient_checkpointing="unsloth",
    )

    # ------------------------------------------------------------------
    # 2. Apply LoRA adapters
    # ------------------------------------------------------------------
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
        random_state=config.get("seed", 3407),
        use_rslora=False,
        loftq_config=None,
    )

    # ------------------------------------------------------------------
    # 3. Build datasets
    # ------------------------------------------------------------------
    dataset_dir: str = config["dataset_dir"]
    train_size: int = config.get("train_size", 1000)
    cache_version: str | None = config.get("cache_version")
    seed: int = config.get("seed", 42)

    train_ds = NmrReasoningDataset(
        dataset_dir,
        split="train",
        train_size=train_size,
        cache_dir=config.get("train_cache_dir"),
        cache_version=cache_version,
        seed=seed,
    )

    eval_ds = NmrReasoningDataset(
        dataset_dir,
        split="test",
        train_size=train_size,
        cache_dir=config.get("eval_cache_dir"),
        cache_version=cache_version,
        seed=seed,
    )

    print(f"Train samples: {len(train_ds)}")
    print(f"Eval samples:  {len(eval_ds)}")

    # ------------------------------------------------------------------
    # 4. Train
    # ------------------------------------------------------------------
    FastVisionModel.for_training(model)

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        data_collator=UnslothVisionDataCollator(model, tokenizer),
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        args=SFTConfig(
            per_device_train_batch_size=config.get("per_device_train_batch_size", 4),
            gradient_accumulation_steps=config.get("gradient_accumulation_steps", 4),
            warmup_steps=config.get("warmup_steps", 5),
            max_steps=config.get("max_steps", 30),
            learning_rate=float(config.get("learning_rate", 2e-4)),
            logging_steps=config.get("logging_steps", 1),
            eval_steps=config.get("eval_steps", 50),
            save_steps=config.get("save_steps", 50),
            optim="adamw_8bit",
            weight_decay=float(config.get("weight_decay", 0.001)),
            lr_scheduler_type="linear",
            seed=seed,
            output_dir=config.get("output_dir", "outputs"),
            report_to="none",
            # Vision fine-tuning requirements
            remove_unused_columns=False,
            dataset_text_field="",
            dataset_kwargs={"skip_prepare_dataset": True},
            max_length=config.get("max_length", 2048),
        ),
    )

    trainer.train()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m src.training.train <config.yaml>")
        sys.exit(1)
    main(load_config(sys.argv[1]))
