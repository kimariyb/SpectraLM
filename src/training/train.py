"""LoRA/QLoRA fine-tuning entrypoint for SpectraLM VLM experiments."""

from __future__ import annotations

import inspect
import os
import sys
from typing import Any

from src.data.dataset import NmrReasoningDataset
from unsloth import FastVisionModel
from unsloth.trainer import UnslothVisionDataCollator
from trl import SFTConfig, SFTTrainer

    "model_path": "/mnt/data/kimariyb/models/Qwen3-VL-8B-Instruct",
    "train_dataset": "dataset/subsets/spectralm_butina_1000_300/train.pkl",
    "eval_dataset": "dataset/subsets/spectralm_butina_1000_300/test.pkl",
    "output_dir": "outputs/spectralm-butina-qwen3-vl-8b",
    "max_seq_length": 2048,
    "per_device_train_batch_size": 1,
    "gradient_accumulation_steps": 8,
    "learning_rate": 2e-5,
    "num_train_epochs": 3,
    "logging_steps": 5,
    "save_steps": 50,
    "eval_steps": 50,
    "lora_r": 16,
    "lora_alpha": 16,
    "seed": 3407,
}

model, tokenizer = FastVisionModel.from_pretrained(
    "unsloth/Llama-3.2-11B-Vision-Instruct",
    load_in_4bit = False, # Use 4bit to reduce memory use. False for 16bit LoRA.
    use_gradient_checkpointing = "unsloth", # True or "unsloth" for long context
)

model = FastVisionModel.get_peft_model(
    model,
    finetune_vision_layers     = True, # False if not finetuning vision layers
    finetune_language_layers   = True, # False if not finetuning language layers
    finetune_attention_modules = True, # False if not finetuning attention layers
    finetune_mlp_modules       = True, # False if not finetuning MLP layers

    r = 16,           # The larger, the higher the accuracy, but might overfit
    lora_alpha = 16,  # Recommended alpha == r at least
    lora_dropout = 0,
    bias = "none",
    random_state = 3407,
    use_rslora = False,  # We support rank stabilized LoRA
    loftq_config = None, # And LoftQ
    # target_modules = "all-linear", # Optional now! Can specify a list if needed
)

dataset = NmrReasoningDataset(
    "unsloth/Radiology_mini"
)