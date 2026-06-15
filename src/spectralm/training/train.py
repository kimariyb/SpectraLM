"""LoRA/QLoRA fine-tuning entrypoint for SpectraLM VLM experiments."""

from __future__ import annotations

import argparse
import inspect
import os
from typing import Any

from spectralm.config import add_config_argument, load_config


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the training CLI parser.

    Returns
    -------
    argparse.ArgumentParser
        Configured parser.
    """
    parser = argparse.ArgumentParser(description="Fine-tune SpectraLM with Qwen-VL and LoRA/QLoRA.")
    add_config_argument(parser)
    parser.add_argument("--model-path", default=None, help="Local or Hugging Face base model path.")
    parser.add_argument("--train-dataset", default=None, help="Training pickle dataset.")
    parser.add_argument("--eval-dataset", default=None, help="Evaluation pickle dataset.")
    parser.add_argument("--output-dir", default=None, help="Checkpoint and adapter output directory.")
    parser.add_argument("--max-seq-length", type=int, default=None)
    parser.add_argument("--per-device-train-batch-size", type=int, default=None)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--num-train-epochs", type=float, default=None)
    parser.add_argument("--logging-steps", type=int, default=None)
    parser.add_argument("--save-steps", type=int, default=None)
    parser.add_argument("--eval-steps", type=int, default=None)
    parser.add_argument("--lora-r", type=int, default=None)
    parser.add_argument("--lora-alpha", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    return parser


def config_value(args: argparse.Namespace, config: dict[str, Any], name: str, default: Any) -> Any:
    """Resolve a CLI option with config fallback.

    Parameters
    ----------
    args
        Parsed CLI arguments.
    config
        Loaded YAML configuration.
    name
        Option name.
    default
        Fallback value.

    Returns
    -------
    Any
        Resolved option value.
    """
    value = getattr(args, name)
    return value if value is not None else config.get(name, default)


def configure_huggingface_env() -> None:
    """Set conservative Hugging Face download environment defaults."""
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "600")
    os.environ.setdefault("HF_HUB_DOWNLOAD_RETRY", "20")


def build_sft_config_kwargs(args: argparse.Namespace, config_cls) -> dict[str, Any]:
    """Build TRL ``SFTConfig`` kwargs compatible with installed versions.

    Parameters
    ----------
    args
        Resolved training arguments.
    config_cls
        TRL SFT configuration class.

    Returns
    -------
    dict[str, Any]
        Keyword arguments supported by the installed class.
    """
    supported = set(inspect.signature(config_cls.__init__).parameters)
    candidates = {
        "output_dir": args.output_dir,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "learning_rate": args.learning_rate,
        "num_train_epochs": args.num_train_epochs,
        "logging_steps": args.logging_steps,
        "save_steps": args.save_steps,
        "eval_steps": args.eval_steps,
        "bf16": True,
        "remove_unused_columns": False,
        "dataset_text_field": "",
        "seed": args.seed,
    }
    if "max_length" in supported:
        candidates["max_length"] = args.max_seq_length
    elif "max_seq_length" in supported:
        candidates["max_seq_length"] = args.max_seq_length
    if "eval_strategy" in supported:
        candidates["eval_strategy"] = "steps"
    elif "evaluation_strategy" in supported:
        candidates["evaluation_strategy"] = "steps"
    return {key: value for key, value in candidates.items() if key in supported}


def build_sft_trainer_kwargs(
    trainer_cls,
    *,
    model,
    tokenizer,
    data_collator,
    train_dataset,
    eval_dataset,
    training_args,
) -> dict[str, Any]:
    """Build TRL ``SFTTrainer`` kwargs compatible with installed versions.

    Parameters
    ----------
    trainer_cls
        TRL trainer class.
    model
        Loaded base model with PEFT adapter.
    tokenizer
        Processor or tokenizer object.
    data_collator
        Vision data collator.
    train_dataset
        Training dataset.
    eval_dataset
        Evaluation dataset.
    training_args
        SFT configuration object.

    Returns
    -------
    dict[str, Any]
        Keyword arguments supported by the installed trainer class.
    """
    supported = set(inspect.signature(trainer_cls.__init__).parameters)
    kwargs = {
        "model": model,
        "args": training_args,
        "data_collator": data_collator,
        "train_dataset": train_dataset,
        "eval_dataset": eval_dataset,
    }
    if "processing_class" in supported:
        kwargs["processing_class"] = tokenizer
    elif "tokenizer" in supported:
        kwargs["tokenizer"] = tokenizer
    return {key: value for key, value in kwargs.items() if key in supported}


def resolve_args(args: argparse.Namespace, config: dict[str, Any]) -> argparse.Namespace:
    """Resolve training arguments from CLI and config values.

    Parameters
    ----------
    args
        Parsed CLI arguments.
    config
        Loaded YAML configuration.

    Returns
    -------
    argparse.Namespace
        Namespace with resolved values.
    """
    defaults = {
        "model_path": "/mnt/data/kimariyb/models/Qwen3-VL-8B-Instruct",
        "train_dataset": "dataset/subsets/spectralm_500_100/train.pkl",
        "eval_dataset": "dataset/subsets/spectralm_500_100/test.pkl",
        "output_dir": "outputs/spectralm-pilot-qwen3-vl-8b",
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
    for name, default in defaults.items():
        setattr(args, name, config_value(args, config, name, default))
    return args


def main() -> None:
    """Run the VLM fine-tuning workflow."""
    args = build_arg_parser().parse_args()
    config = load_config(args.config)
    args = resolve_args(args, config)
    configure_huggingface_env()
    from trl import SFTConfig, SFTTrainer
    from unsloth import FastVisionModel
    from unsloth.trainer import UnslothVisionDataCollator

    from spectralm.training.dataset import NmrReasoningDataset

    print(f"Loading base model: {args.model_path}")
    model, tokenizer = FastVisionModel.from_pretrained(
        args.model_path,
        use_gradient_checkpointing="unsloth",
    )
    print("Configuring LoRA adapter...")
    model = FastVisionModel.get_peft_model(
        model,
        finetune_vision_layers=True,
        finetune_language_layers=True,
        finetune_attention_modules=True,
        finetune_mlp_modules=True,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0,
        bias="none",
        random_state=args.seed,
    )
    train_dataset = NmrReasoningDataset(args.train_dataset, task_probs={"structure_reasoning": 1.0})
    eval_dataset = NmrReasoningDataset(args.eval_dataset, task_probs={"structure_reasoning": 1.0})
    training_args = SFTConfig(**build_sft_config_kwargs(args, SFTConfig))
    trainer = SFTTrainer(
        **build_sft_trainer_kwargs(
            SFTTrainer,
            model=model,
            tokenizer=tokenizer,
            data_collator=UnslothVisionDataCollator(model, tokenizer),
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            training_args=training_args,
        )
    )
    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()
