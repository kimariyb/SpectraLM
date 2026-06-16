"""LoRA/QLoRA fine-tuning entrypoint for SpectraLM VLM experiments."""

from __future__ import annotations

import inspect
import os
import sys
from typing import Any

from spectralm.training.dataset import NmrReasoningDataset
from spectralm.config import load_config
from unsloth import FastVisionModel
from unsloth.trainer import UnslothVisionDataCollator
from trl import SFTConfig, SFTTrainer

_TRAINING_DEFAULTS = {
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


def _resolve_training_config(config: dict[str, Any]) -> dict[str, Any]:
    """Merge user config with training defaults.

    Parameters
    ----------
    config
        User-provided YAML configuration.

    Returns
    -------
    dict[str, Any]
        Resolved configuration with defaults applied.
    """
    merged = dict(_TRAINING_DEFAULTS)
    merged.update(config)
    return merged


def configure_huggingface_env() -> None:
    """Set conservative Hugging Face download environment defaults."""
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "600")
    os.environ.setdefault("HF_HUB_DOWNLOAD_RETRY", "20")


def inner_tokenizer(processing_class: Any) -> Any:
    """Return the tokenizer nested inside a processor when present.

    Parameters
    ----------
    processing_class
        Hugging Face tokenizer or processor object.

    Returns
    -------
    Any
        Tokenizer-like object used for text vocabulary lookup.
    """
    return getattr(processing_class, "tokenizer", processing_class)


def token_exists(tokenizer: Any, token: str | None) -> bool:
    """Check whether a token exists in a tokenizer vocabulary.

    Parameters
    ----------
    tokenizer
        Tokenizer-like object.
    token
        Token string to check.

    Returns
    -------
    bool
        ``True`` when the token can be resolved to a vocabulary id.
    """
    if not token:
        return False
    if hasattr(tokenizer, "get_vocab"):
        try:
            return token in tokenizer.get_vocab()
        except Exception:
            pass
    if hasattr(tokenizer, "convert_tokens_to_ids"):
        try:
            token_id = tokenizer.convert_tokens_to_ids(token)
        except Exception:
            return False
        unknown_id = getattr(tokenizer, "unk_token_id", None)
        return token_id is not None and token_id != unknown_id
    return True


def resolve_eos_token(processing_class: Any) -> str | None:
    """Resolve an EOS token that is valid for a processor or tokenizer.

    Parameters
    ----------
    processing_class
        Hugging Face tokenizer or multimodal processor.

    Returns
    -------
    str | None
        EOS token that exists in the vocabulary, or ``None`` if unavailable.
    """
    tokenizer = inner_tokenizer(processing_class)
    candidates = [
        getattr(tokenizer, "eos_token", None),
        getattr(processing_class, "eos_token", None),
        "<|im_end|>",
        "<|endoftext|>",
    ]
    for token in candidates:
        if token_exists(tokenizer, token):
            return token
    return None


def ensure_padding_token(processing_class: Any) -> None:
    """Ensure tokenizer padding falls back to EOS when padding is undefined.

    Parameters
    ----------
    processing_class
        Hugging Face tokenizer or multimodal processor.
    """
    tokenizer = inner_tokenizer(processing_class)
    if getattr(tokenizer, "pad_token", None) is None and getattr(tokenizer, "eos_token", None) is not None:
        tokenizer.pad_token = tokenizer.eos_token


def set_token_attr(target: Any, name: str, value: str) -> None:
    """Set a tokenizer or processor token attribute when possible.

    Parameters
    ----------
    target
        Tokenizer-like or processor-like object.
    name
        Attribute name to set.
    value
        Token value.
    """
    try:
        setattr(target, name, value)
    except Exception:
        return


def normalize_training_special_tokens(training_args: Any, processing_class: Any) -> None:
    """Normalize TRL special-token fields after ``SFTConfig`` construction.

    Some TRL and Unsloth versions initialize ``SFTConfig.eos_token`` with the
    placeholder ``"<EOS_TOKEN>"`` even when a valid Qwen tokenizer is supplied.
    ``SFTTrainer`` validates that field against the processor vocabulary, so the
    final config object must be patched after construction.

    Parameters
    ----------
    training_args
        Constructed TRL SFT configuration object.
    processing_class
        Hugging Face tokenizer or multimodal processor.
    """
    tokenizer = inner_tokenizer(processing_class)
    eos_token = resolve_eos_token(processing_class)
    if eos_token is None:
        return
    set_token_attr(tokenizer, "eos_token", eos_token)
    set_token_attr(processing_class, "eos_token", eos_token)
    set_token_attr(training_args, "eos_token", eos_token)
    if getattr(tokenizer, "pad_token", None) is None:
        set_token_attr(tokenizer, "pad_token", eos_token)
    if getattr(training_args, "pad_token", None) in (None, "<PAD_TOKEN>"):
        set_token_attr(training_args, "pad_token", getattr(tokenizer, "pad_token", eos_token))


def build_sft_config_kwargs(
    config: dict[str, Any],
    config_cls,
    processing_class: Any | None = None,
) -> dict[str, Any]:
    """Build TRL ``SFTConfig`` kwargs compatible with installed versions.

    Parameters
    ----------
    config
        Resolved training configuration dictionary.
    config_cls
        TRL SFT configuration class.
    processing_class
        Optional Hugging Face tokenizer or multimodal processor used to resolve
        version-specific token settings.

    Returns
    -------
    dict[str, Any]
        Keyword arguments supported by the installed class.
    """
    supported = set(inspect.signature(config_cls.__init__).parameters)
    candidates = {
        "output_dir": config["output_dir"],
        "per_device_train_batch_size": config["per_device_train_batch_size"],
        "gradient_accumulation_steps": config["gradient_accumulation_steps"],
        "learning_rate": config["learning_rate"],
        "num_train_epochs": config["num_train_epochs"],
        "logging_steps": config["logging_steps"],
        "save_steps": config["save_steps"],
        "eval_steps": config["eval_steps"],
        "bf16": True,
        "remove_unused_columns": False,
        "dataset_text_field": "",
        "seed": config["seed"],
    }
    if "max_length" in supported:
        candidates["max_length"] = config["max_seq_length"]
    elif "max_seq_length" in supported:
        candidates["max_seq_length"] = config["max_seq_length"]
    if "eval_strategy" in supported:
        candidates["eval_strategy"] = "steps"
    elif "evaluation_strategy" in supported:
        candidates["evaluation_strategy"] = "steps"
    if processing_class is not None and "eos_token" in supported:
        eos_token = resolve_eos_token(processing_class)
        if eos_token is not None:
            candidates["eos_token"] = eos_token
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


def run(config: dict[str, Any]) -> None:
    """Run the VLM fine-tuning workflow from a configuration dictionary.

    Parameters
    ----------
    config
        Configuration dictionary with training parameters.
    """
    cfg = _resolve_training_config(config)
    configure_huggingface_env()

    print(f"Loading base model: {cfg['model_path']}")
    model, tokenizer = FastVisionModel.from_pretrained(
        cfg["model_path"],
        use_gradient_checkpointing="unsloth",
    )
    ensure_padding_token(tokenizer)
    print("Configuring LoRA adapter...")
    model = FastVisionModel.get_peft_model(
        model,
        finetune_vision_layers=True,
        finetune_language_layers=True,
        finetune_attention_modules=True,
        finetune_mlp_modules=True,
        r=cfg["lora_r"],
        lora_alpha=cfg["lora_alpha"],
        lora_dropout=0,
        bias="none",
        random_state=cfg["seed"],
    )
    train_dataset = NmrReasoningDataset(cfg["train_dataset"], task_probs={"structure_reasoning": 1.0})
    eval_dataset = NmrReasoningDataset(cfg["eval_dataset"], task_probs={"structure_reasoning": 1.0})
    training_args = SFTConfig(**build_sft_config_kwargs(cfg, SFTConfig, tokenizer))
    normalize_training_special_tokens(training_args, tokenizer)
    if getattr(training_args, "eos_token", None):
        print(f"Using EOS token: {training_args.eos_token}")
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
    trainer.save_model(cfg["output_dir"])
    tokenizer.save_pretrained(cfg["output_dir"])


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m spectralm.training.train <config.yaml>")
        sys.exit(1)
    run(load_config(sys.argv[1]))
