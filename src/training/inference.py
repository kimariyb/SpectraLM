"""Prediction entrypoint for SpectraLM fine-tuned models.

Loads a curated JSONL test split, builds
structure-elucidation prompts using the pre-rendered spectrum images, and
generates SMILES predictions.

Formula-conditioned and formula-free runs use identical images and peak tables;
only the molecular-formula line changes.

Usage::

    python -m src.training.inference configs/experiments/infer_stage2_10k.yaml
"""

from __future__ import annotations
import unsloth
import itertools
import json
import sys
import traceback
from pathlib import Path
from typing import Any
import torch
from tqdm import tqdm
from unsloth import FastVisionModel
from peft import PeftModel
from src.config import load_config
from src.data.dataset import (
    load_sample_images,
    load_raw_nmr_samples,
)
from src.data.modalities import (
    build_user_content,
    input_mode_uses_images,
    validate_input_configuration,
)
from src.evaluation.metrics import (
    evaluate_structure_prediction,
    inspect_generation_tokens,
    summarize_generation_behavior,
    summarize_structure_predictions,
)
from src.evaluation.prompts import (
    build_structure_prompt,
    select_structure_prompt,
)
from src.nmr_rules.engine import load_rule_library
from src.training.response_masking import apply_non_thinking_chat_template


def load_model_for_inference(
    base_model_path: str,
    adapter_path: str | Path | None = None,
    max_seq_length: int = 8192,
    load_in_4bit: bool = True,
):
    """Load base VLM and optionally attach a LoRA adapter.

    Parameters
    ----------
    base_model_path
        Path or HuggingFace repo id of the base VLM.
    adapter_path
        Optional path to a LoRA adapter.  When ``None``, *base_model_path*
        is used directly as the fine-tuned model.
    max_seq_length
        Maximum sequence length for the tokenizer.
    load_in_4bit
        Whether to load in 4-bit quantisation.

    Returns
    -------
    tuple
        ``(model, tokenizer)`` ready for inference.
    """
    model, tokenizer = FastVisionModel.from_pretrained(
        base_model_path,
        max_seq_length=max_seq_length,
        load_in_4bit=load_in_4bit,
        use_gradient_checkpointing=False,
        attn_implementation="sdpa"
    )

    if adapter_path is not None:
        model = PeftModel.from_pretrained(
            model,
            str(adapter_path),
            is_trainable=False,
        )

    FastVisionModel.for_inference(model)
    model.eval()

    return model, tokenizer


def build_inference_rows(
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    """Load a bounded curated JSONL split for inference."""
    return load_raw_nmr_samples(
        config["dataset_dir"],
        split=config.get("split", "test"),
    )

@torch.inference_mode()
def generate_one(
    model,
    tokenizer,
    images: list[Any],
    prompt: str,
    max_new_tokens: int = 256,
    temperature: float = 0.0,
    top_p: float = 1.0,
    input_mode: str = "full",
) -> tuple[str, dict[str, int | bool]]:
    """Generate one prediction from paired NMR spectrum images.

    Parameters
    ----------
    model
        The loaded VLM with LoRA adapter.
    tokenizer
        Corresponding tokenizer.
    images
        Two PIL/ndarray images.
    prompt
        Text prompt.
    max_new_tokens
        Maximum tokens to generate.
    temperature
        Sampling temperature (0 = greedy).
    top_p
        Nucleus sampling parameter.

    Returns
    -------
    tuple[str, dict[str, int | bool]]
        Decoded prediction and generation-behavior diagnostics.
    """
    uses_images = input_mode_uses_images(input_mode)
    content = build_user_content(
        prompt,
        input_mode=input_mode,
        images=(None, None) if uses_images else (),
    )

    messages = [{"role": "user", "content": content}]

    input_text = apply_non_thinking_chat_template(tokenizer, messages)

    processor_kwargs: dict[str, Any] = {
        "text": input_text,
        "add_special_tokens": False,
        "return_tensors": "pt",
    }
    if uses_images:
        if len(images) != 2:
            raise ValueError(f"Expected 2 images, but got {len(images)}.")
        processor_kwargs["images"] = images
    elif images:
        raise ValueError(f"{input_mode} input_mode must not receive images")
    inputs = tokenizer(
        **processor_kwargs,
    ).to("cuda")

    do_sample = temperature > 0
    raw_eos_token_ids = getattr(model.generation_config, "eos_token_id", None)
    if raw_eos_token_ids is None:
        raw_eos_token_ids = tokenizer.eos_token_id
    if raw_eos_token_ids is None:
        raise ValueError("Tokenizer and model generation config have no EOS token.")
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = (
            raw_eos_token_ids
            if isinstance(raw_eos_token_ids, int)
            else raw_eos_token_ids[0]
        )

    output_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=temperature if do_sample else None,
        top_p=top_p if do_sample else None,
        use_cache=True,
        eos_token_id=raw_eos_token_ids,
        pad_token_id=pad_token_id,
    )

    input_len = inputs["input_ids"].shape[1]
    generated_ids = output_ids[:, input_len:]
    generated_token_ids = generated_ids[0].detach().cpu().tolist()

    if isinstance(raw_eos_token_ids, int):
        eos_token_ids = {raw_eos_token_ids}
    else:
        eos_token_ids = {
            int(token_id) for token_id in (raw_eos_token_ids or [])
        }
    generation_trace = inspect_generation_tokens(
        generated_token_ids,
        eos_token_ids=eos_token_ids,
        max_new_tokens=max_new_tokens,
    )

    pred = tokenizer.batch_decode(
        generated_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]

    return pred.strip(), generation_trace


@torch.inference_mode()
def generate_many(
    model,
    tokenizer,
    images: list[Any],
    prompt: str,
    *,
    num_return_sequences: int,
    max_new_tokens: int = 128,
    temperature: float = 0.7,
    top_p: float = 0.9,
    input_mode: str = "full",
) -> tuple[list[str], list[dict[str, int | bool]]]:
    """Sample multiple structure candidates from one identical model input."""
    if num_return_sequences < 1:
        raise ValueError("num_return_sequences must be positive")
    if num_return_sequences > 1 and temperature <= 0:
        raise ValueError("Multiple candidates require temperature > 0")
    uses_images = input_mode_uses_images(input_mode)
    content = build_user_content(
        prompt,
        input_mode=input_mode,
        images=(None, None) if uses_images else (),
    )
    messages = [{"role": "user", "content": content}]
    input_text = apply_non_thinking_chat_template(tokenizer, messages)
    processor_kwargs: dict[str, Any] = {
        "text": input_text,
        "add_special_tokens": False,
        "return_tensors": "pt",
    }
    if uses_images:
        if len(images) != 2:
            raise ValueError(f"Expected 2 images, but got {len(images)}.")
        processor_kwargs["images"] = images
    elif images:
        raise ValueError(f"{input_mode} input_mode must not receive images")
    inputs = tokenizer(**processor_kwargs).to("cuda")

    raw_eos_token_ids = getattr(model.generation_config, "eos_token_id", None)
    if raw_eos_token_ids is None:
        raw_eos_token_ids = tokenizer.eos_token_id
    if raw_eos_token_ids is None:
        raise ValueError("Tokenizer and model generation config have no EOS token.")
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = (
            raw_eos_token_ids
            if isinstance(raw_eos_token_ids, int)
            else raw_eos_token_ids[0]
        )
    output_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=temperature > 0,
        temperature=temperature,
        top_p=top_p,
        num_return_sequences=num_return_sequences,
        use_cache=True,
        eos_token_id=raw_eos_token_ids,
        pad_token_id=pad_token_id,
    )
    input_len = inputs["input_ids"].shape[1]
    generated_ids = output_ids[:, input_len:]
    predictions = [
        prediction.strip()
        for prediction in tokenizer.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
    ]
    eos_token_ids = (
        {raw_eos_token_ids}
        if isinstance(raw_eos_token_ids, int)
        else {int(token_id) for token_id in raw_eos_token_ids}
    )
    traces = [
        inspect_generation_tokens(
            token_ids.detach().cpu().tolist(),
            eos_token_ids=eos_token_ids,
            max_new_tokens=max_new_tokens,
        )
        for token_ids in generated_ids
    ]
    return predictions, traces


def main(config: dict[str, Any]) -> None:
    """Run inference over the test split.

    Parameters
    ----------
    config
        Configuration dictionary for one experiment inference run.
    """
    max_samples: int | None = config.get("max_samples", None)
    seed: int = int(config.get("seed", 3407))
    include_formula = bool(config.get("include_formula", True))
    rule_context_enabled = bool(config.get("rule_context_enabled", False))
    input_mode = validate_input_configuration(
        config.get("input_mode", "full"),
        include_formula=include_formula,
        include_rule_context=rule_context_enabled,
    )
    uses_images = input_mode_uses_images(input_mode)
    rule_validation_enabled = bool(config.get("rule_validation_enabled", False))
    rule_library = str(load_rule_library()["library_name"])

    # ---- 1. Load model ---------------------------------------------------
    model, tokenizer = load_model_for_inference(
        base_model_path=config["model_path"],
        adapter_path=config.get("adapter_path"),
        max_seq_length=config.get("max_seq_length", 8192),
        load_in_4bit=config.get("load_in_4bit", True),
    )

    # ---- 2. Load dataset (raw columns, no messages transform) ------------
    test_ds = build_inference_rows(config)

    # ---- 3. Select prompt template ---------------------------------------
    prompt_template_index = int(config.get("prompt_template_index", 0))
    prompt_template = select_structure_prompt(
        prompt_template_index,
        input_mode=input_mode,
    )

    print(f"Input mode: {input_mode}")
    print(f"Formula included: {include_formula}")
    print(f"Rule context enabled: {rule_context_enabled}")
    print(f"Rule validation enabled: {rule_validation_enabled}")
    print(f"Prompt template index: {prompt_template_index}")
    print(f"Prompt template: {prompt_template[:80]}...")

    # ---- 4. Inference loop -----------------------------------------------
    output_path = Path(config["output"])
    output_path.parent.mkdir(parents=True, exist_ok=True)

    n_errors = 0
    metric_rows: list[dict[str, Any]] = []
    predictions: list[str] = []
    generation_traces: list[dict[str, int | bool]] = []

    iterator = enumerate(test_ds)
    if max_samples is not None:
        iterator = itertools.islice(iterator, max_samples)

    with output_path.open("w", encoding="utf-8") as f:
        for idx, row in tqdm(iterator, desc="Infer", total=max_samples):
            # --- extract data from raw row ---
            sample = row
            images: list[Any] = []
            if uses_images:
                images = list(
                    load_sample_images(
                        sample,
                        image_backend=config.get(
                            "image_backend", "lazy_render"
                        ),
                        rendered_image_dir=config.get("rendered_image_dir"),
                        missing_image_policy=config.get(
                            "missing_image_policy", "error"
                        ),
                        h_snr=float(config.get("h_snr", 500.0)),
                        c_snr=float(config.get("c_snr", 300.0)),
                        render_seed=config.get("render_seed", seed),
                        image_size=config.get("image_size"),
                    )
                )
            label = str(sample.get("canonical_smiles", "") or "").strip()

            # --- build prompt ---
            prompt = build_structure_prompt(
                sample,
                prompt_template,
                include_formula=include_formula,
                include_rule_context=rule_context_enabled,
                max_rule_evidence=int(config.get("max_rule_evidence", 12)),
                input_mode=input_mode,
            )

            # --- generate ---
            try:
                pred, generation_trace = generate_one(
                    model=model,
                    tokenizer=tokenizer,
                    images=images,
                    prompt=prompt,
                    max_new_tokens=config.get("max_new_tokens", 256),
                    temperature=config.get("temperature", 0.0),
                    top_p=config.get("top_p", 1.0),
                    input_mode=input_mode,
                )
            except Exception:
                pred = ""
                generation_trace = {
                    "generated_token_count": 0,
                    "generation_terminated_by_eos": False,
                    "generation_hit_max_tokens": False,
                    "generation_repeated_4gram": False,
                }
                n_errors += 1
                traceback.print_exc()

            predictions.append(pred)
            generation_traces.append(generation_trace)

            # --- scoring ---
            structure_metrics = evaluate_structure_prediction(
                pred,
                label,
                sample=sample if rule_validation_enabled else None,
                include_formula=include_formula,
            )
            metric_rows.append(structure_metrics)

            record: dict[str, Any] = {
                "idx": idx,
                "id": sample.get("id"),
                "prompt_template_index": prompt_template_index,
                "include_formula": include_formula,
                "input_mode": input_mode,
                "rule_context_enabled": rule_context_enabled,
                "rule_validation_enabled": rule_validation_enabled,
                "rule_library": (
                    rule_library
                    if rule_context_enabled or rule_validation_enabled
                    else None
                ),
                "prompt": prompt,
                "prediction": pred,
                "label": label,
                **generation_trace,
                **structure_metrics,
            }

            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    # ---- 6. Report -------------------------------------------------------
    print(f"Saved predictions to: {output_path}")
    summary = summarize_structure_predictions(metric_rows)
    summary.update(
        summarize_generation_behavior(predictions, generation_traces)
    )
    summary["generation_errors"] = n_errors
    summary["prompt_template_index"] = prompt_template_index
    summary["input_mode"] = input_mode
    summary["rule_context_enabled"] = rule_context_enabled
    summary["rule_validation_enabled"] = rule_validation_enabled
    if rule_context_enabled or rule_validation_enabled:
        summary["rule_library"] = rule_library
    summary_path = Path(
        config.get(
            "summary_output",
            output_path.with_suffix(".summary.json"),
        )
    )
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Saved summary to: {summary_path}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m src.training.inference <config.yaml>")
        sys.exit(1)
    main(load_config(sys.argv[1]))
