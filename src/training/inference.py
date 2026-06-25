"""Text-only prediction entrypoint for SpectraLM fine-tuned models."""

from __future__ import annotations
import unsloth
import itertools
import json
import sys
import traceback
from pathlib import Path
from typing import Any

import torch
from peft import PeftModel
from tqdm import tqdm
from unsloth import FastLanguageModel

from src.config import load_config
from src.data.dataset import load_raw_nmr_samples
from src.evaluation.metrics import (
    evaluate_structure_prediction,
    inspect_generation_tokens,
    summarize_generation_behavior,
    summarize_structure_predictions,
)
from src.evaluation.prompts import (
    SYSTEM_PROMPT,
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
    """Load a text LLM and optionally attach a LoRA adapter."""
    model, tokenizer = FastLanguageModel.from_pretrained(
        base_model_path,
        max_seq_length=max_seq_length,
        load_in_4bit=load_in_4bit,
        use_gradient_checkpointing=False,
        attn_implementation="sdpa",
    )
    if adapter_path is not None:
        model = PeftModel.from_pretrained(
            model,
            str(adapter_path),
            is_trainable=False,
        )
    FastLanguageModel.for_inference(model)
    model.eval()
    return model, tokenizer


def build_inference_rows(config: dict[str, Any]) -> list[dict[str, Any]]:
    """Load a bounded curated JSONL split for inference."""
    return load_raw_nmr_samples(
        config["dataset_dir"],
        split=config.get("split", "test"),
    )


def _model_device(model: Any) -> str:
    try:
        return str(next(model.parameters()).device)
    except Exception:
        return "cuda" if torch.cuda.is_available() else "cpu"


def _move_inputs(inputs: Any, device: str) -> Any:
    if hasattr(inputs, "to"):
        return inputs.to(device)
    return {
        key: value.to(device) if hasattr(value, "to") else value
        for key, value in inputs.items()
    }


def _eos_token_ids(model: Any, tokenizer: Any) -> tuple[int | list[int], set[int]]:
    raw = getattr(getattr(model, "generation_config", None), "eos_token_id", None)
    if raw is None:
        raw = getattr(tokenizer, "eos_token_id", None)
    if raw is None:
        raise ValueError("Tokenizer and model generation config have no EOS token.")
    eos_set = {int(raw)} if isinstance(raw, int) else {int(token_id) for token_id in raw}
    return raw, eos_set


def _pad_token_id(tokenizer: Any, eos_token_id: int | list[int]) -> int:
    pad = getattr(tokenizer, "pad_token_id", None)
    if pad is not None:
        return int(pad)
    return int(eos_token_id if isinstance(eos_token_id, int) else eos_token_id[0])


def _build_input_text(tokenizer: Any, prompt: str) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]
    return apply_non_thinking_chat_template(tokenizer, messages)


@torch.inference_mode()
def generate_one(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 256,
    temperature: float = 0.0,
    top_p: float = 1.0,
) -> tuple[str, dict[str, int | bool]]:
    """Generate one text prediction from an NMR prompt."""
    input_text = _build_input_text(tokenizer, prompt)
    inputs = tokenizer(
        input_text,
        add_special_tokens=False,
        return_tensors="pt",
    )
    inputs = _move_inputs(inputs, _model_device(model))

    do_sample = temperature > 0
    raw_eos_token_ids, eos_token_ids = _eos_token_ids(model, tokenizer)
    pad_token_id = _pad_token_id(tokenizer, raw_eos_token_ids)
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
    prompt: str,
    *,
    num_return_sequences: int,
    max_new_tokens: int = 128,
    temperature: float = 0.7,
    top_p: float = 0.9,
) -> tuple[list[str], list[dict[str, int | bool]]]:
    """Sample multiple structure candidates from one identical text input."""
    if num_return_sequences < 1:
        raise ValueError("num_return_sequences must be positive")
    if num_return_sequences > 1 and temperature <= 0:
        raise ValueError("Multiple candidates require temperature > 0")
    input_text = _build_input_text(tokenizer, prompt)
    inputs = tokenizer(
        input_text,
        add_special_tokens=False,
        return_tensors="pt",
    )
    inputs = _move_inputs(inputs, _model_device(model))

    raw_eos_token_ids, eos_token_ids = _eos_token_ids(model, tokenizer)
    output_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=temperature > 0,
        temperature=temperature,
        top_p=top_p,
        num_return_sequences=num_return_sequences,
        use_cache=True,
        eos_token_id=raw_eos_token_ids,
        pad_token_id=_pad_token_id(tokenizer, raw_eos_token_ids),
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
    """Run direct text inference over one split."""
    max_samples: int | None = config.get("max_samples", None)
    include_formula = bool(config.get("include_formula", True))
    rule_context_enabled = bool(config.get("rule_context_enabled", False))
    rule_validation_enabled = bool(config.get("rule_validation_enabled", False))
    rule_library = str(load_rule_library()["library_name"])

    model, tokenizer = load_model_for_inference(
        base_model_path=config["model_path"],
        adapter_path=config.get("adapter_path"),
        max_seq_length=config.get("max_seq_length", 8192),
        load_in_4bit=config.get("load_in_4bit", True),
    )
    test_ds = build_inference_rows(config)
    prompt_template_index = int(config.get("prompt_template_index", 0))
    prompt_template = select_structure_prompt(prompt_template_index)

    output_path = Path(config["output"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    n_errors = 0
    metric_rows: list[dict[str, Any]] = []
    predictions: list[str] = []
    generation_traces: list[dict[str, int | bool]] = []

    iterator = enumerate(test_ds)
    if max_samples is not None:
        iterator = itertools.islice(iterator, max_samples)

    with output_path.open("w", encoding="utf-8") as handle:
        for idx, sample in tqdm(iterator, desc="Infer", total=max_samples):
            label = str(sample.get("canonical_smiles", "") or "").strip()
            prompt = build_structure_prompt(
                sample,
                prompt_template,
                include_formula=include_formula,
                include_rule_context=rule_context_enabled,
                max_rule_evidence=int(config.get("max_rule_evidence", 12)),
            )
            try:
                pred, generation_trace = generate_one(
                    model=model,
                    tokenizer=tokenizer,
                    prompt=prompt,
                    max_new_tokens=config.get("max_new_tokens", 256),
                    temperature=config.get("temperature", 0.0),
                    top_p=config.get("top_p", 1.0),
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
            structure_metrics = evaluate_structure_prediction(
                pred,
                label,
                sample=sample if rule_validation_enabled else None,
                include_formula=include_formula,
            )
            metric_rows.append(structure_metrics)
            record = {
                "idx": idx,
                "id": sample.get("id"),
                "prompt_template_index": prompt_template_index,
                "include_formula": include_formula,
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
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    summary = summarize_structure_predictions(metric_rows)
    summary.update(summarize_generation_behavior(predictions, generation_traces))
    summary["generation_errors"] = n_errors
    summary["prompt_template_index"] = prompt_template_index
    summary["rule_context_enabled"] = rule_context_enabled
    summary["rule_validation_enabled"] = rule_validation_enabled
    if rule_context_enabled or rule_validation_enabled:
        summary["rule_library"] = rule_library
    summary_path = Path(
        config.get("summary_output", output_path.with_suffix(".summary.json"))
    )
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Saved predictions to: {output_path}")
    print(f"Saved summary to: {summary_path}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m src.training.inference <config.yaml>")
        sys.exit(1)
    main(load_config(sys.argv[1]))
