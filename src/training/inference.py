"""Prediction entrypoint for SpectraLM fine-tuned models.

Loads the test split via :func:`~src.data.dataset.load_nmr_dataset`, builds
structure-elucidation prompts using the pre-rendered spectrum images, and
generates SMILES predictions.

Supports four ablation modes that progressively remove input information:

``image_table_rule``
    Full input: spectra images + peak tables + NMR interpretation rules.
``image_table``
    Images + peak table.
``table_only``
    Peak tables only, no images.
``image_only``
    Images only, no peak tables or rules.

Usage::

    python -m src.training.inference configs/inference.yaml
"""

from __future__ import annotations

import itertools
import json
import pickle
import random
import sys
import traceback
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm

from src.config import load_config
from src.data.dataset import (
    load_nmr_dataset,
    load_raw_nmr_samples,
    render_sample_images,
)
from src.evaluation.prompts import STRUCTURE_PROMPTS, build_structure_prompt
from unsloth import FastVisionModel
from peft import PeftModel


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


def extract_sample_from_row(row: dict[str, Any]) -> tuple[list[Any], dict[str, Any]]:
    """Extract images and deserialised sample dict from a raw dataset row.

    Parameters
    ----------
    row
        Raw Arrow row with ``"h_image"``, ``"c_image"``, and
        ``"sample_pickle"`` columns.

    Returns
    -------
    tuple[list[Any], dict[str, Any]]
        ``(images, sample)`` — *images* is a list of 0–2 PIL images;
        *sample* is the deserialised sample dictionary.
    """
    h_image = row.get("h_image")
    c_image = row.get("c_image")
    images = [img for img in (h_image, c_image) if img is not None]

    blob = row.get("sample_pickle")
    sample: dict[str, Any]
    if blob is not None:
        if isinstance(blob, memoryview):
            blob = blob.tobytes()
        sample = pickle.loads(blob)
    else:
        sample = {}

    return images, sample


def build_inference_rows(
    config: dict[str, Any],
    *,
    seed: int,
) -> list[dict[str, Any]] | Any:
    """Load inference rows without pre-rendering full JSONL datasets."""
    dataset_backend = config.get("dataset_backend", "hf")
    if dataset_backend == "lazy_jsonl":
        return load_raw_nmr_samples(
            config["dataset_dir"],
            split=config.get("split", "test"),
            train_size=float(config.get("train_size", 0.8)),
        )

    return load_nmr_dataset(
        config["dataset_dir"],
        split=config.get("split", "test"),
        train_size=float(config.get("train_size", 0.8)),
        render_cache_dir=config.get("render_cache_dir", config.get("train_cache_dir")),
        render_cache_version=config.get("cache_version", "1"),
        seed=seed,
        with_messages=False,
    )


# Minimal prompt used when peak tables are not provided (image_only mode).
IMAGE_ONLY_PROMPT: str = (
    "Determine the molecular structure from the 1H and 13C NMR spectra "
    "below.\nOutput the canonical SMILES of the molecule."
)

TABLE_ONLY_PROMPT: str = (
    "Determine the molecular structure from the NMR peak tables below.\n\n"
    "{peak_tables}\n\n"
    "Output the canonical SMILES of the molecule."
)


def extract_label(example: dict[str, Any]) -> str | None:
    """Extract ground-truth answer from your dataset.

    Modify this if your dataset stores labels using another field.
    """

    if "answer" in example:
        return str(example["answer"])

    if "target" in example:
        return str(example["target"])

    if "label" in example:
        return str(example["label"])

    if "messages" in example:
        for msg in example["messages"]:
            if msg.get("role") == "assistant":
                content = msg.get("content", "")
                if isinstance(content, str):
                    return content
                if isinstance(content, list):
                    texts = [
                        x.get("text", "")
                        for x in content
                        if isinstance(x, dict) and x.get("type") == "text"
                    ]
                    return "\n".join(texts)

    return None


@torch.inference_mode()
def generate_one(
    model,
    tokenizer,
    images: list[Any],
    prompt: str,
    max_new_tokens: int = 256,
    temperature: float = 0.0,
    top_p: float = 1.0,
) -> str:
    """Generate one prediction from zero or two NMR spectrum images.

    Parameters
    ----------
    model
        The loaded VLM with LoRA adapter.
    tokenizer
        Corresponding tokenizer.
    images
        List of 0 or 2 PIL/ndarray images.
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
    str
        Decoded prediction string.
    """
    if len(images) not in (0, 2):
        raise ValueError(
            f"Expected 0 or 2 images, but got {len(images)}."
        )

    # Build user-message content dynamically
    content: list[dict[str, Any]] = []
    for _ in images:
        content.append({"type": "image"})
    content.append({"type": "text", "text": prompt})

    messages = [{"role": "user", "content": content}]

    input_text = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
    )

    if images:
        inputs = tokenizer(
            images,
            input_text,
            add_special_tokens=False,
            return_tensors="pt",
        ).to("cuda")
    else:
        inputs = tokenizer(
            input_text,
            add_special_tokens=False,
            return_tensors="pt",
        ).to("cuda")

    do_sample = temperature > 0

    output_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=temperature if do_sample else None,
        top_p=top_p if do_sample else None,
        use_cache=True,
    )

    input_len = inputs["input_ids"].shape[1]
    generated_ids = output_ids[:, input_len:]

    pred = tokenizer.batch_decode(
        generated_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]

    return pred.strip()


def normalize_text(x: str | None) -> str:
    if x is None:
        return ""
    return " ".join(str(x).strip().split())


def main(config: dict[str, Any]) -> None:
    """Run inference over the test split.

    Parameters
    ----------
    config
        Configuration dictionary.  See ``configs/inference.yaml`` for keys.
    """
    mode: str = config.get("mode", "image_table_rule")
    max_samples: int | None = config.get("max_samples", None)
    seed: int = int(config.get("seed", 3407))

    if mode not in ("image_table_rule", "image_table", "table_only", "image_only"):
        raise ValueError(
            f"Unknown mode {mode!r}.  Expected one of: "
            "image_table_rule, image_table, table_only, image_only."
        )

    # ---- 1. Load model ---------------------------------------------------
    model, tokenizer = load_model_for_inference(
        base_model_path=config["model_path"],
        adapter_path=config.get("adapter_path"),
        max_seq_length=config.get("max_seq_length", 8192),
        load_in_4bit=config.get("load_in_4bit", True),
    )

    # ---- 2. Load dataset (raw columns, no messages transform) ------------
    test_ds = build_inference_rows(config, seed=seed)

    # ---- 3. Select prompt template ---------------------------------------
    rng = random.Random(seed)
    prompt_template: str = rng.choice(STRUCTURE_PROMPTS)

    # ---- 4. Ablation-mode flags ------------------------------------------
    include_images: bool = mode != "table_only"
    include_tables: bool = mode != "image_only"
    include_rules: bool = mode == "image_table_rule"
    include_formula: bool = mode != "image_only"

    print(f"Mode: {mode}  |  images={include_images}  |  tables={include_tables}")
    print(f"Prompt template: {prompt_template[:80]}...")

    # ---- 5. Inference loop -----------------------------------------------
    output_path = Path(config["output"])
    output_path.parent.mkdir(parents=True, exist_ok=True)

    n_total = 0
    n_exact = 0
    n_errors = 0

    iterator = enumerate(test_ds)
    if max_samples is not None:
        iterator = itertools.islice(iterator, max_samples)

    with output_path.open("w", encoding="utf-8") as f:
        for idx, row in tqdm(iterator, desc="Infer", total=max_samples):
            # --- extract data from raw row ---
            if config.get("dataset_backend", "hf") == "lazy_jsonl":
                sample = row
                images = list(
                    render_sample_images(
                        sample,
                        h_snr=float(config.get("h_snr", 500.0)),
                        c_snr=float(config.get("c_snr", 300.0)),
                        render_seed=config.get("render_seed", seed),
                        image_size=config.get("image_size"),
                    )
                )
            else:
                images, sample = extract_sample_from_row(row)
            label = str(sample.get("canonical_smiles", "") or "").strip()

            # --- build prompt ---
            if include_tables:
                template = TABLE_ONLY_PROMPT if mode == "table_only" else prompt_template
                prompt = build_structure_prompt(
                    sample,
                    template,
                    include_formula=include_formula,
                    include_rules=include_rules,
                )
            else:
                prompt = IMAGE_ONLY_PROMPT

            # --- select images ---
            selected_images = images if include_images else []

            # --- generate ---
            try:
                pred = generate_one(
                    model=model,
                    tokenizer=tokenizer,
                    images=selected_images,
                    prompt=prompt,
                    max_new_tokens=config.get("max_new_tokens", 256),
                    temperature=config.get("temperature", 0.0),
                    top_p=config.get("top_p", 1.0),
                )
            except Exception:
                pred = ""
                n_errors += 1
                traceback.print_exc()

            # --- scoring ---
            pred_norm = normalize_text(pred)
            label_norm = normalize_text(label)

            exact_match = pred_norm == label_norm if label else None

            if exact_match is not None:
                n_total += 1
                n_exact += int(exact_match)

            record: dict[str, Any] = {
                "idx": idx,
                "mode": mode,
                "prompt": prompt,
                "prediction": pred,
                "label": label,
                "exact_match": exact_match,
            }

            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    # ---- 6. Report -------------------------------------------------------
    print(f"Mode: {mode}")
    print(f"Saved predictions to: {output_path}")
    if n_errors:
        print(f"Errors: {n_errors}")
    if n_total > 0:
        print(f"Exact match: {n_exact}/{n_total} = {n_exact / n_total:.4f}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m src.training.inference <config.yaml>")
        sys.exit(1)
    main(load_config(sys.argv[1]))
