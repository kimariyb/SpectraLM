"""Prediction entrypoint for SpectraLM fine-tuned models.

Renders NMR spectra images from test samples, builds structure-elucidation
prompts, generates SMILES predictions, and writes results as JSONL.

Supports four ablation modes that progressively remove input information:

``image_table_rule``
    Full input: spectra images + peak tables + NMR interpretation rules.
``image_table``
    Images + peak tables, no rules.
``table_only``
    Peak tables only, no images.
``image_only``
    Images only, no peak tables or rules.

Usage::

    python -m src.training.inference configs/inference.yaml
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import torch
from PIL import Image as PILImage
from tqdm import tqdm

from src.config import load_config
from src.io import load_pickle_list
from src.spectra.render import hydrogen_to_spectra, carbon_to_spectra
from src.evaluation.prompts import STRUCTURE_PROMPTS, build_structure_prompt
from unsloth import FastVisionModel

# Constants
PREDICTION_MODES = ("image_table_rule", "image_table", "table_only", "image_only")


def _strip_peak_tables(prompt: str) -> str:
    """Remove peak-table sections while keeping the task contract.

    Parameters
    ----------
    prompt
        Full image-table-rule prompt text.

    Returns
    -------
    str
        Prompt without explicit peak-table blocks.
    """
    sections = prompt.split("\n\n")
    return "\n\n".join(
        section for section in sections if "NMR Peak Table" not in section
    )


def _strip_rule_prompt(prompt: str) -> str:
    """Remove the NMR rule-hint section from a prompt.

    Parameters
    ----------
    prompt
        Full prompt text.

    Returns
    -------
    str
        Prompt text without the rule-hint section.
    """
    sections = prompt.split("\n\n")
    return "\n\n".join(
        section
        for section in sections
        if not section.startswith("NMR rules to consider")
    )


def _render_images(sample: dict[str, Any]) -> tuple[PILImage.Image, PILImage.Image]:
    """Render deterministic 1H and 13C spectrum images for a sample.

    Parameters
    ----------
    sample
        Normalized SpectraLM sample dictionary.

    Returns
    -------
    tuple[PILImage.Image, PILImage.Image]
        RGB images for 1H and 13C spectra respectively.
    """
    h_img = hydrogen_to_spectra(sample, snr=500.0, seed=42)
    c_img = carbon_to_spectra(sample, snr=300.0, seed=42)

    if not isinstance(h_img, PILImage.Image):
        h_img = PILImage.fromarray(h_img)
    h_img = h_img.convert("RGB")

    if not isinstance(c_img, PILImage.Image):
        c_img = PILImage.fromarray(c_img)
    c_img = c_img.convert("RGB")

    return h_img, c_img


# ---------------------------------------------------------------------------
# Prompt building
# ---------------------------------------------------------------------------


def _build_prediction_messages(
    sample: dict[str, Any],
    mode: str,
) -> list[dict[str, Any]]:
    """Build chat-format messages for one ablation mode.

    Parameters
    ----------
    sample
        Normalized SpectraLM sample dictionary.
    mode
        One of ``"image_table_rule"``, ``"image_table"``, ``"table_only"``,
        or ``"image_only"``.

    Returns
    -------
    list[dict[str, Any]]
        Chat messages with ``"user"`` role.

    Raises
    ------
    ValueError
        If *mode* is not a recognised ablation mode.
    """
    if mode not in PREDICTION_MODES:
        raise ValueError(
            f"Unknown prediction mode: {mode!r}.  "
            f"Must be one of {PREDICTION_MODES}."
        )

    prompt = build_structure_prompt(sample, prompt=STRUCTURE_PROMPTS[0])

    # Ablation: progressively strip information
    if mode in ("image_table", "table_only"):
        prompt = _strip_rule_prompt(prompt)
    if mode == "image_only":
        prompt = _strip_rule_prompt(_strip_peak_tables(prompt))

    content: list[dict[str, Any]] = []

    # Include images only when the mode requests them
    if mode in ("image_table_rule", "image_table", "image_only"):
        h_img, c_img = _render_images(sample)
        content.append({"type": "image", "image": h_img})
        content.append({"type": "image", "image": c_img})

    content.append({"type": "text", "text": prompt})

    return [{"role": "user", "content": content}]


# ---------------------------------------------------------------------------
# Model helpers
# ---------------------------------------------------------------------------


def _encode_messages(
    processor: Any,
    messages: list[dict[str, Any]],
) -> dict[str, Any]:
    """Encode chat messages into model inputs.

    Parameters
    ----------
    processor
        HuggingFace multimodal processor.
    messages
        Chat messages with optional ``"image"`` content parts.

    Returns
    -------
    dict[str, Any]
        Tensor inputs ready for ``model.generate``.
    """
    try:
        return processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        )
    except TypeError:
        # Fallback for processors without native chat-template tokenization
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        images = [
            part["image"]
            for message in messages
            for part in message.get("content", [])
            if isinstance(part, dict) and part.get("type") == "image"
        ]
        kwargs: dict[str, Any] = {
            "text": [text],
            "return_tensors": "pt",
            "padding": True,
        }
        if images:
            kwargs["images"] = images
        return processor(**kwargs)


def _move_to_device(inputs: dict[str, Any], device: torch.device) -> dict[str, Any]:
    """Move tensor values in a dict to the target device."""
    return {
        key: value.to(device) if hasattr(value, "to") else value
        for key, value in inputs.items()
    }


def _decode_generated(
    processor: Any,
    inputs: dict[str, Any],
    generated_ids: torch.Tensor,
) -> str:
    """Decode only the newly generated tokens (excluding the prompt)."""
    prompt_len = inputs["input_ids"].shape[-1]
    new_tokens = generated_ids[:, prompt_len:]
    tokenizer = getattr(processor, "tokenizer", processor)
    decoded = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
    return decoded[0].strip() if decoded else ""


def _predict_one(
    model: Any,
    processor: Any,
    sample: dict[str, Any],
    mode: str,
    device: torch.device,
    max_new_tokens: int,
) -> str:
    """Generate a SMILES prediction for a single sample under one ablation mode.

    Parameters
    ----------
    model
        Loaded vision-language model in inference mode.
    processor
        Matching multimodal processor.
    sample
        Normalized SpectraLM sample dictionary.
    mode
        Ablation input mode (see :data:`PREDICTION_MODES`).
    device
        Target device for tensors.
    max_new_tokens
        Maximum number of tokens to generate.

    Returns
    -------
    str
        Raw generated text (typically a SMILES string).
    """
    messages = _build_prediction_messages(sample, mode)
    inputs = _encode_messages(processor, messages)
    inputs = _move_to_device(inputs, device)

    with torch.inference_mode():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
        )

    return _decode_generated(processor, inputs, generated_ids)


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------


def main(config: dict[str, Any]) -> None:
    """Run batch prediction from a configuration dictionary.

    Parameters
    ----------
    config
        Configuration with keys ``model_path``, ``test_dataset``,
        ``mode``, ``output``, ``max_new_tokens``, and optionally
        ``max_samples`` and ``max_seq_length``.
    """
    model_path: str = config["model_path"]
    test_dataset: str = config["test_dataset"]
    mode: str = config.get("mode", "image_table_rule")
    output_path: str = config.get("output", "outputs/predictions.jsonl")
    max_new_tokens: int = int(config.get("max_new_tokens", 256))
    max_samples: int | None = config.get("max_samples")
    if max_samples is not None:
        max_samples = int(max_samples)

    if mode not in PREDICTION_MODES:
        raise ValueError(
            f"Unknown prediction mode: {mode!r}.  "
            f"Must be one of {PREDICTION_MODES}."
        )

    # -- Load model ---------------------------------------------------------
    print(f"Loading model: {model_path}")
    model, processor = FastVisionModel.from_pretrained(
        model_path,
        max_seq_length=config.get("max_seq_length", 8192),
        load_in_4bit=True,
    )
    FastVisionModel.for_inference(model)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # -- Load samples -------------------------------------------------------
    samples = load_pickle_list(test_dataset)
    if max_samples is not None:
        samples = samples[:max_samples]
    print(f"Loaded {len(samples)} test samples  |  mode = {mode}")

    # -- Generate predictions -----------------------------------------------
    results: list[dict[str, str]] = []
    for sample in tqdm(samples, desc=f"Predicting ({mode})"):
        prediction = _predict_one(
            model, processor, sample, mode, device, max_new_tokens
        )
        ref_smiles = (
            sample.get("canonical_smiles")
            or sample.get("smiles")
            or ""
        )
        results.append({
            "id": str(sample.get("id", "")),
            "prediction": prediction,
            "reference_smiles": str(ref_smiles),
            "mode": mode,
        })

    # -- Write output -------------------------------------------------------
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row in results:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Wrote {len(results)} predictions to {output}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m src.training.inference <config.yaml>")
        sys.exit(1)
    main(load_config(sys.argv[1]))
