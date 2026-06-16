"""Prediction entry point for SpectraLM ablation experiments."""

from __future__ import annotations

import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from tqdm import tqdm
from unsloth import FastVisionModel

from spectralm.config import load_config
from spectralm.data.molecules import sample_selfies
from spectralm.io import load_pickle_list
from spectralm.spectra.render import combine_spectra
from spectralm.training.prompts import STRUCTURE_PROMPTS, build_structure_prompt, canonical_smiles

PREDICTION_MODES = ("image_table_rule", "image_table", "table_only", "image_only")


@dataclass(frozen=True)
class PredictionRecord:
    """Serializable model prediction row.

    Parameters
    ----------
    sample_id
        Reference sample identifier.
    prediction
        Raw generated model text.
    reference_smiles
        Reference canonical SMILES.
    mode
        Ablation input mode used for generation.
    """

    sample_id: str
    prediction: str
    reference_smiles: str
    mode: str

    def to_json_row(self) -> dict[str, str]:
        """Convert the record to a JSONL row.

        Returns
        -------
        dict[str, str]
            JSON-serializable row.
        """
        row = asdict(self)
        row["id"] = row.pop("sample_id")
        return row


def strip_peak_tables(prompt: str) -> str:
    """Remove peak-table sections while keeping the task contract.

    Parameters
    ----------
    prompt
        Full image-table-rule prompt.

    Returns
    -------
    str
        Prompt text without explicit peak tables.
    """
    sections = prompt.split("\n\n")
    return "\n\n".join(section for section in sections if "NMR peak table" not in section)


def strip_rule_prompt(prompt: str) -> str:
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
    return "\n\n".join(section for section in sections if not section.startswith("NMR rules to consider"))


def build_prediction_prompt(sample: dict[str, Any], mode: str) -> str:
    """Build the text prompt for one ablation mode.

    Parameters
    ----------
    sample
        Normalized SpectraLM sample.
    mode
        One of ``image_table_rule``, ``image_table``, ``table_only``, or
        ``image_only``.

    Returns
    -------
    str
        Prompt text for model generation.

    Raises
    ------
    ValueError
        If an unknown mode is requested.
    """
    if mode not in PREDICTION_MODES:
        raise ValueError(f"Unknown prediction mode: {mode}")
    prompt = build_structure_prompt(sample, STRUCTURE_PROMPTS[0])
    if mode in ("image_table", "table_only"):
        prompt = strip_rule_prompt(prompt)
    if mode == "image_only":
        prompt = strip_rule_prompt(strip_peak_tables(prompt))
    return prompt


def build_prediction_image(sample: dict[str, Any]) -> Image.Image:
    """Render a deterministic combined spectrum image for prediction.

    Parameters
    ----------
    sample
        Normalized SpectraLM sample.

    Returns
    -------
    PIL.Image.Image
        RGB NMR spectrum image.
    """
    image = combine_spectra(sample=sample, h_snr=500, c_snr=300)
    if not isinstance(image, Image.Image):
        image = Image.fromarray(image)
    return image.convert("RGB")


def build_prediction_example(sample: dict[str, Any], mode: str) -> dict[str, list[dict[str, Any]]]:
    """Build a chat-style prediction example.

    Parameters
    ----------
    sample
        Normalized SpectraLM sample.
    mode
        Ablation input mode.

    Returns
    -------
    dict[str, list[dict[str, Any]]]
        User-only chat example containing text and, when requested, an image.
    """
    prompt = build_prediction_prompt(sample, mode)
    content: list[dict[str, Any]] = []
    if mode in ("image_table_rule", "image_table", "image_only"):
        content.append({"type": "image", "image": build_prediction_image(sample)})
    content.append({"type": "text", "text": prompt})
    return {"messages": [{"role": "user", "content": content}]}


def write_prediction_jsonl(path: str | Path, records: list[PredictionRecord]) -> None:
    """Write prediction records to a JSONL file.

    Parameters
    ----------
    path
        Output JSONL path.
    records
        Prediction rows to serialize.
    """
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record.to_json_row(), ensure_ascii=False) + "\n")


def move_inputs_to_device(inputs: dict[str, Any], device: torch.device) -> dict[str, Any]:
    """Move tensor inputs to a target device.

    Parameters
    ----------
    inputs
        Processor output mapping.
    device
        Target PyTorch device.

    Returns
    -------
    dict[str, Any]
        Mapping with tensors moved to the device.
    """
    return {key: value.to(device) if hasattr(value, "to") else value for key, value in inputs.items()}


def encode_messages(processor: Any, messages: list[dict[str, Any]]) -> dict[str, Any]:
    """Encode Qwen-VL chat messages into model inputs.

    Parameters
    ----------
    processor
        Hugging Face multimodal processor.
    messages
        Chat messages with optional image content.

    Returns
    -------
    dict[str, Any]
        Tensor inputs for generation.
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
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        images = [
            part["image"]
            for message in messages
            for part in message.get("content", [])
            if part.get("type") == "image"
        ]
        kwargs = {"text": [text], "return_tensors": "pt", "padding": True}
        if images:
            kwargs["images"] = images
        return processor(**kwargs)


def decode_generated_text(processor: Any, inputs: dict[str, Any], generated_ids: Any) -> str:
    """Decode only newly generated tokens.

    Parameters
    ----------
    processor
        Hugging Face multimodal processor.
    inputs
        Generation inputs containing ``input_ids``.
    generated_ids
        Output token ids from ``model.generate``.

    Returns
    -------
    str
        Decoded assistant output.
    """
    prompt_len = inputs["input_ids"].shape[-1]
    new_tokens = generated_ids[:, prompt_len:]
    tokenizer = getattr(processor, "tokenizer", processor)
    decoded = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
    return decoded[0].strip() if decoded else ""


def generate_prediction(
    model: Any,
    processor: Any,
    sample: dict[str, Any],
    mode: str,
    max_new_tokens: int,
    device: torch.device,
) -> str:
    """Generate one prediction for a sample.

    Parameters
    ----------
    model
        Loaded vision-language model.
    processor
        Matching multimodal processor.
    sample
        Normalized SpectraLM sample.
    mode
        Ablation input mode.
    max_new_tokens
        Maximum number of newly generated tokens.
    device
        Target device.

    Returns
    -------
    str
        Raw generated text.
    """
    example = build_prediction_example(sample, mode)
    inputs = encode_messages(processor, example["messages"])
    inputs = move_inputs_to_device(inputs, device)
    with torch.inference_mode():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
        )
    return decode_generated_text(processor, inputs, generated_ids)


def run(config: dict[str, Any]) -> None:
    """Run prediction generation from a configuration dictionary.

    Parameters
    ----------
    config
        Configuration dictionary with keys ``model_path``, ``dataset``,
        ``output``, ``mode``, ``max_samples``, and ``max_new_tokens``.
    """
    model_path = config.get("model_path", "outputs/spectralm-butina-qwen3-vl-8b")
    dataset_path = config.get("dataset", "dataset/subsets/spectralm_butina_1000_300/test.pkl")
    output_path = config.get("output", "outputs/predictions.jsonl")
    mode = config.get("mode", "image_table_rule")
    max_samples = config.get("max_samples")
    max_new_tokens = config.get("max_new_tokens", 768)

    print(f"Loading model: {model_path}")
    model, processor = FastVisionModel.from_pretrained(model_path)
    FastVisionModel.for_inference(model)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    samples = load_pickle_list(dataset_path)
    if max_samples is not None:
        samples = samples[: int(max_samples)]

    records = []
    for sample in tqdm(samples, desc=f"Predicting ({mode})"):
        prediction = generate_prediction(model, processor, sample, mode, int(max_new_tokens), device)
        records.append(
            PredictionRecord(
                sample_id=str(sample.get("id", "")),
                prediction=prediction,
                reference_smiles=canonical_smiles(sample),
                mode=mode,
            )
        )
    write_prediction_jsonl(output_path, records)
    print(f"Wrote {len(records)} predictions to {output_path}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m spectralm.inference.predict <config.yaml>")
        sys.exit(1)
    run(load_config(sys.argv[1]))
