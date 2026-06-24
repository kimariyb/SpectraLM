"""Input-modality contracts for controlled NMR ablation experiments."""

from __future__ import annotations

import hashlib
from typing import Any, Iterable, Mapping, Sequence


FULL = "full"
IMAGE_ONLY = "image_only"
PEAK_TABLE_ONLY = "peak_table_only"
FORMULA_ONLY = "formula_only"

INPUT_MODES = (
    FULL,
    IMAGE_ONLY,
    PEAK_TABLE_ONLY,
    FORMULA_ONLY,
)
TRAINING_INPUT_MODES = (FULL, IMAGE_ONLY, PEAK_TABLE_ONLY)


def normalize_input_mode(input_mode: str | None) -> str:
    """Return one validated input-mode name, defaulting to ``full``."""
    normalized = str(input_mode or FULL).strip().lower()
    if normalized not in INPUT_MODES:
        raise ValueError(
            f"input_mode must be one of {', '.join(INPUT_MODES)}, "
            f"got {input_mode!r}"
        )
    return normalized


def normalize_input_mode_weights(
    weights: Mapping[str, float] | None,
) -> dict[str, float]:
    """Validate and normalize the three-mode training curriculum."""
    raw = dict(weights or {FULL: 1.0})
    unknown = sorted(set(raw) - set(TRAINING_INPUT_MODES))
    if unknown:
        raise ValueError(
            "Unsupported training input mode: " + ", ".join(unknown)
        )
    if any(float(value) < 0 for value in raw.values()):
        raise ValueError("input_mode_weights must be non-negative")
    positive = {
        normalize_input_mode(mode): float(value)
        for mode, value in raw.items()
        if float(value) > 0
    }
    if not positive:
        raise ValueError("input_mode_weights must contain a positive weight")
    total = sum(positive.values())
    return {mode: value / total for mode, value in positive.items()}


def select_weighted_input_mode(
    sample_id: str,
    *,
    seed: int,
    weights: Mapping[str, float],
) -> str:
    """Select one reproducible input mode from normalized sample weights."""
    normalized = normalize_input_mode_weights(weights)
    payload = f"{seed}:{sample_id}:input_mode".encode("utf-8")
    point = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") / float(
        2**64
    )
    cumulative = 0.0
    for mode, probability in normalized.items():
        cumulative += probability
        if point < cumulative:
            return mode
    return next(reversed(normalized))


def input_mode_uses_images(input_mode: str | None) -> bool:
    """Return whether a mode exposes spectrum images to the model."""
    return normalize_input_mode(input_mode) in {FULL, IMAGE_ONLY}


def input_mode_uses_peak_tables(input_mode: str | None) -> bool:
    """Return whether a mode exposes numerical peak tables to the model."""
    return normalize_input_mode(input_mode) in {FULL, PEAK_TABLE_ONLY}


def validate_input_configuration(
    input_mode: str | None,
    *,
    include_formula: bool,
    include_rule_context: bool = False,
    task_names: Iterable[str] = ("structure_prediction",),
) -> str:
    """Validate that an experiment cannot leak an ablated input source."""
    mode = normalize_input_mode(input_mode)
    if mode == FORMULA_ONLY and not include_formula:
        raise ValueError("formula_only input_mode requires include_formula=true")
    if mode != FULL and include_rule_context:
        raise ValueError(
            "rule context is only supported for full input_mode because it "
            "derives additional evidence from the peak data"
        )
    tuple(task_names)
    return mode


def build_user_content(
    prompt: str,
    *,
    input_mode: str | None,
    images: Sequence[Any] = (),
) -> list[dict[str, Any]]:
    """Build chat content containing exactly the selected input modalities."""
    mode = normalize_input_mode(input_mode)
    content: list[dict[str, Any]] = []
    if input_mode_uses_images(mode):
        if len(images) != 2:
            raise ValueError(
                f"{mode} input_mode requires exactly two ordered NMR images"
            )
        for image in images:
            item: dict[str, Any] = {"type": "image"}
            if image is not None:
                item["image"] = image
            content.append(item)
    elif images:
        raise ValueError(f"{mode} input_mode must not receive images")
    content.append({"type": "text", "text": str(prompt)})
    return content
