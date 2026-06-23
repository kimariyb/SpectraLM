"""Generation termination, repetition, and collapse diagnostics."""

from __future__ import annotations

from collections import Counter
from typing import Any

import numpy as np


def inspect_generation_tokens(
    token_ids: list[int],
    *,
    eos_token_ids: set[int],
    max_new_tokens: int,
) -> dict[str, int | bool]:
    """Classify one generated token sequence for termination and repetition."""
    ids = [int(token_id) for token_id in token_ids]
    eos_positions = [
        index
        for index, token_id in enumerate(ids)
        if token_id in eos_token_ids
    ]
    terminated = bool(eos_positions)
    if eos_positions:
        ids = ids[: eos_positions[0] + 1]
    ngram_counts = Counter(
        tuple(ids[index : index + 4])
        for index in range(max(0, len(ids) - 3))
    )
    repeated = any(count >= 3 for count in ngram_counts.values())
    return {
        "generated_token_count": len(ids),
        "generation_terminated_by_eos": terminated,
        "generation_hit_max_tokens": (
            not terminated and len(ids) >= int(max_new_tokens)
        ),
        "generation_repeated_4gram": repeated,
    }


def summarize_generation_behavior(
    predictions: list[str],
    traces: list[dict[str, Any]],
) -> dict[str, float | int | None]:
    """Aggregate generation termination, repetition, and collapse diagnostics."""
    if len(predictions) != len(traces):
        raise ValueError("predictions and traces must have equal lengths")
    total = len(predictions)
    if total == 0:
        return {
            "mean_generated_tokens": None,
            "median_generated_tokens": None,
            "eos_termination_rate": None,
            "generation_truncation_rate": None,
            "repeated_4gram_rate": None,
            "unique_prediction_rate": None,
            "duplicate_prediction_rate": None,
        }
    token_counts = np.asarray(
        [int(trace["generated_token_count"]) for trace in traces],
        dtype=np.int64,
    )
    normalized_predictions = [
        str(prediction).strip() for prediction in predictions
    ]
    unique_count = len(set(normalized_predictions))
    return {
        "mean_generated_tokens": float(np.mean(token_counts)),
        "median_generated_tokens": float(np.median(token_counts)),
        "eos_termination_rate": sum(
            bool(trace["generation_terminated_by_eos"]) for trace in traces
        )
        / total,
        "generation_truncation_rate": sum(
            bool(trace["generation_hit_max_tokens"]) for trace in traces
        )
        / total,
        "repeated_4gram_rate": sum(
            bool(trace["generation_repeated_4gram"]) for trace in traces
        )
        / total,
        "unique_prediction_rate": unique_count / total,
        "duplicate_prediction_rate": (total - unique_count) / total,
    }
