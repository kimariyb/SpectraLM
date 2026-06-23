"""Reusable exact, micro, macro, and per-class multilabel metrics."""

from __future__ import annotations

from typing import Any

import numpy as np


def _precision_recall_f1(
    true_positive: int,
    false_positive: int,
    false_negative: int,
) -> tuple[float, float, float]:
    precision = (
        true_positive / (true_positive + false_positive)
        if true_positive + false_positive
        else 0.0
    )
    recall = (
        true_positive / (true_positive + false_negative)
        if true_positive + false_negative
        else 0.0
    )
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )
    return precision, recall, f1


def summarize_multilabel_predictions(
    predicted: list[set[str] | frozenset[str]],
    reference: list[set[str] | frozenset[str]],
    *,
    label_space: list[str] | tuple[str, ...] | None = None,
) -> dict[str, Any]:
    """Summarize exact, micro, macro, and per-class multilabel metrics."""
    if len(predicted) != len(reference):
        raise ValueError("predicted and reference must have equal lengths")
    predicted_sets = [set(labels) for labels in predicted]
    reference_sets = [set(labels) for labels in reference]
    labels = sorted(
        set(label_space or ())
        | {label for values in predicted_sets for label in values}
        | {label for values in reference_sets for label in values}
    )
    per_class: dict[str, dict[str, float | int]] = {}
    total_tp = total_fp = total_fn = 0
    supported_f1: list[float] = []
    for label in labels:
        paired_sets = zip(predicted_sets, reference_sets)
        tp = sum(label in pred and label in ref for pred, ref in paired_sets)
        paired_sets = zip(predicted_sets, reference_sets)
        fp = sum(label in pred and label not in ref for pred, ref in paired_sets)
        paired_sets = zip(predicted_sets, reference_sets)
        fn = sum(label not in pred and label in ref for pred, ref in paired_sets)
        precision, recall, f1 = _precision_recall_f1(tp, fp, fn)
        support = tp + fn
        per_class[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        }
        total_tp += tp
        total_fp += fp
        total_fn += fn
        if support:
            supported_f1.append(f1)

    micro_precision, micro_recall, micro_f1 = _precision_recall_f1(
        total_tp,
        total_fp,
        total_fn,
    )
    total = len(reference_sets)
    return {
        "samples": total,
        "multilabel_exact_match": (
            sum(
                pred == ref
                for pred, ref in zip(predicted_sets, reference_sets)
            )
            / total
            if total
            else None
        ),
        "multilabel_micro_precision": micro_precision,
        "multilabel_micro_recall": micro_recall,
        "multilabel_micro_f1": micro_f1,
        "multilabel_macro_f1": (
            float(np.mean([row["f1"] for row in per_class.values()]))
            if per_class
            else None
        ),
        "multilabel_supported_macro_f1": (
            float(np.mean(supported_f1)) if supported_f1 else None
        ),
        "multilabel_per_class": per_class,
    }
