"""Task definitions for configurable one-model NMR multitask SFT."""

from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any, Mapping, Sequence

from src.data.functional_groups import FUNCTIONAL_GROUP_SMARTS, functional_groups
from src.data.molecules import canonicalize_smiles, sample_smiles
from src.data.spectral_regions import classify_spectral_regions
from src.data.modalities import FULL, validate_input_configuration
from src.evaluation.prompts import (
    build_structure_prompt,
    select_structure_prompt,
)


STRUCTURE_PREDICTION = "structure_prediction"
FUNCTIONAL_GROUP_RECOGNITION = "functional_group_recognition"
CANDIDATE_RANKING = "candidate_ranking"
SPECTRAL_REGION_CLASSIFICATION = "spectral_region_classification"

SUPPORTED_TASKS = (
    STRUCTURE_PREDICTION,
    FUNCTIONAL_GROUP_RECOGNITION,
    CANDIDATE_RANKING,
    SPECTRAL_REGION_CLASSIFICATION,
)
DEFAULT_TASK_WEIGHTS = {STRUCTURE_PREDICTION: 1.0}


@dataclass(frozen=True)
class TaskExample:
    """One task-specific text prompt and supervised target."""

    task: str
    prompt: str
    target: str


def normalize_task_weights(
    task_weights: Mapping[str, float] | None,
) -> dict[str, float]:
    """Validate and normalize multitask sampling weights.

    Parameters
    ----------
    task_weights
        Non-negative weights keyed by supported task name.

    Returns
    -------
    dict[str, float]
        Positive weights normalized to sum to one.
    """
    raw = dict(task_weights or DEFAULT_TASK_WEIGHTS)
    unknown = sorted(set(raw) - set(SUPPORTED_TASKS))
    if unknown:
        raise ValueError(f"Unsupported auxiliary task: {', '.join(unknown)}")
    if any(float(value) < 0 for value in raw.values()):
        raise ValueError("Task weights must be non-negative.")
    positive = {
        task: float(raw.get(task, 0.0))
        for task in SUPPORTED_TASKS
        if float(raw.get(task, 0.0)) > 0
    }
    total = sum(positive.values())
    if total <= 0:
        raise ValueError("At least one task weight must be positive.")
    return {task: weight / total for task, weight in positive.items()}


def _task_prompt(
    sample: dict[str, Any],
    template: str,
    *,
    include_formula: bool,
    include_rule_context: bool,
    max_rule_evidence: int,
    input_mode: str,
) -> str:
    return build_structure_prompt(
        sample,
        template,
        include_formula=include_formula,
        include_rule_context=include_rule_context,
        max_rule_evidence=max_rule_evidence,
        input_mode=input_mode,
    )


def build_task_example(
    sample: dict[str, Any],
    task: str,
    *,
    candidates: Sequence[str] | None = None,
    structure_prompt: str | None = None,
    include_formula: bool = True,
    include_rule_context: bool = False,
    max_rule_evidence: int = 12,
    input_mode: str = FULL,
) -> TaskExample:
    """Build one prompt and target for a supported NMR supervision task.

    Parameters
    ----------
    sample
        Normalized paired-NMR sample.
    task
        One name from ``SUPPORTED_TASKS``.
    candidates
        Candidate SMILES for the ranking task.
    structure_prompt
        Optional selected direct-structure prompt template.
    include_formula
        Whether to include the explicit molecular formula.
    include_rule_context
        Whether to append structured one-dimensional rule evidence.
    max_rule_evidence
        Maximum rule evidence lines.

    Returns
    -------
    TaskExample
        Task prompt and exact supervised target.
    """
    if task not in SUPPORTED_TASKS:
        raise ValueError(f"Unsupported auxiliary task: {task}")
    input_mode = validate_input_configuration(
        input_mode,
        include_formula=include_formula,
        include_rule_context=include_rule_context,
        task_names=(task,),
    )
    target = canonicalize_smiles(sample_smiles(sample))
    if target is None:
        raise ValueError("Task sample requires a valid target structure.")

    if task == STRUCTURE_PREDICTION:
        template = structure_prompt or select_structure_prompt(
            0,
            input_mode=input_mode,
        )
        return TaskExample(
            task=task,
            prompt=_task_prompt(
                sample,
                template,
                include_formula=include_formula,
                include_rule_context=include_rule_context,
                max_rule_evidence=max_rule_evidence,
                input_mode=input_mode,
            ),
            target=target,
        )

    if task == FUNCTIONAL_GROUP_RECOGNITION:
        labels = ", ".join(label for label, _ in FUNCTIONAL_GROUP_SMARTS)
        template = (
            "The first image is a 1H NMR spectrum and the second image is a "
            "13C NMR spectrum. Use the images and peak tables below to identify "
            "the functional groups present in the molecule. Choose labels only "
            f"from this ontology: {labels}.\n\n{{peak_tables}}\n\n"
            "Return only one sorted JSON array of unique functional group labels."
        )
        return TaskExample(
            task=task,
            prompt=_task_prompt(
                sample,
                template,
                include_formula=include_formula,
                include_rule_context=include_rule_context,
                max_rule_evidence=max_rule_evidence,
                input_mode=input_mode,
            ),
            target=json.dumps(
                sorted(functional_groups(target)),
                ensure_ascii=True,
                separators=(",", ":"),
            ),
        )

    if task == SPECTRAL_REGION_CLASSIFICATION:
        template = (
            "The first image is a 1H NMR spectrum and the second image is a "
            "13C NMR spectrum. Classify all observed signals into the controlled "
            "overlapping spectral region labels represented by the peak tables."
            "\n\n{peak_tables}\n\nReturn only a JSON object with keys \"1H\" "
            "and \"13C\" and sorted unique label arrays."
        )
        return TaskExample(
            task=task,
            prompt=_task_prompt(
                sample,
                template,
                include_formula=include_formula,
                include_rule_context=False,
                max_rule_evidence=max_rule_evidence,
                input_mode=input_mode,
            ),
            target=json.dumps(
                classify_spectral_regions(sample),
                ensure_ascii=True,
                separators=(",", ":"),
            ),
        )

    canonical_candidates: list[str] = []
    for candidate in candidates or []:
        canonical = canonicalize_smiles(candidate)
        if canonical is not None and canonical not in canonical_candidates:
            canonical_candidates.append(canonical)
    if target not in canonical_candidates:
        raise ValueError("Candidate set does not contain the target structure.")
    candidate_lines = "\n".join(
        f"{index}. {candidate}"
        for index, candidate in enumerate(canonical_candidates, start=1)
    )
    template = (
        "The first image is a 1H NMR spectrum and the second image is a 13C "
        "NMR spectrum. Select the best candidate structure using all visual "
        "and tabulated spectral evidence.\n\n{peak_tables}\n\nCandidates:\n"
        f"{candidate_lines}\n\nReturn only the canonical SMILES of the best candidate."
    )
    return TaskExample(
        task=task,
        prompt=_task_prompt(
            sample,
            template,
            include_formula=include_formula,
            include_rule_context=include_rule_context,
            max_rule_evidence=max_rule_evidence,
            input_mode=input_mode,
        ),
        target=target,
    )
