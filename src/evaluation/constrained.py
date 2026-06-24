"""Hard-constraint filtering and evaluation for generated structures."""

from __future__ import annotations

from dataclasses import dataclass
from statistics import mean
from typing import Any, Iterable, Mapping

from src.data.molecules import (
    canonicalize_connectivity_smiles,
    canonicalize_smiles,
    inspect_dataset_molecule,
    molecule_formula,
)


@dataclass(frozen=True)
class CandidateFilterResult:
    """Candidate sets retained by successive chemical constraints."""

    raw_count: int
    unique_count: int
    domain_valid_candidates: tuple[str, ...]
    formula_valid_candidates: tuple[str, ...]
    formula_constraint_applicable: bool
    formula_constraint_failed: bool

    @property
    def selectable_candidates(self) -> tuple[str, ...]:
        """Return candidates that may be passed to the ranker."""
        if self.formula_constraint_applicable:
            return self.formula_valid_candidates
        return self.domain_valid_candidates


@dataclass(frozen=True)
class RankedSelection:
    """Final candidate selection and ranker compliance state."""

    prediction: str
    ranking_failed: bool


def filter_generated_candidates(
    generated_candidates: Iterable[str],
    *,
    molecular_formula: str | None,
) -> CandidateFilterResult:
    """Deduplicate candidates and enforce domain and formula constraints.

    Isotope-labelled predictions are rejected rather than silently normalized,
    matching the model-output policy used by the evaluation metrics.
    """
    raw = [str(candidate).strip() for candidate in generated_candidates]
    domain_valid: list[str] = []
    seen: set[str] = set()
    for candidate in raw:
        inspection = inspect_dataset_molecule(candidate)
        if not inspection.accepted or inspection.isotope_label_count:
            continue
        connectivity = canonicalize_connectivity_smiles(
            inspection.canonical_smiles
        )
        if connectivity is None or connectivity in seen:
            continue
        seen.add(connectivity)
        domain_valid.append(connectivity)

    formula = str(molecular_formula or "").strip() or None
    formula_valid = [
        candidate
        for candidate in domain_valid
        if formula is not None and molecule_formula(candidate) == formula
    ]
    applicable = formula is not None
    failed = applicable and not formula_valid
    return CandidateFilterResult(
        raw_count=len(raw),
        unique_count=len(domain_valid),
        domain_valid_candidates=tuple(domain_valid),
        formula_valid_candidates=tuple(formula_valid),
        formula_constraint_applicable=applicable,
        formula_constraint_failed=failed,
    )


def resolve_ranked_candidate(
    candidates: tuple[str, ...],
    ranking_response: str,
) -> RankedSelection:
    """Resolve a ranker response without permitting out-of-set structures."""
    if not candidates:
        return RankedSelection(prediction="", ranking_failed=True)
    ranked = canonicalize_connectivity_smiles(ranking_response)
    if ranked in candidates:
        return RankedSelection(prediction=ranked, ranking_failed=False)
    return RankedSelection(prediction=candidates[0], ranking_failed=True)


def summarize_constrained_predictions(
    rows: Iterable[Mapping[str, Any]],
) -> dict[str, float | int | None]:
    """Aggregate constrained-generation coverage, oracle, and ranking metrics."""
    records = list(rows)
    total = len(records)
    if total == 0:
        return {"num_examples": 0}

    applicable = [
        row for row in records if bool(row["formula_constraint_applicable"])
    ]
    ranking_attempts = [
        row for row in records if bool(row.get("ranking_attempted", False))
    ]

    def rate(key: str, selected: list[Mapping[str, Any]] = records) -> float:
        return mean(float(bool(row.get(key, False))) for row in selected)

    formula_coverage = (
        mean(
            float(not bool(row["formula_constraint_failed"]))
            for row in applicable
        )
        if applicable
        else None
    )
    return {
        "num_examples": total,
        "formula_constraint_applicable_rate": len(applicable) / total,
        "formula_constraint_coverage": formula_coverage,
        "formula_constraint_failure_rate": rate(
            "formula_constraint_failed"
        ),
        "mean_raw_candidate_count": mean(
            float(row.get("raw_candidate_count", 0)) for row in records
        ),
        "mean_unique_candidate_count": mean(
            float(row.get("unique_candidate_count", 0)) for row in records
        ),
        "mean_domain_valid_candidate_count": mean(
            float(row.get("domain_valid_candidate_count", 0))
            for row in records
        ),
        "mean_formula_valid_candidate_count": mean(
            float(row.get("formula_valid_candidate_count", 0))
            for row in records
        ),
        "candidate_oracle_exact_at_32": rate("candidate_oracle_exact"),
        "candidate_oracle_connectivity_at_32": rate(
            "candidate_oracle_connectivity"
        ),
        "ranking_attempt_rate": len(ranking_attempts) / total,
        "ranking_failure_rate": (
            rate("ranking_failed", ranking_attempts)
            if ranking_attempts
            else 0.0
        ),
        "ranked_top1_exact_match": rate("exact_match"),
        "ranked_top1_connectivity_exact_match": rate(
            "connectivity_exact_match"
        ),
    }
