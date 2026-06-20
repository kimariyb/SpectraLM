"""Deterministic evidence generation from 1H and 13C NMR peak tables."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

from src.nmr_rules.formula import calculate_dbe, parse_formula
from src.nmr_rules.models import RuleAnalysis, RuleEvidence


DEFAULT_RULE_PATH = Path(__file__).resolve().parents[2] / "rules" / "nmr_1d.yaml"


@lru_cache(maxsize=4)
def load_rule_library(path: str | Path = DEFAULT_RULE_PATH) -> dict[str, Any]:
    """Load and validate the one-dimensional NMR rule library.

    Parameters
    ----------
    path
        YAML rule-library path.

    Returns
    -------
    dict[str, Any]
        Parsed rule library.
    """
    rule_path = Path(path)
    payload = yaml.safe_load(rule_path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict) or not payload.get("library_name"):
        raise ValueError(f"Invalid NMR rule library: {rule_path}")
    return payload


def _multiplicity(value: Any) -> str:
    return str(value or "s").strip().lower().replace(" ", "")


def _h1_peaks(sample: dict[str, Any]) -> list[dict[str, Any]]:
    peaks: list[dict[str, Any]] = []
    for raw in sample.get("1H_NMR", {}).get("peaks", []):
        if isinstance(raw, dict):
            try:
                peaks.append(
                    {
                        "shift": float(raw["shift"]),
                        "multiplicity": _multiplicity(raw.get("multiplicity", "s")),
                        "J": tuple(float(value) for value in raw.get("J", [])),
                        "integration": float(raw.get("integration", 1.0)),
                    }
                )
            except (KeyError, TypeError, ValueError):
                continue
        else:
            try:
                shift, multiplicity, couplings, integration = raw
                peaks.append(
                    {
                        "shift": float(shift),
                        "multiplicity": _multiplicity(multiplicity),
                        "J": tuple(float(value) for value in couplings),
                        "integration": float(integration),
                    }
                )
            except (TypeError, ValueError):
                continue
    return peaks


def _c13_shifts(sample: dict[str, Any]) -> list[float]:
    shifts: list[float] = []
    for raw in sample.get("13C_NMR", {}).get("peaks", []):
        value = raw.get("shift") if isinstance(raw, dict) else raw
        values = value if isinstance(value, (list, tuple)) else [value]
        for item in values:
            try:
                shifts.append(float(item))
            except (TypeError, ValueError):
                continue
    return shifts


def _rule_evidence(rule: dict[str, Any], metadata: dict[str, Any]) -> RuleEvidence:
    return RuleEvidence(
        rule_id=str(rule["id"]),
        category=str(rule.get("category") or str(rule["id"]).split("_", 1)[0].lower()),
        conclusion=str(rule["conclusion"]),
        confidence=float(rule["confidence"]),
        strength=str(rule["strength"]),
        human_tip=str(rule["human_tip"]),
        metadata=metadata,
    )


def _matching_j(first: dict[str, Any], second: dict[str, Any], tolerance: float) -> float | None:
    matches = [
        (left + right) / 2.0
        for left in first["J"]
        for right in second["J"]
        if abs(left - right) <= tolerance
    ]
    return min(matches) if matches else None


def _matches_peak(peak: dict[str, Any], spec: dict[str, Any], tolerance: float) -> bool:
    if abs(peak["integration"] - float(spec["integration"])) > tolerance:
        return False
    return peak["multiplicity"] in {
        _multiplicity(value) for value in spec.get("multiplicities", [])
    }


def _fragment_evidence(
    peaks: list[dict[str, Any]],
    rule: dict[str, Any],
) -> RuleEvidence | None:
    rule_type = rule.get("type")
    tolerance = float(rule.get("integration_tolerance", 0.5))
    if rule_type == "single_peak":
        spec = {
            "integration": rule["integration"],
            "multiplicities": rule["multiplicities"],
        }
        for peak in peaks:
            if not _matches_peak(peak, spec, tolerance):
                continue
            if "min_ppm" in rule and peak["shift"] < float(rule["min_ppm"]):
                continue
            if "max_ppm" in rule and peak["shift"] > float(rule["max_ppm"]):
                continue
            return _rule_evidence(
                rule,
                {
                    "shift_ppm": peak["shift"],
                    "integration": peak["integration"],
                    "multiplicity": peak["multiplicity"],
                },
            )

    if rule_type == "paired_multiplets":
        first_candidates = [
            peak for peak in peaks if _matches_peak(peak, rule["first"], tolerance)
        ]
        second_candidates = [
            peak for peak in peaks if _matches_peak(peak, rule["second"], tolerance)
        ]
        j_tolerance = float(rule.get("matching_j_tolerance_hz", 0.5))
        for first in first_candidates:
            for second in second_candidates:
                matching_j = _matching_j(first, second, j_tolerance)
                if matching_j is not None:
                    return _rule_evidence(
                        rule,
                        {
                            "matching_j_hz": round(matching_j, 3),
                            "shifts_ppm": [first["shift"], second["shift"]],
                            "integrations": [
                                first["integration"],
                                second["integration"],
                            ],
                        },
                    )
    return None


def analyze_sample(
    sample: dict[str, Any],
    *,
    include_formula: bool = True,
    rule_path: str | Path = DEFAULT_RULE_PATH,
) -> RuleAnalysis:
    """Generate auditable evidence from one 1H/13C NMR sample.

    The function never derives molecular formula from a target SMILES. It
    intentionally has no solvent-peak or multidimensional-NMR rules.

    Parameters
    ----------
    sample
        Normalized sample with optional ``molecular_formula`` and 1D peak
        tables.
    include_formula
        Whether supplied molecular-formula constraints may be used.
    rule_path
        YAML rule-library path.

    Returns
    -------
    RuleAnalysis
        Ordered evidence and warnings.
    """
    library = load_rule_library(rule_path)
    formula_value = sample.get("molecular_formula") if include_formula else None
    formula = str(formula_value).strip() if formula_value else None
    evidence: list[RuleEvidence] = []
    warnings: list[str] = []
    dbe: float | None = None
    formula_counts: dict[str, int] = {}

    if formula is not None:
        try:
            formula_counts = parse_formula(formula)
            dbe = calculate_dbe(formula)
            evidence.append(
                RuleEvidence(
                    rule_id="FORMULA_DBE_001",
                    category="formula",
                    conclusion=f"Molecular formula {formula} gives DBE = {dbe:g}.",
                    confidence=1.0,
                    strength="hard",
                    human_tip="Account for all unsaturation before accepting a structure.",
                    metadata={"formula": formula, "dbe": dbe},
                )
            )
        except ValueError as exc:
            warnings.append(f"Formula constraints skipped: {exc}")
            formula = None

    h_peaks = _h1_peaks(sample)
    c_shifts = _c13_shifts(sample)
    if not h_peaks:
        warnings.append("No valid 1H NMR peaks were available for rule analysis.")
    if not c_shifts:
        warnings.append("No valid 13C NMR peaks were available for rule analysis.")

    carbon_count = formula_counts.get("C")
    if carbon_count is not None and c_shifts:
        observed = len({round(value, 3) for value in c_shifts})
        if observed <= carbon_count:
            evidence.append(
                RuleEvidence(
                    rule_id="C13_SIGNAL_COUNT_001",
                    category="cross_spectrum",
                    conclusion=(
                        f"The {observed} distinct 13C signals are compatible with "
                        f"the {carbon_count} carbons in the supplied formula."
                    ),
                    confidence=0.85,
                    strength="moderate",
                    human_tip=(
                        "Fewer signals than carbons can reflect symmetry, accidental "
                        "overlap, or weak signals; signal intensity is not quantitative."
                    ),
                    metadata={
                        "observed_c13_signals": observed,
                        "formula_carbon_count": carbon_count,
                    },
                )
            )
        else:
            warnings.append(
                f"Observed {observed} distinct 13C signals for a formula with only "
                f"{carbon_count} carbons."
            )

    for rule in library.get("h1_shift_regions", []):
        matching = sorted(
            peak["shift"]
            for peak in h_peaks
            if float(rule["min_ppm"]) <= peak["shift"] <= float(rule["max_ppm"])
        )
        if matching:
            evidence.append(_rule_evidence(rule, {"matching_shifts_ppm": matching}))

    for rule in library.get("c13_shift_regions", []):
        matching = sorted(
            shift
            for shift in c_shifts
            if float(rule["min_ppm"]) <= shift <= float(rule["max_ppm"])
        )
        if matching:
            evidence.append(_rule_evidence(rule, {"matching_shifts_ppm": matching}))

    for rule in library.get("fragment_rules", []):
        match = _fragment_evidence(h_peaks, rule)
        if match is not None:
            evidence.append(match)

    return RuleAnalysis(
        library_name=str(library["library_name"]),
        molecular_formula=formula,
        dbe=dbe,
        evidence=tuple(evidence),
        warnings=tuple(warnings),
    )
