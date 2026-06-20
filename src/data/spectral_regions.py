"""Deterministic multi-label classification of observed 1D NMR regions."""

from __future__ import annotations

from typing import Any

from src.nmr_rules.engine import _c13_shifts, _h1_peaks, load_rule_library


def classify_spectral_regions(sample: dict[str, Any]) -> dict[str, list[str]]:
    """Classify observed 1H and 13C peaks into overlapping soft regions.

    Parameters
    ----------
    sample
        Normalized sample with one-dimensional peak tables.

    Returns
    -------
    dict[str, list[str]]
        Stable region-rule IDs for each nucleus.
    """
    library = load_rule_library()
    h_shifts = [peak["shift"] for peak in _h1_peaks(sample)]
    c_shifts = _c13_shifts(sample)

    def matching_ids(rules: list[dict[str, Any]], shifts: list[float]) -> list[str]:
        return [
            str(rule["id"])
            for rule in rules
            if any(
                float(rule["min_ppm"]) <= shift <= float(rule["max_ppm"])
                for shift in shifts
            )
        ]

    return {
        "1H": matching_ids(library.get("h1_shift_regions", []), h_shifts),
        "13C": matching_ids(library.get("c13_shift_regions", []), c_shifts),
    }
