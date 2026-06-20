"""Soft functional-group consistency checks against observed 1D NMR regions."""

from __future__ import annotations

from typing import Any

from src.data.functional_groups import functional_groups
from src.data.molecules import canonicalize_smiles
from src.nmr_rules.engine import _c13_shifts, _h1_peaks


def evaluate_functional_group_spectral_consistency(
    smiles: str | None,
    sample: dict[str, Any],
) -> dict[str, Any]:
    """Evaluate checkable predicted functional groups against soft 1D evidence.

    Parameters
    ----------
    smiles
        Predicted molecular structure.
    sample
        Sample-visible one-dimensional peak tables.

    Returns
    -------
    dict[str, Any]
        Per-group checks, counts, and optional consistency rate.
    """
    canonical = canonicalize_smiles(smiles)
    if canonical is None:
        return {
            "spectral_functional_group_checks": {},
            "spectral_functional_groups_applicable": 0,
            "spectral_functional_groups_supported": 0,
            "functional_group_spectral_consistency": None,
        }

    groups = functional_groups(canonical)
    h_shifts = [peak["shift"] for peak in _h1_peaks(sample)]
    c_shifts = _c13_shifts(sample)

    def h_between(low: float, high: float) -> bool:
        return any(low <= shift <= high for shift in h_shifts)

    def c_between(low: float, high: float) -> bool:
        return any(low <= shift <= high for shift in c_shifts)

    checks: dict[str, bool] = {}
    for group in sorted(groups):
        if group in {"alcohol", "ether", "amine"}:
            checks[group] = h_between(3.0, 4.6) or c_between(45.0, 90.0)
        elif group == "aldehyde":
            checks[group] = h_between(9.0, 10.6) and c_between(185.0, 220.0)
        elif group == "ketone":
            checks[group] = c_between(185.0, 220.0)
        elif group in {"carboxylic_acid", "ester", "amide"}:
            checks[group] = c_between(160.0, 185.0)
        elif group == "alkene":
            checks[group] = h_between(4.5, 6.8) or c_between(100.0, 165.0)
        elif group == "aromatic_ring":
            checks[group] = h_between(6.0, 9.0) or c_between(100.0, 165.0)
        elif group == "alkyne":
            checks[group] = c_between(65.0, 100.0)
        elif group == "nitrile":
            checks[group] = c_between(105.0, 130.0)

    applicable = len(checks)
    supported = sum(checks.values())
    return {
        "spectral_functional_group_checks": checks,
        "spectral_functional_groups_applicable": applicable,
        "spectral_functional_groups_supported": supported,
        "functional_group_spectral_consistency": (
            supported / applicable if applicable else None
        ),
    }
