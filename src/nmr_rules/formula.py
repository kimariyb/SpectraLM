"""Molecular-formula constraints for one-dimensional NMR analysis."""

from __future__ import annotations

import re


_FORMULA_TOKEN = re.compile(r"([A-Z][a-z]?)(\d*)")
_SUPPORTED_ELEMENTS = {
    "C",
    "H",
    "N",
    "O",
    "F",
    "Si",
    "P",
    "S",
    "Cl",
    "Br",
    "I",
}
_HALOGENS = {"F", "Cl", "Br", "I"}


def parse_formula(formula: str) -> dict[str, int]:
    """Parse a single neutral molecular formula.

    Parameters
    ----------
    formula
        Molecular formula such as ``C8H10O``.

    Returns
    -------
    dict[str, int]
        Element counts.

    Raises
    ------
    ValueError
        If the input is charged, disconnected, malformed, or contains an
        unsupported element.
    """
    text = str(formula).strip()
    if not text or any(marker in text for marker in (".", "+", "-")):
        raise ValueError("Expected a single neutral molecular formula.")

    counts: dict[str, int] = {}
    position = 0
    for match in _FORMULA_TOKEN.finditer(text):
        if match.start() != position:
            raise ValueError("Expected a single neutral molecular formula.")
        element, raw_count = match.groups()
        if element not in _SUPPORTED_ELEMENTS:
            raise ValueError(f"Unsupported element in molecular formula: {element}")
        count = int(raw_count) if raw_count else 1
        if count <= 0:
            raise ValueError("Element counts must be positive integers.")
        counts[element] = counts.get(element, 0) + count
        position = match.end()

    if position != len(text) or not counts:
        raise ValueError("Expected a single neutral molecular formula.")
    return counts


def calculate_dbe(formula: str | None) -> float | None:
    """Calculate double-bond equivalents for a neutral molecular formula.

    Carbon and silicon use valence four; nitrogen and phosphorus use valence
    three; oxygen and divalent sulfur contribute zero; halogens count as
    hydrogen. Missing formulae intentionally return ``None`` so the same rule
    engine can support formula-free inference.

    Parameters
    ----------
    formula
        Optional molecular formula.

    Returns
    -------
    float or None
        DBE, or ``None`` when no formula is supplied.
    """
    if formula is None or not str(formula).strip():
        return None
    counts = parse_formula(str(formula))
    carbon = counts.get("C", 0)
    silicon = counts.get("Si", 0)
    hydrogen = counts.get("H", 0)
    nitrogen = counts.get("N", 0)
    phosphorus = counts.get("P", 0)
    halogens = sum(counts.get(element, 0) for element in _HALOGENS)
    dbe = (
        2 * carbon
        + 2 * silicon
        + 2
        + nitrogen
        + phosphorus
        - hydrogen
        - halogens
    ) / 2.0
    doubled = dbe * 2.0
    if dbe < 0 or abs(doubled - round(doubled)) > 1e-8:
        raise ValueError("DBE must be a non-negative integer or half-integer.")
    return float(dbe)
