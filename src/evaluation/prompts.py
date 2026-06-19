"""Prompts for direct molecular-structure prediction from paired NMR data."""

from __future__ import annotations

from typing import Any

from src.data.molecules import molecule_formula


STRUCTURE_PROMPTS: list[str] = [
    (
        "Given the 1H and 13C NMR data below, determine the molecular "
        "structure.\n\n{peak_tables}\n\nOutput the canonical SMILES of the "
        "molecule."
    ),
    (
        "Elucidate the structure of the compound from the following NMR "
        "data.\n\n{peak_tables}\n\nReturn only the canonical SMILES."
    ),
    (
        "You are an expert NMR spectroscopist. Deduce the molecular "
        "structure from the data below.\n\n{peak_tables}\n\nProvide the "
        "canonical SMILES as your answer."
    ),
    (
        "Solve the structure of this unknown compound using its 1H and 13C "
        "NMR data.\n\n{peak_tables}\n\nAnswer with the canonical SMILES."
    ),
]


def build_structure_prompt(
    sample: dict[str, Any],
    prompt: str,
    *,
    include_formula: bool = True,
) -> str:
    """Fill one structure prompt with peak tables and optional formula."""
    formula = molecule_formula(sample.get("canonical_smiles")) or "unknown"
    peak_tables = _format_peak_tables(
        sample,
        formula if include_formula else None,
    )
    return prompt.format(peak_tables=peak_tables)


def _format_peak_tables(sample: dict[str, Any], formula: str | None) -> str:
    """Build formatted proton and carbon peak tables."""
    parts: list[str] = []
    if formula is not None:
        parts.append(f"Molecular formula: {formula}")

    h_nmr = sample.get("1H_NMR", {})
    parts.append("\n## 1H NMR Peak Table")
    parts.append(
        f"  Solvent: {h_nmr.get('solvent', 'unknown')}  |  "
        f"Frequency: {h_nmr.get('frequency', 'unknown')}"
    )
    parts.append(f"  {'Shift (ppm)':<12} {'Mult.':<8} {'J (Hz)':<16} {'Integ.'}")
    parts.append(f"  {'-' * 12} {'-' * 8} {'-' * 16} {'-' * 6}")

    for peak in h_nmr.get("peaks", []):
        if isinstance(peak, dict):
            shift = float(peak["shift"])
            mult = str(peak.get("multiplicity", "s"))
            j_values = peak.get("J", [])
            integration = float(peak.get("integration", 1))
        else:
            shift, mult, j_values, integration = peak
            shift = float(shift)
            integration = float(integration)
        j_text = ", ".join(f"{j:.1f}" for j in j_values) if j_values else "-"
        parts.append(
            f"  {shift:<12.2f} {mult:<8} {j_text:<16} {integration:.0f}"
        )

    c_nmr = sample.get("13C_NMR", {})
    parts.append("\n## 13C NMR Peak Table")
    parts.append(
        f"  Solvent: {c_nmr.get('solvent', 'unknown')}  |  "
        f"Frequency: {c_nmr.get('frequency', 'unknown')}"
    )
    parts.append(f"  {'Shift (ppm)':<12}")
    parts.append(f"  {'-' * 12}")
    for peak in c_nmr.get("peaks", []):
        shift = peak["shift"] if isinstance(peak, dict) else peak
        if isinstance(shift, (list, tuple)):
            parts.append(f"  {', '.join(f'{float(v):.2f}' for v in shift)}")
        else:
            parts.append(f"  {float(shift):.2f}")
    return "\n".join(parts)
