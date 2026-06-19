"""Prompt builders and target generators for NMR-to-structure prediction.

The module provides:
- Prompt templates for structure and functional-group tasks.
- A builder that injects peak tables and molecular formula.
- Simple target generators (SMILES or functional-group list, no reasoning).
"""

from __future__ import annotations

from typing import Any

from src.data.molecules import molecule_formula


NMR_RULE_HINTS: str = """NMR rules to consider:
- 1H integration constrains the relative number of equivalent protons.
- 1H multiplicity and J values suggest local proton-neighbor environments.
- 13C peak count constrains the number of magnetically distinct carbons.
- Chemical-shift regions suggest functional groups and hybridization states."""


# Prompt templates — structure prediction
STRUCTURE_PROMPTS: list[str] = [
    (
        "Given the 1H and 13C NMR data and molecular formula below, "
        "determine the molecular structure.\n\n"
        "{peak_tables}\n\n"
        "Output the canonical SMILES of the molecule."
    ),
    (
        "Elucidate the structure of the compound with the following "
        "NMR spectra and molecular formula.\n\n"
        "{peak_tables}\n\n"
        "Return only the canonical SMILES."
    ),
    (
        "You are an expert NMR spectroscopist.  Deduce the molecular "
        "structure from the data below.\n\n"
        "{peak_tables}\n\n"
        "Provide the canonical SMILES as your answer."
    ),
    (
        "Solve the structure of this unknown compound using its 1H and "
        "13C NMR data and molecular formula.\n\n"
        "{peak_tables}\n\n"
        "Answer with the canonical SMILES."
    ),
]

# Prompt templates — functional group identification
FUNCTIONAL_GROUP_PROMPTS: list[str] = [
    (
        "Examine the 1H and 13C NMR spectra and molecular formula below "
        "and list ALL functional groups present in the molecule.\n\n"
        "{peak_tables}\n\n"
        "Return a comma-separated list of functional group names "
        "(e.g., alcohol, ketone, aromatic, ester, amine, carboxylic acid, "
        "ether, alkene, alkyne, amide, aldehyde, nitro, nitrile, halide)."
    ),
    (
        "Based on the following NMR data and molecular formula, identify "
        "every functional group in this compound.\n\n"
        "{peak_tables}\n\n"
        "List them as a comma-separated sequence."
    ),
]


def build_structure_prompt(
    sample: dict[str, Any],
    prompt: str,
    *,
    include_formula: bool = True,
    include_rules: bool = False,
) -> str:
    """Build a complete prompt with peak tables and molecular formula.

    Parameters
    ----------
    sample
        Sample dictionary containing ``1H_NMR``, ``13C_NMR``, and
        ``canonical_smiles`` keys.
    prompt
        Template string with a ``{peak_tables}`` placeholder.
    include_formula
        Whether to include molecular formula in the prompt.  Disable this
        for image-only or formula-free ablations.
    include_rules
        Whether to append explicit NMR interpretation hints.

    Returns
    -------
    str
        Filled-in prompt text.
    """
    formula = molecule_formula(sample.get("canonical_smiles")) or "unknown"
    peak_tables = _format_peak_tables(
        sample,
        formula if include_formula else None,
    )
    if include_rules:
        peak_tables = f"{peak_tables}\n\n{NMR_RULE_HINTS}"
    return prompt.format(peak_tables=peak_tables)


def build_reasoning_target(sample: dict[str, Any]) -> str:
    """Build a structured reasoning target for supervised fine-tuning.

    Parameters
    ----------
    sample
        Sample dictionary containing NMR data and molecular structure labels.

    Returns
    -------
    str
        Target text ordered as reasoning, SELFIES, then canonical SMILES.
    """
    smiles = sample.get("canonical_smiles") or sample.get("smiles") or ""
    selfies = sample.get("selfies") or sample.get("SELFIES") or ""
    h_count = len(sample.get("1H_NMR", {}).get("peaks", []))
    c_count = len(sample.get("13C_NMR", {}).get("peaks", []))
    reasoning = (
        "Spectral reasoning: The 1H NMR and 13C NMR spectra provide "
        f"{h_count} proton peak records and {c_count} carbon peak records "
        "that constrain the molecular structure."
    )
    return "\n".join(
        [
            reasoning,
            f"Final SELFIES: {selfies}",
            f"Final canonical SMILES: {smiles}",
        ]
    )

# TODO: Prompt templates — structure reasoning prediction
# 暂时不实现


def _format_peak_tables(sample: dict[str, Any], formula: str | None) -> str:
    """Build a formatted text block of 1H and 13C peak tables with formula.

    Parameters
    ----------
    sample
        Sample dictionary.
    formula
        Molecular formula string. ``None`` omits the formula line.

    Returns
    -------
    str
        Multi-line peak-table text.
    """
    parts: list[str] = []
    if formula is not None:
        parts.append(f"Molecular formula: {formula}")

    # 1H table
    h_nmr = sample.get("1H_NMR", {})
    parts.append("\n## 1H NMR Peak Table")
    h_solvent = h_nmr.get("solvent", "unknown")
    h_freq = h_nmr.get("frequency", "unknown")
    parts.append(f"  Solvent: {h_solvent}  |  Frequency: {h_freq}")
    parts.append(f"  {'Shift (ppm)':<12} {'Mult.':<8} {'J (Hz)':<16} {'Integ.'}")
    parts.append(f"  {'-'*12} {'-'*8} {'-'*16} {'-'*6}")

    for peak in h_nmr.get("peaks", []):
        if isinstance(peak, dict):
            shift = float(peak["shift"])
            mult = str(peak.get("multiplicity", "s"))
            j_vals = peak.get("J", [])
            integ = float(peak.get("integration", 1))
        else:
            shift, mult, j_vals, integ = peak
            shift = float(shift)
            integ = float(integ)
        j_str = ", ".join(f"{j:.1f}" for j in j_vals) if j_vals else "\u2014"
        parts.append(f"  {shift:<12.2f} {mult:<8} {j_str:<16} {integ:.0f}")

    # 13C table
    c_nmr = sample.get("13C_NMR", {})
    parts.append("\n## 13C NMR Peak Table")
    c_solvent = c_nmr.get("solvent", "unknown")
    c_freq = c_nmr.get("frequency", "unknown")
    parts.append(f"  Solvent: {c_solvent}  |  Frequency: {c_freq}")
    parts.append(f"  {'Shift (ppm)':<12}")
    parts.append(f"  {'-'*12}")

    for peak in c_nmr.get("peaks", []):
        shift = peak["shift"] if isinstance(peak, dict) else peak
        if isinstance(shift, (list, tuple)):
            parts.append(f"  {', '.join(f'{float(v):.2f}' for v in shift)}")
        else:
            parts.append(f"  {float(shift):.2f}")

    return "\n".join(parts)
