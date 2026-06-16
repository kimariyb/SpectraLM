"""Prompt builders and target generators for NMR structure elucidation.

The module provides:
- Prompt templates for structure-reasoning and functional-group tasks.
- A builder that injects peak tables and optional NMR rules into templates.
- A target generator that produces structured reasoning chains.
"""

from __future__ import annotations

from typing import Any


# Prompt templates — structure reasoning
STRUCTURE_PROMPTS: list[str] = [
    (
        "You are an expert NMR spectroscopist.  Given the 1H and 13C NMR "
        "data below, determine the molecular structure.\n\n"
        "First, reason step by step through the spectral evidence:\n"
        "- Count the number of unique carbon environments from 13C peaks.\n"
        "- For each 1H signal, assign the chemical shift, multiplicity, "
        "integration, and coupling constants to specific structural fragments.\n"
        "- Piece the fragments together into a complete molecule.\n"
        "- Verify that the assembled structure is consistent with ALL peaks.\n\n"
        "{peak_tables}\n\n"
        "{nmr_rules}"
        "Finally, output the canonical SMILES of the deduced structure."
    ),
    (
        "Analyse the following 1H and 13C NMR spectra and elucidate the "
        "molecular structure.\n\n"
        "{peak_tables}\n\n"
        "{nmr_rules}"
        "Provide a detailed spectral reasoning, then give the final "
        "canonical SMILES."
    ),
    (
        "You receive a set of experimental 1H and 13C NMR peaks.  "
        "Your task is to solve the structure.\n\n"
        "{peak_tables}\n\n"
        "{nmr_rules}"
        "Walk through your analysis step by step: (1) 13C interpretation, "
        "(2) 1H interpretation including multiplicity and integration, "
        "(3) fragment assembly, (4) final structure verification.  "
        "End with the canonical SMILES."
    ),
    (
        "Below are the 1H and 13C NMR data for an unknown organic compound.  "
        "Deduce its molecular structure.\n\n"
        "{peak_tables}\n\n"
        "{nmr_rules}"
        "Think aloud: interpret each signal, identify functional groups and "
        "substructures, then propose the complete molecule.  "
        "Conclude with the canonical SMILES."
    ),
]

# Prompt templates — functional group identification
FUNCTIONAL_GROUP_PROMPTS: list[str] = [
    (
        "Examine the 1H and 13C NMR spectra below and list ALL functional "
        "groups present in the molecule.\n\n"
        "{peak_tables}\n\n"
        "Return a comma-separated list of functional group names "
        "(e.g., alcohol, ketone, aromatic, ester, amine, carboxylic acid, "
        "ether, alkene, alkyne, amide, aldehyde, nitro, nitrile, halide)."
    ),
    (
        "Based on the following NMR data, identify every functional group "
        "in this compound.\n\n"
        "{peak_tables}\n\n"
        "List them as a comma-separated sequence."
    ),
]


# NMR reference rules (injected into prompts)
_NMR_RULES: str = (
    "Key NMR reference ranges:\n"
    "- 1H chemical shifts: alkyl CH3 0.7-1.3, CH2 1.2-1.6, CH 1.4-2.0; "
    "allylic 1.6-2.5; α-to-carbonyl 2.0-2.7; alkyne 2.0-3.0; "
    "O-CH / N-CH 3.0-4.5; alkene 4.5-6.5; aromatic 6.5-8.5; "
    "aldehyde 9.5-10.0; carboxylic acid 10-13; alcohol/phenol 1-6 (broad).\n"
    "- 13C chemical shifts: alkyl C 0-50; C-O / C-N 50-90; "
    "alkene/aromatic 100-160; carbonyl (ester/acid/amide) 160-185; "
    "aldehyde/ketone 190-220.\n"
    "- Common multiplicities: s (singlet), d (doublet), t (triplet), "
    "q (quartet), quin (quintet), sext (sextet), sept (septet), "
    "m (multiplet), dd (doublet of doublets), dt, td, ddd, dq, "
    "brs (broad singlet).\n"
    "- Integration gives relative proton count; coupling constants J "
    "are reported in Hz."
)


def build_structure_prompt(
    sample: dict[str, Any],
    prompt: str,
    include_rules: bool = True,
) -> str:
    """Build a complete structure-reasoning prompt for one sample.

    Parameters
    ----------
    sample
        Sample dictionary containing ``1H_NMR`` and ``13C_NMR`` keys.
    prompt
        Template string with ``{peak_tables}`` and ``{nmr_rules}``
        placeholders.
    include_rules
        Whether to include the NMR reference rules section.

    Returns
    -------
    str
        Filled-in prompt text.
    """
    return prompt.format(
        peak_tables=_format_peak_tables(sample),
        nmr_rules=_NMR_RULES if include_rules else "",
    )


def build_reasoning_target(sample: dict[str, Any]) -> str:
    """Build a step-by-step spectral reasoning target for one sample.

    The output follows a structured chain-of-thought format suitable for
    supervised fine-tuning: 13C → 1H → fragment assembly → final SMILES.

    Parameters
    ----------
    sample
        Sample dictionary.

    Returns
    -------
    str
        Reasoning chain text.
    """
    c_peaks = sample.get("13C_NMR", {}).get("peaks", [])
    h_peaks = sample.get("1H_NMR", {}).get("peaks", [])
    smiles = sample.get("canonical_smiles", "")
    groups = sample.get("functional_groups", [])

    lines: list[str] = []

    # --- 13C analysis --------------------------------------------------------
    lines.append("## 13C NMR Analysis")
    c_count = len(c_peaks)
    lines.append(
        f"The 13C spectrum shows {c_count} distinct carbon environment(s)."
    )
    c_shifts = []
    for peak in c_peaks:
        shift = peak["shift"] if isinstance(peak, dict) else peak
        if isinstance(shift, (list, tuple)):
            c_shifts.extend(float(v) for v in shift)
        else:
            c_shifts.append(float(shift))
    c_shifts.sort()
    lines.append(f"Chemical shifts (ppm): {_fmt_shifts(c_shifts)}")

    # Coarse assignment based on ranges
    assignments: list[str] = []
    for s in c_shifts:
        if s < 50:
            assignments.append(f"{s:.1f} (alkyl C)")
        elif s < 90:
            assignments.append(f"{s:.1f} (C-O / C-N)")
        elif s < 160:
            assignments.append(f"{s:.1f} (alkene / aromatic C)")
        elif s < 190:
            assignments.append(f"{s:.1f} (carbonyl)")
        else:
            assignments.append(f"{s:.1f} (aldehyde / ketone C)")
    lines.append("  " + "; ".join(assignments))

    # --- 1H analysis ---------------------------------------------------------
    lines.append("\n## 1H NMR Analysis")
    h_count = len(h_peaks)
    lines.append(f"The 1H spectrum shows {h_count} distinct signal(s).")
    for peak in h_peaks:
        if isinstance(peak, dict):
            shift = float(peak["shift"])
            mult = str(peak.get("multiplicity", "s"))
            j_vals = peak.get("J", [])
            integ = float(peak.get("integration", 1))
        else:
            shift, mult, j_vals, integ = peak
            shift = float(shift)
            integ = float(integ)

        j_str = f", J = {', '.join(f'{j:.1f}' for j in j_vals)} Hz" if j_vals else ""
        lines.append(
            f"  δ {shift:.2f} ({mult}{j_str}, {integ:.0f}H)"
        )

    # --- Fragment assembly ---------------------------------------------------
    lines.append("\n## Fragment Assembly")
    if groups:
        lines.append(
            f"Functional groups inferred from shifts: {', '.join(groups)}."
        )
    lines.append(
        "Connecting the fragments based on coupling patterns, chemical "
        "shift correlations, and symmetry considerations yields the "
        "proposed structure."
    )

    # --- Final answer --------------------------------------------------------
    lines.append("\n## Final Structure")
    lines.append(f"Canonical SMILES: {smiles}")

    return "\n".join(lines)


def _format_peak_tables(sample: dict[str, Any]) -> str:
    """Build a formatted text block of 1H and 13C peak tables.

    Parameters
    ----------
    sample
        Sample dictionary.

    Returns
    -------
    str
        Multi-line peak-table text.
    """
    parts: list[str] = []

    # 1H table
    h_nmr = sample.get("1H_NMR", {})
    parts.append("## 1H NMR Peak Table")
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
        j_str = ", ".join(f"{j:.1f}" for j in j_vals) if j_vals else "—"
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


def _fmt_shifts(shifts: list[float]) -> str:
    """Format a list of chemical shifts as a compact string."""
    if len(shifts) <= 6:
        return ", ".join(f"{s:.1f}" for s in shifts)
    return ", ".join(f"{s:.1f}" for s in shifts[:6]) + f", ... ({len(shifts)} total)"
