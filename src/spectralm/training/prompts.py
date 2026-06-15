"""Prompt and target builders for NMR reasoning fine-tuning."""

from __future__ import annotations

from typing import Any

from spectralm.data.molecules import canonicalize_smiles, sample_selfies, sample_smiles
from spectralm.data.nmr import format_13c_peak, format_1h_peak, sample_peaks


STRUCTURE_PROMPTS = [
    "Predict the molecular structure from the provided 1H and 13C NMR spectra.",
    "Analyze the multimodal NMR evidence and infer the molecular structure.",
    "Use the spectra and peak tables to determine the molecular structure.",
]

FUNCTIONAL_GROUP_PROMPTS = [
    "Identify the functional groups from the spectra.",
]


def canonical_smiles(sample: dict[str, Any]) -> str:
    """Return canonical SMILES text for a sample.

    Parameters
    ----------
    sample
        Sample dictionary.

    Returns
    -------
    str
        Canonical SMILES or an empty string.
    """
    smiles = sample.get("canonical_smiles") or sample_smiles(sample)
    canonical = canonicalize_smiles(smiles)
    return canonical or str(smiles or "")


def build_structure_prompt(sample: dict[str, Any], prompt: str) -> str:
    """Build the multimodal NMR structure-reasoning prompt.

    Parameters
    ----------
    sample
        Normalized sample dictionary.
    prompt
        Task instruction prefix.

    Returns
    -------
    str
        Full text prompt containing peak tables and rule hints.
    """
    h_nmr = sample.get("1H_NMR", {})
    c_nmr = sample.get("13C_NMR", {})
    h_peaks = [format_1h_peak(peak) for peak in sample_peaks(sample, "1H_NMR")[:30]]
    c_peaks = [format_13c_peak(peak) for peak in sample_peaks(sample, "13C_NMR")[:80]]
    sections = [
        prompt,
        (
            "Return a concise spectral reasoning process first, then provide Final SELFIES "
            "and Final canonical SMILES."
        ),
        (
            f"1H NMR metadata: frequency={h_nmr.get('frequency', 'unknown')}, "
            f"solvent={h_nmr.get('solvent', 'unknown')}\n"
            "1H NMR peak table:\n"
            + "\n".join(h_peaks)
        ),
        (
            f"13C NMR metadata: frequency={c_nmr.get('frequency', 'unknown')}, "
            f"solvent={c_nmr.get('solvent', 'unknown')}\n"
            "13C NMR peak table:\n"
            + ", ".join(c_peaks)
        ),
        (
            "NMR rules to consider:\n"
            "- 1H integration constrains the number of equivalent hydrogens.\n"
            "- Multiplicity and J coupling suggest neighboring proton environments.\n"
            "- 13C peak count approximates distinct carbon environments.\n"
            "- Chemical shift regions suggest functional groups and hybridization.\n"
            "- The final structure must be consistent with both 1H and 13C evidence."
        ),
    ]
    return "\n\n".join(sections)


def build_reasoning_target(sample: dict[str, Any]) -> str:
    """Build a supervised reasoning target followed by SELFIES and SMILES.

    Parameters
    ----------
    sample
        Normalized sample dictionary.

    Returns
    -------
    str
        Training target text.
    """
    h_peaks = sample_peaks(sample, "1H_NMR")
    c_peaks = sample_peaks(sample, "13C_NMR")
    functional_groups = sample.get("functional_groups") or []
    formula = sample.get("molecular_formula", "unknown")
    h_total = sum(float(peak.get("integration", 0)) for peak in h_peaks if isinstance(peak, dict))
    lines = [
        "Spectral reasoning:",
        f"- The 1H NMR spectrum contains {len(h_peaks)} reported proton environments with total integration about {h_total:g}H.",
        f"- The 13C NMR spectrum contains {len(c_peaks)} reported carbon environments.",
    ]
    if functional_groups:
        lines.append(f"- Functional-group evidence is consistent with: {', '.join(functional_groups)}.")
    lines.extend(
        [
            f"- The proposed molecular formula is {formula}.",
            "- The final structure should satisfy the reported 1H integration, splitting patterns, and 13C environment count.",
            "",
            f"Final SELFIES: {sample_selfies(sample)}",
            f"Final canonical SMILES: {canonical_smiles(sample)}",
        ]
    )
    return "\n".join(lines)

