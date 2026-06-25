"""Text-only prompts for molecular-structure prediction from 1D NMR tables."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from src.nmr_rules.engine import analyze_sample


JSON_SMILES_OUTPUT_INSTRUCTION = (
    "Return exactly one JSON object and no other text using this schema: "
    "{{\"smiles\":\"string (canonical SMILES, or null if insufficient data)\"}}."
)


SYSTEM_PROMPT = """
You are a specialized molecular structure elucidation model for one-dimensional NMR data. 
Analyze the provided 1H NMR and/or 13C NMR information, 
including peak tables, chemical shifts, integrations, multiplicities, coupling constants, solvent, frequency, and any explicitly provided molecular formula. 
Predict the most likely molecular structure based only on the supplied information.
Output exactly one JSON object and nothing else: {"smiles":"string (canonical SMILES, or null if insufficient data)"}.
Do not output reasoning.
Do not include explanations, reasoning steps, Markdown, labels, prefixes, suffixes, code blocks, confidence scores, or alternative structures. 
Do not use hidden target structures, database identifiers, IUPAC names, answer labels, or any information not explicitly present in the input. 
Do not describe peak tables as raw FID data. 
If the data are insufficient, set "smiles" to null.
"""

STRUCTURE_PROMPTS: list[str] = [
    (
        "Infer the molecular structure from the one-dimensional 1H NMR and "
        "13C NMR peak tables below. Use the molecular formula when it is "
        "provided as a hard constraint.\n\n{spectral_context}\n\n"
        f"{JSON_SMILES_OUTPUT_INSTRUCTION}"
    ),
    (
        "Determine the unknown compound using the numerical 1H NMR and 13C "
        "NMR evidence. Combine chemical shifts, multiplicities, couplings, "
        "integrations, carbon signal counts, and any molecular formula shown "
        "below.\n\n{spectral_context}\n\n"
        f"{JSON_SMILES_OUTPUT_INSTRUCTION}"
    ),
    (
        "Act as an expert small-molecule NMR structure elucidation model. "
        "Analyze the 1H NMR and 13C NMR peak tables jointly and infer the "
        "structure most consistent with all constraints.\n\n"
        "{spectral_context}\n\n"
        f"{JSON_SMILES_OUTPUT_INSTRUCTION}"
    ),
    (
        "Solve the structure of the unknown compound from paired 1D NMR "
        "measurements. Treat the 1H NMR and 13C NMR tables as complementary evidence "
        "and obey the molecular formula if present.\n\n{spectral_context}\n\n"
        f"{JSON_SMILES_OUTPUT_INSTRUCTION}"
    ),
    (
        "Use all available tabulated one-dimensional 1H NMR and 13C NMR "
        "evidence to predict "
        "the canonical molecular structure. Consider proton environments, "
        "carbon environments, signal counts, and formula constraints.\n\n"
        "{spectral_context}\n\n"
        f"{JSON_SMILES_OUTPUT_INSTRUCTION}"
    ),
    (
        "Perform direct molecular structure prediction from the given 1H NMR "
        "and 13C NMR peak data. Select the structure that best accounts for "
        "the combined spectral measurements.\n\n{spectral_context}\n\n"
        f"{JSON_SMILES_OUTPUT_INSTRUCTION}"
    ),
    (
        "Analyze the tabulated NMR data for an unknown compound. Use the 1H "
        "NMR and 13C NMR peak tables together with any supplied molecular "
        "formula to identify the structure.\n\n{spectral_context}\n\n"
        f"{JSON_SMILES_OUTPUT_INSTRUCTION}"
    ),
    (
        "Identify the molecular structure represented by the 1D NMR evidence. "
        "Use the numerical 1H NMR and 13C NMR data jointly rather than "
        "treating either table independently.\n\n{spectral_context}\n\n"
        f"{JSON_SMILES_OUTPUT_INSTRUCTION}"
    ),
]


def structure_prompts() -> list[str]:
    """Return the active text-only structure-prediction prompt set."""
    return list(STRUCTURE_PROMPTS)


def select_structure_prompt(prompt_template_index: int) -> str:
    """Return one stable inference prompt by explicit template index."""
    index = int(prompt_template_index)
    if index < 0 or index >= len(STRUCTURE_PROMPTS):
        raise ValueError(
            "prompt_template_index must be between 0 and "
            f"{len(STRUCTURE_PROMPTS) - 1}, got {index}"
        )
    return STRUCTURE_PROMPTS[index]


def _iter_h_peaks(sample: dict[str, Any]) -> list[dict[str, Any]]:
    peaks = sample.get("1H_NMR", {}).get("peaks")
    if peaks is None:
        peaks = sample.get("h_nmr_peaks", [])
    normalized: list[dict[str, Any]] = []
    for peak in peaks or []:
        if isinstance(peak, dict):
            normalized.append(peak)
        else:
            shift, multiplicity, couplings, integration = peak
            normalized.append(
                {
                    "shift": shift,
                    "multiplicity": multiplicity,
                    "J": couplings,
                    "integration": integration,
                }
            )
    return sorted(
        normalized,
        key=lambda item: float(item.get("shift", 0.0)),
        reverse=True,
    )


def _iter_c_shifts(sample: dict[str, Any]) -> list[float]:
    peaks = sample.get("13C_NMR", {}).get("peaks")
    if peaks is None:
        peaks = sample.get("c_nmr_peaks", [])
    shifts: list[float] = []
    for peak in peaks or []:
        value = peak.get("shift") if isinstance(peak, dict) else peak
        values: Iterable[Any]
        if isinstance(value, (list, tuple)):
            values = value
        else:
            values = (value,)
        shifts.extend(float(item) for item in values)
    return sorted(shifts, reverse=True)


def format_peak_tables(sample: dict[str, Any]) -> str:
    """Serialize ordered 1H and 13C peak tables deterministically."""
    lines = ["1H NMR:"]
    for peak in _iter_h_peaks(sample):
        couplings = peak.get("J", [])
        j_text = ",".join(f"{float(value):.1f}" for value in couplings) or "-"
        integration = f"{float(peak.get('integration', 1.0)):g}"
        lines.append(
            f"{float(peak['shift']):.2f} ppm | "
            f"{peak.get('multiplicity', 's')} | J={j_text} Hz | "
            f"integration={integration}"
        )

    c_shifts = _iter_c_shifts(sample)
    c_text = ", ".join(f"{value:.2f}" for value in c_shifts) or "-"
    lines.extend(["", "13C NMR:", f"{c_text} ppm"])
    return "\n".join(lines)


def _format_rule_context(
    sample: dict[str, Any],
    *,
    include_formula: bool,
    max_rule_evidence: int,
) -> str:
    """Format bounded, confidence-ordered rule evidence for one prompt."""
    limit = int(max_rule_evidence)
    if limit <= 0:
        raise ValueError("max_rule_evidence must be positive.")
    analysis = analyze_sample(sample, include_formula=include_formula)
    strength_rank = {"hard": 0, "strong": 1, "moderate": 2, "weak": 3}
    ordered = sorted(
        enumerate(analysis.evidence),
        key=lambda item: (
            strength_rank.get(item[1].strength, 99),
            -item[1].confidence,
            item[0],
        ),
    )
    selected = [item for _, item in ordered[:limit]]
    lines = [
        "## Derived 1D NMR Constraints",
        f"Rule library: {analysis.library_name}",
    ]
    lines.extend(
        f"- [{item.strength}; {item.rule_id}] {item.conclusion}"
        for item in selected
    )
    lines.extend(f"- [warning] {warning}" for warning in analysis.warnings)
    return "\n".join(lines)


def build_structure_prompt(
    sample: dict[str, Any],
    prompt: str | None = None,
    *,
    include_formula: bool = True,
    include_rule_context: bool = False,
    max_rule_evidence: int = 12,
) -> str:
    """Build one pure-text NMR structure prompt."""
    template = prompt or STRUCTURE_PROMPTS[0]
    formula = str(sample.get("molecular_formula") or "").strip()
    context: list[str] = []
    if include_formula:
        if not formula:
            raise ValueError("Formula-conditioned input requires molecular_formula")
        context.append(f"Molecular formula: {formula}")
    context.append(format_peak_tables(sample))
    if include_rule_context:
        context.append(
            _format_rule_context(
                sample,
                include_formula=include_formula,
                max_rule_evidence=max_rule_evidence,
            )
        )
    spectral_context = "\n\n".join(context)
    return template.format(
        spectral_context=spectral_context,
        peak_tables=spectral_context,
    )
