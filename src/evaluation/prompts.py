"""Prompts for direct molecular-structure prediction from paired NMR data."""

from __future__ import annotations

from typing import Any

from src.nmr_rules.engine import analyze_sample


STRUCTURE_PROMPTS: list[str] = [
    (
        "The input contains two one-dimensional NMR spectrum images in a "
        "fixed order: the first image is the 1H NMR spectrum, and the second "
        "image is the 13C NMR spectrum. Jointly interpret the visual spectra "
        "and the peak tables below, including the molecular formula when it "
        "is provided, to predict the molecular structure most consistent "
        "with the data.\n\n{peak_tables}\n\nOutput exactly one canonical SMILES "
        "string only."
    ),
    (
        "You are given two ordered NMR spectrum images and accompanying peak "
        "tables. The first image corresponds to the 1H NMR spectrum, and the "
        "second image corresponds to the 13C NMR spectrum. Determine the "
        "unknown compound by combining the image-level spectral patterns "
        "with the tabulated measurements and any molecular formula provided "
        "below.\n\n{peak_tables}\n\nReturn only one canonical SMILES string."
    ),
    (
        "Act as an expert NMR structure-elucidation model. Interpret the "
        "first image as the 1H NMR spectrum and the second image as the 13C "
        "NMR spectrum. Use both spectra together with the peak tables and, "
        "when available, the molecular formula to infer the structure of the "
        "compound.\n\n{peak_tables}\n\nYour answer must contain only a single "
        "canonical SMILES string."
    ),
    (
        "Solve the structure of the unknown compound from the paired NMR "
        "inputs. The first image is the 1H NMR spectrum, and the second image "
        "is the 13C NMR spectrum. Treat the corresponding peak tables and "
        "any supplied molecular formula as additional constraints on the "
        "visual evidence.\n\n{peak_tables}\n\nOutput only one canonical SMILES "
        "string, with no explanation or formatting."
    ),
    (
        "Use all available one-dimensional NMR evidence to predict the "
        "structure of the unknown molecule. The first image shows the 1H NMR "
        "spectrum, while the second image shows the 13C NMR spectrum. Combine "
        "the images with the tabulated peaks and any molecular formula "
        "included below.\n\n{peak_tables}\n\nRespond with exactly one canonical "
        "SMILES string and nothing else."
    ),
    (
        "Perform direct molecular structure prediction from the ordered NMR "
        "inputs. Analyze the first image as the 1H NMR spectrum and the "
        "second image as the 13C NMR spectrum, then integrate their visual "
        "features with the peak tables and any available molecular formula."
        "\n\n{peak_tables}\n\nThe response must contain only one canonical "
        "SMILES string."
    ),
    (
        "Analyze the paired NMR data for an unknown compound. The first image "
        "contains the 1H NMR spectrum, and the second image contains the 13C "
        "NMR spectrum. Use the two visual spectra jointly with the associated "
        "peak tables and the molecular formula if one is present."
        "\n\n{peak_tables}\n\nProvide only a single canonical SMILES string, "
        "without labels, comments, or a code block."
    ),
    (
        "Identify the molecular structure represented by the multimodal NMR "
        "evidence. The first image is the 1H NMR spectrum and the second image "
        "is the 13C NMR spectrum. Select the structure that best accounts for "
        "both images, the numerical peak information, and any molecular "
        "formula supplied below.\n\n{peak_tables}\n\nReturn only one canonical "
        "SMILES string."
    ),
]


def select_structure_prompt(prompt_template_index: int) -> str:
    """Return one stable inference prompt by explicit template index."""
    index = int(prompt_template_index)
    if index < 0 or index >= len(STRUCTURE_PROMPTS):
        raise ValueError(
            "prompt_template_index must be between 0 and "
            f"{len(STRUCTURE_PROMPTS) - 1}, got {index}"
        )
    return STRUCTURE_PROMPTS[index]


def build_structure_prompt(
    sample: dict[str, Any],
    prompt: str,
    *,
    include_formula: bool = True,
    include_rule_context: bool = False,
    max_rule_evidence: int = 12,
) -> str:
    """Fill one structure prompt with peak tables and optional rule evidence.

    Parameters
    ----------
    sample
        Normalized NMR sample. Formula is read only from the explicit
        ``molecular_formula`` input field.
    prompt
        Prompt template containing ``{peak_tables}``.
    include_formula
        Whether the supplied molecular formula may appear in the prompt.
    include_rule_context
        Whether to append evidence from the one-dimensional rule engine.
    max_rule_evidence
        Maximum number of evidence lines appended to the prompt.

    Returns
    -------
    str
        Model-ready structure-prediction prompt.
    """
    formula_value = sample.get("molecular_formula") if include_formula else None
    formula = str(formula_value).strip() if formula_value else None
    peak_tables = _format_peak_tables(
        sample,
        formula,
    )
    if include_rule_context:
        peak_tables += "\n\n" + _format_rule_context(
            sample,
            include_formula=include_formula,
            max_rule_evidence=max_rule_evidence,
        )
    return prompt.format(peak_tables=peak_tables)


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
