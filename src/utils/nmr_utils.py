from typing import Any


def sample_peaks(sample: dict[str, Any], nucleus: str) -> list[Any]:
    nmr = sample.get(nucleus, {})
    return nmr.get("peaks", nmr.get("data", [])) or []


def selfies(sample: dict[str, Any]) -> str:
    return sample.get("selfies") or sample.get("SELFIES") or ""


def canonical_smiles(sample: dict[str, Any]) -> str:
    return (
        sample.get("canonical_smiles")
        or sample.get("canonical_SMILES")
        or sample.get("SMILES")
        or sample.get("smiles")
        or ""
    )


def format_1h_peak(peak: dict[str, Any]) -> str:
    shift = float(peak["shift"])
    mult = peak.get("multiplicity", "s")
    integration = peak.get("integration", 1)
    j_values = peak.get("J", [])
    integration_text = f"{integration:g}H" if isinstance(integration, (int, float)) else str(integration)
    if j_values:
        j_text = ", ".join([f"{float(j):.1f}" for j in j_values])
        return f"{shift:.2f} ppm ({mult}, J={j_text} Hz, {integration_text})"
    return f"{shift:.2f} ppm ({mult}, {integration_text})"


def format_13c_peak(peak: dict[str, Any] | float) -> str:
    shift = peak["shift"] if isinstance(peak, dict) else peak
    if isinstance(shift, list):
        return "/".join(f"{float(value):.1f}" for value in shift)
    return f"{float(shift):.1f}"


def build_structure_prompt(sample: dict[str, Any], prompt: str) -> str:
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
            f"Final SELFIES: {selfies(sample)}",
            f"Final canonical SMILES: {canonical_smiles(sample)}",
        ]
    )
    return "\n".join(lines)
