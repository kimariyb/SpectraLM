"""NMR peak parsing and formatting utilities."""

from __future__ import annotations

import ast
import re
from typing import Any


def safe_literal_eval(value: Any) -> Any:
    """Safely parse Python literal strings.

    Parameters
    ----------
    value
        Input value that may contain a serialized list or dictionary.

    Returns
    -------
    Any
        Parsed value for strings, otherwise the original value.
    """
    if isinstance(value, str):
        return ast.literal_eval(value)
    return value


def parse_float(value: Any, default: float | None = None) -> float | None:
    """Parse the first numeric token from a value.

    Parameters
    ----------
    value
        Input value such as ``"400 MHz"`` or ``"8.2Hz"``.
    default
        Value returned when no number can be parsed.

    Returns
    -------
    float | None
        Parsed floating-point number.
    """
    match = re.search(r"[-+]?\d*\.?\d+", str(value))
    return float(match.group()) if match else default


def parse_frequency_mhz(value: Any, default: float = 400.0) -> float:
    """Parse an NMR spectrometer frequency in MHz.

    Parameters
    ----------
    value
        Frequency value, commonly a string like ``"400 MHz"``.
    default
        Fallback frequency when parsing fails.

    Returns
    -------
    float
        Frequency in MHz.
    """
    parsed = parse_float(value, default=None)
    return float(parsed) if parsed is not None and parsed > 0 else default


def parse_integration(value: Any) -> float:
    """Parse proton integration values.

    Parameters
    ----------
    value
        Integration value such as ``"3H"`` or ``3``.

    Returns
    -------
    float
        Numeric integration with a fallback of ``1.0``.
    """
    parsed = parse_float(value, default=None)
    return float(parsed) if parsed is not None else 1.0


def parse_couplings(couplings: Any) -> list[float]:
    """Normalize coupling constants to a list of Hertz values.

    Parameters
    ----------
    couplings
        Raw coupling values from the source dataset.

    Returns
    -------
    list[float]
        Coupling constants in Hz.
    """
    if couplings is None:
        return []
    if not isinstance(couplings, (list, tuple)):
        couplings = [couplings]
    values = []
    for coupling in couplings:
        parsed = parse_float(coupling, default=None)
        if parsed is not None:
            values.append(float(parsed))
    return values


def normalize_multiplicity(value: Any) -> str:
    """Normalize a proton NMR multiplicity label.

    Parameters
    ----------
    value
        Raw multiplicity string.

    Returns
    -------
    str
        Normalized multiplicity label.
    """
    if value is None:
        return "s"
    text = str(value).strip().lower().replace(" ", "")
    if not text:
        return "s"
    aliases = {
        "brs": "brs",
        "bs": "brs",
        "broad": "br",
        "sept": "hept",
        "quint": "p",
        "quintet": "p",
        "multiplet": "m",
    }
    if text.startswith("app"):
        text = text[3:]
    return aliases.get(text, text)


def sample_peaks(sample: dict[str, Any], nucleus: str) -> list[Any]:
    """Return peak records for one NMR nucleus.

    Parameters
    ----------
    sample
        Sample dictionary.
    nucleus
        Nucleus key such as ``"1H_NMR"`` or ``"13C_NMR"``.

    Returns
    -------
    list[Any]
        Peak records.
    """
    nmr = sample.get(nucleus, {})
    return nmr.get("peaks", nmr.get("data", [])) or []


def peak_count(sample: dict[str, Any], nucleus: str) -> int:
    """Count peaks for one NMR nucleus.

    Parameters
    ----------
    sample
        Sample dictionary.
    nucleus
        Nucleus key such as ``"1H_NMR"`` or ``"13C_NMR"``.

    Returns
    -------
    int
        Number of peak records.
    """
    return len(sample_peaks(sample, nucleus))


def process_13c_peaks(raw_peaks: list[Any]) -> list[dict[str, float | list[float]]]:
    """Normalize raw 13C NMR peak tuples.

    Parameters
    ----------
    raw_peaks
        Raw 13C peak tuples from the source CSV.

    Returns
    -------
    list[dict[str, float | list[float]]]
        Normalized peak dictionaries.
    """
    peaks = []
    for item in raw_peaks or []:
        values = item if isinstance(item, (list, tuple)) else [item]
        shifts = []
        for value in values:
            if value is None:
                continue
            parsed = parse_float(value, default=None)
            if parsed is not None:
                shifts.append(float(parsed))
        if shifts:
            peaks.append({"shift": shifts[0] if len(shifts) == 1 else shifts})
    return peaks


def process_1h_peaks(raw_peaks: list[Any]) -> list[dict[str, Any]]:
    """Normalize raw 1H NMR peak tuples.

    Parameters
    ----------
    raw_peaks
        Raw 1H peak tuples from the source CSV.

    Returns
    -------
    list[dict[str, Any]]
        Normalized proton peak dictionaries.
    """
    peaks = []
    for item in raw_peaks or []:
        if isinstance(item, dict):
            shift = parse_float(item.get("shift"), default=None)
            if shift is None:
                continue
            peaks.append(
                {
                    "shift": float(shift),
                    "multiplicity": normalize_multiplicity(item.get("multiplicity", "s")),
                    "J": parse_couplings(item.get("J", [])),
                    "integration": parse_integration(item.get("integration", 1.0)),
                }
            )
            continue
        values = list(item) if isinstance(item, (list, tuple)) else [item]
        if not values:
            continue
        shift = parse_float(values[0], default=None)
        if shift is None:
            continue
        multiplicity = values[1] if len(values) > 1 else "s"
        couplings = values[2] if len(values) > 2 else []
        integration = values[3] if len(values) > 3 else 1.0
        peaks.append(
            {
                "shift": float(shift),
                "multiplicity": normalize_multiplicity(multiplicity),
                "J": parse_couplings(couplings),
                "integration": parse_integration(integration),
            }
        )
    return peaks


def format_1h_peak(peak: dict[str, Any]) -> str:
    """Format a 1H peak for prompt text.

    Parameters
    ----------
    peak
        Normalized 1H peak dictionary.

    Returns
    -------
    str
        Human-readable peak line.
    """
    shift = float(peak["shift"])
    multiplicity = peak.get("multiplicity", "s")
    integration = peak.get("integration", 1)
    j_values = parse_couplings(peak.get("J", []))
    integration_text = f"{integration:g}H" if isinstance(integration, (int, float)) else str(integration)
    if j_values:
        j_text = ", ".join(f"{float(value):.1f}" for value in j_values)
        return f"{shift:.2f} ppm ({multiplicity}, J={j_text} Hz, {integration_text})"
    return f"{shift:.2f} ppm ({multiplicity}, {integration_text})"


def format_13c_peak(peak: dict[str, Any] | float) -> str:
    """Format a 13C peak for prompt text.

    Parameters
    ----------
    peak
        Normalized 13C peak dictionary or raw shift.

    Returns
    -------
    str
        Human-readable peak shift.
    """
    shift = peak["shift"] if isinstance(peak, dict) else peak
    if isinstance(shift, list):
        return "/".join(f"{float(value):.1f}" for value in shift)
    return f"{float(shift):.1f}"

