"""NMR peak parsing and formatting utilities."""

from __future__ import annotations

import ast
import re
from functools import lru_cache
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


@lru_cache(maxsize=64)
def normalize_multiplicity(value: str | None) -> str:
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
        source_order = (
            len(values) >= 5
            and parse_float(values[0], default=None) is None
            and parse_float(values[3], default=None) is not None
            and parse_float(values[4], default=None) is not None
        )
        if source_order:
            multiplicity = values[0]
            couplings = values[1]
            integration = values[2]
            shift_high = float(parse_float(values[3], default=0.0) or 0.0)
            shift_low = float(parse_float(values[4], default=shift_high) or shift_high)
            shift = (shift_high + shift_low) / 2.0
            peak = {
                "shift": float(shift),
                "shift_range": [min(shift_low, shift_high), max(shift_low, shift_high)],
                "multiplicity": normalize_multiplicity(multiplicity),
                "J": parse_couplings(couplings),
                "integration": parse_integration(integration),
            }
            peaks.append(peak)
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
