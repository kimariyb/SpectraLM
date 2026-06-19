"""Tests for NMR peak parsing utilities."""

from src.data.utils import (
    normalize_multiplicity,
    parse_couplings,
    parse_frequency_mhz,
    process_1h_peaks,
)
from src.spectra.render import SPECTRUM_POINTS, compute_13c


def test_parse_couplings_from_mixed_values() -> None:
    """Coupling parser should extract numeric Hertz values."""
    assert parse_couplings(["7.2Hz", 1.5, None, "bad"]) == [7.2, 1.5]


def test_normalize_multiplicity_aliases() -> None:
    """Multiplicity parser should normalize common labels."""
    assert normalize_multiplicity("app d") == "d"
    assert normalize_multiplicity("sept") == "hept"
    assert normalize_multiplicity("") == "s"


def test_process_1h_peaks_tuple_rows() -> None:
    """Raw proton tuples should become normalized dictionaries."""
    peaks = process_1h_peaks([(1.23, "app t", ["7.1 Hz"], "3H")])
    assert peaks == [
        {
            "shift": 1.23,
            "multiplicity": "t",
            "J": [7.1],
            "integration": 3.0,
        }
    ]


def test_process_1h_peaks_source_tuple_rows() -> None:
    """Source proton tuples should parse multiplicity-first rows."""
    peaks = process_1h_peaks(
        [("m", [], "2H", 7.27, 7.24), ("d", ["3.8Hz"], "2H", 3.68, 3.68)]
    )
    assert peaks == [
        {
            "shift": 7.255,
            "shift_range": [7.24, 7.27],
            "multiplicity": "m",
            "J": [],
            "integration": 2.0,
        },
        {
            "shift": 3.68,
            "shift_range": [3.68, 3.68],
            "multiplicity": "d",
            "J": [3.8],
            "integration": 2.0,
        },
    ]


def test_parse_frequency_uses_default_for_missing_value() -> None:
    """Frequency parsing should fall back for unknown values."""
    assert parse_frequency_mhz(None, default=500.0) == 500.0


def test_render_compute_uses_configured_spectrum_points(ethanol_sample) -> None:
    """Continuous spectra should use the lower point count for faster rendering."""
    import numpy as np

    x_axis, intensity = compute_13c(
        ethanol_sample,
        snr=300.0,
        rng=np.random.default_rng(1),
    )

    assert SPECTRUM_POINTS == 16384
    assert len(x_axis) == SPECTRUM_POINTS
    assert len(intensity) == SPECTRUM_POINTS
