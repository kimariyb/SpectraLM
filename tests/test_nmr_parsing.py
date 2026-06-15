"""Tests for NMR peak parsing utilities."""

from spectralm.data.utils import normalize_multiplicity, parse_couplings, parse_frequency_mhz, process_1h_peaks


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
    assert peaks == [{"shift": 1.23, "multiplicity": "t", "J": [7.1], "integration": 3.0}]


def test_parse_frequency_uses_default_for_missing_value() -> None:
    """Frequency parsing should fall back for unknown values."""
    assert parse_frequency_mhz(None, default=500.0) == 500.0

