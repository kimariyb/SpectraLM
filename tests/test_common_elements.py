"""Tests for the common-element dataset policy."""

from __future__ import annotations

from src.data.manifest import sample_manifest_row
from src.data.molecules import (
    ALLOWED_ELEMENT_SYMBOLS,
    has_only_allowed_elements,
    molecule_elements,
    unsupported_elements,
)


def test_allowed_element_policy_covers_requested_symbols() -> None:
    """The policy should exactly match the requested common elements."""
    assert ALLOWED_ELEMENT_SYMBOLS == frozenset(
        {"H", "C", "N", "O", "F", "Si", "P", "S", "Cl", "Br", "I"}
    )
    assert molecule_elements("C[SiH3]") == frozenset({"C", "H", "Si"})
    assert has_only_allowed_elements("CP(=O)(O)S") is True


def test_allowed_element_policy_rejects_other_elements() -> None:
    """Molecules containing metals, boron, or selenium should be excluded."""
    assert unsupported_elements("C[BH2]") == frozenset({"B"})
    assert unsupported_elements("C[Na]") == frozenset({"Na"})
    assert unsupported_elements("C[SeH]") == frozenset({"Se"})
    assert has_only_allowed_elements("C[Na]") is False
    assert has_only_allowed_elements("not_a_smiles") is False


def test_manifest_marks_unsupported_elements_failed(ethanol_sample) -> None:
    """Manifest QC should expose unsupported symbols for auditable filtering."""
    sample = dict(ethanol_sample)
    sample["canonical_smiles"] = "C[BH2]"
    sample["smiles"] = "C[BH2]"

    row = sample_manifest_row(sample)

    assert row["qc_status"] == "fail"
    assert "unsupported_elements:B" in row["qc_reason"]
    assert row["element_symbols"] == "B;C;H"
