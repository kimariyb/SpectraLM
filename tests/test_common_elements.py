"""Tests for the common-element dataset policy."""

from __future__ import annotations

from src.data.manifest import sample_manifest_row
from src.data import molecules as molecule_utils
from src.data.molecules import (
    ALLOWED_ELEMENT_SYMBOLS,
    has_only_allowed_elements,
    molecule_elements,
    unsupported_elements,
)


def _inspect_dataset_molecule(smiles: str):
    inspector = getattr(molecule_utils, "inspect_dataset_molecule", None)
    assert callable(inspector), "inspect_dataset_molecule must be implemented"
    return inspector(smiles)


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


def test_dataset_policy_rejects_disconnected_salts() -> None:
    """Net-neutral salts still violate the single-component requirement."""
    result = _inspect_dataset_molecule("C[NH3+].[Cl-]")

    assert result.accepted is False
    assert result.component_count == 2
    assert result.formal_charge == 0
    assert result.violations == ("multiple_components",)


def test_dataset_policy_rejects_nonzero_charge_and_radicals() -> None:
    """Charged molecules and structures with radical electrons are excluded."""
    charged = _inspect_dataset_molecule("[NH4+]")
    radical = _inspect_dataset_molecule("[CH3]")

    assert charged.accepted is False
    assert charged.formal_charge == 1
    assert "nonzero_formal_charge" in charged.violations
    assert radical.accepted is False
    assert radical.radical_electron_count == 1
    assert "radical" in radical.violations


def test_dataset_policy_removes_isotope_labels_before_canonicalization() -> None:
    """Isotopologues should collapse onto a natural-abundance structure key."""
    result = _inspect_dataset_molecule("[13CH3]CO")

    assert result.accepted is True
    assert result.canonical_smiles == "CCO"
    assert result.isotope_label_count == 1
    assert result.violations == ()


def test_dataset_policy_keeps_net_neutral_charge_separated_groups() -> None:
    """Formal charge separation is valid when the complete molecule is neutral."""
    result = _inspect_dataset_molecule("C[N+](=O)[O-]")

    assert result.accepted is True
    assert result.component_count == 1
    assert result.formal_charge == 0


def test_manifest_records_structural_policy_audit_fields(ethanol_sample) -> None:
    """Manifest QC should expose every structural-policy measurement."""
    isotope_sample = dict(ethanol_sample)
    isotope_sample["canonical_smiles"] = "[13CH3]CO"

    row = sample_manifest_row(isotope_sample)

    assert row["qc_status"] == "pass"
    assert row["canonical_smiles"] == "CCO"
    assert row["component_count"] == 1
    assert row["formal_charge"] == 0
    assert row["radical_electron_count"] == 0
    assert row["isotope_label_count"] == 1


def test_manifest_marks_disconnected_structure_failed(ethanol_sample) -> None:
    """Manifest generation must catch salts from legacy JSONL data."""
    salt_sample = dict(ethanol_sample)
    salt_sample["canonical_smiles"] = "C[NH3+].[Cl-]"

    row = sample_manifest_row(salt_sample)

    assert row["qc_status"] == "fail"
    assert "multiple_components" in row["qc_reason"]
