"""Molecular representation helpers used by the active JSONL workflow."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import re
from typing import Any

from rdkit import Chem, RDLogger
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem.Scaffolds import MurckoScaffold


RDLogger.DisableLog("rdApp.*")

ALLOWED_ELEMENT_SYMBOLS = frozenset(
    {"H", "C", "N", "O", "F", "Si", "P", "S", "Cl", "Br", "I"}
)
_FORMULA_ELEMENT_PATTERN = re.compile(r"[A-Z][a-z]?")


@dataclass(frozen=True)
class DatasetMoleculeInspection:
    """Normalized structure and dataset-policy audit measurements."""

    canonical_smiles: str | None
    component_count: int
    formal_charge: int
    radical_electron_count: int
    isotope_label_count: int
    violations: tuple[str, ...]

    @property
    def accepted(self) -> bool:
        """Return whether the structure satisfies the dataset policy."""
        return self.canonical_smiles is not None and not self.violations


@lru_cache(maxsize=8192)
def inspect_dataset_molecule(smiles: str | None) -> DatasetMoleculeInspection:
    """Normalize and audit a structure for inclusion in the NMR dataset.

    Isotope labels are removed before canonicalization. Disconnected
    structures, non-zero total formal charge, radicals, and unsupported
    elements are reported as violations. Net-neutral charge-separated
    representations remain valid.

    Parameters
    ----------
    smiles
        Raw molecular SMILES.

    Returns
    -------
    DatasetMoleculeInspection
        Isotope-free canonical structure and auditable policy measurements.
    """
    if not smiles:
        return DatasetMoleculeInspection(None, 0, 0, 0, 0, ("invalid_smiles",))
    try:
        mol = Chem.MolFromSmiles(str(smiles))
    except Exception:
        mol = None
    if mol is None:
        return DatasetMoleculeInspection(None, 0, 0, 0, 0, ("invalid_smiles",))

    component_count = len(Chem.GetMolFrags(mol))
    formal_charge = int(Chem.GetFormalCharge(mol))
    radical_electron_count = sum(
        int(atom.GetNumRadicalElectrons()) for atom in mol.GetAtoms()
    )
    isotope_label_count = sum(
        int(atom.GetIsotope() != 0) for atom in mol.GetAtoms()
    )
    element_symbols = {atom.GetSymbol() for atom in mol.GetAtoms()}
    unsupported = sorted(element_symbols - ALLOWED_ELEMENT_SYMBOLS)

    violations: list[str] = []
    if component_count != 1:
        violations.append("multiple_components")
    if formal_charge != 0:
        violations.append("nonzero_formal_charge")
    if radical_electron_count != 0:
        violations.append("radical")
    if unsupported:
        violations.append(f"unsupported_elements:{','.join(unsupported)}")

    for atom in mol.GetAtoms():
        atom.SetIsotope(0)
    try:
        Chem.SanitizeMol(mol)
        canonical = Chem.MolToSmiles(mol, canonical=True)
    except Exception:
        canonical = None
        if "invalid_smiles" not in violations:
            violations.insert(0, "invalid_smiles")

    return DatasetMoleculeInspection(
        canonical_smiles=canonical,
        component_count=component_count,
        formal_charge=formal_charge,
        radical_electron_count=radical_electron_count,
        isotope_label_count=isotope_label_count,
        violations=tuple(violations),
    )


def sample_smiles(sample: dict[str, Any]) -> str | None:
    """Extract canonical SMILES from a normalized sample."""
    return sample.get("canonical_smiles") or sample.get("smiles")


@lru_cache(maxsize=8192)
def canonicalize_smiles(smiles: str | None) -> str | None:
    """Canonicalize a SMILES string, returning ``None`` when invalid."""
    if not smiles:
        return None
    try:
        mol = Chem.MolFromSmiles(str(smiles))
        if mol is None:
            return None
        return Chem.MolToSmiles(mol, canonical=True)
    except Exception:
        return None


def mol_from_smiles(smiles: str | None) -> Chem.Mol | None:
    """Create an RDKit molecule from SMILES."""
    if not smiles:
        return None
    return Chem.MolFromSmiles(str(smiles))


@lru_cache(maxsize=8192)
def molecule_formula(smiles: str | None) -> str | None:
    """Calculate a molecular formula from SMILES."""
    mol = mol_from_smiles(smiles)
    if mol is None:
        return None
    return rdMolDescriptors.CalcMolFormula(mol)


@lru_cache(maxsize=8192)
def molecule_elements(smiles: str | None) -> frozenset[str]:
    """Return all element symbols in a valid molecule, including hydrogen.

    Parameters
    ----------
    smiles
        Molecular SMILES.

    Returns
    -------
    frozenset[str]
        Element symbols, or an empty set for invalid input.
    """
    formula = molecule_formula(smiles)
    if formula is None:
        return frozenset()
    return frozenset(_FORMULA_ELEMENT_PATTERN.findall(formula))


def unsupported_elements(smiles: str | None) -> frozenset[str]:
    """Return molecule elements outside the common-element policy."""
    return molecule_elements(smiles) - ALLOWED_ELEMENT_SYMBOLS


def has_only_allowed_elements(smiles: str | None) -> bool:
    """Return whether a valid molecule uses only allowed elements."""
    canonical = canonicalize_smiles(smiles)
    return canonical is not None and not unsupported_elements(canonical)


def murcko_scaffold(smiles: str | None) -> str | None:
    """Calculate a Bemis-Murcko scaffold with an acyclic fallback."""
    canonical = canonicalize_smiles(smiles)
    if canonical is None:
        return None
    mol = Chem.MolFromSmiles(canonical)
    scaffold = MurckoScaffold.MurckoScaffoldSmiles(
        mol=mol,
        includeChirality=False,
    )
    return scaffold or f"acyclic:{canonical}"


def heavy_atom_count(smiles: str | None) -> int:
    """Count non-hydrogen atoms, returning zero for invalid molecules."""
    mol = mol_from_smiles(smiles)
    return mol.GetNumHeavyAtoms() if mol is not None else 0
