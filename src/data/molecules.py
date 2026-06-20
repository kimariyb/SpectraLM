"""Molecular representation helpers used by the active JSONL workflow."""

from __future__ import annotations

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
