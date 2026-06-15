"""Molecular representation helpers for SMILES, SELFIES, and RDKit objects."""

from __future__ import annotations

from typing import Any

import selfies as sf
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors
from rdkit.Chem.Scaffolds import MurckoScaffold


def sample_smiles(sample: dict[str, Any]) -> str | None:
    """Return the best available SMILES string from a sample.

    Parameters
    ----------
    sample
        Normalized or raw sample dictionary.

    Returns
    -------
    str | None
        Canonical or raw SMILES value when present.
    """
    return (
        sample.get("canonical_smiles")
        or sample.get("canonical_SMILES")
        or sample.get("SMILES")
        or sample.get("smiles")
    )


def sample_selfies(sample: dict[str, Any]) -> str:
    """Return the SELFIES value from a sample.

    Parameters
    ----------
    sample
        Sample dictionary.

    Returns
    -------
    str
        SELFIES string or an empty string.
    """
    return sample.get("selfies") or sample.get("SELFIES") or ""


def canonicalize_smiles(smiles: str | None) -> str | None:
    """Canonicalize a SMILES string with RDKit.

    Parameters
    ----------
    smiles
        Input SMILES string.

    Returns
    -------
    str | None
        Canonical SMILES, or ``None`` when parsing fails.
    """
    if not smiles:
        return None
    try:
        mol = Chem.MolFromSmiles(str(smiles))
        if mol is None:
            return None
        return Chem.MolToSmiles(mol, canonical=True)
    except Exception:
        return None


def smiles_to_selfies(smiles: str | None) -> str | None:
    """Encode a SMILES string as SELFIES.

    Parameters
    ----------
    smiles
        Input SMILES string.

    Returns
    -------
    str | None
        SELFIES string, or ``None`` when encoding fails.
    """
    if not smiles:
        return None
    try:
        return sf.encoder(smiles)
    except Exception:
        return None


def mol_from_smiles(smiles: str | None):
    """Create an RDKit molecule from SMILES.

    Parameters
    ----------
    smiles
        Input SMILES string.

    Returns
    -------
    rdkit.Chem.Mol | None
        RDKit molecule when parsing succeeds.
    """
    if not smiles:
        return None
    return Chem.MolFromSmiles(str(smiles))


def molecule_formula(smiles: str | None) -> str | None:
    """Calculate a molecular formula from SMILES.

    Parameters
    ----------
    smiles
        Input SMILES string.

    Returns
    -------
    str | None
        Molecular formula or ``None`` for invalid molecules.
    """
    mol = mol_from_smiles(smiles)
    if mol is None:
        return None
    return rdMolDescriptors.CalcMolFormula(mol)


def murcko_scaffold(smiles: str | None) -> str | None:
    """Calculate a Bemis-Murcko scaffold string.

    Parameters
    ----------
    smiles
        Input SMILES string.

    Returns
    -------
    str | None
        Scaffold SMILES. Acyclic molecules use an ``acyclic:`` prefix.
    """
    canonical = canonicalize_smiles(smiles)
    if canonical is None:
        return None
    mol = Chem.MolFromSmiles(canonical)
    scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
    return scaffold or f"acyclic:{canonical}"


def molecule_descriptors(smiles: str | None) -> dict[str, str | None]:
    """Build standard molecule descriptors used by split and sampling workflows.

    Parameters
    ----------
    smiles
        Input SMILES string.

    Returns
    -------
    dict[str, str | None]
        Canonical SMILES, Murcko scaffold, and molecular formula.
    """
    canonical = canonicalize_smiles(smiles)
    if canonical is None:
        return {
            "canonical_smiles": None,
            "murcko_scaffold": None,
            "molecular_formula": None,
        }
    return {
        "canonical_smiles": canonical,
        "murcko_scaffold": murcko_scaffold(canonical),
        "molecular_formula": molecule_formula(canonical),
    }


FUNCTIONAL_GROUP_SMARTS = {
    "aromatic": "a",
    "alkene": "C=C",
    "alkyne": "C#C",
    "alcohol": "[OX2H][#6]",
    "phenol": "c[OX2H]",
    "ether": "[OD2]([#6])[#6]",
    "aldehyde": "[CX3H1](=O)[#6]",
    "ketone": "[#6][CX3](=O)[#6]",
    "carboxylic_acid": "[CX3](=O)[OX2H1]",
    "ester": "[CX3](=O)[OX2][#6]",
    "amide": "[NX3][CX3](=O)",
    "amine": "[NX3;H2,H1,H0;!$(NC=O)]",
    "nitrile": "C#N",
    "nitro": "[$([NX3](=O)=O),$([NX3+](=O)[O-])]",
    "halide": "[F,Cl,Br,I]",
    "boron": "[B]",
    "silicon": "[Si]",
    "phosphorus": "[P]",
    "sulfur": "[S]",
}


def functional_group_labels(smiles: str | None) -> list[str]:
    """Detect coarse functional groups with SMARTS rules.

    Parameters
    ----------
    smiles
        Input SMILES string.

    Returns
    -------
    list[str]
        Functional-group labels, ``["invalid"]``, or ``["none_detected"]``.
    """
    mol = mol_from_smiles(smiles)
    if mol is None:
        return ["invalid"]
    labels = []
    for label, smarts in FUNCTIONAL_GROUP_SMARTS.items():
        pattern = Chem.MolFromSmarts(smarts)
        if pattern is not None and mol.HasSubstructMatch(pattern):
            labels.append(label)
    return labels or ["none_detected"]


def heavy_atom_count(smiles: str | None) -> int:
    """Count heavy atoms for a molecule.

    Parameters
    ----------
    smiles
        Input SMILES string.

    Returns
    -------
    int
        Heavy atom count, or zero for invalid molecules.
    """
    mol = mol_from_smiles(smiles)
    return mol.GetNumHeavyAtoms() if mol is not None else 0


def molecular_weight(smiles: str | None) -> float:
    """Calculate molecular weight for a molecule.

    Parameters
    ----------
    smiles
        Input SMILES string.

    Returns
    -------
    float
        RDKit molecular weight, or zero for invalid molecules.
    """
    mol = mol_from_smiles(smiles)
    return Descriptors.MolWt(mol) if mol is not None else 0.0


def ring_count(smiles: str | None) -> int:
    """Count rings in a molecule.

    Parameters
    ----------
    smiles
        Input SMILES string.

    Returns
    -------
    int
        Ring count, or zero for invalid molecules.
    """
    mol = mol_from_smiles(smiles)
    return mol.GetRingInfo().NumRings() if mol is not None else 0

