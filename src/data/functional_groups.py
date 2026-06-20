"""Functional-group ontology for auxiliary labels and metrics."""

from __future__ import annotations

from functools import lru_cache

from rdkit import Chem

from src.data.molecules import canonicalize_smiles


FUNCTIONAL_GROUP_SMARTS: tuple[tuple[str, str], ...] = (
    ("alkene", "[CX3]=[CX3]"),
    ("alkyne", "[CX2]#[CX2]"),
    ("aromatic_ring", "[a;r]"),
    ("alcohol", "[OX2H;!$([O][C,S,P]=O);!$([O]c)]"),
    ("phenol", "[OX2H]c"),
    ("ether", "[OD2;!$([O][C,S,P]=O)]([#6])[#6]"),
    ("aldehyde", "[CX3H1](=O)[#6,H]"),
    ("ketone", "[#6][CX3](=O)[#6]"),
    ("carboxylic_acid", "[CX3](=O)[OX2H1]"),
    ("ester", "[CX3](=O)[OX2H0][#6]"),
    ("amide", "[CX3](=O)[NX3]"),
    ("amine", "[NX3;!$(N[C,S,P]=O);!$(N=*)]"),
    ("nitrile", "[CX2]#N"),
    ("nitro", "[N+](=O)[O-]"),
    ("organohalogen", "[#6][F,Cl,Br,I]"),
    ("thiol", "[SX2H]"),
    ("thioether", "[SX2]([#6])[#6]"),
    ("sulfoxide", "[#16X3](=O)([#6])[#6]"),
    ("sulfone", "[#16X4](=O)(=O)([#6])[#6]"),
    ("phosphorus_oxygen", "[P](=O)[O,#6]"),
    ("silicon_carbon", "[Si]-[#6]"),
    ("siloxane", "[Si]-[O]-[Si]"),
)


@lru_cache(maxsize=1)
def _compiled_patterns() -> tuple[tuple[str, Chem.Mol], ...]:
    patterns: list[tuple[str, Chem.Mol]] = []
    for label, smarts in FUNCTIONAL_GROUP_SMARTS:
        pattern = Chem.MolFromSmarts(smarts)
        if pattern is None:
            raise ValueError(f"Invalid SMARTS for functional group {label}: {smarts}")
        patterns.append((label, pattern))
    return tuple(patterns)


@lru_cache(maxsize=16384)
def functional_groups(smiles: str | None) -> frozenset[str]:
    """Return controlled functional-group labels for a molecular structure.

    Parameters
    ----------
    smiles
        Molecular SMILES.

    Returns
    -------
    frozenset[str]
        Presence/absence labels from the controlled ontology.
    """
    canonical = canonicalize_smiles(smiles)
    mol = Chem.MolFromSmiles(canonical) if canonical is not None else None
    if mol is None:
        return frozenset()
    return frozenset(
        label for label, pattern in _compiled_patterns() if mol.HasSubstructMatch(pattern)
    )
