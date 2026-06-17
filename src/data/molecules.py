"""Molecular representation helpers for SMILES, SELFIES, and RDKit objects."""

from __future__ import annotations

from functools import lru_cache
from typing import Any
import numpy as np
import selfies as sf
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, rdMolDescriptors
from rdkit.Chem.Scaffolds import MurckoScaffold


RDLogger.DisableLog("rdApp.*")


FUNCTIONAL_GROUP_SMARTS = {
    # ============================================================
    # 基础碳骨架与不饱和键
    # ============================================================
    "aromatic": "a",                         # 任意芳香原子
    "heteroaromatic": "[a;!#6]",             # 芳香杂原子，如吡啶N、呋喃O、噻吩S
    "alkene": "[CX3]=[CX3]",                 # C=C，不包括 C=O / C=N
    "alkyne": "[CX2]#[CX2]",                 # C#C，不包括腈
    "allene": "[CX2](=[CX3])=[CX3]",         # C=C=C 累积二烯

    # ============================================================
    # 含氧官能团
    # ============================================================
    "alcohol": "[OX2H][CX4]",                # 脂肪醇，排除 phenol / enol / carboxylic acid
    "phenol": "[OX2H][cX3]",                 # 酚羟基
    "enol": "[OX2H][CX3]=[CX3]",             # 烯醇
    "ether": "[OD2]([#6;!$(C=O)])[#6;!$(C=O)]",  # 醚，排除酯/酸酐中的 C(=O)-O-C
    "epoxide": "[OX2r3]1[#6r3][#6r3]1",      # 环氧
    "aldehyde": "[$([CX3H1](=O)[#6]),$([CX3H2]=O)]",  # 醛，包括甲醛
    "ketone": "[#6][CX3](=O)[#6]",           # 酮
    "carboxylic_acid": "[CX3](=O)[OX2H1]",   # 羧酸
    "carboxylate": "[CX3](=O)[O-]",          # 羧酸盐 / 去质子化羧酸
    "ester": "[$([#6][CX3](=O)[OX2H0][#6]),$([CX3H1](=O)[OX2H0][#6])]",  # 酯，包括甲酸酯
    "lactone": "[CX3r](=O)[OX2r]",           # 内酯，环状酯
    "acid_anhydride": "[CX3](=[OX1])[OX2][CX3](=[OX1])",  # 酸酐
    "acyl_halide": "[CX3](=[OX1])[F,Cl,Br,I]",            # 酰卤
    "carbonate": "[OX2][CX3](=[OX1])[OX2]",  # 碳酸酯 / 碳酸结构
    "acetal": "[CX4]([OX2][#6])([OX2][#6])", # 缩醛 / 缩酮
    "hemiacetal": "[CX4]([OX2H])([OX2][#6])",# 半缩醛 / 半缩酮
    "peroxide": "[OX2][OX2]",                # 过氧键 O-O

    # ============================================================
    # 含氮官能团
    # ============================================================
    "amine": (
        "[NX3;H2,H1,H0;"
        "!$([N][CX3]=[OX1]);"                # 排除酰胺/氨基甲酸酯/脲
        "!$([N][SX4](=[OX1])(=[OX1]));"      # 排除磺酰胺
        "!$([N+](=O)[O-]);"                  # 排除硝基
        "!$([N](=O)=O);"                     # 排除中性写法硝基
        "!$([N]=[#6]);"                      # 排除亚胺型 N
        "!$([N][O]);"                        # 排除羟胺/肟等 N-O
        "!$([N][N])"                         # 排除肼/腙等 N-N
        "]"
    ),
    "primary_amine": "[NX3H2;!$([N][CX3]=[OX1]);!$([N][SX4](=[OX1])(=[OX1]))][#6]",
    "secondary_amine": "[NX3H1;!$([N][CX3]=[OX1]);!$([N][SX4](=[OX1])(=[OX1]))]([#6])[#6]",
    "tertiary_amine": "[NX3H0;!$([N][CX3]=[OX1]);!$([N][SX4](=[OX1])(=[OX1]))]([#6])([#6])[#6]",

    "amide": "[$([NX3][CX3](=[OX1])[#6]),$([NX3][CX3H1]=[OX1])]",  # 普通羧酰胺，含甲酰胺
    "lactam": "[NX3r][CX3r](=[OX1])",       # 内酰胺，环状酰胺
    "imide": "[CX3](=[OX1])[NX3][CX3](=[OX1])",
    "carbamate": "[NX3][CX3](=[OX1])[OX2][#6]",  # 氨基甲酸酯
    "urea": "[NX3][CX3](=[OX1])[NX3]",      # 脲结构
    "nitrile": "[CX2]#[NX1]",               # 腈
    "isocyanide": "[CX1-]#[NX2+]",          # 异腈
    "isocyanate": "[#6][NX2]=[CX2]=[OX1]",  # 异氰酸酯 R-N=C=O
    "nitro": "[$([NX3](=O)=O),$([NX3+](=O)[O-])]",
    "nitroso": "[#6][NX2]=[OX1]",           # 亚硝基 R-N=O
    "imine": "[CX3]=[NX2;!$([N][O]);!$([N][N])]",  # 亚胺，排除肟/腙
    "oxime": "[CX3]=[NX2][OX2H]",           # 肟
    "hydrazone": "[CX3]=[NX2][NX3]",        # 腙
    "hydrazine": "[#6][NX3][NX3]",          # 有机肼
    "azo": "[#6][NX2]=[NX2][#6]",           # 偶氮 R-N=N-R
    "diazo": "[#6]=[NX2+]=[NX1-]",          # 重氮
    "azide": "[#6][NX2]=[NX2+]=[NX1-]",     # 有机叠氮
    "enamine": "[NX3][CX3]=[CX3]",          # 烯胺

    # ============================================================
    # 含硫官能团
    # ============================================================
    "thiol": "[#6][SX2H]",                  # 硫醇
    "sulfide": "[#6][SX2][#6]",             # 硫醚
    "disulfide": "[#6][SX2][SX2][#6]",      # 二硫键
    "sulfoxide": "[#6][SX3](=[OX1])[#6]",   # 亚砜
    "sulfone": "[#6][SX4](=[OX1])(=[OX1])[#6]",  # 砜
    "sulfonic_acid": "[#6][SX4](=[OX1])(=[OX1])[OX2H]",  # 磺酸
    "sulfonate": "[#6][SX4](=[OX1])(=[OX1])[$([OX2][#6]),$([OX1-])]",  # 磺酸酯/磺酸盐
    "sulfonamide": "[#6][SX4](=[OX1])(=[OX1])[NX3]",     # 磺酰胺
    "sulfonyl_halide": "[#6][SX4](=[OX1])(=[OX1])[F,Cl,Br,I]",  # 磺酰卤
    "thioester": "[CX3](=[OX1])[SX2][#6]",  # 硫酯
    "thioamide": "[NX3][CX3]=[SX1]",        # 硫代酰胺

    # ============================================================
    # 卤素官能团
    # ============================================================
    "halide": "[#6;!$(C=O)][F,Cl,Br,I]",     # 有机卤素，排除酰卤
    "alkyl_halide": "[CX4][F,Cl,Br,I]",      # 烷基卤
    "aryl_halide": "[c][F,Cl,Br,I]",         # 芳基卤
    "vinyl_halide": "[CX3]=[CX3][F,Cl,Br,I]",# 烯基卤

    # ============================================================
    # 磷、硼、硅等
    # ============================================================
    "phosphine": "[PX3;!$([P]=O);!$([P]=S)][#6]",  # 有机膦
    "phosphine_oxide": "[PX4](=[OX1])([#6])([#6])[#6]",  # 膦氧
    "phosphate": "[PX4](=[OX1])([OX2])([OX2])[OX2]",     # 磷酸酯/磷酸
    "phosphonate": "[#6][PX4](=[OX1])([OX2])[OX2]",      # 膦酸/膦酸酯，含 C-P
    "boronic_acid": "[#6][BX3]([OX2H])([OX2H])",         # 有机硼酸
    "boronic_ester": "[#6][BX3]([OX2][#6])([OX2][#6])",  # 有机硼酸酯
    "organosilicon": "[#6][SiX4]",                       # 有机硅
    "silane": "[#6][SiX4;H1,H2,H3]",                     # 含 Si-H 的有机硅烷
    "silyl_ether": "[#6][OX2][SiX4]",                    # 硅醚
}

# Precompiled SMARTS patterns — built once at import time so
# ``functional_group_labels`` does not re-parse ~60 SMARTS on every call.
_PRECOMPILED_SMARTS: list[tuple[str, Chem.Mol]] = []
for _label, _smarts in FUNCTIONAL_GROUP_SMARTS.items():
    _pattern = Chem.MolFromSmarts(_smarts)
    if _pattern is not None:
        _PRECOMPILED_SMARTS.append((_label, _pattern))


def sample_fg(sample: dict[str, Any]) -> list[str]:
    """Return the list of functional-group labels for a sample.

    Parameters
    ----------
    sample
        Sample dictionary.

    Returns
    -------
    list[str]
        Functional-group labels (may be empty).
    """
    return sample.get("functional_groups", [])


def sample_smiles(sample: dict[str, Any]) -> str | None:
    """Extract the canonical SMILES from a sample dictionary.

    Parameters
    ----------
    sample
        SpectraLM sample dictionary.

    Returns
    -------
    str | None
        Canonical SMILES string, or ``None`` if unavailable.
    """
    return sample.get("canonical_smiles") or sample.get("smiles")


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


@lru_cache(maxsize=8192)
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


def mol_from_smiles(smiles: str | None) -> Chem.Mol | None:
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


@lru_cache(maxsize=8192)
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
    for label, pattern in _PRECOMPILED_SMARTS:
        if mol.HasSubstructMatch(pattern):
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

