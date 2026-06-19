"""Structure-only metrics for direct NMR-to-SMILES prediction."""

from __future__ import annotations

import re
from typing import Any

from rdkit import Chem, DataStructs
from rdkit.Chem import rdMolDescriptors

from src.data.molecules import canonicalize_smiles


def extract_final_smiles(text: str) -> str | None:
    """Extract a SMILES candidate from a model response."""
    patterns = [
        r"Final canonical SMILES\s*:\s*(\S+)",
        r"Final SMILES\s*:\s*(\S+)",
        r"SMILES\s*:\s*(\S+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return match.group(1).strip()
    lines = text.strip().splitlines()
    return lines[-1].strip() if lines else None


def predicted_smiles(text: str) -> str | None:
    """Resolve and canonicalize the best structure candidate in a response."""
    return canonicalize_smiles(extract_final_smiles(text))


def tanimoto_similarity(predicted: str | None, reference: str | None) -> float:
    """Calculate radius-2 Morgan fingerprint Tanimoto similarity."""
    pred_canonical = canonicalize_smiles(predicted)
    ref_canonical = canonicalize_smiles(reference)
    if pred_canonical is None or ref_canonical is None:
        return 0.0

    pred_mol = Chem.MolFromSmiles(pred_canonical)
    ref_mol = Chem.MolFromSmiles(ref_canonical)
    if pred_mol is None or ref_mol is None:
        return 0.0
    pred_fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(
        pred_mol,
        radius=2,
        nBits=1024,
    )
    ref_fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(
        ref_mol,
        radius=2,
        nBits=1024,
    )
    return float(DataStructs.TanimotoSimilarity(pred_fp, ref_fp))


def evaluate_structure_prediction(
    prediction_text: str,
    reference_smiles: str | None,
) -> dict[str, Any]:
    """Evaluate one direct structure prediction."""
    predicted = predicted_smiles(prediction_text)
    reference = canonicalize_smiles(reference_smiles)
    return {
        "predicted_smiles": predicted,
        "reference_smiles": reference,
        "valid_smiles": predicted is not None,
        "exact_match": (
            predicted is not None
            and reference is not None
            and predicted == reference
        ),
        "tanimoto": tanimoto_similarity(predicted, reference),
    }


def summarize_structure_predictions(
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    """Aggregate exact match, validity, and Tanimoto metrics."""
    total = len(rows)
    if total == 0:
        return {"samples": 0}
    return {
        "samples": total,
        "exact_match": sum(bool(row["exact_match"]) for row in rows) / total,
        "valid_smiles_rate": (
            sum(bool(row["valid_smiles"]) for row in rows) / total
        ),
        "mean_tanimoto": sum(float(row["tanimoto"]) for row in rows) / total,
    }
