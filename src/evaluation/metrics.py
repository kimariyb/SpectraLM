"""Structure, spectral-consistency, and model-behavior evaluation metrics."""

from __future__ import annotations

import re
from typing import Any

import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem.Scaffolds import MurckoScaffold

from src.data.functional_groups import functional_groups
from src.data.molecules import canonicalize_smiles
from src.evaluation.spectral_consistency import (
    evaluate_functional_group_spectral_consistency,
)
from src.nmr_rules.validator import validate_candidate


def classify_output_behavior(text: str) -> dict[str, bool]:
    """Classify a response into one disjoint output-behavior state."""
    stripped = str(text or "").strip()
    nonempty_lines = [line.strip() for line in stripped.splitlines() if line.strip()]
    bare = (
        bool(stripped)
        and len(nonempty_lines) == 1
        and stripped == nonempty_lines[0]
        and not any(character.isspace() for character in stripped)
        and "`" not in stripped
        and ":" not in stripped
    )
    if not bare:
        return {
            "output_format_compliant": False,
            "illegal_structure": False,
            "non_smiles_output": True,
        }
    if canonicalize_smiles(stripped) is None:
        return {
            "output_format_compliant": False,
            "illegal_structure": True,
            "non_smiles_output": False,
        }
    return {
        "output_format_compliant": True,
        "illegal_structure": False,
        "non_smiles_output": False,
    }


def extract_final_smiles(text: str) -> str | None:
    """Extract a SMILES candidate from a model response."""
    patterns = [
        r"Final canonical SMILES\s*:\s*([^\s`]+)",
        r"Final SMILES\s*:\s*([^\s`]+)",
        r"Canonical SMILES\s*:\s*([^\s`]+)",
        r"SMILES\s*:\s*([^\s`]+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return match.group(1).strip()

    fenced_blocks = re.findall(
        r"```(?:plaintext|text|smiles)?\s*\n?(.*?)```",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    for block in fenced_blocks:
        lines = [line.strip() for line in block.splitlines() if line.strip()]
        if lines:
            return lines[0]

    lines = [
        line.strip()
        for line in text.strip().splitlines()
        if line.strip() and not line.strip().startswith("```")
    ]
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


def _connectivity_smiles(smiles: str | None) -> str | None:
    """Canonicalize molecular connectivity while ignoring stereochemistry."""
    canonical = canonicalize_smiles(smiles)
    mol = Chem.MolFromSmiles(canonical) if canonical is not None else None
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=False)


def _molecular_formula(smiles: str | None) -> str | None:
    """Return the RDKit molecular formula for a valid SMILES string."""
    canonical = canonicalize_smiles(smiles)
    mol = Chem.MolFromSmiles(canonical) if canonical is not None else None
    if mol is None:
        return None
    return rdMolDescriptors.CalcMolFormula(mol)


def _ring_scaffold(smiles: str | None) -> str | None:
    """Return a canonical non-empty Murcko ring scaffold."""
    canonical = canonicalize_smiles(smiles)
    mol = Chem.MolFromSmiles(canonical) if canonical is not None else None
    if mol is None:
        return None
    scaffold = MurckoScaffold.MurckoScaffoldSmiles(
        mol=mol,
        includeChirality=False,
    )
    return scaffold or None


def _functional_group_metrics(
    predicted: str | None,
    reference: str | None,
) -> dict[str, Any]:
    predicted_groups = functional_groups(predicted)
    reference_groups = functional_groups(reference)
    if canonicalize_smiles(predicted) is None or canonicalize_smiles(reference) is None:
        precision = recall = f1 = 0.0
    elif not predicted_groups and not reference_groups:
        precision = recall = f1 = 1.0
    else:
        overlap = len(predicted_groups & reference_groups)
        precision = overlap / len(predicted_groups) if predicted_groups else 0.0
        recall = overlap / len(reference_groups) if reference_groups else 0.0
        f1 = (
            2.0 * precision * recall / (precision + recall)
            if precision + recall > 0
            else 0.0
        )
    return {
        "predicted_functional_groups": sorted(predicted_groups),
        "reference_functional_groups": sorted(reference_groups),
        "functional_group_precision": precision,
        "functional_group_recall": recall,
        "functional_group_f1": f1,
    }


def evaluate_candidate_ranking(
    ranked_candidates: list[str],
    reference_smiles: str | None,
) -> dict[str, float]:
    """Calculate candidate Top-1 accuracy and reciprocal rank."""
    reference = canonicalize_smiles(reference_smiles)
    ranked = [canonicalize_smiles(candidate) for candidate in ranked_candidates]
    try:
        rank = ranked.index(reference) + 1 if reference is not None else 0
    except ValueError:
        rank = 0
    return {
        "candidate_top1_accuracy": float(rank == 1),
        "candidate_mrr": 1.0 / rank if rank else 0.0,
    }


def evaluate_structure_prediction(
    prediction_text: str,
    reference_smiles: str | None,
    *,
    sample: dict[str, Any] | None = None,
    include_formula: bool = True,
) -> dict[str, Any]:
    """Evaluate one direct structure prediction with optional rule checks."""
    predicted = predicted_smiles(prediction_text)
    reference = canonicalize_smiles(reference_smiles)
    predicted_connectivity = _connectivity_smiles(predicted)
    reference_connectivity = _connectivity_smiles(reference)
    predicted_formula = _molecular_formula(predicted)
    reference_formula = _molecular_formula(reference)
    predicted_scaffold = _ring_scaffold(predicted)
    reference_scaffold = _ring_scaffold(reference)
    scaffold_evaluable = reference_scaffold is not None
    formula_match = (
        predicted_formula is not None
        and reference_formula is not None
        and predicted_formula == reference_formula
    )
    result = {
        "predicted_smiles": predicted,
        "reference_smiles": reference,
        "valid_smiles": predicted is not None,
        "exact_match": (
            predicted is not None
            and reference is not None
            and predicted == reference
        ),
        "connectivity_exact_match": (
            predicted_connectivity is not None
            and reference_connectivity is not None
            and predicted_connectivity == reference_connectivity
        ),
        "predicted_formula": predicted_formula,
        "reference_formula": reference_formula,
        "formula_match": formula_match,
        "molecular_formula_match": formula_match,
        "predicted_scaffold": predicted_scaffold,
        "reference_scaffold": reference_scaffold,
        "scaffold_evaluable": scaffold_evaluable,
        "scaffold_match": (
            predicted_scaffold == reference_scaffold
            if scaffold_evaluable
            else None
        ),
        "tanimoto": tanimoto_similarity(predicted, reference),
        **_functional_group_metrics(predicted, reference),
        **classify_output_behavior(prediction_text),
    }
    if sample is not None:
        result.update(
            validate_candidate(
                predicted,
                sample,
                include_formula=include_formula,
            ).to_metrics()
        )
        result.update(
            evaluate_functional_group_spectral_consistency(predicted, sample)
        )
    return result


def summarize_structure_predictions(
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    """Aggregate exact match, validity, and Tanimoto metrics."""
    total = len(rows)
    if total == 0:
        return {"samples": 0}
    similarities = np.asarray(
        [float(row["tanimoto"]) for row in rows],
        dtype=np.float64,
    )
    summary = {
        "samples": total,
        "exact_match": sum(bool(row["exact_match"]) for row in rows) / total,
        "connectivity_exact_match": (
            sum(bool(row["connectivity_exact_match"]) for row in rows) / total
        ),
        "formula_match_rate": (
            sum(bool(row["formula_match"]) for row in rows) / total
        ),
        "molecular_formula_accuracy": (
            sum(bool(row["molecular_formula_match"]) for row in rows) / total
        ),
        "valid_smiles_rate": (
            sum(bool(row["valid_smiles"]) for row in rows) / total
        ),
        "mean_tanimoto": float(np.mean(similarities)),
        "median_tanimoto": float(np.median(similarities)),
        "tanimoto_q25": float(np.quantile(similarities, 0.25)),
        "tanimoto_q75": float(np.quantile(similarities, 0.75)),
        "tanimoto_ge_0_3_rate": float(np.mean(similarities >= 0.3)),
        "tanimoto_ge_0_5_rate": float(np.mean(similarities >= 0.5)),
        "tanimoto_ge_0_7_rate": float(np.mean(similarities >= 0.7)),
        "mean_functional_group_f1": float(
            np.mean([float(row["functional_group_f1"]) for row in rows])
        ),
        "output_format_compliance_rate": (
            sum(bool(row["output_format_compliant"]) for row in rows) / total
        ),
        "illegal_structure_rate": (
            sum(bool(row["illegal_structure"]) for row in rows) / total
        ),
        "non_smiles_output_rate": (
            sum(bool(row["non_smiles_output"]) for row in rows) / total
        ),
    }
    scaffold_rows = [row for row in rows if row["scaffold_evaluable"]]
    summary["scaffold_coverage"] = len(scaffold_rows) / total
    summary["scaffold_match_rate"] = (
        sum(bool(row["scaffold_match"]) for row in scaffold_rows)
        / len(scaffold_rows)
        if scaffold_rows
        else None
    )
    spectral_rows = [
        row
        for row in rows
        if row.get("functional_group_spectral_consistency") is not None
    ]
    summary["functional_group_spectral_consistency_coverage"] = (
        len(spectral_rows) / total
    )
    summary["mean_functional_group_spectral_consistency"] = (
        float(
            np.mean(
                [
                    float(row["functional_group_spectral_consistency"])
                    for row in spectral_rows
                ]
            )
        )
        if spectral_rows
        else None
    )
    rule_rows = [row for row in rows if "rule_consistency_rate" in row]
    if rule_rows:
        summary["mean_rule_consistency_rate"] = float(
            np.mean([float(row["rule_consistency_rate"]) for row in rule_rows])
        )
        summary["rule_contradiction_rate"] = sum(
            int(row.get("rule_contradiction_count", 0)) > 0
            for row in rule_rows
        ) / len(rule_rows)
    return summary
