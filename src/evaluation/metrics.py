"""Evaluation metrics for structure prediction and spectral reasoning outputs."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any

import selfies as sf
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem

from src.config import load_config
from src.data.molecules import canonicalize_smiles, functional_group_labels, sample_smiles
from src.data.utils import peak_count
from src.io import load_pickle_list, write_json


def extract_final_smiles(text: str) -> str | None:
    """Extract the final SMILES string from model output.

    Parameters
    ----------
    text
        Model output text.

    Returns
    -------
    str | None
        Extracted SMILES candidate.
    """
    patterns = [
        r"Final canonical SMILES\s*:\s*(\S+)",
        r"Final SMILES\s*:\s*(\S+)",
        r"SMILES\s*:\s*(\S+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return match.group(1).strip()
    stripped = text.strip().splitlines()
    return stripped[-1].strip() if stripped else None


def extract_final_selfies(text: str) -> str | None:
    """Extract the final SELFIES string from model output.

    Parameters
    ----------
    text
        Model output text.

    Returns
    -------
    str | None
        Extracted SELFIES candidate.
    """
    match = re.search(r"Final SELFIES\s*:\s*(.+)", text, flags=re.IGNORECASE)
    return match.group(1).strip() if match else None


def selfies_to_smiles(value: str | None) -> str | None:
    """Decode SELFIES to SMILES.

    Parameters
    ----------
    value
        SELFIES string.

    Returns
    -------
    str | None
        Decoded SMILES or ``None``.
    """
    if not value:
        return None
    try:
        return sf.decoder(value)
    except Exception:
        return None


def predicted_smiles(text: str) -> str | None:
    """Resolve the best SMILES candidate from model output.

    Parameters
    ----------
    text
        Model output text.

    Returns
    -------
    str | None
        Canonicalized predicted SMILES when valid.
    """
    selfies_value = extract_final_selfies(text)
    smiles = selfies_to_smiles(selfies_value) if selfies_value else None
    smiles = smiles or extract_final_smiles(text)
    return canonicalize_smiles(smiles)


def morgan_fingerprint(smiles: str | None):
    """Build a Morgan fingerprint for a molecule.

    Parameters
    ----------
    smiles
        Input SMILES string.

    Returns
    -------
    rdkit.DataStructs.cDataStructs.ExplicitBitVect | None
        Fingerprint or ``None`` for invalid molecules.
    """
    mol = Chem.MolFromSmiles(smiles) if smiles else None
    if mol is None:
        return None
    return AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)


def tanimoto_similarity(predicted: str | None, reference: str | None) -> float:
    """Calculate Morgan fingerprint Tanimoto similarity.

    Parameters
    ----------
    predicted
        Predicted canonical SMILES.
    reference
        Reference canonical SMILES.

    Returns
    -------
    float
        Tanimoto similarity, or zero for invalid molecules.
    """
    pred_fp = morgan_fingerprint(predicted)
    ref_fp = morgan_fingerprint(reference)
    if pred_fp is None or ref_fp is None:
        return 0.0
    return float(DataStructs.TanimotoSimilarity(pred_fp, ref_fp))


def functional_group_jaccard(predicted: str | None, reference: str | None) -> float:
    """Calculate functional-group label Jaccard similarity.

    Parameters
    ----------
    predicted
        Predicted SMILES.
    reference
        Reference SMILES.

    Returns
    -------
    float
        Jaccard similarity between detected functional groups.
    """
    pred_labels = set(functional_group_labels(predicted))
    ref_labels = set(functional_group_labels(reference))
    if not pred_labels and not ref_labels:
        return 1.0
    if "invalid" in pred_labels:
        return 0.0
    return len(pred_labels & ref_labels) / len(pred_labels | ref_labels)


def reasoning_mentions_required_evidence(text: str) -> bool:
    """Check whether reasoning mentions core NMR evidence types.

    Parameters
    ----------
    text
        Model output text.

    Returns
    -------
    bool
        ``True`` when 1H, 13C, and spectral reasoning are all mentioned.
    """
    lower = text.lower()
    return ("1h" in lower or "proton" in lower) and ("13c" in lower or "carbon" in lower) and (
        "reasoning" in lower or "nmr" in lower
    )


def nmr_rule_violations(prediction_text: str, predicted: str | None, reference: dict[str, Any]) -> list[str]:
    """Detect coarse NMR rule violations in a prediction.

    Parameters
    ----------
    prediction_text
        Raw model output text.
    predicted
        Predicted canonical SMILES.
    reference
        Reference sample dictionary.

    Returns
    -------
    list[str]
        Rule violation labels.
    """
    violations = []
    if predicted is None:
        violations.append("invalid_structure")
    if not reasoning_mentions_required_evidence(prediction_text):
        violations.append("missing_core_reasoning_evidence")
    reference_carbons = peak_count(reference, "13C_NMR")
    mol = Chem.MolFromSmiles(predicted) if predicted else None
    if mol is not None and reference_carbons > 0:
        carbon_atoms = sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() == 6)
        if carbon_atoms < max(1, reference_carbons // 3):
            violations.append("carbon_count_too_low_for_13c_peaks")
    return violations


def evaluate_one(prediction_text: str, reference: dict[str, Any]) -> dict[str, Any]:
    """Evaluate one model output against one reference sample.

    Parameters
    ----------
    prediction_text
        Raw model output text.
    reference
        Reference sample dictionary.

    Returns
    -------
    dict[str, Any]
        Per-sample evaluation metrics.
    """
    pred = predicted_smiles(prediction_text)
    ref = canonicalize_smiles(reference.get("canonical_smiles") or sample_smiles(reference))
    violations = nmr_rule_violations(prediction_text, pred, reference)
    return {
        "id": reference.get("id"),
        "predicted_smiles": pred,
        "reference_smiles": ref,
        "exact_match": pred is not None and ref is not None and pred == ref,
        "tanimoto": tanimoto_similarity(pred, ref),
        "functional_group_jaccard": functional_group_jaccard(pred, ref),
        "reasoning_core_evidence": reasoning_mentions_required_evidence(prediction_text),
        "nmr_rule_violations": violations,
        "nmr_rule_violation_count": len(violations),
    }


def summarize_results(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize per-sample evaluation rows.

    Parameters
    ----------
    rows
        Per-sample metric rows.

    Returns
    -------
    dict[str, Any]
        Aggregate metrics.
    """
    total = len(rows)
    if total == 0:
        return {"samples": 0}
    return {
        "samples": total,
        "exact_match": sum(row["exact_match"] for row in rows) / total,
        "mean_tanimoto": sum(row["tanimoto"] for row in rows) / total,
        "mean_functional_group_jaccard": sum(row["functional_group_jaccard"] for row in rows) / total,
        "reasoning_core_evidence_rate": sum(row["reasoning_core_evidence"] for row in rows) / total,
        "mean_nmr_rule_violation_count": sum(row["nmr_rule_violation_count"] for row in rows) / total,
        "invalid_structure_rate": sum(row["predicted_smiles"] is None for row in rows) / total,
    }


def evaluate_predictions(
    predictions: list[str],
    references: list[dict[str, Any]],
) -> dict[str, Any]:
    """Evaluate predicted model outputs against reference samples.

    Parameters
    ----------
    predictions
        Raw model output strings.
    references
        Reference sample dictionaries.

    Returns
    -------
    dict[str, Any]
        Aggregate and per-sample metrics.
    """
    rows = [evaluate_one(prediction, reference) for prediction, reference in zip(predictions, references)]
    return {"summary": summarize_results(rows), "rows": rows}


def load_prediction_texts(path: str | Path) -> list[str]:
    """Load predictions from a text, JSON, or JSONL file.

    Parameters
    ----------
    path
        Prediction file path.

    Returns
    -------
    list[str]
        Prediction texts.
    """
    prediction_path = Path(path)
    if prediction_path.suffix == ".jsonl":
        texts = []
        with prediction_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                texts.append(row.get("prediction") or row.get("text") or row.get("output") or "")
        return texts
    if prediction_path.suffix == ".json":
        payload = json.loads(prediction_path.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            return [row.get("prediction", row) if isinstance(row, dict) else str(row) for row in payload]
    return prediction_path.read_text(encoding="utf-8").splitlines()


def run(config: dict[str, Any]) -> None:
    """Run evaluation from a configuration dictionary.

    Parameters
    ----------
    config
        Configuration dictionary with keys ``predictions``, ``references``, and ``output``.
    """
    predictions_path = config.get("predictions", "outputs/predictions.jsonl")
    references_path = config.get("references", "dataset/subsets/spectralm_butina_1000_300/test.pkl")
    output_path = config.get("output", "outputs/evaluation_report.json")
    predictions = load_prediction_texts(predictions_path)
    references = load_pickle_list(references_path)
    report = evaluate_predictions(predictions, references)
    write_json(output_path, report)
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m spectralm.evaluation.metrics <config.yaml>")
        sys.exit(1)
    run(load_config(sys.argv[1]))
