"""Structure, spectral-consistency, and model-behavior evaluation metrics."""

from __future__ import annotations

import re
from typing import Any

import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem.Scaffolds import MurckoScaffold

from src.data.functional_groups import FUNCTIONAL_GROUP_SMARTS, functional_groups
from src.data.molecules import (
    canonicalize_smiles,
    has_explicit_stereochemistry,
    inspect_dataset_molecule,
)
from src.evaluation.generation_metrics import (
    inspect_generation_tokens,
    summarize_generation_behavior,
)
from src.evaluation.multilabel_metrics import summarize_multilabel_predictions
from src.evaluation.spectral_consistency import (
    evaluate_functional_group_spectral_support,
)
from src.nmr_rules.validator import validate_candidate


def classify_output_behavior(text: str) -> dict[str, bool]:
    """Classify a response into one disjoint output-behavior state."""
    stripped = str(text or "").strip()
    nonempty_lines = [
        line.strip() for line in stripped.splitlines() if line.strip()
    ]
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
            "rdkit_invalid_bare_output": False,
            "non_bare_output": True,
        }
    if canonicalize_smiles(stripped) is None:
        return {
            "output_format_compliant": False,
            "rdkit_invalid_bare_output": True,
            "non_bare_output": False,
        }
    return {
        "output_format_compliant": True,
        "rdkit_invalid_bare_output": False,
        "non_bare_output": False,
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
        nBits=2048,
        useChirality=False,
    )
    ref_fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(
        ref_mol,
        radius=2,
        nBits=2048,
        useChirality=False,
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


def _domain_validity(smiles: str | None) -> dict[str, bool]:
    """Check whether a valid SMILES satisfies the project molecule policy."""
    inspection = inspect_dataset_molecule(smiles)
    valid = inspection.canonical_smiles is not None
    allowed = valid and not any(
        violation.startswith("unsupported_elements:")
        for violation in inspection.violations
    )
    single_component = valid and inspection.component_count == 1
    neutral = valid and inspection.formal_charge == 0
    no_radicals = valid and inspection.radical_electron_count == 0
    no_isotope_labels = valid and inspection.isotope_label_count == 0
    return {
        "has_only_allowed_elements": allowed,
        "is_single_component": single_component,
        "is_neutral": neutral,
        "has_no_radicals": no_radicals,
        "has_no_isotope_labels": no_isotope_labels,
        "domain_valid_smiles": (
            inspection.accepted and no_isotope_labels
        ),
    }


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
    """Calculate candidate coverage, validity, Hits@k, and reciprocal rank."""
    reference = canonicalize_smiles(reference_smiles)
    ranked = [canonicalize_smiles(candidate) for candidate in ranked_candidates]
    try:
        rank = ranked.index(reference) + 1 if reference is not None else 0
    except ValueError:
        rank = 0
    total = len(ranked)
    return {
        "candidate_reference_covered": float(rank > 0),
        "candidate_valid_rate": (
            sum(candidate is not None for candidate in ranked) / total
            if total
            else 0.0
        ),
        "candidate_top1_accuracy": float(rank == 1),
        "candidate_hits_at_3": float(0 < rank <= 3),
        "candidate_hits_at_5": float(0 < rank <= 5),
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
    reference_scaffold_available = reference_scaffold is not None
    formula_match = (
        predicted_formula is not None
        and reference_formula is not None
        and predicted_formula == reference_formula
    )
    result = {
        "predicted_smiles": predicted,
        "reference_smiles": reference,
        "reference_stereochemistry_present": has_explicit_stereochemistry(
            reference
        ),
        "valid_smiles": predicted is not None,
        **_domain_validity(predicted),
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
        "molecular_formula_match": formula_match,
        "predicted_ring_scaffold": predicted_scaffold,
        "reference_ring_scaffold": reference_scaffold,
        "predicted_ring_scaffold_available": predicted_scaffold is not None,
        "reference_ring_scaffold_available": reference_scaffold_available,
        "ring_scaffold_match": (
            predicted_scaffold == reference_scaffold
            if reference_scaffold_available
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
            evaluate_functional_group_spectral_support(predicted, sample)
        )
    return result


def _upgrade_prediction_row(row: dict[str, Any]) -> dict[str, Any]:
    """Map legacy prediction fields to the current summary schema."""
    upgraded = dict(row)
    legacy_schema = (
        "domain_valid_smiles" not in upgraded
        or "predicted_scaffold" in upgraded
        or "illegal_structure" in upgraded
    )
    if legacy_schema and "prediction" in upgraded and "label" in upgraded:
        upgraded.update(
            evaluate_structure_prediction(
                str(upgraded["prediction"]),
                str(upgraded["label"]),
            )
        )
    if "domain_valid_smiles" not in upgraded:
        upgraded.update(_domain_validity(upgraded.get("predicted_smiles")))
    if "predicted_ring_scaffold" not in upgraded:
        upgraded["predicted_ring_scaffold"] = upgraded.get(
            "predicted_scaffold"
        )
    if "reference_ring_scaffold" not in upgraded:
        upgraded["reference_ring_scaffold"] = upgraded.get(
            "reference_scaffold"
        )
    if "predicted_ring_scaffold_available" not in upgraded:
        upgraded["predicted_ring_scaffold_available"] = (
            upgraded["predicted_ring_scaffold"] is not None
        )
    if "reference_ring_scaffold_available" not in upgraded:
        upgraded["reference_ring_scaffold_available"] = bool(
            upgraded.get("scaffold_evaluable")
        )
    if "ring_scaffold_match" not in upgraded:
        upgraded["ring_scaffold_match"] = upgraded.get("scaffold_match")
    if "rdkit_invalid_bare_output" not in upgraded:
        upgraded["rdkit_invalid_bare_output"] = bool(
            upgraded.get("illegal_structure")
        )
    if "non_bare_output" not in upgraded:
        upgraded["non_bare_output"] = bool(
            upgraded.get("non_smiles_output")
        )
    if "functional_group_spectral_support_rate" not in upgraded:
        upgraded["functional_group_spectral_support_rate"] = upgraded.get(
            "functional_group_spectral_consistency"
        )
    return upgraded


def summarize_structure_predictions(
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    """Aggregate exact match, validity, and Tanimoto metrics."""
    rows = [_upgrade_prediction_row(row) for row in rows]
    total = len(rows)
    if total == 0:
        return {"samples": 0}
    similarities = np.asarray(
        [float(row["tanimoto"]) for row in rows],
        dtype=np.float64,
    )
    valid_similarities = np.asarray(
        [float(row["tanimoto"]) for row in rows if bool(row["valid_smiles"])],
        dtype=np.float64,
    )
    summary = {
        "samples": total,
        "exact_match": sum(bool(row["exact_match"]) for row in rows) / total,
        "connectivity_exact_match": (
            sum(bool(row["connectivity_exact_match"]) for row in rows) / total
        ),
        "molecular_formula_accuracy": (
            sum(bool(row["molecular_formula_match"]) for row in rows) / total
        ),
        "valid_smiles_rate": (
            sum(bool(row["valid_smiles"]) for row in rows) / total
        ),
        "domain_valid_smiles_rate": (
            sum(bool(row["domain_valid_smiles"]) for row in rows) / total
        ),
        "mean_tanimoto": float(np.mean(similarities)),
        "mean_tanimoto_valid_only": (
            float(np.mean(valid_similarities))
            if valid_similarities.size
            else None
        ),
        "median_tanimoto": float(np.median(similarities)),
        "tanimoto_q25": float(np.quantile(similarities, 0.25)),
        "tanimoto_q75": float(np.quantile(similarities, 0.75)),
        "tanimoto_ge_0_3_rate": float(np.mean(similarities >= 0.3)),
        "tanimoto_ge_0_5_rate": float(np.mean(similarities >= 0.5)),
        "tanimoto_ge_0_7_rate": float(np.mean(similarities >= 0.7)),
        "output_format_compliance_rate": (
            sum(bool(row["output_format_compliant"]) for row in rows) / total
        ),
        "rdkit_invalid_bare_output_rate": (
            sum(bool(row["rdkit_invalid_bare_output"]) for row in rows) / total
        ),
        "non_bare_output_rate": (
            sum(bool(row["non_bare_output"]) for row in rows) / total
        ),
    }
    achiral_rows = [
        row for row in rows if not row["reference_stereochemistry_present"]
    ]
    stereo_rows = [
        row for row in rows if row["reference_stereochemistry_present"]
    ]
    summary["achiral_reference_coverage"] = len(achiral_rows) / total
    summary["achiral_exact_match"] = (
        sum(bool(row["exact_match"]) for row in achiral_rows)
        / len(achiral_rows)
        if achiral_rows
        else None
    )
    summary["stereo_present_reference_coverage"] = len(stereo_rows) / total
    summary["stereo_present_connectivity_exact_match"] = (
        sum(bool(row["connectivity_exact_match"]) for row in stereo_rows)
        / len(stereo_rows)
        if stereo_rows
        else None
    )
    predicted_groups = [set(row["predicted_functional_groups"]) for row in rows]
    reference_groups = [set(row["reference_functional_groups"]) for row in rows]
    group_summary = summarize_multilabel_predictions(
        predicted_groups,
        reference_groups,
        label_space=[label for label, _ in FUNCTIONAL_GROUP_SMARTS],
    )
    summary.update(
        {
            "functional_group_sample_macro_precision": float(
                np.mean([float(row["functional_group_precision"]) for row in rows])
            ),
            "functional_group_sample_macro_recall": float(
                np.mean([float(row["functional_group_recall"]) for row in rows])
            ),
            "functional_group_sample_macro_f1": float(
                np.mean([float(row["functional_group_f1"]) for row in rows])
            ),
            "functional_group_micro_precision": group_summary[
                "multilabel_micro_precision"
            ],
            "functional_group_micro_recall": group_summary[
                "multilabel_micro_recall"
            ],
            "functional_group_micro_f1": group_summary["multilabel_micro_f1"],
            "functional_group_macro_f1": group_summary["multilabel_macro_f1"],
            "functional_group_supported_macro_f1": group_summary[
                "multilabel_supported_macro_f1"
            ],
            "functional_group_reference_coverage": sum(
                bool(groups) for groups in reference_groups
            )
            / total,
            "functional_group_per_class": group_summary["multilabel_per_class"],
        }
    )

    scaffold_rows = [
        row for row in rows if row["reference_ring_scaffold_available"]
    ]
    summary["reference_ring_scaffold_coverage"] = len(scaffold_rows) / total
    summary["predicted_ring_scaffold_coverage"] = sum(
        bool(row["predicted_ring_scaffold_available"]) for row in rows
    ) / total
    summary["ring_scaffold_match_rate"] = (
        sum(bool(row["ring_scaffold_match"]) for row in scaffold_rows)
        / len(scaffold_rows)
        if scaffold_rows
        else None
    )
    spectral_rows = [
        row
        for row in rows
        if row.get("functional_group_spectral_support_rate") is not None
    ]
    summary["functional_group_spectral_support_coverage"] = (
        len(spectral_rows) / total
    )
    summary["mean_functional_group_spectral_support_rate_applicable"] = (
        float(
            np.mean(
                [
                    float(row["functional_group_spectral_support_rate"])
                    for row in spectral_rows
                ]
            )
        )
        if spectral_rows
        else None
    )
    summary["end_to_end_functional_group_spectral_support_rate"] = float(
        np.mean(
            [
                float(row.get("functional_group_spectral_support_rate") or 0.0)
                for row in rows
            ]
        )
    )

    rule_rows = [row for row in rows if "rule_checks" in row]
    if rule_rows:
        check_names = sorted(
            {name for row in rule_rows for name in row.get("rule_checks", {})}
        )
        summary["rule_check_pass_rates"] = {
            name: {
                "applicable": sum(
                    name in row.get("rule_checks", {}) for row in rule_rows
                ),
                "passed": sum(
                    bool(row.get("rule_checks", {}).get(name)) for row in rule_rows
                ),
                "rate": (
                    sum(
                        bool(row.get("rule_checks", {}).get(name))
                        for row in rule_rows
                    )
                    / sum(
                        name in row.get("rule_checks", {})
                        for row in rule_rows
                    )
                ),
            }
            for name in check_names
        }
        summary["rule_contradiction_rate"] = sum(
            int(row.get("rule_contradiction_count", 0)) > 0
            for row in rule_rows
        ) / len(rule_rows)
    return summary
