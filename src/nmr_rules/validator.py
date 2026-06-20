"""Candidate-SMILES validation against observable one-dimensional NMR rules."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from rdkit import Chem
from rdkit.Chem import rdMolDescriptors

from src.data.molecules import canonicalize_smiles
from src.nmr_rules.engine import _c13_shifts, analyze_sample
from src.nmr_rules.formula import calculate_dbe, parse_formula


_FRAGMENT_SMARTS = {
    "H1_FRAGMENT_ETHYL_001": "[CH3][CH2]",
    "H1_FRAGMENT_ISOPROPYL_001": "[CH]([CH3])([CH3])",
    "H1_FRAGMENT_TERT_BUTYL_001": "[C;H0]([CH3])([CH3])([CH3])",
    "H1_FRAGMENT_METHOXY_001": "[CH3][O,N,S]",
}


@dataclass(frozen=True)
class CandidateValidation:
    """Rule-consistency result for one candidate structure."""

    valid_smiles: bool
    predicted_smiles: str | None
    checks: dict[str, bool]
    applicable_checks: int
    satisfied_checks: int
    consistency_rate: float
    contradictions: tuple[str, ...]

    def to_metrics(self) -> dict[str, Any]:
        """Return flat fields suitable for prediction JSONL records."""
        return {
            "rule_checks": dict(self.checks),
            "rule_checks_applicable": self.applicable_checks,
            "rule_checks_satisfied": self.satisfied_checks,
            "rule_consistency_rate": self.consistency_rate,
            "rule_contradictions": list(self.contradictions),
            "rule_contradiction_count": len(self.contradictions),
        }


def validate_candidate(
    smiles: str | None,
    sample: dict[str, Any],
    *,
    include_formula: bool = True,
) -> CandidateValidation:
    """Validate a candidate against objective formula and 1D NMR constraints.

    Broad chemical-shift regions are not used as rejection criteria. Fragment
    checks are applied only when the rule engine detects their diagnostic peak
    pattern.

    Parameters
    ----------
    smiles
        Candidate SMILES.
    sample
        Sample-visible formula and one-dimensional peak tables.
    include_formula
        Whether the supplied formula may be used.

    Returns
    -------
    CandidateValidation
        Per-check outcomes and aggregate consistency.
    """
    canonical = canonicalize_smiles(smiles)
    if canonical is None:
        return CandidateValidation(
            valid_smiles=False,
            predicted_smiles=None,
            checks={"valid_smiles": False},
            applicable_checks=1,
            satisfied_checks=0,
            consistency_rate=0.0,
            contradictions=("valid_smiles",),
        )

    mol = Chem.MolFromSmiles(canonical)
    if mol is None:
        raise AssertionError("Canonical SMILES unexpectedly failed RDKit parsing.")
    predicted_formula = rdMolDescriptors.CalcMolFormula(mol)
    predicted_counts = parse_formula(predicted_formula)
    checks: dict[str, bool] = {"valid_smiles": True}

    supplied_value = sample.get("molecular_formula") if include_formula else None
    supplied_formula = str(supplied_value).strip() if supplied_value else None
    if supplied_formula:
        try:
            checks["formula_match"] = predicted_counts == parse_formula(
                supplied_formula
            )
            checks["dbe_match"] = calculate_dbe(predicted_formula) == calculate_dbe(
                supplied_formula
            )
        except ValueError:
            checks["formula_match"] = False
            checks["dbe_match"] = False

    c13_shifts = _c13_shifts(sample)
    if c13_shifts:
        observed_signals = len({round(value, 3) for value in c13_shifts})
        checks["c13_signal_count_feasible"] = (
            observed_signals <= predicted_counts.get("C", 0)
        )

    analysis = analyze_sample(sample, include_formula=include_formula)
    detected_ids = {item.rule_id for item in analysis.evidence}
    for rule_id, smarts in _FRAGMENT_SMARTS.items():
        if rule_id not in detected_ids:
            continue
        pattern = Chem.MolFromSmarts(smarts)
        checks[rule_id] = bool(pattern is not None and mol.HasSubstructMatch(pattern))

    contradictions = tuple(name for name, passed in checks.items() if not passed)
    applicable = len(checks)
    satisfied = sum(checks.values())
    return CandidateValidation(
        valid_smiles=True,
        predicted_smiles=canonical,
        checks=checks,
        applicable_checks=applicable,
        satisfied_checks=satisfied,
        consistency_rate=satisfied / applicable,
        contradictions=contradictions,
    )
