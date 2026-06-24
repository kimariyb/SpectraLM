"""Tests for functional-group, spectral-region, and multitask supervision."""

from __future__ import annotations

import json

import pytest

from src.data.functional_groups import (
    functional_groups,
)
from src.data.spectral_regions import classify_spectral_regions
from src.data.tasks import (
    CANDIDATE_RANKING,
    FUNCTIONAL_GROUP_RECOGNITION,
    SPECTRAL_REGION_CLASSIFICATION,
    STRUCTURE_PREDICTION,
    build_task_example,
    normalize_task_weights,
)


def test_functional_group_ontology_recognizes_common_oxygen_groups() -> None:
    """Mutually informative oxygen labels should be detected from structures."""
    assert functional_groups("CCO") >= {"alcohol"}
    assert functional_groups("Oc1ccccc1") >= {"phenol", "aromatic_ring"}
    assert functional_groups("CC(=O)C") >= {"ketone"}
    assert functional_groups("CC(=O)O") >= {"carboxylic_acid"}
    assert functional_groups("CC(=O)OC") >= {"ester"}
    assert functional_groups("CC(=O)N") >= {"amide"}


def test_structure_task_can_supervise_connectivity_only(ethanol_sample) -> None:
    """New two-stage runs should remove unobservable stereochemistry."""
    sample = dict(ethanol_sample)
    sample["canonical_smiles"] = "F[C@H](Cl)Br"

    example = build_task_example(
        sample,
        STRUCTURE_PREDICTION,
        target_stereochemistry="remove",
    )

    assert "@" not in example.target


def test_functional_group_ontology_covers_requested_hetero_elements() -> None:
    """The ontology should cover N, halogen, S, P, and Si functionality."""
    assert functional_groups("CCN") >= {"amine"}
    assert functional_groups("CC#N") >= {"nitrile"}
    assert functional_groups("CCCl") >= {"organohalogen"}
    assert functional_groups("CS") >= {"thiol"}
    assert functional_groups("CSC") >= {"thioether"}
    assert functional_groups("CS(=O)C") >= {"sulfoxide"}
    assert functional_groups("CS(=O)(=O)C") >= {"sulfone"}
    assert functional_groups("COP(=O)(O)O") >= {"phosphorus_oxygen"}
    assert functional_groups("C[SiH2]C") >= {"silicon_carbon"}
    assert functional_groups("C[Si](C)(C)O[Si](C)(C)C") >= {"siloxane"}


def test_functional_group_ontology_returns_empty_for_invalid_smiles() -> None:
    """Invalid structures should have no inferred functional groups."""
    assert functional_groups("not_a_smiles") == frozenset()


def test_spectral_region_classification_is_multilabel_and_deterministic(
    ethanol_sample,
) -> None:
    """Overlapping soft regions should be retained in stable rule order."""
    regions = classify_spectral_regions(ethanol_sample)

    assert regions == classify_spectral_regions(ethanol_sample)
    assert "H1_SHIFT_ALKYL" in regions["1H"]
    assert "H1_SHIFT_HETEROATOM_SP3" in regions["1H"]
    assert "C13_SHIFT_ALKYL" in regions["13C"]
    assert "C13_SHIFT_HETEROATOM_SP3" in regions["13C"]
    assert len(regions["1H"]) == len(set(regions["1H"]))
    assert len(regions["13C"]) == len(set(regions["13C"]))


def test_task_weights_are_normalized_and_validated() -> None:
    """Multitask sampling should use explicit normalized probabilities."""
    assert normalize_task_weights({STRUCTURE_PREDICTION: 3, "functional_group_recognition": 1}) == {
        STRUCTURE_PREDICTION: 0.75,
        FUNCTIONAL_GROUP_RECOGNITION: 0.25,
    }
    with pytest.raises(ValueError, match="non-negative"):
        normalize_task_weights({STRUCTURE_PREDICTION: -1})
    with pytest.raises(ValueError, match="positive"):
        normalize_task_weights({STRUCTURE_PREDICTION: 0})
    with pytest.raises(ValueError, match="Unsupported auxiliary task"):
        normalize_task_weights({"unknown": 1})


def test_structure_prediction_task_preserves_smiles_target(ethanol_sample) -> None:
    """The main task should remain direct canonical-SMILES prediction."""
    example = build_task_example(ethanol_sample, STRUCTURE_PREDICTION)

    assert example.task == STRUCTURE_PREDICTION
    assert example.target == "CCO"
    assert "canonical SMILES" in example.prompt


def test_functional_group_task_uses_controlled_json_target(ethanol_sample) -> None:
    """Functional-group supervision should use deterministic ontology labels."""
    example = build_task_example(ethanol_sample, FUNCTIONAL_GROUP_RECOGNITION)

    assert json.loads(example.target) == ["alcohol"]
    assert "JSON array" in example.prompt
    assert "functional group" in example.prompt.lower()


def test_spectral_region_task_uses_peak_only_json_target(ethanol_sample) -> None:
    """Region supervision should serialize the deterministic multilabel target."""
    example = build_task_example(ethanol_sample, SPECTRAL_REGION_CLASSIFICATION)

    assert json.loads(example.target) == classify_spectral_regions(ethanol_sample)
    assert "spectral region" in example.prompt.lower()


def test_candidate_ranking_task_selects_target_from_candidates(ethanol_sample) -> None:
    """Candidate supervision should choose the known structure without fake ordering."""
    example = build_task_example(
        ethanol_sample,
        CANDIDATE_RANKING,
        candidates=["COC", "CCO"],
    )

    assert example.target == "CCO"
    assert "1. COC" in example.prompt
    assert "2. CCO" in example.prompt
    assert "best candidate" in example.prompt.lower()


def test_candidate_ranking_requires_target_in_candidate_set(ethanol_sample) -> None:
    """A malformed sidecar must not silently create an incorrect target."""
    with pytest.raises(ValueError, match="target structure"):
        build_task_example(
            ethanol_sample,
            CANDIDATE_RANKING,
            candidates=["COC", "CCN"],
        )
