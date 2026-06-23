"""Tests for prompt generation and structure metrics."""

import pytest

import src.evaluation.metrics as metrics_module
from src.evaluation.metrics import (
    classify_output_behavior,
    evaluate_candidate_ranking,
    evaluate_structure_prediction,
    tanimoto_similarity,
)
from src.evaluation.prompts import (
    STRUCTURE_PROMPTS,
    build_structure_prompt,
    select_structure_prompt,
)


def test_structure_prompt_collection_has_research_grade_constraints() -> None:
    """Every training template should define the same multimodal task."""
    assert len(STRUCTURE_PROMPTS) == 8
    assert len(set(STRUCTURE_PROMPTS)) == len(STRUCTURE_PROMPTS)

    for prompt in STRUCTURE_PROMPTS:
        normalized = " ".join(prompt.split()).lower()
        assert prompt.count("{peak_tables}") == 1
        assert "first image" in normalized
        assert "second image" in normalized
        assert "1h nmr" in normalized
        assert "13c nmr" in normalized
        assert "canonical smiles" in normalized
        assert "only" in normalized or "nothing else" in normalized


def test_inference_selects_current_prompt_by_explicit_index() -> None:
    """Inference should select a stable current template by explicit index."""
    assert select_structure_prompt(0) == STRUCTURE_PROMPTS[0]
    with pytest.raises(ValueError, match="prompt_template_index"):
        select_structure_prompt(len(STRUCTURE_PROMPTS))


@pytest.mark.parametrize(
    ("input_mode", "has_images", "has_peak_tables"),
    [
        ("full", True, True),
        ("image_only", True, False),
        ("peak_table_only", False, True),
        ("formula_only", False, False),
    ],
)
def test_structure_prompts_expose_only_the_selected_input_modalities(
    ethanol_sample,
    input_mode: str,
    has_images: bool,
    has_peak_tables: bool,
) -> None:
    """Each ablation prompt must describe and include exactly its inputs."""
    template = select_structure_prompt(0, input_mode=input_mode)
    prompt = build_structure_prompt(
        ethanol_sample,
        template,
        include_formula=True,
        input_mode=input_mode,
    )
    normalized = " ".join(prompt.lower().split())

    assert ("first image" in normalized) is has_images
    assert ("## 1h nmr peak table" in prompt.lower()) is has_peak_tables
    assert ("## 13c nmr peak table" in prompt.lower()) is has_peak_tables
    assert "Molecular formula: C2H6O" in prompt


@pytest.mark.parametrize("input_mode", ["image_only", "peak_table_only", "formula_only"])
def test_non_full_modalities_reject_rule_context(
    ethanol_sample,
    input_mode: str,
) -> None:
    """Derived rules must not leak omitted spectral evidence into ablations."""
    template = select_structure_prompt(0, input_mode=input_mode)

    with pytest.raises(ValueError, match="rule context"):
        build_structure_prompt(
            ethanol_sample,
            template,
            include_formula=True,
            include_rule_context=True,
            input_mode=input_mode,
        )


def test_formula_only_requires_an_explicit_formula(ethanol_sample) -> None:
    """The prior-only control cannot be constructed with an empty input."""
    template = select_structure_prompt(0, input_mode="formula_only")

    with pytest.raises(ValueError, match="requires include_formula"):
        build_structure_prompt(
            ethanol_sample,
            template,
            include_formula=False,
            input_mode="formula_only",
        )


def test_structure_prompt_can_omit_formula_for_ablation(ethanol_sample) -> None:
    """Formula-free ablations should not leak formula from labels."""
    prompt = build_structure_prompt(
        ethanol_sample,
        "Predict.\n\n{peak_tables}",
        include_formula=False,
    )
    assert "Molecular formula:" not in prompt


def test_structure_prompt_reads_formula_without_target_smiles(ethanol_sample) -> None:
    """Formula-conditioned prompts should use the explicit input field."""
    sample = dict(ethanol_sample)
    sample.pop("canonical_smiles")
    sample.pop("smiles")

    prompt = build_structure_prompt(
        sample,
        "Predict.\n\n{peak_tables}",
        include_formula=True,
    )

    assert "Molecular formula: C2H6O" in prompt


def test_structure_prompt_never_derives_formula_from_target(ethanol_sample) -> None:
    """A missing input formula must stay missing even when a label exists."""
    sample = dict(ethanol_sample)
    sample.pop("molecular_formula")

    prompt = build_structure_prompt(
        sample,
        "Predict.\n\n{peak_tables}",
        include_formula=True,
    )

    assert "Molecular formula:" not in prompt


def test_structure_prompt_can_include_compact_rule_context(ethanol_sample) -> None:
    """Rule-context experiments should add bounded auditable evidence."""
    prompt = build_structure_prompt(
        ethanol_sample,
        "Predict.\n\n{peak_tables}",
        include_formula=True,
        include_rule_context=True,
        max_rule_evidence=3,
    )

    assert "## Derived 1D NMR Constraints" in prompt
    assert "DBE = 0" in prompt
    assert "ethyl fragment" in prompt
    assert prompt.count("- [") == 3


def test_formula_free_rule_context_does_not_emit_dbe(ethanol_sample) -> None:
    """Formula-free rule evidence should remain useful without leaking DBE."""
    prompt = build_structure_prompt(
        ethanol_sample,
        "Predict.\n\n{peak_tables}",
        include_formula=False,
        include_rule_context=True,
    )

    assert "Molecular formula:" not in prompt
    assert "DBE" not in prompt
    assert "ethyl fragment" in prompt


def test_structure_evaluation_handles_invalid_smiles(ethanol_sample) -> None:
    """Evaluation should not fail on invalid model structures."""
    row = evaluate_structure_prediction(
        "Final canonical SMILES: not_a_smiles",
        ethanol_sample["canonical_smiles"],
    )
    assert row["predicted_smiles"] is None
    assert row["valid_smiles"] is False
    assert row["tanimoto"] == 0.0


def test_structure_evaluation_extracts_smiles_from_markdown_fence() -> None:
    """Zero-shot Markdown formatting should not be scored as invalid chemistry."""
    row = evaluate_structure_prediction(
        "```plaintext\nCCO\n```",
        "CCO",
    )

    assert row["predicted_smiles"] == "CCO"
    assert row["valid_smiles"] is True
    assert row["exact_match"] is True


def test_structure_evaluation_reports_formula_and_connectivity_matches() -> None:
    """Metrics should separate stereochemical exactness from connectivity."""
    row = evaluate_structure_prediction(
        "F[C@H](Cl)Br",
        "F[C@@H](Cl)Br",
    )

    assert row["exact_match"] is False
    assert row["connectivity_exact_match"] is True
    assert row["molecular_formula_match"] is True
    assert "formula_match" not in row


def test_tanimoto_similarity_scores_exact_match() -> None:
    """Identical valid SMILES should score perfect Tanimoto similarity."""
    assert tanimoto_similarity("CCO", "CCO") == 1.0


def test_structure_summary_reports_direct_prediction_metrics() -> None:
    """Direct SMILES experiments should report structure-level outcomes."""
    evaluate_structure = getattr(
        metrics_module,
        "evaluate_structure_prediction",
        None,
    )
    summarize_structure = getattr(
        metrics_module,
        "summarize_structure_predictions",
        None,
    )

    assert callable(evaluate_structure)
    assert callable(summarize_structure)
    rows = [
        evaluate_structure("CCO", "CCO"),
        evaluate_structure("not_a_smiles", "CCN"),
    ]
    summary = summarize_structure(rows)

    assert summary["samples"] == 2
    assert summary["exact_match"] == 0.5
    assert summary["connectivity_exact_match"] == 0.5
    assert summary["valid_smiles_rate"] == 0.5
    assert summary["domain_valid_smiles_rate"] == 0.5
    assert summary["mean_tanimoto"] == 0.5
    assert summary["mean_tanimoto_valid_only"] == 1.0
    assert summary["median_tanimoto"] == 0.5
    assert summary["tanimoto_ge_0_5_rate"] == 0.5
    assert summary["molecular_formula_accuracy"] == 0.5
    assert summary["output_format_compliance_rate"] == 0.5
    assert summary["rdkit_invalid_bare_output_rate"] == 0.5
    assert summary["non_bare_output_rate"] == 0.0
    assert "formula_match_rate" not in summary
    assert "illegal_structure_rate" not in summary
    assert "non_smiles_output_rate" not in summary


def test_structure_summary_upgrades_legacy_prediction_rows() -> None:
    """Existing prediction JSONL rows should remain re-summarizable."""
    legacy = evaluate_structure_prediction("CCO", "CCO")
    for key in (
        "has_only_allowed_elements",
        "is_single_component",
        "is_neutral",
        "domain_valid_smiles",
    ):
        legacy.pop(key)
    legacy["predicted_scaffold"] = legacy.pop("predicted_ring_scaffold")
    legacy["reference_scaffold"] = legacy.pop("reference_ring_scaffold")
    legacy["scaffold_evaluable"] = legacy.pop(
        "reference_ring_scaffold_available"
    )
    legacy.pop("predicted_ring_scaffold_available")
    legacy["scaffold_match"] = legacy.pop("ring_scaffold_match")
    legacy["illegal_structure"] = legacy.pop("rdkit_invalid_bare_output")
    legacy["non_smiles_output"] = legacy.pop("non_bare_output")
    legacy["prediction"] = "CCO"
    legacy["label"] = "CCO"
    legacy["tanimoto"] = 0.0

    summary = metrics_module.summarize_structure_predictions([legacy])

    assert summary["domain_valid_smiles_rate"] == 1.0
    assert summary["mean_tanimoto"] == 1.0
    assert summary["reference_ring_scaffold_coverage"] == 0.0
    assert summary["output_format_compliance_rate"] == 1.0


def test_domain_validity_enforces_the_dataset_molecule_policy() -> None:
    """Domain validity should be stricter than generic RDKit parsing."""
    unsupported = evaluate_structure_prediction("C[Ge](C)(C)C", "CCO")
    charged = evaluate_structure_prediction("C[NH3+]", "CCN")
    disconnected = evaluate_structure_prediction("CC.O", "CCO")
    radical = evaluate_structure_prediction("[CH3]", "C")
    isotope = evaluate_structure_prediction("[13CH3]CO", "CCO")
    neutral_nitro = evaluate_structure_prediction("C[N+](=O)[O-]", "CCO")

    for row in (unsupported, charged, disconnected, radical, isotope):
        assert row["valid_smiles"] is True
        assert row["domain_valid_smiles"] is False
    assert unsupported["has_only_allowed_elements"] is False
    assert charged["is_neutral"] is False
    assert disconnected["is_single_component"] is False
    assert radical["has_no_radicals"] is False
    assert isotope["has_no_isotope_labels"] is False
    assert neutral_nitro["domain_valid_smiles"] is True


def test_structure_metrics_can_include_rule_consistency(ethanol_sample) -> None:
    """Rule-enabled evaluation should report candidate contradictions."""
    exact = evaluate_structure_prediction(
        "CCO",
        "CCO",
        sample=ethanol_sample,
        include_formula=True,
    )
    wrong_formula = evaluate_structure_prediction(
        "CCCO",
        "CCO",
        sample=ethanol_sample,
        include_formula=True,
    )

    assert exact["rule_consistency_rate"] == 1.0
    assert exact["rule_contradiction_count"] == 0
    assert wrong_formula["rule_consistency_rate"] < 1.0
    assert wrong_formula["rule_contradiction_count"] >= 1

    summary = metrics_module.summarize_structure_predictions(
        [exact, wrong_formula]
    )
    assert summary["rule_contradiction_rate"] == 0.5
    assert summary["rule_check_pass_rates"]["formula_match"] == {
        "applicable": 2,
        "passed": 1,
        "rate": 0.5,
    }
    assert "mean_rule_consistency_rate" not in summary


def test_structure_metrics_report_scaffold_match_and_coverage() -> None:
    """Murcko matching should exclude acyclic references from its denominator."""
    ring_match = evaluate_structure_prediction("Cc1ccccc1", "c1ccccc1")
    acyclic = evaluate_structure_prediction("CCN", "CCO")

    assert ring_match["exact_match"] is False
    assert ring_match["reference_ring_scaffold_available"] is True
    assert ring_match["ring_scaffold_match"] is True
    assert acyclic["reference_ring_scaffold_available"] is False
    assert acyclic["ring_scaffold_match"] is None

    summary = metrics_module.summarize_structure_predictions([ring_match, acyclic])
    assert summary["reference_ring_scaffold_coverage"] == 0.5
    assert summary["predicted_ring_scaffold_coverage"] == 0.5
    assert summary["ring_scaffold_match_rate"] == 1.0
    assert "scaffold_coverage" not in summary


def test_structure_metrics_report_functional_group_f1() -> None:
    """Functional-group similarity should compare controlled ontology sets."""
    exact = evaluate_structure_prediction("CCO", "CCO")
    mismatch = evaluate_structure_prediction("COC", "CCO")

    assert exact["predicted_functional_groups"] == ["alcohol"]
    assert exact["functional_group_f1"] == 1.0
    assert mismatch["predicted_functional_groups"] == ["ether"]
    assert mismatch["functional_group_f1"] == 0.0

    summary = metrics_module.summarize_structure_predictions([exact, mismatch])
    assert summary["functional_group_sample_macro_f1"] == 0.5
    assert summary["functional_group_micro_precision"] == 0.5
    assert summary["functional_group_micro_recall"] == 0.5
    assert summary["functional_group_micro_f1"] == 0.5
    assert summary["functional_group_per_class"]["alcohol"] == {
        "precision": 1.0,
        "recall": 0.5,
        "f1": 2.0 / 3.0,
        "support": 2,
    }


def test_functional_group_spectral_support_uses_soft_1d_signatures(
    ethanol_sample,
) -> None:
    """Predicted functional groups should be checked against observable regions."""
    supported = evaluate_structure_prediction(
        "CCO",
        "CCO",
        sample=ethanol_sample,
    )
    unsupported = evaluate_structure_prediction(
        "CC(=O)C",
        "CCO",
        sample=ethanol_sample,
    )

    assert supported["functional_group_spectral_support_rate"] == 1.0
    assert supported["spectral_functional_group_checks"]["alcohol"] is True
    assert unsupported["functional_group_spectral_support_rate"] == 0.0
    assert unsupported["spectral_functional_group_checks"]["ketone"] is False


def test_output_behavior_states_are_disjoint() -> None:
    """Formatting, illegal structures, and non-SMILES text need distinct rates."""
    compliant = classify_output_behavior("CCO")
    illegal = classify_output_behavior("not_a_smiles")
    prose = classify_output_behavior("Final SMILES: CCO")
    fenced = classify_output_behavior("```smiles\nCCO\n```")

    assert compliant == {
        "output_format_compliant": True,
        "rdkit_invalid_bare_output": False,
        "non_bare_output": False,
    }
    assert illegal == {
        "output_format_compliant": False,
        "rdkit_invalid_bare_output": True,
        "non_bare_output": False,
    }
    assert prose["non_bare_output"] is True
    assert fenced["non_bare_output"] is True
    for result in (compliant, illegal, prose, fenced):
        assert sum(bool(value) for value in result.values()) == 1


def test_candidate_ranking_metrics_report_top1_and_mrr() -> None:
    """Candidate selection should expose Top-1 accuracy and reciprocal rank."""
    first = evaluate_candidate_ranking(["CCO", "COC", "CCN"], "CCO")
    second = evaluate_candidate_ranking(["COC", "CCO", "CCN"], "CCO")
    missing = evaluate_candidate_ranking(["COC", "CCN"], "CCO")

    assert first == {
        "candidate_reference_covered": 1.0,
        "candidate_valid_rate": 1.0,
        "candidate_top1_accuracy": 1.0,
        "candidate_hits_at_3": 1.0,
        "candidate_hits_at_5": 1.0,
        "candidate_mrr": 1.0,
    }
    assert second["candidate_top1_accuracy"] == 0.0
    assert second["candidate_hits_at_3"] == 1.0
    assert second["candidate_mrr"] == 0.5
    assert missing["candidate_reference_covered"] == 0.0
    assert missing["candidate_mrr"] == 0.0


def test_generation_behavior_reports_eos_truncation_and_repetition() -> None:
    """Generation diagnostics should distinguish normal EOS from capped loops."""
    inspect_generation = getattr(metrics_module, "inspect_generation_tokens", None)
    assert callable(inspect_generation)

    stopped = inspect_generation([10, 11, 2], eos_token_ids={2}, max_new_tokens=8)
    looped = inspect_generation(
        [4, 5, 6, 7] * 3,
        eos_token_ids={2},
        max_new_tokens=12,
    )

    assert stopped == {
        "generated_token_count": 3,
        "generation_terminated_by_eos": True,
        "generation_hit_max_tokens": False,
        "generation_repeated_4gram": False,
    }
    assert looped["generation_terminated_by_eos"] is False
    assert looped["generation_hit_max_tokens"] is True
    assert looped["generation_repeated_4gram"] is True


def test_generation_summary_reports_collapse_and_termination_rates() -> None:
    """Generation summary should expose within- and across-sample collapse."""
    summarize_generation = getattr(metrics_module, "summarize_generation_behavior", None)
    assert callable(summarize_generation)
    summary = summarize_generation(
        ["CCO", "CCO", "CCN"],
        [
            {"generated_token_count": 4, "generation_terminated_by_eos": True, "generation_hit_max_tokens": False, "generation_repeated_4gram": False},
            {"generated_token_count": 8, "generation_terminated_by_eos": False, "generation_hit_max_tokens": True, "generation_repeated_4gram": True},
            {"generated_token_count": 5, "generation_terminated_by_eos": True, "generation_hit_max_tokens": False, "generation_repeated_4gram": False},
        ],
    )
    assert summary["eos_termination_rate"] == 2 / 3
    assert summary["generation_truncation_rate"] == 1 / 3
    assert summary["repeated_4gram_rate"] == 1 / 3
    assert summary["unique_prediction_rate"] == 2 / 3
    assert summary["duplicate_prediction_rate"] == 1 / 3


def test_multilabel_summary_reports_micro_macro_and_per_class_metrics() -> None:
    """Auxiliary classification tasks need dataset-level multilabel metrics."""
    summarize_multilabel = getattr(metrics_module, "summarize_multilabel_predictions", None)
    assert callable(summarize_multilabel)
    summary = summarize_multilabel(
        predicted=[{"alcohol"}, {"ether"}],
        reference=[{"alcohol"}, {"alcohol"}],
        label_space=["alcohol", "ether"],
    )
    assert summary["samples"] == 2
    assert summary["multilabel_exact_match"] == 0.5
    assert summary["multilabel_micro_f1"] == 0.5
    assert summary["multilabel_macro_f1"] == 1.0 / 3.0
    assert summary["multilabel_per_class"]["ether"]["support"] == 0
