"""Formula-constrained text candidate generation and model-based ranking."""

from __future__ import annotations

import itertools
import json
import sys
import traceback
from pathlib import Path
from typing import Any, Callable, Sequence

from tqdm import tqdm

from src.config import load_config
from src.data.molecules import (
    canonicalize_connectivity_smiles,
    canonicalize_smiles,
    sample_smiles,
)
from src.data.tasks import build_candidate_ranking_prompt
from src.evaluation.constrained import (
    filter_generated_candidates,
    resolve_ranked_candidate,
    summarize_constrained_predictions,
)
from src.evaluation.metrics import (
    evaluate_structure_prediction,
    summarize_generation_behavior,
    summarize_structure_predictions,
)
from src.evaluation.prompts import build_structure_prompt, select_structure_prompt


def rule_presort_candidates(
    sample: dict[str, Any],
    candidates: Sequence[str],
    *,
    include_formula: bool,
) -> tuple[str, ...]:
    """Sort candidates by lightweight 1D-NMR consistency before LLM ranking."""
    scored: list[tuple[float, int, str]] = []
    for index, candidate in enumerate(candidates):
        metrics = evaluate_structure_prediction(
            candidate,
            candidate,
            sample=sample,
            include_formula=include_formula,
        )
        support = metrics.get("functional_group_spectral_support_rate")
        score = float(support) if support is not None else 0.0
        scored.append((score, index, candidate))
    scored.sort(key=lambda item: (-item[0], item[1]))
    return tuple(candidate for _, _, candidate in scored)


def constrain_and_rank_sample(
    sample: dict[str, Any],
    generated_candidates: Sequence[str],
    *,
    ranker: Callable[[tuple[str, ...]], str],
    include_formula: bool = True,
) -> dict[str, Any]:
    """Apply hard constraints, rule pre-sorting, and rank one candidate set."""
    formula = sample.get("molecular_formula") or sample.get("formula")
    filtered = filter_generated_candidates(
        generated_candidates,
        molecular_formula=str(formula) if include_formula and formula else None,
    )
    candidates = rule_presort_candidates(
        sample,
        filtered.selectable_candidates,
        include_formula=include_formula,
    )
    reference = canonicalize_smiles(sample_smiles(sample))
    reference_connectivity = canonicalize_connectivity_smiles(reference)
    candidate_oracle_exact = reference in candidates
    candidate_oracle_connectivity = reference_connectivity in candidates

    ranking_attempted = len(candidates) > 1
    ranking_failed = False
    ranking_response = ""
    if not candidates:
        prediction = ""
    elif len(candidates) == 1:
        prediction = candidates[0]
    else:
        ranking_response = str(ranker(candidates)).strip()
        selection = resolve_ranked_candidate(candidates, ranking_response)
        prediction = selection.prediction
        ranking_failed = selection.ranking_failed

    metrics = evaluate_structure_prediction(
        prediction,
        reference,
        sample=sample,
        include_formula=include_formula,
    )
    return {
        "prediction": prediction,
        "ranking_response": ranking_response,
        "raw_candidates": list(generated_candidates),
        "selectable_candidates": list(candidates),
        "raw_candidate_count": filtered.raw_count,
        "unique_candidate_count": filtered.unique_count,
        "domain_valid_candidate_count": len(filtered.domain_valid_candidates),
        "formula_valid_candidate_count": len(filtered.formula_valid_candidates),
        "formula_constraint_applicable": (
            filtered.formula_constraint_applicable
        ),
        "formula_constraint_failed": filtered.formula_constraint_failed,
        "candidate_oracle_exact": candidate_oracle_exact,
        "candidate_oracle_connectivity": candidate_oracle_connectivity,
        "ranking_attempted": ranking_attempted,
        "ranking_failed": ranking_failed,
        **metrics,
    }


def main(config: dict[str, Any]) -> None:
    """Run constrained generation and candidate ranking over one split."""
    import torch

    from src.training.inference import (
        build_inference_rows,
        generate_many,
        generate_one,
        load_model_for_inference,
    )

    max_samples = config.get("max_samples")
    seed = int(config.get("seed", 3407))
    include_formula = bool(config.get("include_formula", True))
    rule_context_enabled = bool(config.get("rule_context_enabled", False))
    model, tokenizer = load_model_for_inference(
        config["model_path"],
        config.get("adapter_path"),
        config.get("max_seq_length", 8192),
        config.get("load_in_4bit", True),
    )
    rows = build_inference_rows(config)
    template = select_structure_prompt(int(config.get("prompt_template_index", 0)))
    output_path = Path(config["output"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    metric_rows: list[dict[str, Any]] = []
    selected_predictions: list[str] = []
    generation_traces: list[dict[str, int | bool]] = []
    errors = 0
    iterator = enumerate(rows)
    if max_samples is not None:
        iterator = itertools.islice(iterator, int(max_samples))

    with output_path.open("w", encoding="utf-8") as handle:
        for index, sample in tqdm(iterator, desc="Constrained infer"):
            prompt = build_structure_prompt(
                sample,
                template,
                include_formula=include_formula,
                include_rule_context=rule_context_enabled,
                max_rule_evidence=int(config.get("max_rule_evidence", 12)),
            )
            try:
                torch.manual_seed(seed + index)
                candidates, traces = generate_many(
                    model,
                    tokenizer,
                    prompt,
                    num_return_sequences=int(config.get("num_candidates", 32)),
                    max_new_tokens=int(config.get("max_new_tokens", 128)),
                    temperature=float(
                        config.get(
                            "candidate_temperature",
                            config.get("temperature", 0.7),
                        )
                    ),
                    top_p=float(
                        config.get(
                            "candidate_top_p",
                            config.get("top_p", 0.9),
                        )
                    ),
                )

                def ranker(selectable: tuple[str, ...]) -> str:
                    ranking_prompt = build_candidate_ranking_prompt(
                        sample,
                        selectable,
                        include_formula=include_formula,
                        include_rule_context=rule_context_enabled,
                        max_rule_evidence=int(
                            config.get("max_rule_evidence", 12)
                        ),
                    )
                    ranked, _ = generate_one(
                        model,
                        tokenizer,
                        ranking_prompt,
                        max_new_tokens=int(
                            config.get("ranking_max_new_tokens", 128)
                        ),
                        temperature=float(
                            config.get("ranking_temperature", 0.0)
                        ),
                        top_p=float(config.get("ranking_top_p", 1.0)),
                    )
                    return ranked

                result = constrain_and_rank_sample(
                    sample,
                    candidates,
                    ranker=ranker,
                    include_formula=include_formula,
                )
                trace = traces[0]
            except Exception:
                errors += 1
                traceback.print_exc()
                result = constrain_and_rank_sample(
                    sample,
                    [],
                    ranker=lambda _: "",
                    include_formula=include_formula,
                )
                trace = {
                    "generated_token_count": 0,
                    "generation_terminated_by_eos": False,
                    "generation_hit_max_tokens": False,
                    "generation_repeated_4gram": False,
                }
            result.update(
                {
                    "idx": index,
                    "id": sample.get("id"),
                    "include_formula": include_formula,
                    "prompt": prompt,
                    **trace,
                }
            )
            metric_rows.append(result)
            selected_predictions.append(str(result["prediction"]))
            generation_traces.append(trace)
            handle.write(json.dumps(result, ensure_ascii=False) + "\n")

    summary = summarize_structure_predictions(metric_rows)
    summary.update(summarize_constrained_predictions(metric_rows))
    summary.update(
        summarize_generation_behavior(
            selected_predictions,
            generation_traces,
        )
    )
    summary["generation_errors"] = errors
    summary["num_candidates"] = int(config.get("num_candidates", 32))
    summary["include_formula"] = include_formula
    summary_path = Path(
        config.get("summary_output", output_path.with_suffix(".summary.json"))
    )
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m src.training.constrained_inference <config.yaml>")
        raise SystemExit(1)
    main(load_config(sys.argv[1]))
