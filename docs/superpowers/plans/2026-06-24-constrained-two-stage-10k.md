# Constrained Two-Stage 10k Workflow Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement a reproducible 9k-train/1k-validation two-stage Qwen3.5-9B workflow with deterministic three-modality training, formula-matched hard-negative ranking, non-isomeric targets, and formula-constrained candidate inference.

**Architecture:** Extend the existing lazy JSONL pipeline rather than adding a second dataset backend. Pure molecule, modality, candidate-filtering, and metric functions remain independently testable; CUDA orchestration lives in a dedicated constrained-inference entrypoint. Stage 2 continues the trainable Stage 1 LoRA adapter, while the existing greedy inference remains the direct-generation baseline.

**Tech Stack:** Python 3.10, RDKit, NumPy, PyTorch, Hugging Face Transformers/TRL, PEFT, Unsloth, YAML, pytest, Bash.

---

## File Map

- Modify `src/data/molecules.py`: non-isomeric canonicalization and explicit-stereochemistry detection.
- Modify `src/data/tasks.py`: configurable target stereochemistry and mode-aware auxiliary/ranking prompts.
- Modify `src/data/modalities.py`: validated deterministic weighted modality selection.
- Modify `src/data/dataset.py`: per-sample modality selection, image avoidance, and target-policy propagation.
- Modify `script/curate_jsonl_subsets.py`: total-cohort sizes with a validation fraction.
- Modify `script/build_full_jsonl.sh`: current 10k-study defaults.
- Modify `script/build_candidate_sidecar.py`: non-isomeric formula-matched hard negatives ranked by scaffold, functional groups, and Tanimoto.
- Create `src/training/model_setup.py`: new-adapter versus continuation-adapter selection.
- Modify `src/training/train.py`: continuation adapter, distinct train/eval modality policies, and provenance logging.
- Create `src/evaluation/constrained.py`: pure candidate filtering and constrained-summary metrics.
- Create `src/training/constrained_inference.py`: 32-candidate CUDA generation, hard filtering, ranking, and JSONL reporting.
- Modify `src/evaluation/metrics.py`: stereochemistry strata in direct and constrained summaries.
- Rename `configs/experiments/train_scale_10k.yaml` to `configs/experiments/train_stage2_10k.yaml`.
- Rename `configs/experiments/infer_scale_10k.yaml` to `configs/experiments/infer_stage2_10k.yaml`.
- Create `configs/experiments/train_stage1_10k.yaml` and `configs/experiments/infer_constrained_10k.yaml`.
- Modify `script/run_experiment.sh`, `tests/`, and `README.md`, and rename `docs/experiments_50k.md` to `docs/experiments.md` to expose the single current workflow.

---

### Task 1: Non-Isomeric Targets and Stereochemistry-Stratified Metrics

**Files:**
- Modify: `src/data/molecules.py`
- Modify: `src/data/tasks.py`
- Modify: `src/evaluation/metrics.py`
- Test: `tests/test_common_elements.py`
- Test: `tests/test_auxiliary_tasks.py`
- Test: `tests/test_prompts_metrics.py`

- [ ] **Step 1: Write failing molecule-target tests**

Add tests that define the target-policy API:

```python
from src.data.molecules import (
    canonicalize_connectivity_smiles,
    has_explicit_stereochemistry,
)


def test_connectivity_canonicalization_removes_tetrahedral_stereo() -> None:
    first = canonicalize_connectivity_smiles("F[C@H](Cl)Br")
    second = canonicalize_connectivity_smiles("F[C@@H](Cl)Br")
    assert first == second
    assert first is not None and "@" not in first


def test_explicit_stereochemistry_detection_distinguishes_targets() -> None:
    assert has_explicit_stereochemistry("F[C@H](Cl)Br") is True
    assert has_explicit_stereochemistry("FC(Cl)Br") is False
```

- [ ] **Step 2: Write failing task-target and summary tests**

Add one task test and one summary test:

```python
def test_structure_task_can_supervise_connectivity_only(chiral_sample) -> None:
    example = build_task_example(
        chiral_sample,
        "structure_prediction",
        target_stereochemistry="remove",
    )
    assert "@" not in example.target


def test_structure_summary_stratifies_stereochemistry() -> None:
    achiral = evaluate_structure_prediction("CCO", "CCO")
    stereo = evaluate_structure_prediction("FC(Cl)Br", "F[C@H](Cl)Br")
    summary = summarize_structure_predictions([achiral, stereo])
    assert summary["achiral_reference_coverage"] == 0.5
    assert summary["achiral_exact_match"] == 1.0
    assert summary["stereo_present_reference_coverage"] == 0.5
    assert summary["stereo_present_connectivity_exact_match"] == 1.0
```

- [ ] **Step 3: Run focused tests and verify RED**

Run:

```bash
conda activate ml
pytest -q tests/test_common_elements.py tests/test_auxiliary_tasks.py tests/test_prompts_metrics.py
```

Expected: failures report missing `canonicalize_connectivity_smiles`, missing `target_stereochemistry`, and absent stereochemistry summary fields.

- [ ] **Step 4: Implement the molecule and task policy**

Add the following public helpers in `src/data/molecules.py`:

```python
@lru_cache(maxsize=8192)
def canonicalize_connectivity_smiles(smiles: str | None) -> str | None:
    canonical = canonicalize_smiles(smiles)
    mol = Chem.MolFromSmiles(canonical) if canonical is not None else None
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=False)


@lru_cache(maxsize=8192)
def has_explicit_stereochemistry(smiles: str | None) -> bool:
    canonical = canonicalize_smiles(smiles)
    connectivity = canonicalize_connectivity_smiles(smiles)
    return canonical is not None and connectivity is not None and canonical != connectivity
```

Extend `build_task_example` with `target_stereochemistry: str = "preserve"`, validate it against `{"preserve", "remove"}`, and select the target as follows:

```python
raw_target = canonicalize_smiles(sample_smiles(sample))
if raw_target is None:
    raise ValueError("Task sample requires a valid target structure.")
target = (
    canonicalize_connectivity_smiles(raw_target)
    if target_stereochemistry == "remove"
    else raw_target
)
```

- [ ] **Step 5: Implement stereochemistry fields and aggregate strata**

In `evaluate_structure_prediction`, add:

```python
"reference_stereochemistry_present": has_explicit_stereochemistry(reference),
```

In `summarize_structure_predictions`, aggregate with explicit denominators:

```python
achiral_rows = [row for row in rows if not row["reference_stereochemistry_present"]]
stereo_rows = [row for row in rows if row["reference_stereochemistry_present"]]
summary["achiral_reference_coverage"] = len(achiral_rows) / total
summary["achiral_exact_match"] = (
    sum(bool(row["exact_match"]) for row in achiral_rows) / len(achiral_rows)
    if achiral_rows else None
)
summary["stereo_present_reference_coverage"] = len(stereo_rows) / total
summary["stereo_present_connectivity_exact_match"] = (
    sum(bool(row["connectivity_exact_match"]) for row in stereo_rows)
    / len(stereo_rows)
    if stereo_rows else None
)
```

- [ ] **Step 6: Run focused tests and commit**

Run:

```bash
conda activate ml
pytest -q tests/test_common_elements.py tests/test_auxiliary_tasks.py tests/test_prompts_metrics.py
git add src/data/molecules.py src/data/tasks.py src/evaluation/metrics.py tests/test_common_elements.py tests/test_auxiliary_tasks.py tests/test_prompts_metrics.py
git commit -m "Add connectivity targets and stereo-aware metrics"
```

Expected: focused tests pass.

---

### Task 2: Exact 9k/1k 10k-Study Curation

**Files:**
- Modify: `script/curate_jsonl_subsets.py`
- Modify: `script/build_full_jsonl.sh`
- Test: `tests/test_curate_jsonl_subsets.py`

- [ ] **Step 1: Write a failing ratio test**

Create enough fixture rows in the existing train, val, and test mother splits, then add:

```python
def test_total_size_and_val_fraction_build_exact_9k_1k_protocol(tmp_path) -> None:
    rows = [
        _manifest_row(f"train-{idx}", "train", f"train-scaffold-{idx}")
        for idx in range(9000)
    ]
    rows.extend(
        _manifest_row(f"val-{idx}", "val", f"val-scaffold-{idx}")
        for idx in range(1000)
    )
    rows.extend(
        _manifest_row(f"test-{idx}", "test", f"test-scaffold-{idx}")
        for idx in range(5000)
    )
    summary = build_subsets(
        rows,
        tmp_path,
        subset_sizes=[10_000],
        val_fraction=0.1,
        test_size=5_000,
        seed=3407,
    )
    assert summary["subsets"]["clean_10k"]["train"]["samples"] == 9000
    assert summary["subsets"]["clean_10k"]["val"]["samples"] == 1000
    assert summary["subsets"]["clean_10k"]["test"]["samples"] == 5000
    assert len((tmp_path / "clean_10k_train_ids.txt").read_text().splitlines()) == 9000
    assert len((tmp_path / "clean_10k_val_ids.txt").read_text().splitlines()) == 1000
```

- [ ] **Step 2: Verify RED**

Run:

```bash
conda activate ml
pytest -q tests/test_curate_jsonl_subsets.py::test_total_size_and_val_fraction_build_exact_9k_1k_protocol
```

Expected: `build_subsets()` rejects the unknown `val_fraction` argument.

- [ ] **Step 3: Implement total-cohort semantics without breaking legacy calls**

Add `val_fraction: float | None = None` to `build_subsets`. Validate and select sizes as follows:

```python
if val_fraction is not None and not 0.0 < float(val_fraction) < 1.0:
    raise ValueError("val_fraction must be between 0 and 1")

if val_fraction is None:
    train_count = requested_size
    selected_val = ranked["val"][: min(val_size, len(ranked["val"]))]
else:
    validation_count = int(round(requested_size * float(val_fraction)))
    train_count = requested_size - validation_count
    selected_val = ranked["val"][: min(validation_count, len(ranked["val"]))]

selected_train = ranked["train"][: min(train_count, len(ranked["train"]))]
```

Add CLI `--val-fraction` and include `requested_total_size`, `requested_train_size`, and `requested_val_size` in each subset summary.

Set `DEFAULT_SUBSET_SIZES = [10_000]` and update the corresponding default-design test so running the curation script without explicit sizes creates only the current 10k cohort.

- [ ] **Step 4: Change the rebuild defaults to the current 10k study**

In `script/build_full_jsonl.sh`, use:

```bash
SUBSET_SIZES="${SUBSET_SIZES:-10000}"
VAL_FRACTION="${VAL_FRACTION:-0.1}"
TEST_SIZE="${TEST_SIZE:-5000}"
```

and call curation with `--val-fraction "${VAL_FRACTION}"` instead of `--val-size`.

- [ ] **Step 5: Run tests and commit**

Run:

```bash
conda activate ml
pytest -q tests/test_curate_jsonl_subsets.py
git add script/curate_jsonl_subsets.py script/build_full_jsonl.sh tests/test_curate_jsonl_subsets.py
git commit -m "Curate the 9k train 1k validation study"
```

Expected: all curation tests pass and legacy tests without `val_fraction` retain their former behavior.

---

### Task 3: Deterministic Three-Modality Sampling and Mode-Aware Tasks

**Files:**
- Modify: `src/data/modalities.py`
- Modify: `src/data/tasks.py`
- Modify: `src/data/dataset.py`
- Modify: `src/training/train.py`
- Test: `tests/test_dataset_transform.py`
- Test: `tests/test_auxiliary_tasks.py`
- Test: `tests/test_training_arguments.py`

- [ ] **Step 1: Write failing modality-normalization and selection tests**

Add:

```python
def test_modality_weights_normalize_to_approved_three_modes() -> None:
    assert normalize_input_mode_weights(
        {"full": 0.5, "image_only": 0.25, "peak_table_only": 0.25}
    ) == {"full": 0.5, "image_only": 0.25, "peak_table_only": 0.25}


def test_modality_selection_is_stable_per_sample() -> None:
    weights = {"full": 0.5, "image_only": 0.25, "peak_table_only": 0.25}
    first = select_weighted_input_mode("sample-17", seed=3407, weights=weights)
    second = select_weighted_input_mode("sample-17", seed=3407, weights=weights)
    assert first == second
    assert first in weights


def test_formula_only_is_rejected_from_training_mixture() -> None:
    with pytest.raises(ValueError, match="formula_only"):
        normalize_input_mode_weights({"full": 0.5, "formula_only": 0.5})
```

- [ ] **Step 2: Write failing dataset and auxiliary-prompt tests**

Add a transform test that passes `input_mode_weights`, verifies deterministic content types, and verifies that an image-only candidate-ranking prompt says no peak tables are available. Add a lazy-dataset test that monkeypatches `load_sample_images` and proves a hash-selected `peak_table_only` row does not load images.

Use this assertion contract:

```python
assert "No numerical peak tables are available" in image_only_prompt
assert "No spectrum images are available" in peak_only_prompt
assert image_loader_calls == []
```

- [ ] **Step 3: Verify RED**

Run:

```bash
conda activate ml
pytest -q tests/test_dataset_transform.py tests/test_auxiliary_tasks.py
```

Expected: missing weight helpers and unsupported transform arguments fail.

- [ ] **Step 4: Implement deterministic weighted selection**

In `src/data/modalities.py`, add:

```python
TRAINING_INPUT_MODES = (FULL, IMAGE_ONLY, PEAK_TABLE_ONLY)


def normalize_input_mode_weights(weights: Mapping[str, float] | None) -> dict[str, float]:
    raw = dict(weights or {FULL: 1.0})
    unknown = sorted(set(raw) - set(TRAINING_INPUT_MODES))
    if unknown:
        raise ValueError(f"Unsupported training input mode: {', '.join(unknown)}")
    positive = {normalize_input_mode(k): float(v) for k, v in raw.items() if float(v) > 0}
    if any(float(v) < 0 for v in raw.values()) or not positive:
        raise ValueError("input_mode_weights must contain positive non-negative weights")
    total = sum(positive.values())
    return {mode: value / total for mode, value in positive.items()}


def select_weighted_input_mode(sample_id: str, *, seed: int, weights: Mapping[str, float]) -> str:
    normalized = normalize_input_mode_weights(weights)
    digest = hashlib.sha256(f"{seed}:{sample_id}:input_mode".encode()).digest()
    point = int.from_bytes(digest[:8], "big") / float(2**64)
    cumulative = 0.0
    for mode, probability in normalized.items():
        cumulative += probability
        if point < cumulative:
            return mode
    return next(reversed(normalized))
```

Import `hashlib` and `Mapping` explicitly.

- [ ] **Step 5: Make task prompts describe the selected evidence**

Add a private evidence helper in `src/data/tasks.py`:

```python
def _evidence_description(input_mode: str) -> str:
    if input_mode == FULL:
        return "Use both ordered NMR images and the numerical peak tables."
    if input_mode == IMAGE_ONLY:
        return "Use the two ordered NMR images. No numerical peak tables are available."
    if input_mode == PEAK_TABLE_ONLY:
        return "Use the numerical 1H and 13C peak tables. No spectrum images are available."
    raise ValueError(f"Unsupported auxiliary-task input mode: {input_mode}")
```

Use it in functional-group, spectral-region, and candidate-ranking templates. Remove the existing restriction that non-full modes can only run structure prediction, while retaining the rule-context restriction for non-full modes.

- [ ] **Step 6: Propagate train and validation modality policies**

Extend `NMRMessageTransform`, `LazyNMRJsonlDataset`, and `load_lazy_nmr_dataset` with `input_mode_weights` and `target_stereochemistry`. Add `input_mode_for_sample(sample)` to the transform and call it before loading images in `LazyNMRJsonlDataset.__getitem__`.

In `train.py`, build separate kwargs:

```python
train_dataset_kwargs = {
    **dataset_kwargs,
    "input_mode_weights": config.get("input_mode_weights"),
    "target_stereochemistry": config.get("target_stereochemistry", "preserve"),
}
eval_dataset_kwargs = {
    **dataset_kwargs,
    "input_mode_weights": config.get("eval_input_mode_weights", {"full": 1.0}),
    "target_stereochemistry": config.get("target_stereochemistry", "preserve"),
}
```

Log normalized train and eval modality weights.

- [ ] **Step 7: Run focused tests and commit**

Run:

```bash
conda activate ml
pytest -q tests/test_dataset_transform.py tests/test_auxiliary_tasks.py tests/test_training_arguments.py
git add src/data/modalities.py src/data/tasks.py src/data/dataset.py src/training/train.py tests/test_dataset_transform.py tests/test_auxiliary_tasks.py tests/test_training_arguments.py
git commit -m "Add deterministic three-modality training"
```

Expected: focused tests pass and `src/training/train.py` still has `import unsloth` as its first normal import.

---

### Task 4: Formula-Matched Hard-Negative Sidecars

**Files:**
- Modify: `script/build_candidate_sidecar.py`
- Modify: `src/data/dataset.py`
- Test: `tests/test_candidate_sidecar.py`

- [ ] **Step 1: Write a failing hardness-order test**

Extend the fixture with same-formula candidates that differ in scaffold and functional groups, then assert each negative metadata row contains the complete ordering evidence:

```python
assert row["target"] == canonicalize_connectivity_smiles(row["target"])
assert all(
    candidate == canonicalize_connectivity_smiles(candidate)
    for candidate in row["candidates"]
)
hardness = row["negative_hardness"]
assert hardness == sorted(
    hardness,
    key=lambda item: (
        item["same_ring_scaffold"],
        item["functional_group_jaccard"],
        item["tanimoto"],
        item["smiles"],
    ),
    reverse=True,
)
```

Also assert the report includes:

```python
assert report["candidate_coverage"] == report["candidate_sets"] / report["input_samples"]
```

- [ ] **Step 2: Verify RED**

Run:

```bash
conda activate ml
pytest -q tests/test_candidate_sidecar.py
```

Expected: missing `negative_hardness`, non-isomeric target policy, and coverage fail.

- [ ] **Step 3: Implement the hardness tuple**

Canonicalize targets and pool entries with `canonicalize_connectivity_smiles`. Precompute fingerprints, `functional_groups`, and `_ring_scaffold` equivalents. Score each negative as:

```python
same_ring_scaffold = bool(
    target_scaffold
    and candidate_scaffold
    and target_scaffold == candidate_scaffold
)
union = target_groups | candidate_groups
functional_group_jaccard = len(target_groups & candidate_groups) / len(union) if union else 1.0
tanimoto = DataStructs.TanimotoSimilarity(target_fp, candidate_fp)
hardness_key = (
    int(same_ring_scaffold),
    functional_group_jaccard,
    tanimoto,
    candidate_smiles,
)
```

Sort descending, retain at most seven negatives, deterministically shuffle displayed candidates, and emit `negative_hardness` in pre-shuffle hardness order. Keep `negative_tanimoto` for compatibility.

- [ ] **Step 4: Report realized supervision coverage**

Return:

```python
return {
    "input_samples": len(samples),
    "candidate_sets": candidate_sets,
    "omitted_without_negatives": omitted,
    "candidate_coverage": candidate_sets / len(samples) if samples else 0.0,
}
```

- [ ] **Step 5: Run tests and commit**

Run:

```bash
conda activate ml
pytest -q tests/test_candidate_sidecar.py tests/test_dataset_transform.py::test_lazy_dataset_loads_candidate_sidecar_for_ranking
git add script/build_candidate_sidecar.py src/data/dataset.py tests/test_candidate_sidecar.py
git commit -m "Strengthen formula-matched ranking negatives"
```

Expected: candidate tests pass.

---

### Task 5: Continue Training from the Stage 1 Adapter

**Files:**
- Create: `src/training/model_setup.py`
- Modify: `src/training/train.py`
- Create: `tests/test_model_setup.py`
- Modify: `tests/test_experiment_design.py`

- [ ] **Step 1: Write failing pure model-setup tests**

Use small fake classes rather than importing CUDA modules:

```python
def test_setup_lora_loads_initial_adapter_as_trainable(tmp_path) -> None:
    adapter = tmp_path / "stage1" / "best_model"
    adapter.mkdir(parents=True)
    peft = FakePeftModel()
    result = setup_lora_model(
        object(),
        {"initial_adapter_path": str(adapter)},
        fast_vision_model=FakeFastVisionModel(),
        peft_model_class=peft,
    )
    assert result is peft.loaded_model
    assert peft.calls == [(str(adapter), True)]


def test_setup_lora_fails_when_initial_adapter_is_missing(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="initial_adapter_path"):
        setup_lora_model(
            object(),
            {"initial_adapter_path": str(tmp_path / "missing")},
            fast_vision_model=FakeFastVisionModel(),
            peft_model_class=FakePeftModel(),
        )
```

Add a separate test proving the no-path branch calls `get_peft_model` with the existing LoRA parameters.

- [ ] **Step 2: Verify RED**

Run:

```bash
conda activate ml
pytest -q tests/test_model_setup.py
```

Expected: import fails because `src.training.model_setup` does not exist.

- [ ] **Step 3: Implement the isolated setup function**

Create `src/training/model_setup.py`:

```python
from pathlib import Path
from typing import Any


def setup_lora_model(
    model: Any,
    config: dict[str, Any],
    *,
    fast_vision_model: Any,
    peft_model_class: Any,
) -> Any:
    initial_adapter = config.get("initial_adapter_path")
    if initial_adapter:
        path = Path(initial_adapter)
        if not path.exists():
            raise FileNotFoundError(f"initial_adapter_path does not exist: {path}")
        return peft_model_class.from_pretrained(
            model,
            str(path),
            is_trainable=True,
        )
    return fast_vision_model.get_peft_model(
        model,
        finetune_vision_layers=True,
        finetune_language_layers=True,
        finetune_attention_modules=True,
        finetune_mlp_modules=True,
        r=int(config.get("lora_r", 16)),
        lora_alpha=int(config.get("lora_alpha", 16)),
        lora_dropout=float(config.get("lora_dropout", 0)),
        bias="none",
        random_state=int(config.get("seed", 3407)),
        use_rslora=False,
        loftq_config=None,
    )
```

- [ ] **Step 4: Integrate without violating import order**

In `train.py`, retain:

```python
from __future__ import annotations
import unsloth
```

as the first normal import sequence. Import `PeftModel` and `setup_lora_model` afterward, replace the direct `get_peft_model` call with `setup_lora_model`, and add `initial_adapter_path` to the training log config.

- [ ] **Step 5: Run tests and commit**

Run:

```bash
conda activate ml
pytest -q tests/test_model_setup.py tests/test_experiment_design.py::test_train_imports_unsloth_before_the_cuda_training_stack
git add src/training/model_setup.py src/training/train.py tests/test_model_setup.py tests/test_experiment_design.py
git commit -m "Support trainable adapter continuation"
```

Expected: model-setup and import-order tests pass.

---

### Task 6: Pure Formula-Constrained Candidate Filtering and Metrics

**Files:**
- Create: `src/evaluation/constrained.py`
- Create: `tests/test_constrained_candidates.py`
- Modify: `src/evaluation/metrics.py`

- [ ] **Step 1: Write failing hard-constraint tests**

Add:

```python
def test_filter_candidates_enforces_domain_and_formula() -> None:
    result = filter_generated_candidates(
        ["CCO", "OCC", "CCN", "C.C", "not_smiles"],
        molecular_formula="C2H6O",
    )
    assert result.raw_count == 5
    assert result.unique_count == 2
    assert result.domain_valid_candidates == ("CCO", "CCN")
    assert result.formula_valid_candidates == ("CCO",)
    assert result.formula_constraint_applicable is True
    assert result.formula_constraint_failed is False


def test_filter_candidates_returns_empty_on_hard_constraint_failure() -> None:
    result = filter_generated_candidates(["CCN"], molecular_formula="C2H6O")
    assert result.formula_valid_candidates == ()
    assert result.formula_constraint_failed is True


def test_filter_candidates_without_formula_uses_domain_only() -> None:
    result = filter_generated_candidates(["CCO"], molecular_formula=None)
    assert result.formula_constraint_applicable is False
    assert result.selectable_candidates == ("CCO",)
```

- [ ] **Step 2: Write failing ranking-fallback and summary tests**

Add:

```python
def test_resolve_ranked_candidate_rejects_out_of_set_output() -> None:
    selection = resolve_ranked_candidate(("CCO", "COC"), "CCN")
    assert selection.prediction == "CCO"
    assert selection.ranking_failed is True


def test_constrained_summary_counts_empty_failures_in_denominator() -> None:
    summary = summarize_constrained_predictions([
        constrained_row(formula_failed=False, oracle=True, exact=True),
        constrained_row(formula_failed=True, oracle=False, exact=False),
    ])
    assert summary["formula_constraint_coverage"] == 0.5
    assert summary["candidate_oracle_exact_at_32"] == 0.5
    assert summary["ranked_top1_exact_match"] == 0.5
```

- [ ] **Step 3: Verify RED**

Run:

```bash
conda activate ml
pytest -q tests/test_constrained_candidates.py
```

Expected: module import fails.

- [ ] **Step 4: Implement immutable filtering results**

Create dataclasses in `src/evaluation/constrained.py`:

```python
@dataclass(frozen=True)
class CandidateFilterResult:
    raw_count: int
    unique_count: int
    domain_valid_candidates: tuple[str, ...]
    formula_valid_candidates: tuple[str, ...]
    formula_constraint_applicable: bool
    formula_constraint_failed: bool

    @property
    def selectable_candidates(self) -> tuple[str, ...]:
        return (
            self.formula_valid_candidates
            if self.formula_constraint_applicable
            else self.domain_valid_candidates
        )


@dataclass(frozen=True)
class RankedSelection:
    prediction: str
    ranking_failed: bool
```

`filter_generated_candidates` must canonicalize with `canonicalize_connectivity_smiles`, deduplicate in generation order, call `inspect_dataset_molecule`, require `accepted` and zero isotope labels, and compare `molecule_formula(candidate)` to the supplied formula. `resolve_ranked_candidate` canonicalizes the ranking response and requires membership; otherwise it returns the first candidate with failure true.

- [ ] **Step 5: Implement constrained aggregate metrics**

Add `summarize_constrained_predictions(rows)` with all samples in the denominator. Report:

```python
{
    "formula_constraint_applicable_rate": ...,
    "formula_constraint_coverage": ...,
    "formula_constraint_failure_rate": ...,
    "mean_raw_candidate_count": ...,
    "mean_unique_candidate_count": ...,
    "mean_domain_valid_candidate_count": ...,
    "mean_formula_valid_candidate_count": ...,
    "candidate_oracle_exact_at_32": ...,
    "candidate_oracle_connectivity_at_32": ...,
    "ranking_failure_rate": ...,
    "ranked_top1_exact_match": ...,
    "ranked_top1_connectivity_exact_match": ...,
}
```

- [ ] **Step 6: Run tests and commit**

Run:

```bash
conda activate ml
pytest -q tests/test_constrained_candidates.py tests/test_prompts_metrics.py
git add src/evaluation/constrained.py src/evaluation/metrics.py tests/test_constrained_candidates.py tests/test_prompts_metrics.py
git commit -m "Add formula-constrained candidate evaluation"
```

Expected: constrained and existing metric tests pass.

---

### Task 7: Constrained CUDA Generation and Candidate Ranking Entrypoint

**Files:**
- Create: `src/training/constrained_inference.py`
- Modify: `src/training/inference.py`
- Modify: `src/data/tasks.py`
- Create: `tests/test_constrained_inference.py`
- Modify: `tests/test_auxiliary_tasks.py`

- [ ] **Step 1: Write a failing inference-safe ranking-prompt test**

The inference prompt builder must not require the reference target:

```python
def test_candidate_ranking_prompt_does_not_require_reference(ethanol_sample) -> None:
    sample = dict(ethanol_sample)
    sample.pop("canonical_smiles", None)
    prompt = build_candidate_ranking_prompt(
        sample,
        ["CCO", "COC"],
        include_formula=True,
        input_mode="full",
    )
    assert "1. CCO" in prompt
    assert "2. COC" in prompt
```

- [ ] **Step 2: Write failing orchestration tests with injected callables**

Define a pure `constrain_and_rank_sample` API and test it without CUDA:

```python
def test_constrain_and_rank_sample_returns_empty_when_formula_has_no_candidate() -> None:
    result = constrain_and_rank_sample(
        sample={"molecular_formula": "C2H6O", "canonical_smiles": "CCO"},
        generated_texts=["CCN"],
        ranker=lambda candidates: candidates[0],
    )
    assert result["prediction"] == ""
    assert result["formula_constraint_failed"] is True
    assert result["ranking_attempted"] is False


def test_constrain_and_rank_sample_uses_ranker_for_multiple_candidates() -> None:
    result = constrain_and_rank_sample(
        sample={"molecular_formula": "C2H6O", "canonical_smiles": "CCO"},
        generated_texts=["CCO", "COC"],
        ranker=lambda candidates: "COC",
    )
    assert result["prediction"] == "COC"
    assert result["ranking_attempted"] is True
    assert result["ranking_failed"] is False
```

- [ ] **Step 3: Verify RED**

Run:

```bash
conda activate ml
pytest -q tests/test_constrained_inference.py tests/test_auxiliary_tasks.py
```

Expected: missing module and prompt builder failures.

- [ ] **Step 4: Extract an inference-safe ranking prompt**

Create `build_candidate_ranking_prompt` in `src/data/tasks.py`. It canonicalizes and deduplicates candidates, requires at least one candidate, builds numbered lines, uses `_evidence_description(input_mode)`, and calls `build_structure_prompt`. Refactor the training ranking branch to call this function and keep its separate target-in-candidate validation.

- [ ] **Step 5: Add batched candidate generation to the existing inference module**

Add `generate_many` beside `generate_one` with the same prompt/image preparation. Its generation call must include:

```python
output_ids = model.generate(
    **inputs,
    max_new_tokens=max_new_tokens,
    do_sample=True,
    temperature=temperature,
    top_p=top_p,
    num_return_sequences=num_candidates,
    use_cache=True,
    eos_token_id=raw_eos_token_ids,
    pad_token_id=pad_token_id,
)
```

Decode only generated tokens after the expanded prompt length and return `list[str]` plus one trace per sequence. Reject `num_candidates < 1`, `temperature <= 0`, and invalid `top_p`.

- [ ] **Step 6: Implement the constrained entrypoint**

Create `src/training/constrained_inference.py` with:

```python
def constrain_and_rank_sample(sample, generated_texts, ranker):
    filtered = filter_generated_candidates(
        generated_texts,
        molecular_formula=sample.get("molecular_formula"),
    )
    candidates = filtered.selectable_candidates
    if not candidates:
        prediction = ""
        selection = RankedSelection("", False)
        ranking_attempted = False
    elif len(candidates) == 1:
        prediction = candidates[0]
        selection = RankedSelection(prediction, False)
        ranking_attempted = False
    else:
        ranking_attempted = True
        selection = resolve_ranked_candidate(candidates, ranker(candidates))
        prediction = selection.prediction
    return build_constrained_record(sample, filtered, selection, ranking_attempted)
```

The CUDA `main(config)` loads the final adapter once, loads test rows, seeds Torch with `seed + idx`, samples 32 candidates, filters them, and uses greedy full-input `generate_one` for the ranking prompt. It writes per-sample JSONL and merges `summarize_structure_predictions`, `summarize_generation_behavior`, and `summarize_constrained_predictions` into the sibling summary JSON.

- [ ] **Step 7: Run focused tests and commit**

Run:

```bash
conda activate ml
pytest -q tests/test_constrained_inference.py tests/test_auxiliary_tasks.py tests/test_prompts_metrics.py
git add src/training/constrained_inference.py src/training/inference.py src/data/tasks.py tests/test_constrained_inference.py tests/test_auxiliary_tasks.py tests/test_prompts_metrics.py
git commit -m "Add constrained candidate generation and ranking"
```

Expected: pure orchestration tests pass without loading a GPU model.

---

### Task 8: Replace the 10k Experiment Matrix and Document Exact Commands

**Files:**
- Create: `configs/experiments/train_stage1_10k.yaml`
- Rename: `configs/experiments/train_scale_10k.yaml` to `configs/experiments/train_stage2_10k.yaml`
- Rename: `configs/experiments/infer_scale_10k.yaml` to `configs/experiments/infer_stage2_10k.yaml`
- Create: `configs/experiments/infer_constrained_10k.yaml`
- Modify: `script/run_experiment.sh`
- Modify: `tests/test_experiment_design.py`
- Modify: `README.md`
- Rename: `docs/experiments_50k.md` to `docs/experiments.md`

- [ ] **Step 1: Write failing experiment-contract tests**

Replace the old `scale-10k` assertions with:

```python
def test_two_stage_10k_configs_share_the_approved_protocol() -> None:
    stage1 = _read_yaml("experiments/train_stage1_10k.yaml")
    stage2 = _read_yaml("experiments/train_stage2_10k.yaml")
    assert stage1["train_split_name"] == stage2["train_split_name"] == "clean_10k_train"
    assert stage1["eval_split_name"] == stage2["eval_split_name"] == "clean_10k_val"
    assert stage1["max_eval_samples"] == stage2["max_eval_samples"] == 1000
    assert stage1["input_mode_weights"] == stage2["input_mode_weights"] == {
        "full": 0.50,
        "image_only": 0.25,
        "peak_table_only": 0.25,
    }
    assert stage1["task_weights"] == {
        "structure_prediction": 0.40,
        "functional_group_recognition": 0.20,
        "candidate_ranking": 0.30,
        "spectral_region_classification": 0.10,
    }
    assert stage2["task_weights"] == {"structure_prediction": 1.0}
    assert stage2["initial_adapter_path"] == f"{stage1['output_dir']}/best_model"
    assert stage1["target_stereochemistry"] == stage2["target_stereochemistry"] == "remove"
```

Add tests that old `train_scale_10k.yaml` and `infer_scale_10k.yaml` are absent, constrained inference uses 32 candidates at 0.7/0.9, and `run_experiment.sh list` contains `stage1-10k`, `stage2-10k`, and `constrained-10k`.

- [ ] **Step 2: Verify RED**

Run:

```bash
conda activate ml
pytest -q tests/test_experiment_design.py
```

Expected: new configs and run names are missing.

- [ ] **Step 3: Create the Stage 1 configuration**

Base `configs/experiments/train_stage1_10k.yaml` on the current multitask config with these defining values:

```yaml
model_path: /mnt/data/kimariyb/models/Qwen3.5-9B
train_split_name: clean_10k_train
eval_split_name: clean_10k_val
target_stereochemistry: remove
input_mode: full
input_mode_weights:
  full: 0.50
  image_only: 0.25
  peak_table_only: 0.25
eval_input_mode_weights:
  full: 1.0
task_weights:
  structure_prediction: 0.40
  functional_group_recognition: 0.20
  candidate_ranking: 0.30
  spectral_region_classification: 0.10
train_candidate_sidecar_path: dataset/paired_jsonl_full/candidate_sets_clean_10k_train.jsonl
eval_candidate_sidecar_path: dataset/paired_jsonl_full/candidate_sets_clean_10k_val.jsonl
max_eval_samples: 1000
num_train_epochs: 1
learning_rate: 1.0e-4
output_dir: outputs/experiments/multitask/stage1-10k-seed3407
```

Retain the active QLoRA, image, DataLoader, early-stopping, and 48GB batch controls.

- [ ] **Step 4: Replace the old 10k configs with Stage 2 and constrained inference**

Rename the existing files and set Stage 2 to:

```yaml
initial_adapter_path: outputs/experiments/multitask/stage1-10k-seed3407/best_model
train_split_name: clean_10k_train
eval_split_name: clean_10k_val
target_stereochemistry: remove
input_mode_weights:
  full: 0.50
  image_only: 0.25
  peak_table_only: 0.25
eval_input_mode_weights:
  full: 1.0
task_weights:
  structure_prediction: 1.0
max_eval_samples: 1000
num_train_epochs: 2
learning_rate: 5.0e-5
output_dir: outputs/experiments/structure/stage2-10k-seed3407
```

Direct inference points to the Stage 2 best model. Create constrained inference with:

```yaml
adapter_path: outputs/experiments/structure/stage2-10k-seed3407/best_model
split: clean_10k_test
include_formula: true
num_candidates: 32
candidate_temperature: 0.7
candidate_top_p: 0.9
ranking_temperature: 0.0
max_samples: 5000
output: outputs/experiments/structure/predictions/stage2-10k-constrained.jsonl
```

- [ ] **Step 5: Expose preparation and run commands**

Extend `script/run_experiment.sh` with stages:

```text
prepare split-10k
prepare candidates-10k-train
prepare candidates-10k-val
train stage1-10k
train stage2-10k
infer stage2-10k
infer constrained-10k
```

Preparation commands must execute the existing Python scripts with the approved exact parameters. The constrained branch executes `python -m src.training.constrained_inference`; other inference continues to use `src.training.inference`.

- [ ] **Step 6: Rewrite the active protocol documentation**

Update README and the single experiment document to show this exact order:

```bash
conda activate ml
bash script/run_experiment.sh prepare split-10k
bash script/run_experiment.sh prepare candidates-10k-train
bash script/run_experiment.sh prepare candidates-10k-val
CUDA_VISIBLE_DEVICES=0 bash script/run_experiment.sh train stage1-10k
CUDA_VISIBLE_DEVICES=0 bash script/run_experiment.sh train stage2-10k
CUDA_VISIBLE_DEVICES=0 bash script/run_experiment.sh infer stage2-10k
CUDA_VISIBLE_DEVICES=0 bash script/run_experiment.sh infer constrained-10k
```

Document that constrained inference may output an empty prediction and that this is a counted hard-constraint failure, not a missing sample.

- [ ] **Step 7: Run configuration tests and commit**

Run:

```bash
conda activate ml
pytest -q tests/test_experiment_design.py
git add configs/experiments script/run_experiment.sh tests/test_experiment_design.py README.md docs/experiments.md
git commit -m "Define the constrained two-stage 10k experiment"
```

Expected: experiment-design tests pass and old 10k config names are absent.

---

### Task 9: Full Verification and Server Dry Runs

**Files:**
- Modify only if verification reveals a defect in a file changed above.

- [ ] **Step 1: Run syntax verification**

Run:

```bash
conda activate ml
python -m compileall src script tests
```

Expected: exit code 0 with no syntax errors.

- [ ] **Step 2: Run the full test suite**

Run:

```bash
conda activate ml
pytest
```

Expected: all tests pass with zero failures.

- [ ] **Step 3: Run repository hygiene checks**

Run:

```bash
git diff --check
rg -n "train_scale_10k|infer_scale_10k|scale-10k" configs script README.md docs/experiments.md tests
```

Expected: `git diff --check` is empty and the legacy 10k names have no active references.

- [ ] **Step 4: Verify the named command matrix**

Run:

```bash
bash script/run_experiment.sh list
```

Expected: output includes all three preparation commands plus Stage 1, Stage 2, direct inference, and constrained inference.

- [ ] **Step 5: Run server-side data dry runs after synchronization**

On the CUDA server:

```bash
conda activate ml
bash script/run_experiment.sh prepare split-10k
bash script/run_experiment.sh prepare candidates-10k-train
bash script/run_experiment.sh prepare candidates-10k-val
CUDA_VISIBLE_DEVICES=0 python - <<'PY'
from src.config import load_config
from src.training.train import main

config = load_config("configs/experiments/train_stage1_10k.yaml")
config["dry_run"] = True
main(config)
PY
```

Inspect the printed roles, selected modality, target, and candidate-sidecar coverage. Then run formal Stage 1 with `bash script/run_experiment.sh train stage1-10k`. Do not start Stage 2 until Stage 1 writes `best_model/adapter_config.json`.
