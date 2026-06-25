# SpectraLM Text-Only LLM Migration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the active VLM workflow with a single Qwen3-8B text-only NMR peak-table instruction-tuning workflow, including formula/no-formula experiments and constrained candidate ranking.

**Architecture:** Lazy JSONL samples are serialized into deterministic 1H/13C text tables and system/user/assistant messages. Qwen3-8B is loaded through Unsloth `FastLanguageModel`, trained with a project-owned response-only collator and explicit non-thinking chat templates, then used for direct or Top-k generation followed by molecular-policy filtering, optional formula filtering, rule pre-ranking, and LLM candidate selection.

**Tech Stack:** Python, PyTorch, Unsloth `FastLanguageModel`, TRL `SFTTrainer`, PEFT LoRA, RDKit, NumPy, PyYAML, pytest.

---

## File Structure

### Create

- `src/training/text_collator.py`: text chat rendering, response-only token masking, and strict no-thinking validation.
- `tests/test_text_collator.py`: collator red/green tests.
- `tests/test_text_inference.py`: text-only generation and chat-template tests.
- Formula/no-formula training and inference YAML files listed in Task 8.

### Rewrite

- `src/evaluation/prompts.py`: peak-table-only prompts and deterministic serialization.
- `src/data/tasks.py`: auxiliary tasks without modality branches.
- `src/data/dataset.py`: retain offset caching and reusable handles; remove images.
- `src/training/response_masking.py`: strict no-thinking helpers and preflight validation.
- `src/training/model_setup.py`: FastLanguageModel LoRA setup.
- `src/training/arguments.py`: text SFT/config validation.
- `src/training/train.py`: text-only QLoRA training.
- `src/training/inference.py`: text-only direct and sampled inference.
- `src/training/constrained_inference.py`: formula-optional filtering and rule pre-ranking.
- `script/curate_jsonl_subsets.py`: make 10k mean 8k/1k/1k total and expose grouped random control splits.
- `script/run_experiment.sh`, `script/run_train_cuda_48g.sh`: one text workflow.
- `README.md`, `docs/experiments.md`, `docs/research_design.md`: text-only study.

### Delete

- `src/spectra/`
- `src/data/modalities.py`
- `script/pre_render_jsonl_images.py`
- `tests/test_pre_render_jsonl_images.py`
- Old VLM plans/specs and `docs/vlm_nmr_feasibility_review.md`
- Old YAML files after their replacements exist
- `outputs/`, every `dataset/**/rendered/`, and `img/`

## Task 1: Correct the 10k Cohort Contract

**Files:**
- Modify: `tests/test_curate_jsonl_subsets.py`
- Modify: `script/curate_jsonl_subsets.py`

- [ ] **Step 1: Change the existing protocol test to require an exact 8k/1k/1k total**

```python
def test_total_size_builds_exact_8k_1k_1k_protocol(tmp_path: Path) -> None:
    rows = [
        _manifest_row(f"train-{idx}", "train", f"train-{idx}")
        for idx in range(8000)
    ]
    rows.extend(
        _manifest_row(f"val-{idx}", "val", f"val-{idx}")
        for idx in range(1000)
    )
    rows.extend(
        _manifest_row(f"test-{idx}", "test", f"test-{idx}")
        for idx in range(1000)
    )

    summary = build_subsets(
        rows,
        tmp_path,
        subset_sizes=[10_000],
        val_fraction=0.1,
        test_fraction=0.1,
        seed=3407,
    )

    cohort = summary["subsets"]["clean_10k"]
    assert cohort["train"]["samples"] == 8000
    assert cohort["val"]["samples"] == 1000
    assert cohort["test"]["samples"] == 1000
    assert sum(cohort[f"requested_{key}_size"] for key in ("train", "val", "test")) == 10_000
```

- [ ] **Step 2: Run the test and verify RED**

Run: `conda run -n ml pytest tests/test_curate_jsonl_subsets.py::test_total_size_builds_exact_8k_1k_1k_protocol -v`

Expected: FAIL because current code requests 9,000 train plus 1,000 validation plus 1,000 test.

- [ ] **Step 3: Make subset size represent the total cohort**

Replace the size calculation with:

```python
requested_test_size = int(round(requested_size * float(test_fraction)))
requested_val_size = (
    int(round(requested_size * float(val_fraction)))
    if val_fraction is not None
    else int(val_size)
)
requested_train_size = (
    requested_size - requested_val_size - requested_test_size
)
if requested_train_size <= 0:
    raise ValueError(
        "subset size must exceed validation and test allocations"
    )
```

- [ ] **Step 4: Add deterministic grouped-random split coverage**

Add a test that calls a new `assign_grouped_random_splits(rows, seed, val_fraction, test_fraction)` helper twice and asserts identical assignments, no canonical structure crosses splits, and the resulting counts sum to the input count.

- [ ] **Step 5: Implement grouped random assignment**

Group rows by `canonical_smiles`, shuffle groups with `random.Random(seed)`, and greedily assign each complete group to train/val/test targets. Keep the existing manifest `split` values as the scaffold-disjoint source; random assignment is a separate preparation strategy and never overwrites the manifest.

- [ ] **Step 6: Run curation tests**

Run: `conda run -n ml pytest tests/test_curate_jsonl_subsets.py -v`

Expected: all curation tests PASS.

- [ ] **Step 7: Commit**

```bash
git add script/curate_jsonl_subsets.py tests/test_curate_jsonl_subsets.py
git commit -m "Define exact text study cohorts"
```

## Task 2: Replace Visual Prompts with Stable Peak-Table Text

**Files:**
- Modify: `tests/test_prompts_metrics.py`
- Modify: `tests/test_auxiliary_tasks.py`
- Modify: `src/evaluation/prompts.py`
- Modify: `src/data/tasks.py`

- [ ] **Step 1: Write failing prompt tests**

Add tests with these assertions:

```python
from src.evaluation.prompts import (
    SYSTEM_PROMPT,
    build_structure_prompt,
    format_peak_tables,
)


def test_system_prompt_forbids_reasoning_output() -> None:
    assert "one-dimensional NMR" in SYSTEM_PROMPT
    assert "do not output reasoning" in SYSTEM_PROMPT.lower()


def test_peak_tables_are_stable_and_text_only(ethanol_sample) -> None:
    rendered = format_peak_tables(ethanol_sample)
    assert "1H NMR:" in rendered
    assert "13C NMR:" in rendered
    assert "ppm" in rendered
    assert "image" not in rendered.lower()


def test_formula_ablation_removes_the_formula_line(ethanol_sample) -> None:
    with_formula = build_structure_prompt(ethanol_sample, include_formula=True)
    without_formula = build_structure_prompt(ethanol_sample, include_formula=False)
    assert "Molecular formula:" in with_formula
    assert "Molecular formula:" not in without_formula
    assert "1H NMR:" in without_formula
    assert "13C NMR:" in without_formula
```

Rewrite candidate-ranking tests to assert the prompt says `Use the numerical 1H and 13C peak tables` and never contains `image`.

- [ ] **Step 2: Run prompt and task tests and verify RED**

Run: `conda run -n ml pytest tests/test_prompts_metrics.py tests/test_auxiliary_tasks.py -v`

Expected: FAIL because prompts currently require images and task builders require `input_mode`.

- [ ] **Step 3: Implement the text prompt API**

Use these public contracts:

```python
SYSTEM_PROMPT = (
    "You are a molecular structure elucidation model for one-dimensional "
    "NMR data. Follow the requested output format exactly and do not output "
    "reasoning."
)


def format_peak_tables(sample: dict[str, Any]) -> str:
    """Serialize ordered 1H and 13C peak tables deterministically."""
    lines = ["1H NMR:"]
    h_peaks = sorted(
        sample.get("1H_NMR", {}).get("peaks", []),
        key=lambda peak: float(peak["shift"]),
        reverse=True,
    )
    for peak in h_peaks:
        couplings = peak.get("J", [])
        j_text = ",".join(f"{float(value):.1f}" for value in couplings) or "-"
        integration = f"{float(peak.get('integration', 1.0)):g}"
        lines.append(
            f"{float(peak['shift']):.2f} ppm | "
            f"{peak.get('multiplicity', 's')} | J={j_text} Hz | "
            f"integration={integration}"
        )
    c_values = []
    for peak in sample.get("13C_NMR", {}).get("peaks", []):
        value = peak["shift"] if isinstance(peak, dict) else peak
        c_values.extend(value if isinstance(value, list) else [value])
    c_text = ", ".join(
        f"{float(value):.2f}"
        for value in sorted(c_values, reverse=True)
    )
    lines.extend(["", "13C NMR:", f"{c_text} ppm"])
    return "\n".join(lines)


def build_structure_prompt(
    sample: dict[str, Any],
    prompt: str | None = None,
    *,
    include_formula: bool = True,
    include_rule_context: bool = False,
    max_rule_evidence: int = 12,
) -> str:
    """Build one pure-text NMR structure prompt."""
    template = prompt or STRUCTURE_PROMPTS[0]
    context = []
    if include_formula:
        formula = str(sample.get("molecular_formula") or "").strip()
        if not formula:
            raise ValueError("Formula-conditioned input requires molecular_formula")
        context.append(f"Molecular formula: {formula}")
    context.append(format_peak_tables(sample))
    if include_rule_context:
        context.append(
            _format_rule_context(
                sample,
                include_formula=include_formula,
                max_rule_evidence=max_rule_evidence,
            )
        )
    return template.format(spectral_context="\n\n".join(context))
```

Keep a small list of text-only structure prompt variants for training, and use explicit `prompt_template_index` for inference.

- [ ] **Step 4: Remove modality parameters from auxiliary tasks**

Change both functions to text-only signatures:

```python
def build_candidate_ranking_prompt(
    sample: dict[str, Any],
    candidates: Sequence[str],
    *,
    include_formula: bool = True,
    include_rule_context: bool = False,
    max_rule_evidence: int = 12,
) -> str:
    canonical = []
    for candidate in candidates:
        value = canonicalize_smiles(candidate)
        if value is not None and value not in canonical:
            canonical.append(value)
    if not canonical:
        raise ValueError("Candidate ranking requires a valid candidate")
    candidate_text = "\n".join(
        f"{index}. {value}"
        for index, value in enumerate(canonical, start=1)
    )
    template = (
        "Use the numerical 1H and 13C peak tables to select the candidate "
        "most consistent with the spectral evidence.\n\n"
        "{spectral_context}\n\nCandidates:\n"
        f"{candidate_text}\n\nReturn only the selected canonical SMILES."
    )
    return build_structure_prompt(
        sample,
        template,
        include_formula=include_formula,
        include_rule_context=include_rule_context,
        max_rule_evidence=max_rule_evidence,
    )


def build_task_example(
    sample: dict[str, Any],
    task: str,
    *,
    candidates: Sequence[str] | None = None,
    structure_prompt: str | None = None,
    include_formula: bool = True,
    include_rule_context: bool = False,
    max_rule_evidence: int = 12,
    target_stereochemistry: str = "preserve",
) -> TaskExample:
    if task not in SUPPORTED_TASKS:
        raise ValueError(f"Unsupported auxiliary task: {task}")
    if target_stereochemistry not in {"preserve", "remove"}:
        raise ValueError(
            "target_stereochemistry must be 'preserve' or 'remove'"
        )
    raw_target = canonicalize_smiles(sample_smiles(sample))
    if raw_target is None:
        raise ValueError("Task sample requires a valid target structure")
    target = (
        canonicalize_connectivity_smiles(raw_target)
        if target_stereochemistry == "remove"
        else raw_target
    )
    if target is None:
        raise ValueError("Task sample requires a valid target structure")
    if task == STRUCTURE_PREDICTION:
        template = structure_prompt or STRUCTURE_PROMPTS[0]
        return TaskExample(
            task=task,
            prompt=build_structure_prompt(
                sample,
                template,
                include_formula=include_formula,
                include_rule_context=include_rule_context,
                max_rule_evidence=max_rule_evidence,
            ),
            target=str(target),
        )
    if task == FUNCTIONAL_GROUP_RECOGNITION:
        labels = ", ".join(label for label, _ in FUNCTIONAL_GROUP_SMARTS)
        template = (
            "Use the numerical 1H and 13C peak tables to identify functional "
            f"groups from this ontology: {labels}.\n\n"
            "{spectral_context}\n\nReturn only one sorted JSON array."
        )
        return TaskExample(
            task=task,
            prompt=build_structure_prompt(
                sample,
                template,
                include_formula=include_formula,
                include_rule_context=include_rule_context,
                max_rule_evidence=max_rule_evidence,
            ),
            target=json.dumps(
                sorted(functional_groups(str(target))),
                ensure_ascii=True,
                separators=(",", ":"),
            ),
        )
    if task == SPECTRAL_REGION_CLASSIFICATION:
        template = (
            "Classify all observed 1H and 13C signals into the controlled "
            "spectral-region labels.\n\n{spectral_context}\n\nReturn only a "
            "JSON object with keys \"1H\" and \"13C\"."
        )
        return TaskExample(
            task=task,
            prompt=build_structure_prompt(
                sample,
                template,
                include_formula=include_formula,
                include_rule_context=False,
                max_rule_evidence=max_rule_evidence,
            ),
            target=json.dumps(
                classify_spectral_regions(sample),
                ensure_ascii=True,
                separators=(",", ":"),
            ),
        )
    canonical_candidates = []
    for candidate in candidates or []:
        value = canonicalize_smiles(candidate)
        if value is not None and value not in canonical_candidates:
            canonical_candidates.append(value)
    if target not in canonical_candidates:
        raise ValueError("Candidate set does not contain the target structure")
    return TaskExample(
        task=task,
        prompt=build_candidate_ranking_prompt(
            sample,
            canonical_candidates,
            include_formula=include_formula,
            include_rule_context=include_rule_context,
            max_rule_evidence=max_rule_evidence,
        ),
        target=str(target),
    )
```

- [ ] **Step 5: Run prompt/task tests**

Run: `conda run -n ml pytest tests/test_prompts_metrics.py tests/test_auxiliary_tasks.py -v`

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/evaluation/prompts.py src/data/tasks.py tests/test_prompts_metrics.py tests/test_auxiliary_tasks.py
git commit -m "Replace visual prompts with NMR text tables"
```

## Task 3: Convert the Lazy Dataset to Plain Chat Messages

**Files:**
- Modify: `tests/test_dataset_transform.py`
- Modify: `src/data/dataset.py`

- [ ] **Step 1: Replace image tests with pure-text message tests**

Retain offset-cache, file-handle, split-resolution, candidate-sidecar, and pickle-state tests. Delete resize/render/pre-render assertions. Add:

```python
def test_message_transform_emits_system_user_assistant_strings(ethanol_sample) -> None:
    transform = NMRMessageTransform(include_formula=True, seed=3407)
    row = transform({"sample": [ethanol_sample]})["messages"][0]
    assert [message["role"] for message in row] == ["system", "user", "assistant"]
    assert all(isinstance(message["content"], str) for message in row)
    assert "Molecular formula:" in row[1]["content"]
    assert "image" not in row[1]["content"].lower()


def test_message_transform_omits_formula_without_hiding_peaks(ethanol_sample) -> None:
    transform = NMRMessageTransform(include_formula=False, seed=3407)
    row = transform({"sample": [ethanol_sample]})["messages"][0]
    assert "Molecular formula:" not in row[1]["content"]
    assert "1H NMR:" in row[1]["content"]
    assert "13C NMR:" in row[1]["content"]
```

- [ ] **Step 2: Run dataset tests and verify RED**

Run: `conda run -n ml pytest tests/test_dataset_transform.py -v`

Expected: FAIL because current messages contain image items and no system message.

- [ ] **Step 3: Simplify `NMRMessageTransform`**

Remove `input_mode`, `input_mode_weights`, image batches, and modality selection. Build:

```python
messages_batch.append(
    [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": example.prompt},
        {"role": "assistant", "content": example.target},
    ]
)
```

- [ ] **Step 4: Simplify `LazyNMRJsonlDataset`**

Keep only data/task arguments. `__getitem__` becomes:

```python
def __getitem__(self, idx: int) -> dict[str, Any]:
    sample = self._load_sample_at(self.offsets[idx])
    transformed = self.transform({"sample": [sample]})
    return {"messages": transformed["messages"][0]}
```

Remove PIL, rendering imports and all image functions. Preserve offset cache and reusable file-handle behavior unchanged.

- [ ] **Step 5: Run dataset tests**

Run: `conda run -n ml pytest tests/test_dataset_transform.py -v`

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/data/dataset.py tests/test_dataset_transform.py
git commit -m "Make lazy NMR dataset text only"
```

## Task 4: Add a Strict Non-Thinking Response-Only Collator

**Files:**
- Create: `tests/test_text_collator.py`
- Create: `src/training/text_collator.py`
- Modify: `tests/test_response_masking.py`
- Modify: `src/training/response_masking.py`

- [ ] **Step 1: Write failing collator tests with a deterministic fake tokenizer**

Tests must prove:

```python
def test_collator_masks_system_and_user_tokens() -> None:
    batch = collator([chat_sample("CCO")])
    supervised = batch["labels"][0][batch["labels"][0] != -100]
    assert tokenizer.decode(supervised, skip_special_tokens=True).strip() == "CCO"


def test_collator_passes_enable_thinking_false() -> None:
    collator([chat_sample("CCO")])
    assert tokenizer.template_calls
    assert all(call["enable_thinking"] is False for call in tokenizer.template_calls)


def test_collator_rejects_thinking_target() -> None:
    with pytest.raises(RuntimeError, match="thinking tags"):
        collator([chat_sample("<think>guess</think>CCO")])


def test_collator_rejects_non_prefix_template_boundary() -> None:
    tokenizer.break_prompt_prefix = True
    with pytest.raises(RuntimeError, match="not a prefix"):
        collator([chat_sample("CCO")])
```

- [ ] **Step 2: Run collator tests and verify RED**

Run: `conda run -n ml pytest tests/test_text_collator.py tests/test_response_masking.py -v`

Expected: import failure because `src.training.text_collator` does not exist.

- [ ] **Step 3: Implement strict chat rendering**

```python
def apply_non_thinking_chat_template(
    tokenizer: Any,
    messages: list[dict[str, Any]],
    *,
    add_generation_prompt: bool,
) -> str:
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=add_generation_prompt,
        enable_thinking=False,
    )
    if "<think>" in text or "</think>" in text:
        raise RuntimeError("Non-thinking chat template emitted thinking tags.")
    return str(text)
```

- [ ] **Step 4: Implement `TextResponseOnlyCollator`**

For each row, render full messages and `messages[:-1]` with `add_generation_prompt=True`; tokenize both without special tokens; require prompt IDs to be an exact prefix of full IDs; mask the prefix and padding with `-100`; reject truncation that removes the entire response. Set right padding and support `pad_to_multiple_of=8`.

- [ ] **Step 5: Make preflight strict**

Remove the old empty-thinking-prefix tolerance from `validate_response_only_batch`. Supervised decoding must match the target after whitespace normalization, and any `<think>` tag is an error.

- [ ] **Step 6: Run collator/masking tests**

Run: `conda run -n ml pytest tests/test_text_collator.py tests/test_response_masking.py -v`

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add src/training/text_collator.py src/training/response_masking.py tests/test_text_collator.py tests/test_response_masking.py
git commit -m "Add text response-only collator"
```

## Task 5: Migrate Training to FastLanguageModel

**Files:**
- Modify: `tests/test_model_setup.py`
- Modify: `tests/test_training_arguments.py`
- Modify: `tests/test_experiment_design.py`
- Modify: `src/training/model_setup.py`
- Modify: `src/training/arguments.py`
- Modify: `src/training/train.py`

- [ ] **Step 1: Write failing language-model setup tests**

Use a fake class recording `get_peft_model` calls and assert:

```python
assert kwargs["target_modules"] == [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]
assert "finetune_vision_layers" not in kwargs
```

Retain continued-adapter and missing-adapter tests.

- [ ] **Step 2: Add legacy visual-config rejection tests**

```python
@pytest.mark.parametrize("field", [
    "image_backend", "rendered_image_dir", "missing_image_policy",
    "image_size", "h_snr", "c_snr", "render_seed",
    "input_mode", "input_mode_weights", "eval_input_mode_weights",
])
def test_reject_legacy_visual_config_fields(field: str) -> None:
    with pytest.raises(ValueError, match=field):
        reject_legacy_visual_config({field: "legacy"})
```

- [ ] **Step 3: Run setup/argument tests and verify RED**

Run: `conda run -n ml pytest tests/test_model_setup.py tests/test_training_arguments.py tests/test_experiment_design.py -v`

Expected: FAIL because training still imports FastVisionModel and visual collator helpers.

- [ ] **Step 4: Implement language LoRA setup**

```python
LANGUAGE_LORA_TARGETS = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]


def setup_lora_model(model, config, *, fast_language_model, peft_model_class):
    initial_adapter = config.get("initial_adapter_path")
    if initial_adapter:
        path = Path(initial_adapter)
        if not path.exists():
            raise FileNotFoundError(
                f"initial_adapter_path does not exist: {path}"
            )
        return peft_model_class.from_pretrained(
            model,
            str(path),
            is_trainable=True,
        )
    return fast_language_model.get_peft_model(
        model,
        target_modules=LANGUAGE_LORA_TARGETS,
        r=int(config.get("lora_r", 16)),
        lora_alpha=int(config.get("lora_alpha", 32)),
        lora_dropout=float(config.get("lora_dropout", 0.0)),
        bias="none",
        use_gradient_checkpointing=config.get(
            "use_gradient_checkpointing", False
        ),
        random_state=int(config.get("seed", 3407)),
        use_rslora=False,
        loftq_config=None,
    )
```

- [ ] **Step 5: Migrate `train.py`**

Keep `import unsloth` as the first non-future import. Import `FastLanguageModel`, use `TextResponseOnlyCollator`, call `reject_legacy_visual_config(config)`, remove all image dataset arguments, load the exact `model_path` from YAML, and call `FastLanguageModel.for_training(model)`.

The preflight remains mandatory:

```python
preflight_sample = train_ds[0]
masking_stats = validate_response_only_batch(
    data_collator([preflight_sample]),
    tokenizer,
    expected_response=assistant_response_text(preflight_sample),
)
```

- [ ] **Step 6: Run setup/argument/experiment tests**

Run: `conda run -n ml pytest tests/test_model_setup.py tests/test_training_arguments.py tests/test_experiment_design.py -v`

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add src/training/model_setup.py src/training/arguments.py src/training/train.py tests/test_model_setup.py tests/test_training_arguments.py tests/test_experiment_design.py
git commit -m "Migrate training to FastLanguageModel"
```

## Task 6: Migrate Direct and Sampled Inference

**Files:**
- Create: `tests/test_text_inference.py`
- Modify: `src/training/inference.py`

- [ ] **Step 1: Write failing inference tests**

Test that `generate_one(model, tokenizer, prompt, 128, 0.0, 1.0)` and `generate_many(model, tokenizer, prompt, num_return_sequences=4, max_new_tokens=128, temperature=0.7, top_p=0.9)` have no image argument, call the chat template with `enable_thinking=False`, and decode only newly generated tokens. Test that `load_model_for_inference` receives a fake `FastLanguageModel` and calls `for_inference`.

- [ ] **Step 2: Run and verify RED**

Run: `conda run -n ml pytest tests/test_text_inference.py -v`

Expected: FAIL because current signatures require images and import FastVisionModel.

- [ ] **Step 3: Implement the pure-text inference input path**

```python
messages = [
    {"role": "system", "content": SYSTEM_PROMPT},
    {"role": "user", "content": prompt},
]
input_text = apply_non_thinking_chat_template(
    tokenizer,
    messages,
    add_generation_prompt=True,
)
inputs = tokenizer(
    input_text,
    add_special_tokens=False,
    return_tensors="pt",
).to("cuda")
```

Keep generation token diagnostics. Remove every image and `input_mode` branch. Store `include_formula` in prediction provenance.

- [ ] **Step 4: Run inference tests**

Run: `conda run -n ml pytest tests/test_text_inference.py -v`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/training/inference.py tests/test_text_inference.py
git commit -m "Make NMR inference text only"
```

## Task 7: Make Constrained Inference Formula-Optional and Rule-Presorted

**Files:**
- Modify: `tests/test_constrained_inference.py`
- Modify: `src/training/constrained_inference.py`

- [ ] **Step 1: Add failing formula/no-formula and rule-order tests**

```python
def test_no_formula_constrained_inference_keeps_domain_valid_candidates() -> None:
    result = constrain_and_rank_sample(
        sample_without_formula,
        ["CCO", "CCN"],
        include_formula=False,
        ranker=lambda candidates: candidates[0],
    )
    assert result["formula_constraint_applicable"] is False
    assert result["selectable_candidates"]


def test_candidates_are_rule_presorted_before_ranker(ethanol_sample) -> None:
    observed = []
    constrain_and_rank_sample(
        ethanol_sample,
        ["COC", "CCO"],
        include_formula=True,
        ranker=lambda candidates: observed.extend(candidates) or candidates[0],
    )
    assert observed[0] == "CCO"
```

- [ ] **Step 2: Run and verify RED**

Run: `conda run -n ml pytest tests/test_constrained_inference.py -v`

Expected: FAIL because `include_formula` is not an argument and candidates are not rule-presorted.

- [ ] **Step 3: Add deterministic rule pre-ranking**

```python
def rule_presort_candidates(
    sample: dict[str, Any],
    candidates: Sequence[str],
    *,
    include_formula: bool,
) -> tuple[str, ...]:
    scored = []
    for candidate in candidates:
        result = validate_candidate(
            candidate,
            sample,
            include_formula=include_formula,
        )
        scored.append((
            len(result.contradictions),
            -result.satisfied_checks,
            candidate,
        ))
    return tuple(item[2] for item in sorted(scored))
```

Pass the pre-sorted tuple to the LLM ranker. `resolve_ranked_candidate` already provides the required first-candidate fallback.

- [ ] **Step 4: Remove images and modality configuration from constrained main**

Call `generate_many(model, tokenizer, prompt, num_return_sequences=num_candidates, max_new_tokens=max_new_tokens, temperature=candidate_temperature, top_p=candidate_top_p)` and `generate_one(model, tokenizer, ranking_prompt, max_new_tokens=ranking_max_new_tokens, temperature=ranking_temperature, top_p=ranking_top_p)`. Pass `molecular_formula=None` when `include_formula` is false.

- [ ] **Step 5: Run constrained tests**

Run: `conda run -n ml pytest tests/test_constrained_inference.py tests/test_constrained_candidates.py -v`

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/training/constrained_inference.py tests/test_constrained_inference.py
git commit -m "Add text candidate constraint and rule ranking"
```

## Task 8: Replace YAMLs and Experiment Runner

**Files:**
- Delete: current files under `configs/`
- Create: the nine YAML files from the approved spec
- Modify: `tests/test_experiment_design.py`
- Modify: `script/run_experiment.sh`
- Modify: `script/run_train_cuda_48g.sh`

- [ ] **Step 1: Change config contract tests**

Assert the exact active config set, Qwen3-8B model path, formula/no-formula output separation, matched training controls, 8k/1k/1k splits, and absence of every legacy visual field.

```python
LEGACY_VISUAL_FIELDS = {
    "image_backend", "rendered_image_dir", "missing_image_policy",
    "image_size", "h_snr", "c_snr", "render_seed", "input_mode",
    "input_mode_weights", "eval_input_mode_weights",
}

for config in configs:
    assert config["model_path"] == "/mnt/data/kimariyb/models/Qwen3-8B"
    assert not LEGACY_VISUAL_FIELDS.intersection(config)
```

- [ ] **Step 2: Run and verify RED**

Run: `conda run -n ml pytest tests/test_experiment_design.py -v`

Expected: FAIL against the current five VLM configs.

- [ ] **Step 3: Create formula and no-formula configs**

Use output roots:

```text
outputs/experiments/formula/
outputs/experiments/no_formula/
```

Use scaffold-disjoint `clean_10k_*` for the first formal run. Keep random IDs prepared under `random_10k_*` for the later matched comparison. Formula candidate inference enables hard formula filtering; no-formula candidate inference sets `include_formula: false`.

- [ ] **Step 4: Rewrite the runner command surface**

Expose only:

```text
prepare split-10k
prepare candidates-formula-10k-train
prepare candidates-formula-10k-val
train smoke
train stage1-formula-10k
train stage2-formula-10k
train stage1-no-formula-10k
train stage2-no-formula-10k
infer direct-formula-10k
infer candidates-formula-10k
infer direct-no-formula-10k
infer candidates-no-formula-10k
```

- [ ] **Step 5: Run config and shell syntax tests**

Run: `conda run -n ml pytest tests/test_experiment_design.py -v && bash -n script/run_experiment.sh && bash -n script/run_train_cuda_48g.sh`

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add configs script/run_experiment.sh script/run_train_cuda_48g.sh tests/test_experiment_design.py
git commit -m "Define the text-only experiment workflow"
```

## Task 9: Remove Visual Code and Historical Artifacts

**Files:**
- Delete: `src/spectra/`, `src/data/modalities.py`, `script/pre_render_jsonl_images.py`, `tests/test_pre_render_jsonl_images.py`
- Modify: `tests/test_nmr_parsing.py`
- Delete: old VLM docs/plans/specs except the approved text migration spec and plan
- Delete: `outputs/`, `img/`, and every `dataset/**/rendered/`
- Modify: `README.md`, `docs/experiments.md`, `docs/research_design.md`

- [ ] **Step 1: Add a repository cleanliness contract**

Add to `tests/test_experiment_design.py`:

```python
def test_active_codebase_has_no_visual_training_pipeline() -> None:
    forbidden = [
        "FastVisionModel", "UnslothVisionDataCollator",
        "load_sample_images", "image_backend", "rendered_image_dir",
    ]
    active_paths = [ROOT / "src", ROOT / "script", ROOT / "configs"]
    text = "\n".join(
        path.read_text(encoding="utf-8", errors="ignore")
        for base in active_paths
        for path in base.rglob("*")
        if path.is_file() and path.suffix in {".py", ".yaml", ".sh"}
    )
    for token in forbidden:
        assert token not in text
```

- [ ] **Step 2: Run and verify RED**

Run: `conda run -n ml pytest tests/test_experiment_design.py::test_active_codebase_has_no_visual_training_pipeline -v`

Expected: FAIL with current visual modules.

- [ ] **Step 3: Delete visual source and obsolete tests**

Use `apply_patch` for tracked source/test deletions. Remove only the rendering-specific test from `tests/test_nmr_parsing.py`; retain NMR parsing tests that do not depend on `src.spectra`.

- [ ] **Step 4: Delete explicitly approved generated artifacts**

Remove `outputs/`, `img/`, and directories named `rendered` below `dataset/`. Do not delete CSV, JSONL, manifest, ID list, candidate sidecar, or offset cache files.

- [ ] **Step 5: Rewrite active documentation**

README must document only the text pipeline and first server commands. `docs/experiments.md` must list the exact 10k formula/no-formula sequence. `docs/research_design.md` must replace all VLM claims with text LLM hypotheses, baselines, split protocol, gates, and publication claims.

- [ ] **Step 6: Run cleanliness and targeted tests**

Run: `conda run -n ml pytest tests/test_experiment_design.py tests/test_nmr_parsing.py -v`

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add -A src script tests configs docs README.md img outputs dataset
git commit -m "Remove the visual NMR workflow"
```

Before committing, inspect `git status --short` and ensure `.gitignore` is not staged because it contains pre-existing user changes.

## Task 10: Full Verification and First Research Handoff

**Files:**
- Modify only files required by failures found in this task.

- [ ] **Step 1: Compile all source and tests**

Run: `conda run -n ml python -m compileall src tests script`

Expected: exit 0 with no syntax errors.

- [ ] **Step 2: Run the full test suite**

Run: `conda run -n ml pytest`

Expected: all tests PASS.

- [ ] **Step 3: Scan active code and configs for visual remnants**

Run:

```bash
rg -n -i "FastVisionModel|UnslothVisionDataCollator|image_backend|rendered_image_dir|load_sample_images|input_mode_weights" src script configs README.md docs/experiments.md docs/research_design.md
```

Expected: no matches.

- [ ] **Step 4: Verify destructive cleanup boundaries**

Run:

```bash
find dataset -type d -name rendered -print
test ! -d outputs
test ! -d img
```

Expected: no rendered directories and both directory tests exit 0.

- [ ] **Step 5: Run text-only dry-run**

Run: `conda run -n ml python -m src.training.train configs/train_smoke.yaml` only when `dry_run: true` is set temporarily through an approved smoke config; the normal smoke config remains a 20-step CUDA run.

Expected: sample roles are system/user/assistant, all content is text, and no model is loaded during dry-run.

- [ ] **Step 6: Produce server command handoff**

The first server sequence is:

```bash
conda activate ml
bash script/run_experiment.sh prepare split-10k
bash script/run_experiment.sh prepare candidates-formula-10k-train
bash script/run_experiment.sh prepare candidates-formula-10k-val
CUDA_VISIBLE_DEVICES=0 bash script/run_experiment.sh train smoke
```

Ask the user to return the smoke log, response-only preflight line, GPU memory, train/eval loss, runtime, and first rendered text sample before starting Stage 1 formula training.

- [ ] **Step 7: Final commit after any verification fixes**

```bash
git add src script tests configs README.md docs
git commit -m "Verify text-only SpectraLM workflow"
```

Do not stage `.gitignore` unless the user separately asks to include its existing modification.
