# NMR Input-Modality Ablation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add reproducible full, image-only, peak-table-only, and formula-only structure-prediction experiments.

**Architecture:** `src/data/modalities.py` defines and validates modality behavior. Dataset transformation, prompt formatting, training, and inference consume that API so omitted modalities cannot leak through another path. YAML files and `run_experiment.sh` expose matched 5k and 50k runs.

**Tech Stack:** Python, PyTorch, Unsloth, Qwen3-VL, YAML, pytest

---

### Task 1: Modality Contract and Prompt Evidence

**Files:**
- Create: `src/data/modalities.py`
- Modify: `src/evaluation/prompts.py`
- Test: `tests/test_prompts_metrics.py`

- [x] Write tests asserting mode validation and exact presence/absence of image wording, peak tables, rules, and formula.
- [x] Run focused tests and confirm missing modality APIs fail.
- [x] Implement constants, validation helpers, mode-specific prompts, and evidence formatting.
- [x] Re-run focused tests and confirm all modality prompt cases pass.

### Task 2: Training Dataset and Inference Input Assembly

**Files:**
- Modify: `src/data/dataset.py`
- Modify: `src/data/tasks.py`
- Modify: `src/training/train.py`
- Modify: `src/training/inference.py`
- Test: `tests/test_dataset_transform.py`
- Test: `tests/test_response_masking.py`

- [x] Write tests showing image modes emit two images and text-only modes emit none.
- [x] Write tests showing lazy text-only datasets do not require rendered image files.
- [x] Implement shared mode propagation, conditional image loading, text-only generation, and provenance logging.
- [x] Confirm `import unsloth` remains the first normal import in `train.py`.

### Task 3: Matched Experiment Matrix

**Files:**
- Modify: `configs/experiments/train_scale_5k.yaml`
- Modify: `configs/experiments/infer_scale_5k.yaml`
- Create: `configs/experiments/train_modality_{image_only,peak_table_only,formula_only}_5k.yaml`
- Create: `configs/experiments/infer_modality_{image_only,peak_table_only,formula_only}_5k.yaml`
- Create: `configs/experiments/{train,infer}_modality_{image_only,peak_table_only}_50k.yaml`
- Modify: `script/run_experiment.sh`
- Modify: `tests/test_experiment_design.py`

- [x] Write failing configuration tests for four matched 5k runs and two 50k modality runs.
- [x] Add explicit input mode, fixed prompt index, shared split, seed, formula, and isolated output paths.
- [x] Add train/infer run-name routing and verify `list` displays every condition.

### Task 4: Documentation and Verification

**Files:**
- Modify: `docs/experiments_50k.md`

- [x] Document the modality matrix, execution order, and interpretation policy.
- [x] Run `python -m compileall -q src script tests`, `pytest -q`, direct runner listing, and `git diff --check` in the `ml` environment.
