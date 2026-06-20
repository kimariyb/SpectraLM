# NMR Multitask And Evaluation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add configurable four-task VLM supervision, common-element dataset enforcement, generalized DBE, and publication-ready structure, spectroscopy, and model-behavior metrics.

**Architecture:** Chemistry labels and allowed-element policy live in focused data modules. A task builder produces task-specific prompts and targets without changing image loading. Candidate negatives are precomputed in a sidecar, while evaluation derives all structure labels independently from model outputs.

**Tech Stack:** Python 3.10, RDKit, PyYAML, NumPy, pytest, existing lazy JSONL and Unsloth pipeline.

---

### Task 1: Common Elements And DBE

**Files:**
- Modify: `src/data/molecules.py`
- Modify: `src/data/manifest.py`
- Modify: `src/nmr_rules/formula.py`
- Modify: `script/build_paired_jsonl.py`
- Test: `tests/test_common_elements.py`
- Test: `tests/test_nmr_rules.py`

- [ ] Add tests for allowed Si/P/S molecules, rejected B/Na/Se molecules, manifest reasons, and filtering during dataset construction.
- [ ] Run focused tests under the `ml` environment and verify failures from missing behavior.
- [ ] Implement the centralized element policy and generalized DBE expression.
- [ ] Run focused tests and verify passing behavior.

### Task 2: Chemistry Labels And Spectral Regions

**Files:**
- Create: `src/data/functional_groups.py`
- Create: `src/data/spectral_regions.py`
- Test: `tests/test_auxiliary_tasks.py`

- [ ] Add tests for representative oxygen, nitrogen, halogen, sulfur, phosphorus, and silicon groups and deterministic overlapping region labels.
- [ ] Run focused tests and verify failures because the modules do not exist.
- [ ] Implement current SMARTS labels and region classification.
- [ ] Run focused tests and verify passing behavior.

### Task 3: Auxiliary Task Builder

**Files:**
- Create: `src/data/tasks.py`
- Modify: `src/data/dataset.py`
- Test: `tests/test_auxiliary_tasks.py`
- Modify: `tests/test_dataset_transform.py`

- [ ] Add tests for prompts and exact targets for all four tasks, normalized task weights, candidate fallback, deterministic seeded selection, and structure-only compatibility.
- [ ] Run focused tests and verify expected failures.
- [ ] Implement task definitions and integrate them into `NMRMessageTransform` and lazy dataset arguments.
- [ ] Run focused tests and verify passing behavior.

### Task 4: Formula-Matched Candidate Sidecar

**Files:**
- Create: `script/build_candidate_sidecar.py`
- Modify: `src/data/dataset.py`
- Test: `tests/test_candidate_sidecar.py`

- [ ] Add tests for same-formula negatives, target inclusion, deterministic similarity ordering, insufficient-group omission, and sidecar loading.
- [ ] Run focused tests and verify expected failures.
- [ ] Implement sidecar generation and lazy ID lookup.
- [ ] Run focused tests and verify passing behavior.

### Task 5: Requested Metrics

**Files:**
- Modify: `src/evaluation/metrics.py`
- Create: `src/evaluation/spectral_consistency.py`
- Modify: `tests/test_prompts_metrics.py`

- [ ] Add tests for scaffold coverage/match, functional-group F1, molecular-formula alias, functional-group spectral consistency, and three disjoint output-behavior states.
- [ ] Run focused tests and verify expected failures.
- [ ] Implement per-sample fields and aggregate metrics while retaining existing keys.
- [ ] Run focused tests and verify passing behavior.

### Task 6: Training Configuration And Documentation

**Files:**
- Modify: `src/training/train.py`
- Create: `configs/experiments/train_multitask_50k.yaml`
- Modify: `script/run_experiment.sh`
- Modify: `tests/test_experiment_design.py`
- Modify: `docs/experiments_50k.md`

- [ ] Add failing configuration tests for task weights, sidecar, independent protocol configuration, and output directory.
- [ ] Run experiment tests and verify expected failures.
- [ ] Propagate multitask configuration and add the named training run.
- [ ] Document dataset filtering, sidecar generation, training, and evaluation commands.
- [ ] Run experiment tests and verify passing behavior.

### Task 7: Verification

**Files:**
- Verify all changed source, scripts, configs, docs, and tests.

- [ ] Run `python -m compileall src tests script` in the `ml` environment.
- [ ] Run the full `pytest` suite in the `ml` environment.
- [ ] Load every YAML file and the SMARTS ontology.
- [ ] Run `git diff --check` and inspect the final status without reverting unrelated user changes.
