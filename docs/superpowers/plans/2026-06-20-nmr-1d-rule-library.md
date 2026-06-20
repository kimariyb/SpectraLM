# One-Dimensional NMR Rule Library Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a current, auditable rule library for molecular-formula-conditioned or formula-free 1H/13C NMR structure prediction without solvent-peak or two-dimensional NMR assumptions.

**Architecture:** A typed rule engine converts only sample-visible fields into compact evidence for the VLM. Prompt integration is opt-in and current so existing `current structure prompts` experiments remain unchanged. A separate candidate validator measures whether generated SMILES satisfy formula, carbon-count, DBE, and supported motif constraints.

**Tech Stack:** Python 3.10, PyYAML, RDKit, pytest, existing lazy JSONL and Unsloth training pipeline.

---

### Task 1: Rule Data Model And Formula Constraints

**Files:**
- Create: `src/nmr_rules/__init__.py`
- Create: `src/nmr_rules/models.py`
- Create: `src/nmr_rules/formula.py`
- Test: `tests/test_nmr_rules.py`

- [ ] Write tests for formula parsing, DBE, missing formula, and evidence serialization.
- [ ] Run `conda run -n ml pytest tests/test_nmr_rules.py -q` and verify failure because the package does not exist.
- [ ] Implement immutable evidence/result dataclasses and neutral closed-shell DBE calculation.
- [ ] Run the focused tests and verify they pass.

### Task 2: 1H, 13C, Fragment, And Cross-Spectrum Rules

**Files:**
- Create: `rules/nmr_1d.yaml`
- Create: `src/nmr_rules/engine.py`
- Modify: `tests/test_nmr_rules.py`

- [ ] Add failing tests for ethyl detection, diagnostic shift regions, signal-count constraints, formula-free operation, and the absence of solvent/2D evidence.
- [ ] Run the focused tests and verify expected failures.
- [ ] Implement deterministic rules using soft ranges and explicit confidence/caveat metadata.
- [ ] Run the focused tests and verify they pass.

### Task 3: Prompt And Dataset Integration

**Files:**
- Modify: `src/evaluation/prompts.py`
- Modify: `src/data/dataset.py`
- Modify: `src/training/train.py`
- Modify: `src/training/inference.py`
- Modify: `tests/test_prompts_metrics.py`
- Modify: `tests/test_dataset_transform.py`

- [ ] Add failing tests proving formula comes from `molecular_formula`, the label is never consulted, rule context is opt-in, and formula-free rule prompts work.
- [ ] Run the prompt and dataset tests and verify expected failures.
- [ ] Add `current rule-context prompts`, rule-context formatting, and configuration propagation while preserving `current structure prompts` behavior.
- [ ] Run the prompt and dataset tests and verify they pass.

### Task 4: Candidate Validation And Metrics

**Files:**
- Create: `src/nmr_rules/validator.py`
- Modify: `src/evaluation/metrics.py`
- Modify: `src/training/inference.py`
- Modify: `tests/test_nmr_rules.py`
- Modify: `tests/test_prompts_metrics.py`

- [ ] Add failing tests for formula match, DBE match, carbon signal feasibility, rule consistency, invalid SMILES, and aggregate metrics.
- [ ] Run focused tests and verify expected failures.
- [ ] Implement candidate validation and add optional rule metrics to inference records and summaries.
- [ ] Run focused tests and verify they pass.

### Task 5: Reproducible Rule Experiments And Human Guide

**Files:**
- Create: `configs/experiments/train_rules_50k.yaml`
- Create: `configs/experiments/infer_rules_50k.yaml`
- Create: `configs/experiments/train_rules_no_formula_50k.yaml`
- Create: `configs/experiments/infer_rules_no_formula_50k.yaml`
- Modify: `script/run_experiment.sh`
- Modify: `tests/test_experiment_design.py`
- Create: `docs/nmr_1d_rulebook.md`

- [ ] Add failing experiment-matrix tests for separate outputs, prompt configurations, and rule settings.
- [ ] Run the experiment tests and verify expected failures.
- [ ] Add isolated rule-context configurations and runner entries without changing baseline configs.
- [ ] Write the human guide from the implemented rules, including uncertainty and failure modes.
- [ ] Run experiment tests and verify they pass.

### Task 6: Verification

**Files:**
- Verify all changed source, rule, config, documentation, and test files.

- [ ] Run `conda run -n ml python -m compileall src tests script`.
- [ ] Run `conda run -n ml pytest`.
- [ ] Inspect `git diff --check` and the final diff for unintended changes.
- [ ] Report exact verification results and remote-server commands.
