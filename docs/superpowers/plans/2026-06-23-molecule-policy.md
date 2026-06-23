# Dataset Molecule Policy Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Enforce isotope-free, single-component, neutral, non-radical molecular targets throughout paired NMR dataset construction.

**Architecture:** A typed RDKit policy result in `src/data/molecules.py` is the single source of truth. CSV indexing normalizes and filters before molecular grouping, while pair building and manifest generation repeat validation for auditability.

**Tech Stack:** Python, RDKit, pandas, SQLite, pytest

---

### Task 1: Define Molecular Policy

**Files:**
- Modify: `src/data/molecules.py`
- Test: `tests/test_common_elements.py`

- [x] Add failing tests for salts, net charge, radicals, isotope removal, neutral charge-separated structures, and unsupported elements.
- [x] Run `pytest tests/test_common_elements.py -q` in the `ml` environment and confirm policy API failures.
- [x] Add a frozen policy-result dataclass and one RDKit inspection function returning canonical isotope-free SMILES, structural counts, violations, and acceptance status.
- [x] Re-run `pytest tests/test_common_elements.py -q` and confirm all cases pass.

### Task 2: Enforce Policy During Dataset Construction

**Files:**
- Modify: `script/build_paired_jsonl.py`
- Test: `tests/test_build_paired_jsonl.py`

- [x] Add a failing end-to-end test containing a salt, charged molecule, radical, isotope-labelled pair, and ordinary molecule.
- [x] Replace raw canonicalization with the shared policy before SQLite insertion and expose rejection/isotope-normalization counters.
- [x] Revalidate canonical keys during pair construction so old reused indexes cannot emit rejected structures.
- [x] Re-run `pytest tests/test_build_paired_jsonl.py -q` and confirm normalized output and counters.

### Task 3: Add Manifest Audit Fields

**Files:**
- Modify: `src/data/manifest.py`
- Test: `tests/test_common_elements.py`

- [x] Add failing assertions for component count, net charge, radical electrons, isotope labels, and policy QC reasons.
- [x] Populate the fields from the shared policy result and use policy violations in `qc_reason`.
- [x] Run focused tests, then `python -m compileall src tests` and `pytest` in the `ml` environment.

### Task 4: Document Rebuild Requirement

**Files:**
- Modify: `docs/experiments_50k.md`

- [x] Document that the SQLite index, JSONL, subsets, candidate sidecars, and rendered images must be rebuilt because canonical grouping changes after isotope removal.
- [x] Run `git diff --check` and inspect the final diff.
