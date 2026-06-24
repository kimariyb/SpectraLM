# Constrained Two-Stage 10k NMR Structure Training Design

## Objective

Build one reproducible 10k-study workflow that improves NMR-to-structure
prediction through formula-constrained candidate inference, sequential
multitask and structure-specialization training, formula-matched hard
negatives, controlled input-modality dropout, and stereochemistry-aware
evaluation.

## Data Protocol

The curated 10k study cohort contains 9,000 training samples and 1,000
validation samples. The existing scaffold-disjoint mother train and validation
partitions remain separate: training rows are selected only from the mother
train partition and validation rows only from the mother validation partition.
The independent test split remains outside the 10k cohort and uses 5,000
samples by default.

The active split names are:

- `clean_10k_train`: 9,000 samples.
- `clean_10k_val`: 1,000 samples.
- `clean_10k_test`: 5,000 samples selected from the independent test partition.

## Molecular Target Policy

Both training stages supervise non-isomeric canonical SMILES. Isotope labels
are removed by the existing molecule policy, while atom connectivity is
preserved. Candidate structures are also canonicalized without stereochemistry
before deduplication.

Evaluation retains the original isomeric reference structure. Stereo-sensitive
exact match remains a secondary metric, while connectivity exact match is the
primary structure-recovery metric. Results are stratified into references with
and without explicit stereochemistry.

## Deterministic Modality Sampling

Only three formula-conditioned input combinations are used:

- Full, containing two images, peak tables, and formula: 50%.
- Image plus formula, without peak tables: 25%.
- Peak tables plus formula, without images: 25%.

Training modality assignment is derived from a stable hash of the experiment
seed and sample ID. This keeps the assignment reproducible across worker counts
and repeated runs. Validation always uses Full input so evaluation loss is
comparable across checkpoints.

## Stage 1: Multitask Representation Training

Stage 1 starts from `/mnt/data/kimariyb/models/Qwen3.5-9B` and trains one LoRA
adapter for one epoch at learning rate `1e-4`. The task mixture is:

- Direct non-isomeric structure prediction: 40%.
- Formula-matched hard candidate ranking: 30%.
- Functional-group recognition: 20%.
- Spectral-region classification: 10%.

All task prompts are compatible with the three selected modality combinations.
If a sample lacks a hard-negative candidate set, a selected ranking task falls
back to direct structure prediction. Sidecar construction reports the eligible
candidate coverage so the realized ranking supervision can be audited.

## Hard-Negative Construction

Each ranking set contains at most eight unique non-isomeric structures,
including the target. Every negative must have exactly the same molecular
formula as the target and must come from the same active train or validation
partition as the target sidecar.

Negatives are ordered by difficulty using the following evidence in order:

1. Same non-empty Bemis-Murcko ring scaffold.
2. Higher functional-group Jaccard similarity.
3. Higher radius-2 Morgan fingerprint Tanimoto similarity.

Candidate display order is deterministically shuffled per sample. No test
structure is introduced into a training sidecar.

## Stage 2: Structure Specialization

Stage 2 loads the Stage 1 adapter as trainable and continues updating the same
adapter. It uses only direct non-isomeric structure prediction, retains the
50/25/25 modality mixture, trains for two epochs, and lowers the learning rate
to `5e-5`. The Stage 1 checkpoint is retained so ranking retention can be
measured after Stage 2.

The training entrypoint must keep `import unsloth` as the first normal import.
An `initial_adapter_path` configuration field selects continuation training;
without it, training follows the existing new-adapter path.

## Formula-Constrained Candidate Inference

The existing greedy direct inference remains the baseline. The new constrained
path uses the final Stage 2 adapter and a fixed seed to sample 32 candidates per
test sample with `temperature=0.7` and `top_p=0.9`.

Candidate processing is ordered as follows:

1. Extract and canonicalize generated SMILES without stereochemistry.
2. Remove duplicate structures.
3. Require the project domain policy: supported elements, one component, net
   neutral, no radicals, and no isotope labels.
4. When a user formula is supplied, retain only exact formula matches.

If no candidate survives an applicable formula constraint, the prediction is
an empty string and `formula_constraint_failed` is true. The system never
returns a formula-mismatched fallback. With no user formula, only domain
filtering is applied and `formula_constraint_applicable` is false.

One surviving candidate is selected directly. Multiple surviving candidates
are presented to the final adapter using the candidate-ranking prompt and full
NMR input. If the ranking response is not one of the candidates, the first
surviving candidate is selected and `ranking_failed` is true.

## Evaluation

The constrained inference output retains all existing structure, functional
group, validity, and generation diagnostics and adds:

- Formula-constraint applicability, success, failure, and coverage.
- Raw, unique, domain-valid, and formula-valid candidate counts.
- Candidate Oracle Exact@32 and Connectivity@32.
- Ranked Top-1 Exact and Connectivity Exact.
- Ranking failure rate.
- Achiral-reference Exact Match.
- Stereo-present-reference Connectivity Exact Match.
- Coverage for both stereochemistry strata.

Formula-constraint coverage and candidate oracle coverage are reported
separately. An empty prediction caused by hard-constraint failure scores as an
incorrect structure rather than being removed from the denominator.

## Configuration and Commands

The experiment matrix contains one Stage 1 training configuration, one Stage 2
training configuration, one direct Stage 2 inference configuration, and one
constrained Stage 2 inference configuration. The former 10k direct training
configuration becomes the Stage 2 configuration rather than being retained as
a competing old workflow. No numbered release labels are introduced.

`script/run_experiment.sh` exposes commands for:

1. Building the 9k/1k curated split.
2. Building train and validation hard-negative sidecars.
3. Running Stage 1.
4. Running Stage 2.
5. Running direct inference.
6. Running constrained candidate inference.

## Failure Handling

Dataset construction fails on missing split files, invalid modality weights, or
an absent continuation adapter. Candidate generation records per-sample model
exceptions without silently substituting an unconstrained structure. Ranking
failures use the documented first-candidate fallback and remain visible in
summary metrics.

## Verification

Implementation follows test-driven development. Tests cover the exact 9k/1k
split, deterministic modality assignment, mode-aware task prompts,
non-isomeric targets, hard-negative ordering, continuation-adapter selection,
formula filtering, empty hard-constraint failure, ranking fallback, candidate
coverage, and stereochemistry-stratified metrics. Final verification runs:

```bash
conda activate ml
python -m compileall src script tests
pytest
```
