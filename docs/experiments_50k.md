# 50k NMR-to-Structure Experiment Protocol

The scientific interpretation policy is documented in
[`nmr_1d_rulebook.md`](nmr_1d_rulebook.md).

This protocol trains nested 5k, 10k, 25k, and 50k formula-conditioned
subsets, three seeds for the 50k main model, one formula-free ablation, and a
zero-shot baseline. All runs use the same scaffold-disjoint 5k validation and
5k test samples.

## 1. Environment

```bash
conda activate ml
```

Run every command from the repository root.

For a complete rebuild from the raw CSV, the supported top-level entrypoint is:

```bash
bash script/build_full_jsonl.sh \
  dataset/NMRexp_10to24_1_1004.csv \
  dataset/paired_jsonl_full
```

The builder applies the common-element policy before writing samples and then
creates the nested clean subsets. The allowed symbols are
`H C N O F Si P S Cl Br I`; unsupported molecules are excluded during this
build rather than repaired in a later migration step.

The same build also rejects disconnected salts, structures with non-zero net
formal charge, and radicals. Isotope labels are removed before canonical
SMILES grouping; net-neutral charge-separated representations remain valid.
Inspect `index_report.json` for `molecule_rejection_counts`,
`isotope_normalized_rows`, and `isotope_labels_removed`.

After changing this molecular policy, rebuild from the raw CSV without
`--reuse-index`. Isotope normalization changes canonical keys and sample IDs,
so an old SQLite index, JSONL, subset IDs, candidate sidecars, and rendered
images are not compatible with the rebuilt dataset.

## 2. Curate Nested Subsets

```bash
python script/curate_jsonl_subsets.py dataset/paired_jsonl_full \
  --subset-sizes 5000 10000 25000 50000 \
  --val-size 5000 \
  --test-size 5000 \
  --seed 3407
```

The training sets are nested prefixes. The validation and test IDs are shared
across every scale. Inspect
`dataset/paired_jsonl_full/subsets/curation_summary.json` before training.

Build formula-matched hard candidate sets for multitask training:

```bash
python script/build_candidate_sidecar.py dataset/paired_jsonl_full \
  --split clean_50k_train \
  --output dataset/paired_jsonl_full/candidate_sets_clean_50k_train.jsonl \
  --candidates-per-sample 8 \
  --seed 3407

python script/build_candidate_sidecar.py dataset/paired_jsonl_full \
  --split clean_50k_val \
  --output dataset/paired_jsonl_full/candidate_sets_clean_50k_val.jsonl \
  --candidates-per-sample 8 \
  --seed 3407
```

## 3. Pre-render Images

The 50k training set contains every smaller training subset, so one render pass
covers the full experiment matrix.

```bash
python script/pre_render_jsonl_images.py dataset/paired_jsonl_full \
  --splits clean_50k_train clean_50k_val clean_50k_test \
  --image-size 768 432 \
  --num-workers 32 \
  --overwrite
```

`--overwrite` is required for the first render after a molecular-policy
rebuild because a stable sample ID may now resolve to a different best paired
record after isotope-labelled and unlabelled candidates are merged.

## 4. CUDA Smoke Test

```bash
CUDA_VISIBLE_DEVICES=0 bash script/run_train_cuda_48g.sh \
  configs/train_cuda_48g_smoke.yaml
```

## 5. Training Matrix

All training configurations monitor `eval_loss` with early stopping. Training
stops after three consecutive evaluations without an improvement of at least
`0.001`, then restores the checkpoint with the lowest validation loss. The
patience is counted in evaluation calls, so its step span is
`3 * eval_steps` for each run.

List all named runs:

```bash
bash script/run_experiment.sh list
```

Run the two core comparisons first:

```bash
CUDA_VISIBLE_DEVICES=0 bash script/run_experiment.sh train main-3407
CUDA_VISIBLE_DEVICES=1 bash script/run_experiment.sh train no-formula
```

Run the matched rule-context comparisons after the baselines:

```bash
CUDA_VISIBLE_DEVICES=0 bash script/run_experiment.sh train rules-50k
CUDA_VISIBLE_DEVICES=1 bash script/run_experiment.sh train rules-no-formula
```

Run the approved single-model multitask experiment separately:

```bash
CUDA_VISIBLE_DEVICES=0 bash script/run_experiment.sh train multitask-50k
```

Run the data-scaling curve:

```bash
CUDA_VISIBLE_DEVICES=0 bash script/run_experiment.sh train scale-5k
CUDA_VISIBLE_DEVICES=1 bash script/run_experiment.sh train scale-10k
CUDA_VISIBLE_DEVICES=0 bash script/run_experiment.sh train scale-25k
```

Run the remaining 50k seeds:

```bash
CUDA_VISIBLE_DEVICES=0 bash script/run_experiment.sh train main-42
CUDA_VISIBLE_DEVICES=1 bash script/run_experiment.sh train main-2026
```

Each command is blocking. Use separate terminal sessions when assigning two
GPUs concurrently.

## 6. Shared Test Evaluation

Every inference run uses greedy decoding, the same prompt seed, and
`clean_50k_test`. The formula-free run retains images and peak tables and only
removes the molecular formula.

```bash
CUDA_VISIBLE_DEVICES=0 bash script/run_experiment.sh infer zero-shot
CUDA_VISIBLE_DEVICES=0 bash script/run_experiment.sh infer scale-5k
CUDA_VISIBLE_DEVICES=0 bash script/run_experiment.sh infer scale-10k
CUDA_VISIBLE_DEVICES=0 bash script/run_experiment.sh infer scale-25k
CUDA_VISIBLE_DEVICES=0 bash script/run_experiment.sh infer main-3407
CUDA_VISIBLE_DEVICES=0 bash script/run_experiment.sh infer main-42
CUDA_VISIBLE_DEVICES=0 bash script/run_experiment.sh infer main-2026
CUDA_VISIBLE_DEVICES=0 bash script/run_experiment.sh infer no-formula
CUDA_VISIBLE_DEVICES=0 bash script/run_experiment.sh infer rules-50k
CUDA_VISIBLE_DEVICES=0 bash script/run_experiment.sh infer rules-no-formula
CUDA_VISIBLE_DEVICES=0 bash script/run_experiment.sh infer multitask-50k
```

Predictions are written under `outputs/experiments/structure/`,
`outputs/experiments/rules/`, or `outputs/experiments/multitask/`. Each JSONL
has a sibling `*.summary.json` containing the metrics defined below.

### Evaluation metric contract

- `exact_match` compares stereochemistry-aware RDKit canonical SMILES.
- `connectivity_exact_match` canonicalizes with stereochemistry disabled.
- `valid_smiles_rate` measures whether a SMILES candidate can be extracted and
  parsed by RDKit. `domain_valid_smiles_rate` additionally requires one neutral
  component containing only `H C N O F Si P S Cl Br I`.
- `molecular_formula_accuracy` compares formulas calculated by RDKit from the
  predicted and reference structures.
- Morgan-fingerprint Tanimoto uses radius 2, 2,048 bits, and no chirality. The
  summary reports both the end-to-end mean, where invalid predictions score
  zero, and the conditional mean over RDKit-valid predictions.
- Ring-scaffold matching uses an achiral Bemis-Murcko scaffold.
  `reference_ring_scaffold_coverage` is the fraction of references with a
  non-empty ring scaffold; `predicted_ring_scaffold_coverage` is the equivalent
  prediction-side rate. Acyclic references are excluded from the ring-scaffold
  match denominator.
- Functional groups use the fixed SMARTS ontology and report sample-macro,
  micro, macro, supported-macro, and per-class precision/recall/F1 statistics.
- Strict behavior states are mutually exclusive: compliant bare SMILES,
  RDKit-invalid bare output, and non-bare output.
- Generation diagnostics report EOS termination, max-token truncation,
  repeated token 4-grams, unique predictions, and duplicate predictions.
- Rule-enabled runs report each rule's applicable count and pass rate plus the
  fraction of candidates with at least one contradiction. Soft functional-group
  spectral support is reported separately with its applicability coverage; it
  is not treated as proof of structural correctness.

## 7. Reporting

Report the zero-shot, 5k, 10k, 25k, and 50k seed-3407 results as the data
scaling curve. Report mean and standard deviation across the three 50k seeds.
Use the seed-3407 formula and no-formula runs for the controlled ablation.
