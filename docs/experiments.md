# SpectraLM Two-Stage 10k Experiment

This is the only active experiment protocol. The 10k development cohort contains
9,000 training samples and 1,000 validation samples. A separate scaffold-disjoint
test split contains up to 5,000 samples and is not counted in the 10k cohort.

## Fixed Protocol

- Base model: `/mnt/data/kimariyb/models/Qwen3.5-9B`
- Spectrum images: pre-rendered `512 x 288` paired 1H/13C images
- Formula: supplied in every active training modality and at inference
- Target: canonical connectivity SMILES without stereochemistry
- Molecule domain: one neutral, non-radical, isotope-free component containing
  only H, C, N, O, F, Si, P, S, Cl, Br, or I
- Training modalities: full 0.50, image-only 0.25, peak-table-only 0.25
- Validation modality: full input only

Stage 1 runs one epoch with four tasks: direct structure prediction 0.40,
candidate ranking 0.30, functional-group recognition 0.20, and spectral-region
classification 0.10. Stage 2 continues the same trainable adapter for two epochs
using direct structure prediction only.

## Data Preparation

Activate the required environment before every command:

```bash
conda activate ml
```

If the paired JSONL mother dataset does not yet exist, build it once:

```bash
bash script/build_full_jsonl.sh \
  dataset/NMRexp_10to24_1_1004.csv \
  dataset/paired_jsonl_full
```

Create or refresh the exact 9k/1k/5k split and formula-matched hard-negative
sidecars:

```bash
bash script/run_experiment.sh prepare split-10k
bash script/run_experiment.sh prepare candidates-10k-train
bash script/run_experiment.sh prepare candidates-10k-val
```

Pre-render images when the `512x288` cache is absent or incomplete:

```bash
python script/pre_render_jsonl_images.py dataset/paired_jsonl_full \
  --splits clean_10k_train clean_10k_val clean_10k_test \
  --image-size 512 288 \
  --num-workers 32
```

Before training, verify that the split files contain 9,000, 1,000, and 5,000
IDs and inspect `candidate_coverage` printed by each sidecar command. Samples
without a same-formula negative fall back to direct structure supervision for
that draw rather than aborting training.

## Training

Run smoke first, then the two stages sequentially on one 48GB CUDA GPU:

```bash
CUDA_VISIBLE_DEVICES=0 bash script/run_experiment.sh train smoke
CUDA_VISIBLE_DEVICES=0 bash script/run_experiment.sh train stage1-10k
CUDA_VISIBLE_DEVICES=0 bash script/run_experiment.sh train stage2-10k
```

Do not start Stage 2 until this file exists:

```text
outputs/experiments/multitask/stage1-10k-seed3407/best_model/adapter_config.json
```

Stage 2 loads that adapter with `is_trainable=True`; it does not initialize a
new LoRA adapter. All training uses response-only supervision, so prompt text,
peak tables, and images are context while only assistant targets contribute to
the language-model loss.

## Inference

Run direct greedy decoding first, followed by constrained inference:

```bash
CUDA_VISIBLE_DEVICES=0 bash script/run_experiment.sh infer stage2-10k
CUDA_VISIBLE_DEVICES=0 bash script/run_experiment.sh infer constrained-10k
```

Constrained inference samples 32 structures at temperature 0.7 and top-p 0.9,
rejects invalid or out-of-domain structures, applies an exact molecular-formula
constraint, and asks the final model to rank the survivors. If no generated
candidate satisfies the supplied formula, the prediction is deliberately empty.
This is recorded as a hard-constraint failure and remains in every metric
denominator; it is not treated as a missing sample.

Primary reporting uses connectivity exact match because one-dimensional spectra
do not generally identify stereochemistry. Report stereo exact match separately,
along with validity, molecular-formula accuracy, Tanimoto similarity, scaffold
match, functional-group F1, output behavior, candidate oracle@32, constraint
coverage/failure, and ranking failure.

## Outputs

- Stage 1: `outputs/experiments/multitask/stage1-10k-seed3407/`
- Stage 2: `outputs/experiments/structure/stage2-10k-seed3407/`
- Direct predictions: `outputs/experiments/structure/predictions/stage2-10k-direct.jsonl`
- Constrained predictions: `outputs/experiments/structure/predictions/stage2-10k-constrained.jsonl`

Each prediction JSONL has a sibling `.summary.json`. Preserve configs, split ID
files, sidecars, training summaries, and prediction files for manuscript tables.
