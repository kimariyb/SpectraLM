# SpectraLM Text-Only 10k Experiment

This is the only active experiment protocol. The 10k cohort contains 8,000
training samples, 1,000 validation samples, and 1,000 test samples.

## Fixed Protocol

- Base model: `/mnt/data/kimariyb/models/Qwen3-8B`
- Input: serialized `1H` and `13C` NMR peak tables
- Formula setting: formula-conditioned and no-formula experiments are trained
  separately
- Target: canonical connectivity SMILES without stereochemistry
- Molecule domain: one neutral, non-radical, isotope-free component containing
  only H, C, N, O, F, Si, P, S, Cl, Br, or I
- System prompt: non-thinking structure elucidation, no reasoning output
- Output schema: `{"smiles":"string (canonical SMILES, or null if insufficient data)"}`
- Supervision: response-only labels; prompt, peak tables, formula, and rules
  are context and do not contribute to the loss

## Data Preparation

```bash
conda activate ml

bash script/build_full_jsonl.sh \
  dataset/NMRexp_10to24_1_1004.csv \
  dataset/paired_jsonl_full

bash script/run_experiment.sh prepare split-10k
bash script/run_experiment.sh prepare candidates-formula-10k-train
bash script/run_experiment.sh prepare candidates-formula-10k-val
```

After `prepare split-10k`, verify that the generated split files contain
8,000 train IDs, 1,000 validation IDs, and 1,000 test IDs.

## Training

Run smoke first:

```bash
CUDA_VISIBLE_DEVICES=0 bash script/run_experiment.sh train smoke
```

Formula-conditioned path:

```bash
CUDA_VISIBLE_DEVICES=0 bash script/run_experiment.sh train stage1-formula-10k
CUDA_VISIBLE_DEVICES=0 bash script/run_experiment.sh train stage2-formula-10k
```

No-formula ablation:

```bash
CUDA_VISIBLE_DEVICES=0 bash script/run_experiment.sh train stage1-no-formula-10k
CUDA_VISIBLE_DEVICES=0 bash script/run_experiment.sh train stage2-no-formula-10k
```

Stage 1 uses direct structure prediction, candidate ranking, functional-group
recognition, and spectral-region classification. Stage 2 continues the Stage 1
adapter with direct structure prediction only.

## Inference

Formula-conditioned evaluation:

```bash
CUDA_VISIBLE_DEVICES=0 bash script/run_experiment.sh infer direct-formula-10k
CUDA_VISIBLE_DEVICES=0 bash script/run_experiment.sh infer candidates-formula-10k
```

No-formula evaluation:

```bash
CUDA_VISIBLE_DEVICES=0 bash script/run_experiment.sh infer direct-no-formula-10k
CUDA_VISIBLE_DEVICES=0 bash script/run_experiment.sh infer candidates-no-formula-10k
```

Candidate inference samples 32 structures at temperature 0.7 and top-p 0.9. In
formula-conditioned runs, invalid, out-of-domain, and formula-mismatched
candidates are rejected before rule pre-ranking and model-based candidate
selection. In no-formula runs, formula filtering is skipped but domain validity
and rule/model ranking remain active.

## Outputs

- Multitask adapters: `outputs/experiments/multitask/`
- Structure adapters: `outputs/experiments/structure/`
- Predictions: `outputs/experiments/structure/predictions/`

Primary reporting uses connectivity exact match. Also report exact match, valid
SMILES rate, molecular-formula accuracy, Tanimoto similarity, scaffold match,
functional-group F1, functional-group spectral support, output-format
compliance, invalid-structure rate, non-SMILES output rate, candidate oracle@32,
constraint failure, and ranking failure.
