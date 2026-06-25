# SpectraLM

SpectraLM fine-tunes a text LLM to predict molecular connectivity from
one-dimensional `1H` and `13C` NMR peak tables, with an optional molecular
formula.

## Current Workflow

The repository has one supported research workflow:

1. Build a paired JSONL mother dataset from the raw CSV files.
2. Curate a 10k cohort with 8,000 train, 1,000 validation, and 1,000 test
   samples.
3. Build formula-matched hard negatives for candidate-ranking supervision.
4. Train Qwen3-8B text adapters for formula-conditioned and no-formula
   settings.
5. Evaluate direct greedy prediction and Top-k candidate generation followed by
   formula filtering, NMR-rule pre-ranking, and model-based candidate ranking.

The complete commands are documented in
[`docs/experiments.md`](docs/experiments.md). The NMR interpretation policy is
documented in [`docs/nmr_1d_rulebook.md`](docs/nmr_1d_rulebook.md).

## Main Commands

```bash
conda activate ml

bash script/build_full_jsonl.sh \
  dataset/NMRexp_10to24_1_1004.csv \
  dataset/paired_jsonl_full

bash script/run_experiment.sh list
bash script/run_experiment.sh prepare split-10k
bash script/run_experiment.sh prepare candidates-formula-10k-train
bash script/run_experiment.sh prepare candidates-formula-10k-val
CUDA_VISIBLE_DEVICES=0 bash script/run_experiment.sh train smoke
CUDA_VISIBLE_DEVICES=0 bash script/run_experiment.sh train stage1-formula-10k
CUDA_VISIBLE_DEVICES=0 bash script/run_experiment.sh train stage2-formula-10k
CUDA_VISIBLE_DEVICES=0 bash script/run_experiment.sh train stage1-no-formula-10k
CUDA_VISIBLE_DEVICES=0 bash script/run_experiment.sh train stage2-no-formula-10k
CUDA_VISIBLE_DEVICES=0 bash script/run_experiment.sh infer direct-formula-10k
CUDA_VISIBLE_DEVICES=0 bash script/run_experiment.sh infer candidates-formula-10k
CUDA_VISIBLE_DEVICES=0 bash script/run_experiment.sh infer direct-no-formula-10k
CUDA_VISIBLE_DEVICES=0 bash script/run_experiment.sh infer candidates-no-formula-10k
```

## Repository Layout

- `src/data/`: molecule normalization, JSONL datasets, and task construction.
- `src/nmr_rules/`: deterministic NMR evidence and candidate validation.
- `src/training/`: CUDA text training and inference entrypoints.
- `src/evaluation/`: prompts and evaluation metrics.
- `script/`: supported preprocessing and experiment commands.
- `configs/`: the smoke and text-only 10k experiment configurations.
- `rules/nmr_1d.yaml`: machine-readable NMR rule library.

## Verification

```bash
conda activate ml
python -m compileall src tests script
pytest
```
