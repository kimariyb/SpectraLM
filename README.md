# SpectraLM

SpectraLM fine-tunes a vision-language model to predict a molecular structure
from paired one-dimensional `1H` and `13C` NMR spectrum images, peak tables,
and an optional molecular formula.

## Current Workflow

The repository has one supported research workflow:

1. Curate a 10k development cohort: 9,000 train and 1,000 validation samples.
2. Keep a separate scaffold-disjoint test set of up to 5,000 samples.
3. Build formula-matched hard negatives for candidate-ranking supervision.
4. Train one adapter sequentially: multitask Stage 1, then structure-only Stage 2.
5. Evaluate both greedy prediction and 32-candidate formula-constrained inference.

The complete commands and experiment matrix are documented in
[`docs/experiments.md`](docs/experiments.md). The NMR interpretation
policy is documented in [`docs/nmr_1d_rulebook.md`](docs/nmr_1d_rulebook.md).

## Main Commands

```bash
conda activate ml

bash script/build_full_jsonl.sh \
  dataset/NMRexp_10to24_1_1004.csv \
  dataset/paired_jsonl_full

python script/pre_render_jsonl_images.py dataset/paired_jsonl_full \
  --splits clean_10k_train clean_10k_val clean_10k_test \
  --image-size 512 288 \
  --num-workers 32

bash script/run_experiment.sh list
bash script/run_experiment.sh prepare split-10k
bash script/run_experiment.sh prepare candidates-10k-train
bash script/run_experiment.sh prepare candidates-10k-val
CUDA_VISIBLE_DEVICES=0 bash script/run_experiment.sh train smoke
CUDA_VISIBLE_DEVICES=0 bash script/run_experiment.sh train stage1-10k
CUDA_VISIBLE_DEVICES=0 bash script/run_experiment.sh train stage2-10k
CUDA_VISIBLE_DEVICES=0 bash script/run_experiment.sh infer stage2-10k
CUDA_VISIBLE_DEVICES=0 bash script/run_experiment.sh infer constrained-10k
```

## Repository Layout

- `src/data/`: molecule normalization, JSONL datasets, task construction.
- `src/spectra/`: one-dimensional NMR spectrum rendering.
- `src/nmr_rules/`: deterministic NMR evidence and candidate validation.
- `src/training/`: CUDA training and inference entrypoints.
- `src/evaluation/`: prompts and evaluation metrics.
- `script/`: supported preprocessing and experiment commands.
- `configs/`: the smoke and two-stage 10k experiment configurations.
- `rules/nmr_1d.yaml`: machine-readable NMR rule library.

## Verification

```bash
conda activate ml
python -m compileall src tests
pytest
```
