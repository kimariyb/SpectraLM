# SpectraLM

SpectraLM fine-tunes a vision-language model to predict a molecular structure
from paired one-dimensional `1H` and `13C` NMR spectrum images, peak tables,
and an optional molecular formula.

## Current Workflow

The repository has one supported research workflow:

1. Build a paired, scaffold-disjoint JSONL dataset from the raw CSV.
2. Curate nested 5k, 10k, 25k, and 50k training subsets.
3. Pre-render `512 x 288` spectrum images.
4. Build formula-matched candidates for the ranking auxiliary task.
5. Train and evaluate structure, rule-context, and multitask experiments.

The complete commands and experiment matrix are documented in
[`docs/experiments_50k.md`](docs/experiments_50k.md). The NMR interpretation
policy is documented in [`docs/nmr_1d_rulebook.md`](docs/nmr_1d_rulebook.md).

## Main Commands

```bash
conda activate ml

bash script/build_full_jsonl.sh \
  dataset/NMRexp_10to24_1_1004.csv \
  dataset/paired_jsonl_full

python script/pre_render_jsonl_images.py dataset/paired_jsonl_full \
  --splits clean_50k_train clean_50k_val clean_50k_test \
  --image-size 512 288 \
  --num-workers 32

CUDA_VISIBLE_DEVICES=0 bash script/run_train_cuda_48g.sh \
  configs/train_cuda_48g_smoke.yaml

bash script/run_experiment.sh list
```

## Repository Layout

- `src/data/`: molecule normalization, JSONL datasets, task construction.
- `src/spectra/`: one-dimensional NMR spectrum rendering.
- `src/nmr_rules/`: deterministic NMR evidence and candidate validation.
- `src/training/`: CUDA training and inference entrypoints.
- `src/evaluation/`: prompts and evaluation metrics.
- `script/`: supported preprocessing and experiment commands.
- `configs/`: smoke, baseline, ablation, rule, and multitask configurations.
- `rules/nmr_1d.yaml`: machine-readable NMR rule library.

## Verification

```bash
conda activate ml
python -m compileall src tests
pytest
```
