#!/usr/bin/env bash
set -euo pipefail

CSV_PATH="${1:-dataset/NMRexp_10to24_1_1004.csv}"
OUT_DIR="${2:-dataset/paired_jsonl_full}"
DB_PATH="${3:-${OUT_DIR}/candidates.sqlite}"
SUBSET_SIZES="${SUBSET_SIZES:-10000}"
VAL_FRACTION="${VAL_FRACTION:-0.1}"
TEST_SIZE="${TEST_SIZE:-5000}"

python script/build_paired_jsonl.py "${CSV_PATH}" \
  --out-dir "${OUT_DIR}" \
  --db "${DB_PATH}" \
  --chunksize 100000 \
  --top-k 3 \
  --train-ratio 0.8 \
  --val-ratio 0.1 \
  --test-ratio 0.1 \
  --seed 3407

python script/curate_jsonl_subsets.py "${OUT_DIR}" \
  --subset-sizes ${SUBSET_SIZES} \
  --val-fraction "${VAL_FRACTION}" \
  --test-size "${TEST_SIZE}" \
  --prefix clean \
  --seed 3407 \
  --max-heavy-atoms 60 \
  --max-h-peaks 80 \
  --max-c-peaks 120 \
  --solvent-policy any
