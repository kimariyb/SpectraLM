#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

usage() {
  cat <<'EOF'
Usage:
  bash script/run_experiment.sh list
  bash script/run_experiment.sh prepare split-10k
  bash script/run_experiment.sh prepare candidates-formula-10k-train
  bash script/run_experiment.sh prepare candidates-formula-10k-val
  bash script/run_experiment.sh train smoke
  CUDA_VISIBLE_DEVICES=0 bash script/run_experiment.sh train stage1-formula-10k
  CUDA_VISIBLE_DEVICES=0 bash script/run_experiment.sh train stage2-formula-10k
  CUDA_VISIBLE_DEVICES=0 bash script/run_experiment.sh train stage1-no-formula-10k
  CUDA_VISIBLE_DEVICES=0 bash script/run_experiment.sh train stage2-no-formula-10k
  CUDA_VISIBLE_DEVICES=0 bash script/run_experiment.sh infer direct-formula-10k
  CUDA_VISIBLE_DEVICES=0 bash script/run_experiment.sh infer candidates-formula-10k
  CUDA_VISIBLE_DEVICES=0 bash script/run_experiment.sh infer direct-no-formula-10k
  CUDA_VISIBLE_DEVICES=0 bash script/run_experiment.sh infer candidates-no-formula-10k
EOF
}

stage="${1:-list}"
run_name="${2:-}"
dataset_dir="dataset/paired_jsonl_full"

if [[ "${stage}" == "list" ]]; then
  usage
  exit 0
fi

case "${stage}:${run_name}" in
  prepare:split-10k)
    exec python script/curate_jsonl_subsets.py "${dataset_dir}" \
      --subset-sizes 10000 \
      --val-fraction 0.1 \
      --test-fraction 0.1 \
      --prefix clean \
      --seed 3407 \
      --max-heavy-atoms 60 \
      --max-h-peaks 80 \
      --max-c-peaks 120 \
      --solvent-policy any ;;
  prepare:candidates-formula-10k-train)
    exec python script/build_candidate_sidecar.py "${dataset_dir}" \
      --split clean_10k_train \
      --output "${dataset_dir}/candidate_sets_clean_10k_train.jsonl" \
      --candidates-per-sample 8 \
      --max-pool-size 512 \
      --seed 3407 ;;
  prepare:candidates-formula-10k-val)
    exec python script/build_candidate_sidecar.py "${dataset_dir}" \
      --split clean_10k_val \
      --output "${dataset_dir}/candidate_sets_clean_10k_val.jsonl" \
      --candidates-per-sample 8 \
      --max-pool-size 512 \
      --seed 3407 ;;
  train:smoke)
    config="configs/train_smoke.yaml" ;;
  train:stage1-formula-10k)
    config="configs/experiments/train_stage1_formula_10k.yaml" ;;
  train:stage2-formula-10k)
    config="configs/experiments/train_stage2_formula_10k.yaml" ;;
  train:stage1-no-formula-10k)
    config="configs/experiments/train_stage1_no_formula_10k.yaml" ;;
  train:stage2-no-formula-10k)
    config="configs/experiments/train_stage2_no_formula_10k.yaml" ;;
  infer:direct-formula-10k)
    config="configs/experiments/infer_direct_formula_10k.yaml" ;;
  infer:candidates-formula-10k)
    config="configs/experiments/infer_candidates_formula_10k.yaml" ;;
  infer:direct-no-formula-10k)
    config="configs/experiments/infer_direct_no_formula_10k.yaml" ;;
  infer:candidates-no-formula-10k)
    config="configs/experiments/infer_candidates_no_formula_10k.yaml" ;;
  *)
    usage >&2
    exit 2 ;;
esac

if [[ "${stage}" == "train" ]]; then
  exec bash script/run_train_cuda_48g.sh "${config}"
fi

: "${CUDA_VISIBLE_DEVICES:=0}"
export CUDA_VISIBLE_DEVICES
if [[ "${run_name}" == candidates-* ]]; then
  exec python -m src.training.constrained_inference "${config}"
fi
exec python -m src.training.inference "${config}"
