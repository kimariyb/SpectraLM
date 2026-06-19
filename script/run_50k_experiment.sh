#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

usage() {
  cat <<'EOF'
Usage:
  bash script/run_50k_experiment.sh list
  bash script/run_50k_experiment.sh train <run-name>
  bash script/run_50k_experiment.sh infer <run-name>

Training runs:
  scale-5k  scale-10k  scale-25k
  main-3407  main-42  main-2026  no-formula

Inference runs:
  zero-shot  scale-5k  scale-10k  scale-25k
  main-3407  main-42  main-2026  no-formula
EOF
}

stage="${1:-list}"
run_name="${2:-}"

if [[ "${stage}" == "list" ]]; then
  usage
  exit 0
fi

if [[ "${CONDA_DEFAULT_ENV:-}" != "ml" ]]; then
  echo "Activate the ml conda environment before running experiments." >&2
  exit 2
fi

case "${stage}:${run_name}" in
  train:scale-5k)
    config="configs/experiments/train_scale_5k.yaml" ;;
  train:scale-10k)
    config="configs/experiments/train_scale_10k.yaml" ;;
  train:scale-25k)
    config="configs/experiments/train_scale_25k.yaml" ;;
  train:main-3407)
    config="configs/train_cuda_48g_jsonl.yaml" ;;
  train:main-42)
    config="configs/experiments/train_main_50k_seed42.yaml" ;;
  train:main-2026)
    config="configs/experiments/train_main_50k_seed2026.yaml" ;;
  train:no-formula)
    config="configs/train_cuda_48g_no_formula.yaml" ;;
  infer:zero-shot)
    config="configs/experiments/infer_zero_shot_50k.yaml" ;;
  infer:scale-5k)
    config="configs/experiments/infer_scale_5k.yaml" ;;
  infer:scale-10k)
    config="configs/experiments/infer_scale_10k.yaml" ;;
  infer:scale-25k)
    config="configs/experiments/infer_scale_25k.yaml" ;;
  infer:main-3407)
    config="configs/experiments/infer_main_50k_seed3407.yaml" ;;
  infer:main-42)
    config="configs/experiments/infer_main_50k_seed42.yaml" ;;
  infer:main-2026)
    config="configs/experiments/infer_main_50k_seed2026.yaml" ;;
  infer:no-formula)
    config="configs/experiments/infer_no_formula_50k.yaml" ;;
  *)
    usage >&2
    exit 2 ;;
esac

if [[ "${stage}" == "train" ]]; then
  exec bash script/run_train_cuda_48g.sh "${config}"
fi

: "${CUDA_VISIBLE_DEVICES:=0}"
export CUDA_VISIBLE_DEVICES
exec python -m src.training.inference "${config}"
