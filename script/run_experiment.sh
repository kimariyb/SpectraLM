#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

usage() {
  cat <<'EOF'
Usage:
  bash script/run_experiment.sh list
  bash script/run_experiment.sh train <run-name>
  bash script/run_experiment.sh infer <run-name>

Training runs:
  scale-5k  scale-10k  scale-25k
  main-3407  main-42  main-2026  no-formula
  rules-50k  rules-no-formula  multitask-50k
  modality-image-only-5k
  modality-peak-table-only-5k  modality-formula-only-5k
  modality-image-only-50k
  modality-peak-table-only-50k

Inference runs:
  zero-shot  scale-5k  scale-10k  scale-25k
  main-3407  main-42  main-2026  no-formula
  rules-50k  rules-no-formula  multitask-50k
  modality-image-only-5k
  modality-peak-table-only-5k  modality-formula-only-5k
  modality-image-only-50k
  modality-peak-table-only-50k
EOF
}

stage="${1:-list}"
run_name="${2:-}"

if [[ "${stage}" == "list" ]]; then
  usage
  exit 0
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
  train:rules-50k)
    config="configs/experiments/train_rules_50k.yaml" ;;
  train:rules-no-formula)
    config="configs/experiments/train_rules_no_formula_50k.yaml" ;;
  train:multitask-50k)
    config="configs/experiments/train_multitask_50k.yaml" ;;
  train:modality-image-only-5k)
    config="configs/experiments/train_modality_image_only_5k.yaml" ;;
  train:modality-peak-table-only-5k)
    config="configs/experiments/train_modality_peak_table_only_5k.yaml" ;;
  train:modality-formula-only-5k)
    config="configs/experiments/train_modality_formula_only_5k.yaml" ;;
  train:modality-image-only-50k)
    config="configs/experiments/train_modality_image_only_50k.yaml" ;;
  train:modality-peak-table-only-50k)
    config="configs/experiments/train_modality_peak_table_only_50k.yaml" ;;
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
  infer:rules-50k)
    config="configs/experiments/infer_rules_50k.yaml" ;;
  infer:rules-no-formula)
    config="configs/experiments/infer_rules_no_formula_50k.yaml" ;;
  infer:multitask-50k)
    config="configs/experiments/infer_multitask_50k.yaml" ;;
  infer:modality-image-only-5k)
    config="configs/experiments/infer_modality_image_only_5k.yaml" ;;
  infer:modality-peak-table-only-5k)
    config="configs/experiments/infer_modality_peak_table_only_5k.yaml" ;;
  infer:modality-formula-only-5k)
    config="configs/experiments/infer_modality_formula_only_5k.yaml" ;;
  infer:modality-image-only-50k)
    config="configs/experiments/infer_modality_image_only_50k.yaml" ;;
  infer:modality-peak-table-only-50k)
    config="configs/experiments/infer_modality_peak_table_only_50k.yaml" ;;
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
