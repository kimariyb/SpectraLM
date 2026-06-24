#!/usr/bin/env bash
set -euo pipefail

CONFIG="${1:-configs/train_smoke.yaml}"
: "${CUDA_VISIBLE_DEVICES:=0}"
export CUDA_VISIBLE_DEVICES

python - <<'PY'
import torch

print(f"torch={torch.__version__}")
print(f"cuda_available={torch.cuda.is_available()}")
print(f"visible_cuda_devices={torch.cuda.device_count()}")
if not torch.cuda.is_available():
    raise SystemExit("CUDA is not available. Check NVIDIA driver, CUDA_VISIBLE_DEVICES, and PyTorch build.")
for idx in range(torch.cuda.device_count()):
    prop = torch.cuda.get_device_properties(idx)
    print(f"cuda:{idx} {prop.name} {prop.total_memory / 1024**3:.1f} GB")
PY

python -m src.training.train "${CONFIG}"
