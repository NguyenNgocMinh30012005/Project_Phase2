#!/usr/bin/env bash
set -euo pipefail

python -m pip install \
  "huggingface_hub==0.25.2" \
  "diffusers==0.25.0" \
  "transformers==4.36.2" \
  "accelerate==0.25.0" \
  "einops==0.7.0" \
  "torchmetrics==1.2.1" \
  fvcore \
  cloudpickle \
  omegaconf \
  pycocotools \
  basicsr \
  av

# IDM-VTON's reference environment does not require PEFT. Newer PEFT releases
# import APIs missing from accelerate==0.25.0 and can break diffusers loading.
python -m pip uninstall -y peft >/dev/null 2>&1 || true

python - <<'PY'
import accelerate
import diffusers
import einops
import torch
import torchvision
import transformers

print("IDM-VTON runtime ready")
print(f"torch={torch.__version__}")
print(f"torchvision={torchvision.__version__}")
print(f"diffusers={diffusers.__version__}")
print(f"transformers={transformers.__version__}")
print(f"accelerate={accelerate.__version__}")
print(f"einops={einops.__version__}")
PY
