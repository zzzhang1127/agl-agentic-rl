#!/usr/bin/env bash
# Install a CUDA 12.4 GPU stack into the project .venv.
# Driver 550.x cannot load cu128/cu129/cu130 wheels from scripts/setup_verl.sh.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
VENV="${1:-$ROOT/.venv}"
PYTHON_BIN="$VENV/bin/python"

if [ ! -x "$PYTHON_BIN" ]; then
  echo "ERROR: expected $PYTHON_BIN. Run uv sync first." >&2
  exit 1
fi

export PATH="${HOME}/.local/bin:/workspace/.local/bin:${PATH}"
export UV_INDEX_URL="${UV_INDEX_URL:-https://mirrors.aliyun.com/pypi/simple/}"
export UV_DEFAULT_INDEX="${UV_DEFAULT_INDEX:-https://mirrors.aliyun.com/pypi/simple/}"
export PIP_INDEX_URL="${PIP_INDEX_URL:-https://mirrors.aliyun.com/pypi/simple/}"
export PIP_TRUSTED_HOST="${PIP_TRUSTED_HOST:-mirrors.aliyun.com}"
export UV_HTTP_TIMEOUT="${UV_HTTP_TIMEOUT:-300}"
export UV_LINK_MODE="${UV_LINK_MODE:-copy}"
echo "Python: $PYTHON_BIN"
echo "Installing torch 2.6.0+cu124, vllm 0.8.5, verl 0.7.1, swanlab"

uv pip install --python "$PYTHON_BIN" pip
# Aliyun's cu124 tree is a wheel dump, not a PEP 503 index, so pin the files.
uv pip install --python "$PYTHON_BIN" \
  "https://mirrors.aliyun.com/pytorch-wheels/cu124/torch-2.6.0+cu124-cp312-cp312-linux_x86_64.whl" \
  "https://mirrors.aliyun.com/pytorch-wheels/cu124/torchvision-0.21.0+cu124-cp312-cp312-linux_x86_64.whl" \
  "https://mirrors.aliyun.com/pytorch-wheels/cu124/torchaudio-2.6.0+cu124-cp312-cp312-linux_x86_64.whl"

# Keep the cu124 torch; verl's published extras may try to pull a newer CUDA build.
uv pip install --python "$PYTHON_BIN" "vllm==0.8.5" "swanlab" "tensorboard"
uv pip install --python "$PYTHON_BIN" "verl==0.7.1" --no-deps
uv pip install --python "$PYTHON_BIN" \
  hydra-core omegaconf ray tensordict datasets transformers accelerate \
  pyarrow pandas peft codetiming pylatexenc torchdata wandb

"$PYTHON_BIN" - <<'PY'
import torch
print("torch", torch.__version__, "cuda", torch.version.cuda, "available", torch.cuda.is_available())
import vllm
print("vllm", vllm.__version__)
import verl
print("verl", getattr(verl, "__version__", verl.__file__))
import swanlab
print("swanlab", swanlab.__version__)
PY
