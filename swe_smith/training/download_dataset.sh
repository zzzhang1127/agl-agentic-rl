#!/usr/bin/env bash
# Official SWE-smith splits: https://drive.google.com/file/d/1q19DP53l4rldvBR2dkUhbaPI_mHVBVL1
set -euo pipefail

EXAMPLE_DIR="$(cd "$(dirname "$0")" && pwd)"
OUT_DIR="${1:-$EXAMPLE_DIR}"
FILE_ID="1q19DP53l4rldvBR2dkUhbaPI_mHVBVL1"
ZIP="$OUT_DIR/swe_smith_splits.zip"
PY="${PYTHON_BIN:-/workspace/projects/agent-lightning/.venv/bin/python}"

mkdir -p "$OUT_DIR"
export PATH="/workspace/.local/bin:${HOME}/.local/bin:${PATH}"
if [ ! -x "$PY" ]; then
  PY="$(command -v python3)"
fi

export UV_INDEX_URL="${UV_INDEX_URL:-https://mirrors.aliyun.com/pypi/simple/}"
export UV_DEFAULT_INDEX="${UV_DEFAULT_INDEX:-https://mirrors.aliyun.com/pypi/simple/}"
export PIP_INDEX_URL="${PIP_INDEX_URL:-https://mirrors.aliyun.com/pypi/simple/}"
export PIP_TRUSTED_HOST="${PIP_TRUSTED_HOST:-mirrors.aliyun.com}"
uv pip install --python "$PY" gdown
"$PY" -m gdown "$FILE_ID" -O "$ZIP"
python3 - <<PY
import zipfile
from pathlib import Path
zip_path = Path("$ZIP")
out = Path("$OUT_DIR")
with zipfile.ZipFile(zip_path) as zf:
    zf.extractall(out)
    print("extracted", zf.namelist())
for name in ("train_dataset_mixed.jsonl", "val_dataset_filtered.jsonl"):
    path = out / name
    print(name, "exists" if path.exists() else "MISSING", path.stat().st_size if path.exists() else 0)
PY
