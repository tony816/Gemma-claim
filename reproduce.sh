#!/usr/bin/env bash
# Reproduce validation, conversion, training and evaluation from a dataset ZIP.
#   ./reproduce.sh /path/to/final_dataset_v112.zip [output_dir]
set -euo pipefail

ZIP="${1:?usage: reproduce.sh <dataset zip> [output dir]}"
export WORKSPACE="${2:-$(pwd)/run}"
export OUT_DIR="$WORKSPACE/outputs"
export DATA_ROOT="$WORKSPACE/data/v112"
export CODE_DIR="${CODE_DIR:-$(cd "$(dirname "$0")" && pwd)}"
export HF_HOME="${HF_HOME:-$WORKSPACE/hf_home}"
export PYTHONPATH="$CODE_DIR/pipeline:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

export BASE_MODEL="${BASE_MODEL:-google/gemma-4-31B-it}"
export BASE_MODEL_REVISION="${BASE_MODEL_REVISION:-main}"
export SEED="${SEED:-42}"
export MICRO_BS="${MICRO_BS:-1}"
export GRAD_ACCUM="${GRAD_ACCUM:-4}"
export EPOCHS="${EPOCHS:-6}"
export LR="${LR:-1e-4}"

mkdir -p "$WORKSPACE"
cp -n "$ZIP" "$WORKSPACE/final_dataset_v112.zip" 2>/dev/null || true

python -m pip install -r "$CODE_DIR/requirements.txt"

python "$CODE_DIR/pipeline/fetch_dataset.py"     # sha256 gate + extract
python "$CODE_DIR/pipeline/preflight.py"         # data contract gate
python "$CODE_DIR/pipeline/token_audit.py"       # zero-truncation gate
python "$CODE_DIR/pipeline/train.py"             # bf16 LoRA SFT
EVAL_SPLITS=validation,test python "$CODE_DIR/pipeline/evaluate.py"
python "$CODE_DIR/pipeline/report.py"

echo "Artifacts in $OUT_DIR"
