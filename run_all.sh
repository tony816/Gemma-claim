#!/usr/bin/env bash
# Pod-side orchestrator. Stages are idempotent: each writes a marker under
# $OUT/.stage/, so a restarted container resumes instead of redoing work.
set -uo pipefail

export WORKSPACE="${WORKSPACE:-/workspace}"
export OUT_DIR="${OUT_DIR:-$WORKSPACE/outputs}"
export DATA_ROOT="${DATA_ROOT:-$WORKSPACE/data/v112}"
export CODE_DIR="${CODE_DIR:-$WORKSPACE/code}"
export HF_HOME="${HF_HOME:-$WORKSPACE/hf_home}"
export PYTHONPATH="$CODE_DIR/pipeline:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

MARK="$OUT_DIR/.stage"
mkdir -p "$OUT_DIR" "$MARK" "$HF_HOME" "$WORKSPACE/data"

banner() { echo; echo "=================== $* ==================="; echo; }
fail()   { echo; echo "### PIPELINE_FAILED_AT $1"; echo "PIPELINE_STATUS=FAILED"; }

run_stage() {
  local name="$1"; shift
  if [ -f "$MARK/$name.done" ] && [ "${FORCE_STAGE:-}" != "$name" ] && [ "${FORCE_ALL:-0}" != "1" ]; then
    echo "[stage] $name already complete - skipping"
    return 0
  fi
  banner "STAGE $name"
  STAGE="$name" "$@"
  local rc=$?
  if [ $rc -ne 0 ]; then
    fail "$name (exit $rc)"
    return $rc
  fi
  touch "$MARK/$name.done"
  echo "[stage] $name OK"
  return 0
}

setup() {
  python -m pip install -q --upgrade pip
  # gemma4 needs a transformers that knows the architecture.
  if ! python -m pip install -q "transformers>=5.5.0"; then
    echo "[setup] pinned transformers unavailable on PyPI; installing from git main"
    python -m pip install -q "git+https://github.com/huggingface/transformers.git"
  fi
  python -m pip install -q \
    "accelerate>=1.10.0" "peft>=0.17.0" "safetensors>=0.4.5" "sentencepiece>=0.2.0" \
    "pillow>=10.4.0" "huggingface_hub[hf_transfer]>=0.34.0" "gdown>=5.2.0" \
    "rouge-score>=0.1.2" "sacrebleu>=2.4.3" "pyyaml>=6.0.2" || return 1
  python - <<'PY'
import transformers, torch
print("transformers", transformers.__version__, "torch", torch.__version__)
from transformers import AutoConfig
assert torch.cuda.is_available(), "no CUDA device visible"
p = torch.cuda.get_device_properties(0)
print("gpu", p.name, round(p.total_memory / 2**30, 1), "GiB")
PY
}

main() {
  banner "ENVIRONMENT"
  nvidia-smi || true
  df -h "$WORKSPACE" || true
  free -g || true

  run_stage setup     setup                                        || return 1
  run_stage fetch     python "$CODE_DIR/pipeline/fetch_dataset.py"  || return 1
  run_stage preflight python "$CODE_DIR/pipeline/preflight.py"      || return 1
  run_stage audit     python "$CODE_DIR/pipeline/token_audit.py"    || return 1
  run_stage train     python "$CODE_DIR/pipeline/train.py"          || return 1

  # Configuration and checkpoint are frozen by this point; test is scored once.
  run_stage evaluate  env EVAL_SPLITS=validation,test \
                      python "$CODE_DIR/pipeline/evaluate.py"       || return 1

  cp -f "$CODE_DIR/pipeline/inference.py" "$OUT_DIR/inference.py" 2>/dev/null || true
  cp -f "$CODE_DIR/reproduce.sh"          "$OUT_DIR/reproduce.sh"  2>/dev/null || true

  run_stage push      python "$CODE_DIR/pipeline/push_hub.py"       || return 1
  run_stage report    python "$CODE_DIR/pipeline/report.py"         || return 1

  banner "FINAL REPORT"
  cat "$OUT_DIR/FINAL_TRAINING_REPORT.md" || true
  echo
  echo "PIPELINE_STATUS=SUCCESS"
}

main
rc=$?
echo "PIPELINE_EXIT=$rc"
echo "PIPELINE_DONE_MARKER"
# Stay alive so the run can be inspected and stages re-driven via a restart.
sleep infinity
