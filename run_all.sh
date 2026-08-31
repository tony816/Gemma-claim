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

wait_for_dataset() {
  local deadline=$(( SECONDS + ${DATASET_WAIT_SECONDS:-1800} ))
  local attempt=0
  while :; do
    attempt=$((attempt + 1))
    echo "[fetch] attempt $attempt"
    if python "$CODE_DIR/pipeline/fetch_dataset.py"; then
      return 0
    fi
    # A hash mismatch means the wrong bytes, not a permission problem: never
    # sit in a retry loop on it.
    if [ -f "$OUT_DIR/.hash_mismatch" ]; then
      echo "[fetch] hash mismatch - not retrying"
      return 1
    fi
    if [ $SECONDS -ge $deadline ]; then
      echo "[fetch] DATASET_WAIT_TIMEOUT after ${DATASET_WAIT_SECONDS:-1800}s"
      return 1
    fi
    echo "[fetch] dataset not reachable yet; retrying in 60s"
    echo "DATASET_ACCESS_PENDING"
    sleep 60
  done
}

main() {
  banner "ENVIRONMENT"
  nvidia-smi || true
  df -h "$WORKSPACE" || true
  free -g || true

  run_stage setup     setup                                        || return 1

  # 62.5 GB of weights is the long pole and is needed regardless of when the
  # dataset becomes reachable, so warm the cache before waiting on Drive.
  run_stage prefetch  python "$CODE_DIR/pipeline/prefetch_model.py" || return 1

  # The dataset ZIP may still be private when the pod starts. Rather than dying
  # on a permission denial, poll until the owner makes it link-readable, so the
  # run continues by itself the moment access is granted.
  run_stage fetch     wait_for_dataset                             || return 1
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

# Hard spend ceiling. An unattended GPU pod is the expensive failure mode, so a
# watchdog kills PID 1 after MAX_RUNTIME_SECONDS no matter what the pipeline is
# doing. Billing stops when the container exits.
watchdog() {
  local limit="${MAX_RUNTIME_SECONDS:-18000}"
  sleep "$limit"
  echo "### WATCHDOG_TIMEOUT after ${limit}s - killing container to stop GPU billing"
  kill -9 1 2>/dev/null || true
}
watchdog &

# Restart-loop guard: a container that exits is restarted by Runpod, and a
# bootstrap that can never succeed would bill in a loop. A few attempts without
# reaching the training stage is treated as unrecoverable.
#
# The counter is keyed to BOOT_EPOCH so that shipping a new code payload starts
# a fresh count -- otherwise attempts made against code that has since been
# fixed keep the guard latched and no amount of fixing can get past it.
#
# Nothing inside the container can stop billing (exiting just gets it
# restarted), so the abort path holds before exiting rather than spinning at
# the restart interval: same cost per hour, legible logs, and a window to pull
# them. Only stop-pod from outside actually stops the meter.
# The code payload is injected as base64 env chunks and overlaid onto
# /workspace/code, so a payload that arrives corrupted can leave a partially
# rewritten tree behind. Syntax-check every module before doing anything
# expensive: a broken tree must fail loudly here, not halfway through training.
verify_code() {
  local bad=0 f
  for f in "$CODE_DIR"/pipeline/*.py; do
    if ! python3 -c "import ast,sys; ast.parse(open(sys.argv[1]).read())" "$f" 2>/dev/null; then
      echo "### CODE_TREE_CORRUPT $f"
      bad=1
    fi
  done
  for f in common dataset preflight token_audit train evaluate push_hub report inference prefetch_model fetch_dataset; do
    [ -f "$CODE_DIR/pipeline/$f.py" ] || { echo "### CODE_TREE_MISSING pipeline/$f.py"; bad=1; }
  done
  if [ "$bad" -ne 0 ]; then
    echo "### CODE_TREE_UNUSABLE - refusing to run. Reship the affected files."
    sleep "${ABORT_HOLD_SECONDS:-600}"
    exit 4
  fi
  echo "[boot] code tree verified"
}
verify_code

BOOTS="$OUT_DIR/.boot_count.${BOOT_EPOCH:-0}"
count=$(( $(cat "$BOOTS" 2>/dev/null || echo 0) + 1 ))
echo "$count" > "$BOOTS"
echo "[boot] attempt $count (epoch ${BOOT_EPOCH:-0}, limit ${BOOT_LIMIT:-3})"
if [ "$count" -gt "${BOOT_LIMIT:-3}" ] && [ ! -f "$MARK/train.done" ]; then
  echo "### BOOTSTRAP_LOOP_ABORT: $count boots in epoch ${BOOT_EPOCH:-0} without reaching training."
  echo "### Holding ${ABORT_HOLD_SECONDS:-600}s so the loop does not hammer, then exiting."
  sleep "${ABORT_HOLD_SECONDS:-600}"
  exit 3
fi

main
rc=$?
echo "PIPELINE_EXIT=$rc"
echo "PIPELINE_DONE_MARKER"

# On failure hold briefly so the logs can be pulled, then exit -- never idle
# indefinitely on a paid GPU. On success exit immediately.
if [ $rc -ne 0 ]; then
  echo "Holding ${FAILURE_HOLD_SECONDS:-300}s for log retrieval, then exiting."
  sleep "${FAILURE_HOLD_SECONDS:-300}"
fi
echo "PIPELINE_CONTAINER_EXITING rc=$rc"
exit $rc
