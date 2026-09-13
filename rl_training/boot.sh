#!/bin/bash
set -euo pipefail
umask 077
cd /workspace/Gemma-claim
set -a
source /workspace/pod.env
set +a
export HF_HOME=/workspace/hf-cache
export PYTHONUNBUFFERED=1
export PYTHONPATH=/workspace/Gemma-claim
export RL_OUT
export TOKENIZERS_PARALLELISM=false
mkdir -p "$RL_OUT"
python -u -m rl_training.followup_guard > /workspace/remote-guard.log 2>&1 < /dev/null &
trap 'rc=$?; if [ "$rc" -ne 0 ]; then printf "{\"stage\":\"failed\",\"boot_exit_code\":%s,\"time\":%s}\n" "$rc" "$(date +%s)" > "$RL_OUT/status.json"; fi' EXIT
python -m venv --system-site-packages /workspace/rl-venv
/workspace/rl-venv/bin/python -m pip install -r rl_training/requirements.txt
/workspace/rl-venv/bin/python -m pip freeze > "$RL_OUT/pip-freeze.txt"
/workspace/rl-venv/bin/python -u -m "${RL_TRAINING_MODULE:-rl_training.train}"
