#!/usr/bin/env bash
# Terminate this pod when it stops earning its keep.
#
# The previous run lost $10.33 to a pod nobody turned off. A pod bills by the
# hour whether or not the GPU is doing anything, and the failure mode is not
# "forgot to stop it" so much as "the session that would have stopped it ended".
# So the pod has to be able to stop itself.
#
# Two independent triggers, because they catch different failures:
#   · idle   — the GPU has been below a utilisation floor for a while. This is
#              the one that catches an abandoned pod, which is what actually
#              happened.
#   · deadline — a wall-clock cap. This catches a run that is busy but wrong,
#              e.g. a loop that never converges.
#
# Run it from the pod's boot script, in the background:
#     nohup bash tools/pod_guard.sh >> /workspace/outputs/pod_guard.log 2>&1 &
#
# Environment:
#   GUARD_DEADLINE_HOURS   hard cap in hours            (default 8)
#   GUARD_IDLE_MINUTES     idle minutes before stopping (default 20)
#   GUARD_IDLE_PERCENT     GPU% at or below = idle      (default 5)
#   GUARD_POLL_SECONDS     how often to look            (default 60)
#   GUARD_DRY_RUN          1 = log the decision, do not terminate
#   RUNPOD_API_KEY         needed to actually stop the pod (see the note below)
#   RUNPOD_POD_ID          set automatically on RunPod pods
#
# On the API key: terminating from inside needs a credential, and a RunPod key
# is account-wide. Weigh that against a pod that bills all night. If you would
# rather not put a key on the pod, run with GUARD_DRY_RUN=1 — the guard still
# tells you, loudly and in the log, that the pod should be stopped, but you have
# to stop it yourself. Silence is not an option; unattended billing is.
set -uo pipefail

DEADLINE_HOURS="${GUARD_DEADLINE_HOURS:-8}"
IDLE_MINUTES="${GUARD_IDLE_MINUTES:-20}"
IDLE_PERCENT="${GUARD_IDLE_PERCENT:-5}"
POLL="${GUARD_POLL_SECONDS:-60}"
POD_ID="${RUNPOD_POD_ID:-}"

started=$(date +%s)
deadline=$(( started + $(printf '%.0f' "$(echo "$DEADLINE_HOURS * 3600" | bc -l 2>/dev/null || echo $((DEADLINE_HOURS * 3600)))") ))
idle_needed=$(( IDLE_MINUTES * 60 ))
idle_for=0

say() { echo "[pod_guard $(date -u +%FT%TZ)] $*"; }

gpu_util() {
  command -v nvidia-smi >/dev/null 2>&1 || { echo -1; return; }
  nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null \
    | awk 'BEGIN{m=0} {if ($1+0 > m) m=$1+0} END{print m+0}'
}

terminate() {
  local why="$1"
  say "STOPPING POD — $why"

  if [ "${GUARD_DRY_RUN:-0}" = "1" ]; then
    say "GUARD_DRY_RUN=1 — not terminating. THIS POD IS STILL BILLING. Stop it yourself."
    return 0
  fi
  if [ -z "$POD_ID" ]; then
    say "RUNPOD_POD_ID is not set, cannot identify this pod. STILL BILLING — stop it manually."
    return 1
  fi
  if [ -z "${RUNPOD_API_KEY:-}" ]; then
    say "RUNPOD_API_KEY is not set, cannot call the API. STILL BILLING — stop pod $POD_ID manually."
    return 1
  fi

  # Prefer stop (keeps the volume) over terminate (destroys it): a run that hit
  # a deadline usually still has artefacts worth collecting.
  local code
  code=$(curl -s -o /tmp/pod_guard_resp -w '%{http_code}' -X POST \
    -H "Authorization: Bearer $RUNPOD_API_KEY" \
    "https://api.runpod.ai/v2/pods/$POD_ID/stop" || echo 000)
  if [ "$code" = "200" ] || [ "$code" = "204" ]; then
    say "pod $POD_ID stopped (HTTP $code)"
    return 0
  fi

  say "stop failed (HTTP $code): $(head -c 300 /tmp/pod_guard_resp 2>/dev/null)"
  if command -v runpodctl >/dev/null 2>&1; then
    say "trying runpodctl"
    runpodctl stop pod "$POD_ID" && { say "stopped via runpodctl"; return 0; }
  fi
  say "COULD NOT STOP THIS POD. IT IS STILL BILLING. Stop $POD_ID manually now."
  return 1
}

say "watching: deadline ${DEADLINE_HOURS}h, idle ${IDLE_MINUTES}m at <=${IDLE_PERCENT}% GPU, poll ${POLL}s"
[ -z "$POD_ID" ] && say "note: RUNPOD_POD_ID unset — will warn but cannot self-stop"
[ -z "${RUNPOD_API_KEY:-}" ] && say "note: RUNPOD_API_KEY unset — will warn but cannot self-stop"

while true; do
  sleep "$POLL"
  now=$(date +%s)

  if [ "$now" -ge "$deadline" ]; then
    terminate "wall-clock deadline of ${DEADLINE_HOURS}h reached"
    exit 0
  fi

  util=$(gpu_util)
  if [ "$util" -lt 0 ]; then
    continue                      # no nvidia-smi: deadline still applies
  fi
  if [ "$util" -le "$IDLE_PERCENT" ]; then
    idle_for=$(( idle_for + POLL ))
    # Say something when idle starts and then periodically, so a log being read
    # later shows when the GPU went quiet rather than only the moment it stopped.
    if [ "$idle_for" -le "$POLL" ] || [ $(( idle_for % 300 )) -lt "$POLL" ]; then
      say "GPU at ${util}% — idle $(( idle_for / 60 ))m of ${IDLE_MINUTES}m allowed"
    fi
    if [ "$idle_for" -ge "$idle_needed" ]; then
      terminate "GPU idle at <=${IDLE_PERCENT}% for ${IDLE_MINUTES}m"
      exit 0
    fi
  elif [ "$idle_for" -ne 0 ]; then
    say "GPU back to ${util}% — idle timer reset"
    idle_for=0
  fi
done
