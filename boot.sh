#!/usr/bin/env bash
# Container entrypoint. Everything that has to survive the env-var code
# transport intact lives here: this is the smallest file that can be shipped,
# and a payload carrying only boot.sh is short enough to verify by length.
mkdir -p /workspace/outputs
exec > >(tee -a /workspace/outputs/pod_run.log) 2>&1
echo "BOOT $(date -u +%FT%TZ)"
O=/workspace/outputs
C=/workspace/code

# Restart-loop guard. Runpod restarts a container that exits, so a bootstrap
# that can never succeed would bill in a loop. The counter is keyed to
# BOOT_EPOCH: shipping a new payload starts a fresh count, otherwise attempts
# made against code that has since been fixed keep the guard latched forever.
# Nothing in here can stop billing -- exiting just gets the container
# restarted -- so the abort path holds instead of spinning at the restart
# interval. Only stop-pod from outside actually stops the meter.
B="$O/.boot_count.${BOOT_EPOCH:-0}"
n=$(( $(cat "$B" 2>/dev/null || echo 0) + 1 ))
echo "$n" > "$B"
echo "[boot] attempt $n (epoch ${BOOT_EPOCH:-0}, limit ${BOOT_LIMIT:-6})"
if [ "$n" -gt "${BOOT_LIMIT:-6}" ] && [ ! -f "$O/.stage/train.done" ]; then
  echo "### BOOTSTRAP_LOOP_ABORT after $n boots without reaching training"
  sleep "${ABORT_HOLD_SECONDS:-600}"
  exit 3
fi

# The pipeline is untarred over /workspace/code, so a payload that arrives
# corrupted can leave a partially rewritten tree behind. Syntax-check every
# module before doing anything expensive.
bad=0
bash -n "$C/run_all.sh" 2>/dev/null || { echo "### CODE_TREE_CORRUPT run_all.sh"; bad=1; }
for f in "$C"/pipeline/*.py; do
  python3 -c 'import ast,sys; ast.parse(open(sys.argv[1]).read())' "$f" 2>/dev/null \
    || { echo "### CODE_TREE_CORRUPT $f"; bad=1; }
done
for m in common dataset preflight token_audit train evaluate push_hub report \
         inference prefetch_model fetch_dataset; do
  [ -f "$C/pipeline/$m.py" ] || { echo "### CODE_TREE_MISSING $m.py"; bad=1; }
done
if [ "$bad" -ne 0 ]; then
  echo "### CODE_TREE_UNUSABLE - reship the affected files"
  sleep "${ABORT_HOLD_SECONDS:-600}"
  exit 4
fi
echo "[boot] code tree verified"

# The stale in-tree guard from an older payload counts on a fixed path; keep it
# from latching now that boot.sh owns the restart-loop guard.
echo 0 > "$O/.boot_count"

# A pod bills by the hour whether or not it is doing anything, and the session
# that would have stopped it may end first. That is how the previous run lost
# $10.33. The guard watches GPU utilisation and the clock and stops the pod
# itself; without RUNPOD_API_KEY it can only warn, loudly, in its log.
GUARD="$C/tools/pod_guard.sh"
[ -f "$GUARD" ] || GUARD="$(dirname "$C")/tools/pod_guard.sh"
if [ "${GUARD_DISABLE:-0}" != "1" ] && [ -f "$GUARD" ]; then
  nohup bash "$GUARD" >> "$O/pod_guard.log" 2>&1 &
  echo "[boot] pod_guard started (deadline ${GUARD_DEADLINE_HOURS:-8}h, idle ${GUARD_IDLE_MINUTES:-20}m) -> $O/pod_guard.log"
else
  echo "[boot] WARNING: pod_guard not started - this pod will bill until stopped by hand"
fi

bash "$C/run_all.sh"
