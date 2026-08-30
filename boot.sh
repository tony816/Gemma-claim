#!/usr/bin/env bash
# Container entrypoint: keep the start command free of nested quoting by
# putting everything that needs quotes in here instead.
mkdir -p /workspace/outputs
exec > >(tee -a /workspace/outputs/pod_run.log) 2>&1
echo "BOOT $(date -u +%FT%TZ)"
bash /workspace/code/run_all.sh
