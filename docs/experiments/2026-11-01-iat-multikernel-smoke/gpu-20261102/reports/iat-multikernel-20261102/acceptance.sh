#!/bin/bash
set -u
workspace=/root/autodl-tmp/ant-workspace
report="$workspace/reports/iat-multikernel-20261102"
cd "$workspace/src/AgentPrint-Pro-IAT" || exit 1
export TMPDIR="$workspace/tmp/iat" TEMP="$workspace/tmp/iat" TMP="$workspace/tmp/iat"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export XDG_CACHE_HOME="$workspace/runtime-cache/iat-20261102"
export MPLCONFIGDIR="$workspace/runtime-cache/matplotlib-iat-20261102"
date -Iseconds > "$report/start.txt"
"$workspace/envs/optim-agent/bin/python" -u -m agentprint.iat_smoke \
  --config "$workspace/reports/iat-multikernel-20261101/base.yaml" \
  --workspace-root "$workspace" \
  --work-root "$workspace/acceptance/iat-multikernel-20261102-gpu" \
  > "$report/smoke.log" 2> "$report/smoke-stderr.log"
status=$?
echo "$status" > "$report/smoke.exit"
date -Iseconds > "$report/end.txt"
exit "$status"
