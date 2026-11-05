#!/usr/bin/env bash
# Run the vLLM perf benchmark. Self-contained: clone this dir on the GPU box,
# set VLLM_GIT (or paths.vllm_git in config.yaml), then:
#   ./run.sh                 # default set from config.yaml (interesting)
#   ./run.sh all             # all 202 PRs
#   ./run.sh 21222 21627     # specific PRs
#   ./run.sh interesting     # a named set from config.yaml
#   ./run.sh workloads       # run per-PR workload/ images by cluster (GPU box)
#
# Per PR: build image at base_sha, run REAL test_patch for base(FAIL) +
# gold(PASS) + each enabled solver, measure latency with repeats->variance,
# dump results/<pr>.json (the HTML live-reads these).
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"; cd "$HERE"
mkdir -p .work results

# workloads mode: iterate clusters in order, download weights once per cluster,
# build+run each per-PR workload/ image for both arms. Delegated to run_workloads.sh.
if [ "${1:-}" = "workloads" ]; then
  shift
  exec bash "$HERE/run_workloads.sh" "$@"
fi

if [ $# -eq 0 ]; then
  default=$(python3 conf.py get run.default)
  PRS=$(python3 conf.py set "${default:-interesting}")
elif [ $# -eq 1 ] && ! [[ "$1" =~ ^[0-9]+$ ]]; then
  PRS=$(python3 conf.py set "$1")
else
  PRS="$*"
fi
echo "running PRs: $PRS"

for PR in $PRS; do
  echo "================= PR $PR ================="
  bash "$HERE/run_one.sh" "$PR" || echo "PR $PR failed, continuing"
done

python3 aggregate.py
echo 'run ./serve.sh then open http://localhost:8000/web/ (live reads results/all.json)'
