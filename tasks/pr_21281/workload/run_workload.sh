#!/usr/bin/env bash
# AUTO-GENERATED — PR #21281. Mimic the author's workload for BOTH arms.
# Mounts: /src = the arm's vllm tree (base or gold). Prints parseable lines.
set -euo pipefail
ARM="${ARM:-unknown}"
echo "# pr=21281 arm=$ARM cmd='python3 benchmarks/kv_cache/benchmark_block_pool.py --num-gpu-blocks 100000 --num-iteration 1000 --allocate-blocks 10 50 100 500 1000'"
cd /src
export PYTHONPATH=/src:${PYTHONPATH:-}
python3 benchmarks/kv_cache/benchmark_block_pool.py --num-gpu-blocks 100000 --num-iteration 1000 --allocate-blocks 10 50 100 500 1000
