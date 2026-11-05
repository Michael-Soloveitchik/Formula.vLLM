#!/usr/bin/env bash
# AUTO-GENERATED — PR #12583 (GPU). Author workload: 'python3 benchmarks/kernels/benchmark_moe.py --model mistralai/Mixtral-8x7B-Instruct-v0.1 --tp-size 2 --dtype auto --seed 0'
# Requires CUDA + (maybe) a model mounted at /weights. run.sh only calls this
# when a GPU is present; otherwise the row stays reported-only/pending.
set -euo pipefail
ARM="${ARM:-unknown}"
if ! python3 -c "import torch;assert torch.cuda.is_available()" 2>/dev/null; then
  echo "# pr=12583 arm=$ARM SKIP=no_cuda — GPU workload, run on a CUDA box"
  exit 0
fi
echo "# pr=12583 arm=$ARM cmd='python3 benchmarks/kernels/benchmark_moe.py --model mistralai/Mixtral-8x7B-Instruct-v0.1 --tp-size 2 --dtype auto --seed 0'"
cd /src
export PYTHONPATH=/src:${PYTHONPATH:-}
export VLLM_MODEL="/weights"  # mounted from .weights/mixtral
python3 benchmarks/kernels/benchmark_moe.py --model mistralai/Mixtral-8x7B-Instruct-v0.1 --tp-size 2 --dtype auto --seed 0
