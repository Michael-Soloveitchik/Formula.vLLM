#!/usr/bin/env bash
# AUTO-GENERATED — PR #22940 (GPU). Author workload: 'VLLM_USE_V1=1 python3 benchmarks/benchmark_throughput.py --model facebook/opt-125m --backend vllm --input-len 800 --output-len 75 --num-prompts 30000'
# Requires CUDA + (maybe) a model mounted at /weights. run.sh only calls this
# when a GPU is present; otherwise the row stays reported-only/pending.
set -euo pipefail
ARM="${ARM:-unknown}"
if ! python3 -c "import torch;assert torch.cuda.is_available()" 2>/dev/null; then
  echo "# pr=22940 arm=$ARM SKIP=no_cuda — GPU workload, run on a CUDA box"
  exit 0
fi
echo "# pr=22940 arm=$ARM cmd='VLLM_USE_V1=1 python3 benchmarks/benchmark_throughput.py --model facebook/opt-125m --backend vllm --input-len 800 --output-len 75 --num-prompts 30000'"
cd /src
export PYTHONPATH=/src:${PYTHONPATH:-}
export VLLM_MODEL="/weights"  # mounted from .weights/opt125m
VLLM_USE_V1=1 python3 benchmarks/benchmark_throughput.py --model facebook/opt-125m --backend vllm --input-len 800 --output-len 75 --num-prompts 30000
