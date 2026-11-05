#!/usr/bin/env bash
# AUTO-GENERATED — PR #37225 (GPU). Author workload: 'python benchmarks/benchmark_topk_topp.py --batch-sizes 128 2048 --vocab-sizes 32768 131072 --warmup-iters 5 --benchmark-iters 20'
# Requires CUDA + (maybe) a model mounted at /weights. run.sh only calls this
# when a GPU is present; otherwise the row stays reported-only/pending.
set -euo pipefail
ARM="${ARM:-unknown}"
if ! python3 -c "import torch;assert torch.cuda.is_available()" 2>/dev/null; then
  echo "# pr=37225 arm=$ARM SKIP=no_cuda — GPU workload, run on a CUDA box"
  exit 0
fi
echo "# pr=37225 arm=$ARM cmd='python benchmarks/benchmark_topk_topp.py --batch-sizes 128 2048 --vocab-sizes 32768 131072 --warmup-iters 5 --benchmark-iters 20'"
cd /src
export PYTHONPATH=/src:${PYTHONPATH:-}
python benchmarks/benchmark_topk_topp.py --batch-sizes 128 2048 --vocab-sizes 32768 131072 --warmup-iters 5 --benchmark-iters 20
