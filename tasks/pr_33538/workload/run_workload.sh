#!/usr/bin/env bash
# AUTO-GENERATED — PR #33538 (GPU). Author workload: 'python benchmarks/benchmark_topk_topp.py'
# Requires CUDA + (maybe) a model mounted at /weights. run.sh only calls this
# when a GPU is present; otherwise the row stays reported-only/pending.
set -euo pipefail
ARM="${ARM:-unknown}"
if ! python3 -c "import torch;assert torch.cuda.is_available()" 2>/dev/null; then
  echo "# pr=33538 arm=$ARM SKIP=no_cuda — GPU workload, run on a CUDA box"
  exit 0
fi
echo "# pr=33538 arm=$ARM cmd='python benchmarks/benchmark_topk_topp.py'"
cd /src
export PYTHONPATH=/src:${PYTHONPATH:-}
python benchmarks/benchmark_topk_topp.py
