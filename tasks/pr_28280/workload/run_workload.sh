#!/usr/bin/env bash
# AUTO-GENERATED — PR #28280 (GPU). Author workload: 'vllm bench serve --num-prompts 100 --dataset-name sharegpt'
# Requires CUDA + (maybe) a model mounted at /weights. run.sh only calls this
# when a GPU is present; otherwise the row stays reported-only/pending.
set -euo pipefail
ARM="${ARM:-unknown}"
if ! python3 -c "import torch;assert torch.cuda.is_available()" 2>/dev/null; then
  echo "# pr=28280 arm=$ARM SKIP=no_cuda — GPU workload, run on a CUDA box"
  exit 0
fi
echo "# pr=28280 arm=$ARM cmd='vllm bench serve --num-prompts 100 --dataset-name sharegpt'"
cd /src
export PYTHONPATH=/src:${PYTHONPATH:-}
export VLLM_MODEL="/weights"  # mounted from .weights/qwen05b
vllm bench serve --num-prompts 100 --dataset-name sharegpt
