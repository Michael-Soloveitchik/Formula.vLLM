#!/usr/bin/env bash
# AUTO-GENERATED — PR #27793 (GPU). Author workload: 'pytest -v tests/v1/core/test_kv_cache_metrics.py'
# Requires CUDA + (maybe) a model mounted at /weights. run.sh only calls this
# when a GPU is present; otherwise the row stays reported-only/pending.
set -euo pipefail
ARM="${ARM:-unknown}"
if ! python3 -c "import torch;assert torch.cuda.is_available()" 2>/dev/null; then
  echo "# pr=27793 arm=$ARM SKIP=no_cuda — GPU workload, run on a CUDA box"
  exit 0
fi
echo "# pr=27793 arm=$ARM cmd='pytest -v tests/v1/core/test_kv_cache_metrics.py'"
cd /src
export PYTHONPATH=/src:${PYTHONPATH:-}
pytest -v tests/v1/core/test_kv_cache_metrics.py
