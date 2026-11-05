#!/usr/bin/env bash
# AUTO-GENERATED — PR #27793. Mimic the author's workload for BOTH arms.
# Mounts: /src = the arm's vllm tree (base or gold). Prints parseable lines.
set -euo pipefail
ARM="${ARM:-unknown}"
echo "# pr=27793 arm=$ARM cmd='pytest -v tests/v1/core/test_kv_cache_metrics.py'"
cd /src
export PYTHONPATH=/src:${PYTHONPATH:-}
pytest -v tests/v1/core/test_kv_cache_metrics.py
