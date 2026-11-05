#!/usr/bin/env bash
# AUTO-GENERATED — PR #32975. Mimic the author's workload for BOTH arms.
# Mounts: /src = the arm's vllm tree (base or gold). Prints parseable lines.
set -euo pipefail
ARM="${ARM:-unknown}"
echo "# pr=32975 arm=$ARM cmd='python3 tasks/pr_32975/bench.py'"
cd /src
export PYTHONPATH=/src:${PYTHONPATH:-}
python3 tasks/pr_32975/bench.py
