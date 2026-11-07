# layout.sh — source this from a bash script; thin wrapper over gen/layout.py (the single owner of paths).
#   . "$ROOT/gen/layout.sh"
#   layout_env 37225          # exports PR_DIR TASK_JSON PROBLEM_MD WORK_DIR RESULTS_JSON LOGS_DIR
#                             #         WORKLOAD_DIR WORKLOAD_DEFAULT_DIR HW_SLUG WORK_ROOT RUN_LOCK AGG_DIR
#   lay results_json 37225    # any layout.py function -> one line on stdout
#   lay task_get 37225 workload.cmd
LAYOUT_PY="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/layout.py"
ROOT="$(cd "$(dirname "$LAYOUT_PY")/.." && pwd)"
lay() { python3 "$LAYOUT_PY" "$@"; }
layout_env() { eval "$(python3 "$LAYOUT_PY" env "$@")"; }
