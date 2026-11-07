#!/usr/bin/env bash
# Sequential reproduce queue: each line of QUEUE is one ./run.sh --reproduce invocation, run one after another.
# Usage: nohup ./queue.sh "6442" "21222 21281 6779" "--next 16" "13365 29499 22940" > .work/queue.log 2>&1 &
# A line is PR numbers or --next N / --retry-failed. No polling, no pgrep: run.sh's flock already serialises.
# Per-job logs: .work/reproduce_<job>_<stamp>.log
HERE="$(cd "$(dirname "$0")" && pwd)"; ROOT="$(cd "$HERE/.." && pwd)"; cd "$ROOT"
mkdir -p .work
for job in "$@"; do
  LOG=".work/reproduce_$(echo "$job" | tr -c 'A-Za-z0-9' '_')_$(date +%Y%m%d_%H%M).log"
  echo "== $(date +%H:%M) start: ./run.sh --reproduce $job  -> $LOG"
  CLEAN_IMAGES=1 bash "$HERE/run.sh" --reproduce $job >"$LOG" 2>&1; rc=$?
  echo "== $(date +%H:%M) done rc=$rc: $job"; grep -E "^SKIP|base=|failed rc|timed out" "$LOG" | tail -12 | cut -c1-110
done
echo "== QUEUE FINISHED $(date +%H:%M)"; python3 "$HERE/next_prs.py" --status
