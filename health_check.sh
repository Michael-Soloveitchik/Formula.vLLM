#!/usr/bin/env bash
# Health check ONE pr: requirements -> start the real reproduce run -> watch its log
# for HEALTH_SECS -> stop it -> verdict. Nothing is recorded into results/.
#   ./health_check.sh <pr> [secs]           (default 180s)
# Verdict (last line, machine-readable):  HEALTH <pr> OK|FAIL|UNMET <stage reached> -- <why>
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"; cd "$HERE"
PR="${1:?usage: health_check.sh <pr> [secs]}"; SECS="${2:-180}"
mkdir -p ".work" "logs/pr_$PR"; LOG="logs/pr_$PR/health_$(date +%Y%m%d_%H%M%S).log"
say(){ echo "[health $PR] $*"; }

if ! python3 requirements.py --quiet --install "$PR" >"$LOG.req" 2>&1; then
  say "requirements unmet:"; sed 's/^/    /' "$LOG.req"
  echo "HEALTH $PR UNMET requirements -- $(grep -c NO "$LOG.req") unmet"; exit 2
fi
say "requirements ok; starting run for ${SECS}s (log: $LOG)"

# own process group so the whole pipeline (bash + docker clients) can be stopped at once
RUN_SOLVERS=0 FETCH_SOURCES=1 REPEATS=1 setsid bash run_one.sh "$PR" >"$LOG" 2>&1 &
PID=$!
stage="start"; why=""; verdict=""
for ((t=0; t<SECS; t+=5)); do
  sleep 5
  # stage = furthest pipeline step seen in the log
  grep -q "sources ready\|have [0-9a-f]\{40\}" "$LOG" && stage="fetch"
  grep -q "building vllm-task" "$LOG" && stage="build"
  grep -q "^\[$PR\] tests:" "$LOG" && stage="tests"
  grep -q "workload base run" "$LOG" && stage="workload:baseline"
  grep -q "workload gold run" "$LOG" && stage="workload:pr-solution"
  grep -q "done -> results" "$LOG" && { stage="done"; verdict=OK; why="full run finished"; break; }
  if grep -qiE "ERROR:|error: failed to solve|No such file|command not found|CUDA error|OutOfMemory|out of memory|Traceback \(most recent" "$LOG"; then
    verdict=FAIL; why="$(grep -iE -m1 'ERROR:|error: failed|No such file|command not found|CUDA error|OutOfMemory|out of memory|Traceback' "$LOG" | cut -c1-160)"; break
  fi
  if ! kill -0 "$PID" 2>/dev/null; then
    wait "$PID"; rc=$?
    [ "$rc" = 0 ] && { verdict=OK; why="run_one exited 0"; } || { verdict=FAIL; why="run_one exited $rc: $(tail -n1 "$LOG" | cut -c1-160)"; }
    break
  fi
done
[ -n "$verdict" ] || { verdict=OK; why="no error in ${SECS}s, reached stage '$stage'"; }

# stop: the process group, then any container/build of this PR's image
kill -TERM -- "-$PID" 2>/dev/null; sleep 2; kill -KILL -- "-$PID" 2>/dev/null
docker ps -q --filter "ancestor=vllm-task:$PR" | xargs -r docker kill >/dev/null 2>&1
# a run interrupted mid-way must not leave a half-written ledger behind
[ "$stage" = done ] || { rm -f ".work/pr_$PR/arms.json"; }
say "stopped; last log lines:"; tail -n 5 "$LOG" | sed 's/^/    /'
echo "HEALTH $PR $verdict $stage -- $why"
[ "$verdict" = OK ]
