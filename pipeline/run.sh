#!/usr/bin/env bash
# Run the vLLM perf benchmark. Self-contained: clone this dir on the GPU box (docker + NVIDIA
# container toolkit), then:
#   ./run.sh                 # default set from config.yaml (interesting)
#   ./run.sh all             # all 202 PRs
#   ./run.sh 21222 21627     # specific PRs
#   ./run.sh interesting     # a named set from config.yaml
#   ./run.sh workloads       # run per-PR workload/ images by cluster (GPU box)
#   ./run.sh --reproduce <pr>...     # ONLY baseline vs PR's solution on this machine:
#                                    # no claude/codex, no API key, no tokens. The per-PR image
#                                    # fetches the exact commits itself. Each PR's requirements
#                                    # (data/pr_<n>/task.json:requirements: GPU/VRAM/compute cap, docker,
#                                    # weights, disk...) are checked first; no-sudo software is
#                                    # installed, unmet hardware -> PR skipped with the reason.
#                                    # Prints the reported-vs-measured table at the end.
#                                    # CLEAN_IMAGES=1 ./run.sh --reproduce ...  removes each PR's
#                                    # image after it is measured (disk: 13-25 GB per image).
#   ./run.sh --reproduce --next 10   # AUTONOMIC: next 10 not-yet-measured PRs (data/pr_*/task.json with a
#                                    # workload): vendor task (gen_tasks/gen_workloads/gen_requirements) ->
#                                    # requirements (+auto-install) -> run -> dump (record/derive) -> report.
#                                    # Unmet PRs are remembered in data/pr_<n>/work/unmet.txt;
#                                    # --retry-failed reruns 0-measured ones. status: python3 pipeline/next_prs.py --status
#   ./run.sh --reproduce --health-check <pr>...
#                                    # same checks, then start the real run for HEALTH_SECS
#                                    # (default 180) per PR, watch the log, stop it, verdict.
#                                    # Nothing recorded. Checklist only: python3 pipeline/requirements.py <pr>
#   ./run.sh --run-formula-harnesses <pr>...
#                                    # the claude/codex solver arms (config.yaml:solvers) on PRs
#                                    # already reproduced (reuses data/pr_<n>/work, KEEP_WORK=1).
#                                    # Needs the API keys. NEVER part of --reproduce.
#   ./run.sh --run-formula-harnesses --health-check <pr>...
#                                    # solver CLIs on PATH + their env vars + refs reproduced.
#                                    # Invokes NO solver (no tokens); different from the
#                                    # --reproduce health check, which starts the real run.
#
# Per PR: build the per-PR image, run REAL test_patch for baseline (FAIL) +
# PR's solution (PASS) + each enabled solver, measure latency with repeats->variance,
# dump data/pr_<n>/results.<hw-slug>.json (the HTML live-reads these via results/manifest.json).
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"; ROOT="$(cd "$HERE/.." && pwd)"; cd "$ROOT"
. "$ROOT/gen/layout.sh"
WORK_ROOT="$(lay WORK_ROOT 2>/dev/null || echo "$ROOT/.work")"; WORK_ROOT="$ROOT/.work"
mkdir -p "$WORK_ROOT" results
# one benchmark run at a time on this machine (two PRs sharing the GPU = noisy numbers): queued callers block here.
# flock is Linux-only; on macOS (no flock) the lock is skipped — fine for CPU apply/dev runs.
if command -v flock >/dev/null 2>&1; then exec 9>"$(lay run_lock)"; flock 9; fi

# flags: --reproduce (refs only, no solvers) | --health-check (with --reproduce) |
#        --run-formula-harnesses (solver arms only, on already-reproduced PRs)
REPRODUCE=0; HEALTH=0; HARNESSES=0; APPLY_AI=0; NEXT=""; RETRY=""; ARGS=()
PREV=""
for a in "$@"; do
  case "$a" in
    --reproduce) REPRODUCE=1 ;;
    --next) NEXT="10" ;;
    --retry-failed|--retry-unmet) RETRY="$a" ;;
    --health-check) HEALTH=1 ;;
    --run-formula-harnesses) HARNESSES=1 ;;
    --apply-ai-generated-solutions) APPLY_AI=1 ;;
    *) if [ "$PREV" = "--next" ] && [[ "$a" =~ ^[0-9]+$ ]]; then NEXT="$a"; else ARGS+=("$a"); fi ;;
  esac
  PREV="$a"
done
set -- ${ARGS[@]+"${ARGS[@]}"}
if [ "$REPRODUCE" = 1 ] && [ "$HARNESSES" = 1 ]; then echo "--reproduce and --run-formula-harnesses are separate runs; pick one" >&2; exit 2; fi
if [ "$HEALTH" = 1 ] && [ "$REPRODUCE" != 1 ] && [ "$HARNESSES" != 1 ]; then echo "--health-check goes with --reproduce or --run-formula-harnesses" >&2; exit 2; fi
if [ "$REPRODUCE" = 1 ]; then export RUN_SOLVERS=0 CLEAN_IMAGE="${CLEAN_IMAGES:-0}"; echo "REPRODUCE: baseline vs PR's solution only, solver arms disabled (CLEAN_IMAGES=${CLEAN_IMAGES:-0}, hw=$(lay hw_slug))"; fi
# apply-ai mode: take the saved data/pr_<n>/files/<arm>/ solutions, overlay them on the
# baseline tree and re-measure on CPU with the SAME instrument (bench.py + parse_workload.py).
# No GPU, no solver CLI, no API key: just applies what is already in files/ and times it.
if [ "$APPLY_AI" = 1 ]; then
  [ $# -gt 0 ] || { echo "usage: ./run.sh --apply-ai-generated-solutions <pr> [<pr> ...] [--arm <name>]" >&2; exit 2; }
  echo "APPLY AI SOLUTIONS: overlay data/pr_<n>/files/<arm>/ on baseline, re-measure on CPU (hw=$(lay hw_slug))"
  python3 "$HERE/apply_ai_solutions.py" "$@"
  exit $?
fi

if [ "$HARNESSES" = 1 ]; then
  [ $# -gt 0 ] || { echo "usage: ./run.sh --run-formula-harnesses <pr> [<pr> ...]" >&2; exit 2; }
  export RUN_SOLVERS=1 KEEP_WORK=1
  echo "FORMULA HARNESSES: solver arms ($(python3 gen/conf.py solvers | cut -f1 | paste -sd, -)) on already-reproduced PRs"
  if [ "$HEALTH" = 1 ]; then
    # harness health = solver CLIs + their API env vars + reproduced refs per PR. No solver is invoked
    # (that would spend API tokens); a run only starts when all of this is green.
    bad=0
    while IFS=$'\t' read -r HN MODEL CMD GUIDED; do
      [ -n "${HN:-}" ] || continue
      bin=$(printf '%s' "$CMD" | awk '{print $1}')
      command -v "$bin" >/dev/null && echo "  [ok] solver '$HN': $bin found ($(command -v "$bin"))" || { echo "  [NO] solver '$HN': '$bin' not on PATH"; bad=1; }
      for v in $(python3 - "$HN" <<'PY'
import sys, os
sys.path.insert(0, os.getcwd())
from gen import conf
for s in conf.load().get("solvers", []):
    if s.get("harness") == sys.argv[1]:
        print(" ".join(s.get("env", []) if isinstance(s.get("env"), list) else []))
PY
); do [ -n "${!v:-}" ] && echo "  [ok]   env $v set" || { echo "  [NO]   env $v missing (API key/endpoint for '$HN')"; bad=1; }; done
    done < <(python3 gen/conf.py solvers)
    for PR in "$@"; do
      A="$(lay work_file "$PR" arms.json)"
      [ -f "$A" ] && echo "  [ok] PR $PR reproduced ($A)" || { echo "  [NO] PR $PR not reproduced yet -> ./run.sh --reproduce $PR"; bad=1; }
    done
    [ "$bad" = 0 ] && echo "HEALTH harnesses OK -- ready: ./run.sh --run-formula-harnesses $*" || echo "HEALTH harnesses FAIL -- fix the [NO] lines above"
    exit $bad
  fi
  for PR in "$@"; do
    A="$(lay work_file "$PR" arms.json)"
    [ -f "$A" ] || { echo "SKIP $PR: not reproduced yet (no $A) -> run ./run.sh --reproduce $PR first"; continue; }
    echo "================= PR $PR (harnesses) ================="
    bash "$HERE/run_one.sh" "$PR" || echo "PR $PR failed, continuing"
  done
  python3 "$HERE/aggregate.py"; exit 0
fi

# workloads mode: iterate clusters in order, download weights once per cluster,
# build+run each per-PR workload/ image for both arms. Delegated to run_workloads.sh.
if [ "${1:-}" = "workloads" ]; then
  shift
  exec bash "$HERE/run_workloads.sh" "$@"
fi

if [ $# -eq 0 ] && [ "$REPRODUCE" != 1 ]; then
  default=$(python3 gen/conf.py get run.default)
  PRS=$(python3 gen/conf.py set "${default:-interesting}")
elif [ $# -eq 1 ] && [ "$REPRODUCE" != 1 ] && ! [[ "$1" =~ ^[0-9]+$ ]]; then
  PRS=$(python3 gen/conf.py set "$1")
else
  PRS="$*"
fi
# reproduce mode: explicit PR numbers only; check each PR's requirements first
if [ "$REPRODUCE" = 1 ]; then
  if [ -n "$NEXT" ] || [ -n "$RETRY" ]; then
    set -- $(python3 "$HERE/next_prs.py" ${RETRY:---next "$NEXT"})
    echo "AUTONOMIC pick (${RETRY:---next $NEXT}): ${*:-nothing to do}"; [ $# -gt 0 ] || exit 0
    # vendor whatever is missing for these PRs, then refresh workload dirs + requirements
    python3 gen/gen_tasks.py "$@" | grep -v "already set, skip" || true
    python3 gen/gen_workloads.py "$@" >/dev/null && python3 gen/gen_requirements.py >/dev/null
    PRS="$*"
  fi
  [ $# -gt 0 ] || { echo "usage: ./run.sh --reproduce <pr> [<pr> ...]   (or --next N)" >&2; exit 2; }
  for a in "$@"; do [[ "$a" =~ ^[0-9]+$ ]] || { echo "--reproduce takes PR numbers only (got '$a')" >&2; exit 2; }; done
  if [ "$HEALTH" = 1 ]; then
    for PR in $PRS; do mkdir -p "$(lay work_dir "$PR")"; bash "$HERE/health_check.sh" "$PR" "${HEALTH_SECS:-180}" | tee "$(lay work_file "$PR" health.out)" | tail -n 1; done
    echo "---- health summary ----"; for PR in $PRS; do tail -n 1 "$(lay work_file "$PR" health.out)"; done
    exit 0
  fi
  KEEP=""; SKIPPED=""
  for PR in $PRS; do
    mkdir -p "$(lay work_dir "$PR")"; REQ="$(lay work_file "$PR" req.txt)"; UNMET="$(lay work_file "$PR" unmet.txt)"
    if python3 "$HERE/requirements.py" --quiet --install "$PR" >"$REQ"; then KEEP="$KEEP $PR"; rm -f "$UNMET"
    else echo "SKIP $PR - cannot reproduce on this machine:"; sed 's/^/    /' "$REQ"; SKIPPED="$SKIPPED $PR"; cp "$REQ" "$UNMET"; fi
  done
  PRS="${KEEP# }"
  [ -n "$SKIPPED" ] && echo "skipped (unmet requirements):$SKIPPED"
fi
echo "running PRs: $PRS"

for PR in $PRS; do
  echo "================= PR $PR ================="
  bash "$HERE/run_one.sh" "$PR" || echo "PR $PR failed, continuing"
done

python3 "$HERE/aggregate.py"
if [ "$REPRODUCE" = 1 ]; then
  python3 gen/gen_workloads.py $PRS >/dev/null 2>&1 || true     # refresh the "Reproduced here" section of WORKLOAD.md
  python3 stats/build_stats.py >/dev/null 2>&1 || echo "stats/build_stats.py failed (numbers are still in data/pr_*/results.*.json)"
  python3 "$HERE/reproduce_report.py" $PRS; echo "---- status of all workload PRs ----"; python3 "$HERE/next_prs.py" --status
fi
echo 'run ./serve.sh then open http://localhost:8000/web/ (live reads results/manifest.json + data/pr_<n>/results.<hw>.json)'
