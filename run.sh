#!/usr/bin/env bash
# Run the vLLM perf benchmark. Self-contained: clone this dir on the GPU box,
# set VLLM_GIT (or paths.vllm_git in config.yaml), then:
#   ./run.sh                 # default set from config.yaml (interesting)
#   ./run.sh all             # all 202 PRs
#   ./run.sh 21222 21627     # specific PRs
#   ./run.sh interesting     # a named set from config.yaml
#   ./run.sh workloads       # run per-PR workload/ images by cluster (GPU box)
#   ./run.sh --reproduce <pr>...     # ONLY baseline vs PR's solution on this machine:
#                                    # no claude/codex, no API key, no tokens. Clones vLLM +
#                                    # fetches the exact commits itself. Each PR's requirements
#                                    # (index.json[pr].requirements: GPU/VRAM/compute cap, docker,
#                                    # weights, disk...) are checked first; no-sudo software is
#                                    # installed, unmet hardware -> PR skipped with the reason.
#                                    # Prints the reported-vs-measured table at the end.
#                                    # CLEAN_IMAGES=1 ./run.sh --reproduce ...  removes each PR's
#                                    # image after it is measured (disk: 13-25 GB per image).
#   ./run.sh --reproduce --next 10   # AUTONOMIC: next 10 not-yet-measured PRs from workloads.json:
#                                    # vendor task (gen_tasks/gen_workloads/gen_requirements) -> requirements
#                                    # (+auto-install) -> run -> dump (record/derive) -> report. Unmet PRs are
#                                    # remembered in .work/unmet_<pr>.txt; --retry-failed reruns 0-measured ones.
#                                    # status: python3 next_prs.py --status
#   ./run.sh --reproduce --health-check <pr>...
#                                    # same checks, then start the real run for HEALTH_SECS
#                                    # (default 180) per PR, watch the log, stop it, verdict.
#                                    # Nothing recorded. Checklist only: python3 requirements.py <pr>
#   ./run.sh --run-formula-harnesses <pr>...
#                                    # the claude/codex solver arms (config.yaml:solvers) on PRs
#                                    # already reproduced (reuses .work/pr_<n>, KEEP_WORK=1).
#                                    # Needs the API keys. NEVER part of --reproduce.
#   ./run.sh --run-formula-harnesses --health-check <pr>...
#                                    # solver CLIs on PATH + their env vars + refs reproduced.
#                                    # Invokes NO solver (no tokens); different from the
#                                    # --reproduce health check, which starts the real run.
#
# Per PR: build image at base_sha, run REAL test_patch for baseline (FAIL) +
# PR's solution (PASS) + each enabled solver, measure latency with repeats->variance,
# dump results/<pr>.json (the HTML live-reads these).
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"; cd "$HERE"
mkdir -p .work results

# flags: --reproduce (refs only, no solvers) | --health-check (with --reproduce) |
#        --run-formula-harnesses (solver arms only, on already-reproduced PRs)
REPRODUCE=0; HEALTH=0; HARNESSES=0; NEXT=""; RETRY=""; ARGS=()
PREV=""
for a in "$@"; do
  case "$a" in
    --reproduce) REPRODUCE=1 ;;
    --next) NEXT="10" ;;
    --retry-failed|--retry-unmet) RETRY="$a" ;;
    --health-check) HEALTH=1 ;;
    --run-formula-harnesses) HARNESSES=1 ;;
    *) if [ "$PREV" = "--next" ] && [[ "$a" =~ ^[0-9]+$ ]]; then NEXT="$a"; else ARGS+=("$a"); fi ;;
  esac
  PREV="$a"
done
set -- ${ARGS[@]+"${ARGS[@]}"}
if [ "$REPRODUCE" = 1 ] && [ "$HARNESSES" = 1 ]; then echo "--reproduce and --run-formula-harnesses are separate runs; pick one" >&2; exit 2; fi
if [ "$HEALTH" = 1 ] && [ "$REPRODUCE" != 1 ] && [ "$HARNESSES" != 1 ]; then echo "--health-check goes with --reproduce or --run-formula-harnesses" >&2; exit 2; fi
if [ "$REPRODUCE" = 1 ]; then export RUN_SOLVERS=0 FETCH_SOURCES=1 CLEAN_IMAGE="${CLEAN_IMAGES:-0}"; echo "REPRODUCE: baseline vs PR's solution only, solver arms disabled (CLEAN_IMAGES=${CLEAN_IMAGES:-0})"; fi
if [ "$HARNESSES" = 1 ]; then
  [ $# -gt 0 ] || { echo "usage: ./run.sh --run-formula-harnesses <pr> [<pr> ...]" >&2; exit 2; }
  export RUN_SOLVERS=1 KEEP_WORK=1
  echo "FORMULA HARNESSES: solver arms ($(python3 conf.py solvers | cut -f1 | paste -sd, -)) on already-reproduced PRs"
  if [ "$HEALTH" = 1 ]; then
    # harness health = solver CLIs + their API env vars + reproduced refs per PR. No solver is invoked
    # (that would spend API tokens); a run only starts when all of this is green.
    bad=0
    while IFS=$'\t' read -r HN MODEL CMD GUIDED; do
      [ -n "${HN:-}" ] || continue
      bin=$(printf '%s' "$CMD" | awk '{print $1}')
      command -v "$bin" >/dev/null && echo "  [ok] solver '$HN': $bin found ($(command -v "$bin"))" || { echo "  [NO] solver '$HN': '$bin' not on PATH"; bad=1; }
      for v in $(python3 - "$HN" <<'PY'
import sys, conf
for s in conf.load().get("solvers", []):
    if s.get("harness") == sys.argv[1]:
        print(" ".join(s.get("env", []) if isinstance(s.get("env"), list) else []))
PY
); do [ -n "${!v:-}" ] && echo "  [ok]   env $v set" || { echo "  [NO]   env $v missing (API key/endpoint for '$HN')"; bad=1; }; done
    done < <(python3 conf.py solvers)
    for PR in "$@"; do
      [ -f ".work/pr_${PR}/arms.json" ] && echo "  [ok] PR $PR reproduced (.work/pr_${PR}/arms.json)" || { echo "  [NO] PR $PR not reproduced yet -> ./run.sh --reproduce $PR"; bad=1; }
    done
    [ "$bad" = 0 ] && echo "HEALTH harnesses OK -- ready: ./run.sh --run-formula-harnesses $*" || echo "HEALTH harnesses FAIL -- fix the [NO] lines above"
    exit $bad
  fi
  for PR in "$@"; do
    [ -f ".work/pr_${PR}/arms.json" ] || { echo "SKIP $PR: not reproduced yet (no .work/pr_${PR}/arms.json) -> run ./run.sh --reproduce $PR first"; continue; }
    echo "================= PR $PR (harnesses) ================="
    bash "$HERE/run_one.sh" "$PR" || echo "PR $PR failed, continuing"
  done
  python3 aggregate.py; exit 0
fi

# workloads mode: iterate clusters in order, download weights once per cluster,
# build+run each per-PR workload/ image for both arms. Delegated to run_workloads.sh.
if [ "${1:-}" = "workloads" ]; then
  shift
  exec bash "$HERE/run_workloads.sh" "$@"
fi

if [ $# -eq 0 ] && [ "$REPRODUCE" != 1 ]; then
  default=$(python3 conf.py get run.default)
  PRS=$(python3 conf.py set "${default:-interesting}")
elif [ $# -eq 1 ] && [ "$REPRODUCE" != 1 ] && ! [[ "$1" =~ ^[0-9]+$ ]]; then
  PRS=$(python3 conf.py set "$1")
else
  PRS="$*"
fi
# reproduce mode: explicit PR numbers only; check each PR's requirements first
if [ "$REPRODUCE" = 1 ]; then
  if [ -n "$NEXT" ] || [ -n "$RETRY" ]; then
    set -- $(python3 next_prs.py ${RETRY:---next "$NEXT"})
    echo "AUTONOMIC pick (${RETRY:---next $NEXT}): ${*:-nothing to do}"; [ $# -gt 0 ] || exit 0
    # vendor whatever is missing for these PRs, then refresh requirements
    python3 gen_tasks.py "$@" | grep -v "exists, skip" || true
    python3 gen_workloads.py >/dev/null && python3 gen_requirements.py >/dev/null
    PRS="$*"
  fi
  [ $# -gt 0 ] || { echo "usage: ./run.sh --reproduce <pr> [<pr> ...]   (or --next N)" >&2; exit 2; }
  for a in "$@"; do [[ "$a" =~ ^[0-9]+$ ]] || { echo "--reproduce takes PR numbers only (got '$a')" >&2; exit 2; }; done
  if [ "$HEALTH" = 1 ]; then
    for PR in $PRS; do bash "$HERE/health_check.sh" "$PR" "${HEALTH_SECS:-180}" | tee ".work/health_$PR.out" | tail -n 1; done
    echo "---- health summary ----"; for PR in $PRS; do tail -n 1 ".work/health_$PR.out"; done
    exit 0
  fi
  KEEP=""; SKIPPED=""
  for PR in $PRS; do
    if python3 requirements.py --quiet --install "$PR" >.work/req_$PR.txt; then KEEP="$KEEP $PR"; rm -f ".work/unmet_$PR.txt"
    else echo "SKIP $PR - cannot reproduce on this machine:"; sed 's/^/    /' .work/req_$PR.txt; SKIPPED="$SKIPPED $PR"; cp ".work/req_$PR.txt" ".work/unmet_$PR.txt"; fi
  done
  PRS="${KEEP# }"
  [ -n "$SKIPPED" ] && echo "skipped (unmet requirements):$SKIPPED"
fi
echo "running PRs: $PRS"

for PR in $PRS; do
  echo "================= PR $PR ================="
  bash "$HERE/run_one.sh" "$PR" || echo "PR $PR failed, continuing"
done

python3 aggregate.py
if [ "$REPRODUCE" = 1 ]; then python3 reproduce_report.py $PRS; echo "---- status of all workload PRs ----"; python3 next_prs.py --status; fi
echo 'run ./serve.sh then open http://localhost:8000/web/ (live reads results/all.json)'
