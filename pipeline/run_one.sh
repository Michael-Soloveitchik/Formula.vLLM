#!/usr/bin/env bash
# Build + verify + measure ONE pr, self-contained. Dumps data/pr_<n>/results.<hw-slug>.json.
#   ./run_one.sh <pr>
#   DRY_RUN=1 ./run_one.sh <pr>     # validate every step, no docker needed
#
# Image: ONE per PR, data/pr_<n>/workload/<hw-slug>/Dockerfile (falls back to workload/default/),
# rendered by gen/gen_workloads.py from templates/Dockerfile.gpu-wheel + task.json (wheel, shas,
# torch_fix, extra_pip baked in; task.json:hardware_overrides[<hw-slug>] applied). The image
# installs the PyPI wheel named by task.json:wheel AND clones/checks out the PR's base_sha +
# merge_sha INSIDE the image (/base, /gold). Nothing vLLM-related is needed on the host.
# A task without `wheel` is refused (python3 gen/gen_tasks.py <pr> vendors it).
#
# Pipeline (all in code):
#   1. checkout  : export baseline + PR's-solution trees from the image
#   2. build     : one image per task (per-PR Dockerfile regenerated first so it matches task.json)
#   3. isolate   : each arm = a .py-tree swap INSIDE that same image, so the
#                  environment is byte-identical across arms (FormulaCode-style)
#   4. run       : the hidden pytest oracle — baseline must FAIL, PR's solution must PASS
#   5. solve     : each enabled solver edits a baseline tree, is re-verified
#   6. measure   : run the task's workload cmd per arm, `repeats` times, inside
#                  the same image -> ms per scenario, mean ± sd
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"; ROOT="$(cd "$HERE/.." && pwd)"; cd "$ROOT"
. "$ROOT/gen/layout.sh"
PR="${1:?usage: run_one.sh <pr>}"
DRY_RUN="${DRY_RUN:-0}"
layout_env "$PR"    # PR_DIR TASK_JSON PROBLEM_MD WORK_DIR RESULTS_JSON LOGS_DIR WORKLOAD_DIR WORKLOAD_DEFAULT_DIR HW_SLUG ...

ENGINE="${CONTAINER_ENGINE:-docker}"
GPU="${GPU_FLAG:---gpus device=0}"      # pin one GPU; CUDA_VISIBLE_DEVICES=0 inside (BASELINE_GAP.md)
WORK="$WORK_DIR"
TASK="$PR_DIR"
REPEATS="${REPEATS:-$(python3 gen/conf.py get measure.repeats)}"; REPEATS="${REPEATS:-5}"
# a task may pin its own repeat count (task.json "repeats", or hardware_overrides[<hw>].repeats, e.g. 1 for a 16-minute throughput run)
SOLVER_TIMEOUT="${SOLVER_TIMEOUT:-15m}"   # hard cap per solver arm (claude -p / codex exec)
KEEP_WORK="${KEEP_WORK:-0}"             # 1 = reuse existing work/ refs (baseline / PR-solution logs), run solvers only
# RUN_SOLVERS=0 -> measure baseline / PR solution only (fast numbers first); run solvers later with KEEP_WORK=1

say(){ echo "[$PR] $*"; }
die(){ echo "[$PR] ERROR: $*" >&2; exit 1; }

# ---- task.json (hardware override of THIS slug applied): shas, tests, wheel, workload ----
[ -f "$TASK_JSON" ] || die "no $TASK_JSON (pr $PR is not in data/)"
meta(){ lay task_get "$PR" "$1" "$HW_SLUG"; }
BASE="$(meta base_sha)"; MERGE="$(meta merge_sha)"
[ -n "$BASE" ] && [ -n "$MERGE" ] || die "missing baseline/PR commit sha in $TASK_JSON"
TESTS="$(meta test_files)"
WHEEL="$(meta wheel)"
SWAP="$(meta swap)"; SWAP="${SWAP:-tree}"      # tree = whole vllm/ .py tree per arm; files = only source_files
SRC_FILES="$(meta source_files)"
WL_CMD="$(meta workload.cmd)"
TASK_REPEATS="$(meta repeats)"; [ -n "$TASK_REPEATS" ] && [ -z "${REPEATS_FORCED:-}" ] && REPEATS="$TASK_REPEATS"
OBJ="$(cat "$PROBLEM_MD" 2>/dev/null || echo "Optimize PR #$PR; make the hidden tests pass.")"
IMG="vllm-task:${PR}"
[ -n "$WHEEL" ] || die "task.json has no 'wheel' (vendor it: python3 gen/gen_tasks.py $PR); run_one.sh builds wheel-mode images only"
DF="$WORKLOAD_DIR/Dockerfile"; CTX="$WORKLOAD_DIR"

# ---- preflight: fail early with a clear message -----------------------------
preflight() {
  "$ENGINE" info >/dev/null 2>&1 || die "container engine '$ENGINE' not available/running"
}

# =============================== DRY RUN =====================================
if [ "$DRY_RUN" = "1" ]; then
  say "DRY_RUN — validating pipeline (no docker) hw=$HW_SLUG wheel=$WHEEL"
  say "0. generate : python3 gen/gen_workloads.py $PR  (-> $DF from templates/Dockerfile.gpu-wheel + task.json)"
  [ -f "$DF" ] || say "   WARN: $DF not generated yet (gen_workloads.py writes it; the real run does that first)"
  say "2. build    : $ENGINE build -t $IMG -f $DF $CTX   (wheel $WHEEL, BASE_SHA=$BASE, MERGE_SHA=$MERGE baked in)"
  say "1. checkout : $ENGINE cp <container>:/base -> $WORK/base_full ; /gold -> $WORK/gold_full"
  say "3. isolate  : $ENGINE run --rm $GPU -v <tree>:/arm:ro -v <pr_solution_tree>:/gold:ro $IMG  (wipe site-packages/vllm/*.py, cp /arm/vllm -> site-packages, tests+benchmarks copied to /t)"
  say "4. run      : python3 -m pytest -q ${TESTS:-<none>}  (baseline expect FAIL, PR's solution expect PASS)"
  [ -n "$TESTS" ] || say "   WARN: no test_files in $TASK_JSON"
  n=0
  while IFS=$'\t' read -r HN MODEL CMD GUIDED; do
    [ -n "${HN:-}" ] || continue
    mode=$([ "${GUIDED:-0}" = "1" ] && echo guided || echo blind)
    say "5. solve    : [$HN/$MODEL $mode] (cd base_tree && $CMD \"<objective:$mode>\")"
    n=$((n+1))
  done < <([ "${RUN_SOLVERS:-1}" = 1 ] && python3 gen/conf.py solvers)
  say "   enabled solvers: $n (RUN_SOLVERS=${RUN_SOLVERS:-1})"
  if [ -n "$WL_CMD" ]; then say "6. measure  : per arm x$REPEATS inside $IMG: $WL_CMD  -> parse_workload.py"
  else say "6. measure  : no workload cmd in task.json — latency omitted"; fi
  say "7. record   : record.py -> $RESULTS_JSON (+ logs copied to $LOGS_DIR)"
  say "DRY_RUN OK"
  exit 0
fi

# =============================== REAL RUN ====================================
preflight
say "hw=$HW_SLUG swap=$SWAP base=$BASE merge=$MERGE wheel=$WHEEL engine=$ENGINE repeats=$REPEATS"

if [ "$KEEP_WORK" = 1 ] && [ -f "$WORK/arms.json" ]; then
  say "KEEP_WORK=1: reusing refs in $WORK (skipping checkout/build/baseline + PR-solution measurement)"
else
rm -rf "$WORK"; mkdir -p "$WORK/base_full" "$WORK/gold_full"
# 0+2. regenerate THIS PR's workload dir so the Dockerfile matches task.json, then build
# (one image per task: wheel + both PR trees cloned inside).
python3 gen/gen_workloads.py "$PR" >/dev/null
layout_env "$PR"; DF="$WORKLOAD_DIR/Dockerfile"; CTX="$WORKLOAD_DIR"
[ -f "$DF" ] || die "no Dockerfile at $DF (gen/gen_workloads.py failed?)"
say "building $IMG from $DF (wheel $WHEEL, trees cloned in-image) ..."
$ENGINE build -t "$IMG" -f "$DF" "$CTX"
# 1. checkout: export the trees from the image (host keeps no vLLM clone)
CID=$($ENGINE create "$IMG")
$ENGINE cp "$CID:/base/." "$WORK/base_full/"
$ENGINE cp "$CID:/gold/." "$WORK/gold_full/"
$ENGINE rm "$CID" >/dev/null
[ "$(cat "$WORK/base_full/.sha")" = "$BASE" ] || die "image /base sha mismatch"
fi  # KEEP_WORK (checkout/build)

# machine snapshot (GPU, clocks, other tenants) next to the measurement, folded into results.<hw>.json by record.py
python3 "$HERE/machine.py" snapshot >"$WORK/machine.json" 2>/dev/null || true
# ---- in-container prelude: swap the arm's .py tree over the installed vllm --
# Wipes the wheel's own .py files first so a file the arm tree lacks (e.g. a
# kernel the PR introduces) cannot leak into the baseline arm; compiled .so stay.
# tests/ + benchmarks/ are copied to /t WITHOUT a sibling vllm/ package, so
# `import vllm` always resolves to site-packages (= the swapped arm).
if [ "$SWAP" = files ]; then
# files mode: keep the wheel's own .py tree (consistent with its compiled ops) and
# overlay ONLY the PR's source files from the arm (missing in arm -> removed).
PRELUDE='
set -e
site=$(python3 -c "import vllm,os;print(os.path.dirname(os.path.dirname(vllm.__file__)))")
for f in '"$SRC_FILES"'; do
  if [ -f "/arm/$f" ]; then mkdir -p "$site/$(dirname "$f")"; cp "/arm/$f" "$site/$f"; else rm -f "$site/$f"; fi
done
'
else
PRELUDE='
set -e
site=$(python3 -c "import vllm,os;print(os.path.dirname(os.path.dirname(vllm.__file__)))")
find "$site/vllm" -name "*.py" -delete
cp -a /arm/vllm/. "$site/vllm/"
'
fi
PRELUDE="$PRELUDE"'
ln -sfn /arm /src   # vendored data/pr_<n>/bench.py read the arm tree at /src (ARM env = arm name)
# ~2 s GPU warm-up so the first timed iteration is not on a P8/180 MHz card (BASELINE_GAP.md rec. 6)
python3 - <<"WARM" 2>/dev/null || true
import torch, time
if torch.cuda.is_available():
    x = torch.randn(4096, 4096, device="cuda", dtype=torch.float16); t = time.time()
    while time.time() - t < 2.0: x = x @ x; x = x / x.norm()
    torch.cuda.synchronize()
WARM
rm -rf /t && mkdir -p /t && cd /gold
for d in tests benchmarks pyproject.toml conftest.py; do [ -e "$d" ] && cp -a "$d" /t/; done
cd /t
'

# Common mounts: /arm = the arm's tree, /gold = merge tree (tests/benchmarks),
# /task = data/pr_<n> (vendored bench.py, tests/), HF cache shared across PRs at
# .weights/hf (model weights download once, never committed). --shm-size for
# vLLM engine workloads.
mkdir -p "$ROOT/.weights/hf"
# WL_TIMEOUT: hard cap per workload run (a hung bench must not block the queue); container gets killed with it.
WL_TIMEOUT="${WL_TIMEOUT:-20m}"
run_in_img() { # run_in_img <tree> <script>   (TIMED=1 -> capped by WL_TIMEOUT)
  local cname="vllm-${PR}-$$-$RANDOM"
  [ "${TIMED:-0}" = 1 ] && { timeout -k 20s "$WL_TIMEOUT" "$ENGINE" run --rm --name "$cname" $GPU --shm-size=2g ${ARM_ENV:-} \
    -v "$1":/arm:ro -v "$WORK/gold_full":/gold:ro -v "$TASK":/task:ro \
    -v "$ROOT/.weights/hf":/root/.cache/huggingface \
    -e HF_HUB_DISABLE_PROGRESS_BARS=1 -e CUDA_VISIBLE_DEVICES=0 -e CUDA_DEVICE_ORDER=PCI_BUS_ID \
    "$IMG" -lc "$2"; local rc=$?; [ $rc = 124 ] && { $ENGINE kill "$cname" >/dev/null 2>&1; say "workload run hit WL_TIMEOUT=$WL_TIMEOUT -> killed"; }; return $rc; }
  $ENGINE run --rm $GPU --shm-size=2g ${ARM_ENV:-} \
    -v "$1":/arm:ro -v "$WORK/gold_full":/gold:ro -v "$TASK":/task:ro \
    -v "$ROOT/.weights/hf":/root/.cache/huggingface \
    -e HF_HUB_DISABLE_PROGRESS_BARS=1 -e CUDA_VISIBLE_DEVICES=0 -e CUDA_DEVICE_ORDER=PCI_BUS_ID \
    "$IMG" -lc "$2"
}

# contention gate (BASELINE_GAP.md rec. 3): wait up to CONTENTION_WAIT (default 10m) while another tenant uses
# the GPU (>5 % util or a foreign compute process >1 GiB) or the CPU (load1 > nproc/2); then run anyway, flagged.
CONTENTION_WAIT="${CONTENTION_WAIT:-600}"
contended() { python3 - <<'PY'
import json, os, subprocess, sys
try:
    u = subprocess.run("nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits", shell=True, capture_output=True, text=True, timeout=10).stdout.strip().splitlines()[0]
    apps = subprocess.run("nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits", shell=True, capture_output=True, text=True, timeout=10).stdout.strip().splitlines()
    big = [a for a in apps if a.strip() and a.split(",")[1].strip().isdigit() and int(a.split(",")[1]) > 1024]
    load1 = os.getloadavg()[0]; n = os.cpu_count() or 1
    why = []
    if int(float(u or 0)) > 5: why.append(f"gpu util {u}%")
    if big: why.append(f"{len(big)} foreign gpu proc >1GiB")
    if load1 > n / 2: why.append(f"load1 {load1:.1f} > {n/2:.0f}")
    print(";".join(why))
except Exception as e:
    print("")
PY
}
wait_for_quiet() { # -> sets CONTENDED="" or the reason
  local waited=0 why
  while why="$(contended)"; [ -n "$why" ] && [ "$waited" -lt "$CONTENTION_WAIT" ]; do
    [ "$waited" = 0 ] && say "contention: $why — waiting up to ${CONTENTION_WAIT}s"
    sleep 30; waited=$((waited+30))
  done
  CONTENDED="$why"; [ -n "$why" ] && say "still contended ($why) — running anyway, flagged"
}

# 3+4. isolate an arm's .py tree in the same image, run the oracle -> PASS/FAIL
run_arm() {
  local arm="$1" tree="$2"
  [ -n "$TESTS" ] || { echo SKIP; return 0; }
  run_in_img "$tree" "$PRELUDE"'
    python3 -m pytest -q -p no:cacheprovider '"$TESTS"' >/tmp/t.log 2>&1 && echo PASS || echo FAIL
    cat /tmp/t.log >&2
  ' 2>"$WORK/test_${arm}.log"
}

# 6. workload: one run = one `docker run` of the author's bench cmd. Repeats are INTERLEAVED ABBA across
# arms (A,B,B,A,A,B,...) so clock/thermal/tenant drift lands on both arms equally
# (BASELINE_GAP.md rec. 4). A machine snapshot + contention flag is stored next to every run.
run_workload_once() { # <arm> <tree> <i>
  local arm="$1" tree="$2" i="$3"
  wait_for_quiet
  python3 "$HERE/machine.py" snapshot >"$WORK/machine_${arm}_${i}.json" 2>/dev/null || true
  [ -n "$CONTENDED" ] && echo "$CONTENDED" >"$WORK/contended_${arm}_${i}.txt"
  say "workload $arm run $i/$REPEATS${CONTENDED:+ (contended: $CONTENDED)}"
  TIMED=1 ARM_ENV="-e ARM=$arm" run_in_img "$tree" "$PRELUDE $WL_CMD" >"$WORK/wl_${arm}_${i}.log" 2>&1
  local rc=$?
  [ $rc = 0 ] || { say "workload $arm run $i failed rc=$rc (see $WORK/wl_${arm}_${i}.log)"; rm -f "$WORK/wl_${arm}_${i}.log"; [ $rc = 124 ] && touch "$WORK/.wl_timeout"; return 1; }
}
run_workloads_abba() { # <armA> <treeA> <armB> <treeB>
  [ -n "$WL_CMD" ] || return 0
  local i failA=0 failB=0
  rm -f "$WORK/.wl_timeout"
  for i in $(seq 1 "$REPEATS"); do
    [ -f "$WORK/.wl_timeout" ] && { say "a workload run timed out: bench is stuck, aborting both arms"; break; }
    if [ $((i % 2)) = 1 ]; then
      [ $failA = 1 ] || run_workload_once "$1" "$2" "$i" || failA=1
      [ $failB = 1 ] || run_workload_once "$3" "$4" "$i" || failB=1
    else
      [ $failB = 1 ] || run_workload_once "$3" "$4" "$i" || failB=1
      [ $failA = 1 ] || run_workload_once "$1" "$2" "$i" || failA=1
    fi
  done
}
run_workload() { # single-arm (solver arms): plain repeats
  local arm="$1" tree="$2" i
  [ -n "$WL_CMD" ] || return 0
  for i in $(seq 1 "$REPEATS"); do run_workload_once "$arm" "$tree" "$i" || break; done
}

if [ "$KEEP_WORK" = 1 ] && [ -f "$WORK/arms.json" ]; then
  say "KEEP_WORK=1: refs already measured ($(ls "$WORK" | grep -c "^wl_") workload logs)"
else
say "tests: ${TESTS:-<none-known>}"
BASE_RES=$(run_arm base "$WORK/base_full" || echo FAIL)
GOLD_RES=$(run_arm gold "$WORK/gold_full" || echo FAIL)
say "base=$BASE_RES gold=$GOLD_RES"
run_workloads_abba base "$WORK/base_full" gold "$WORK/gold_full"
# arms.json is the running ledger record.py folds into results.<hw>.json
python3 - "$WORK/arms.json" "$BASE_RES" "$GOLD_RES" <<'PY'
import json,sys
json.dump({"base":{"tests":sys.argv[2],"slug":"base"},"gold":{"tests":sys.argv[3],"slug":"gold"}},open(sys.argv[1],"w"),indent=1)
PY
python3 "$HERE/record.py" "$PR"          # measured baseline / PR-solution numbers are live on the dashboard from here on
fi  # KEEP_WORK (refs)

# blind objective = problem.md with the "## PR's solution" section removed
OBJ_BLIND="$(printf '%s\n' "$OBJ" | awk '
  /^## PR.s solution/{skip=1; next}
  skip && /^## /{skip=0}
  !skip{print}')"

# 5. solvers: each starts from the baseline tree, edits ./vllm via `cmd + objective`.
SOLVERS=""; ARMS="base gold"
if [ "${RUN_SOLVERS:-1}" = 1 ]; then
while IFS=$'\t' read -r HN MODEL CMD GUIDED; do
  [ -n "${HN:-}" ] || continue
  SLUG=$(printf '%s' "$HN" | tr -c 'A-Za-z0-9._-' '_')
  SDIR="$WORK/solutions/$SLUG"
  rm -rf "$SDIR"; mkdir -p "$SDIR"
  cp -a "$WORK/base_full/." "$SDIR/"
  if [ "${GUIDED:-0}" = "1" ]; then THIS_OBJ="$OBJ"; mode=guided; else THIS_OBJ="$OBJ_BLIND"; mode=blind; fi
  say "solver $HN ($MODEL $mode): $CMD"
  rm -f "$WORK"/wl_"$SLUG"_*.log "$WORK/test_$SLUG.log"
  ( cd "$SDIR" && eval "timeout -k 30s $SOLVER_TIMEOUT $CMD \"\$THIS_OBJ\"" </dev/null ) >"$WORK/solve_$SLUG.log" 2>&1 \
    || say "solver $HN exited nonzero/timeout $SOLVER_TIMEOUT (see $WORK/solve_$SLUG.log)"
  ( cd "$SDIR" && diff -ru "$WORK/base_full/vllm" vllm >"$WORK/solve_$SLUG.patch" ) || true
  if grep -qiE "spend limit|usage limit|rate limit|quota" "$WORK/solve_$SLUG.log" && [ ! -s "$WORK/solve_$SLUG.patch" ]; then
    say "solver $HN hit an API limit — arm recorded as LIMIT (no measurement)"
    R=LIMIT
  else
  R=$(run_arm "$SLUG" "$SDIR" || echo FAIL)
  fi
  say "solver $HN tests=$R patch=$(wc -c <"$WORK/solve_$SLUG.patch")B"
  [ "$R" = LIMIT ] || run_workload "$SLUG" "$SDIR"
  python3 - "$WORK/arms.json" "$HN" "$R" "$SLUG" <<'PY'
import json,sys
d=json.load(open(sys.argv[1])); d[sys.argv[2]]={"tests":sys.argv[3],"slug":sys.argv[4]}; json.dump(d,open(sys.argv[1],"w"),indent=1)
PY
  python3 "$HERE/record.py" "$PR"        # dashboard gets this arm right away
done < <(python3 gen/conf.py solvers)
else say "RUN_SOLVERS=0: refs only"; fi

# 6. results: record.py already wrote results.<hw>.json after refs and after each solver arm
say "done -> $RESULTS_JSON"
# CLEAN_IMAGE=1 (run.sh --reproduce sets it when CLEAN_IMAGES=1): drop this PR's image (13-25 GB each)
# once measured; a later --run-formula-harnesses rebuilds it from the cached wheel layer.
if [ "${CLEAN_IMAGE:-0}" = 1 ] && [ "$DRY_RUN" != 1 ]; then $ENGINE rmi "$IMG" >/dev/null 2>&1 && say "image $IMG removed (CLEAN_IMAGE=1)"; fi
