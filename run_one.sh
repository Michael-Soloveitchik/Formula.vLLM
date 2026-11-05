#!/usr/bin/env bash
# Build + verify + measure ONE pr, self-contained. Dumps results/<pr>.json.
#   ./run_one.sh <pr>
#   DRY_RUN=1 ./run_one.sh <pr>     # validate every step, no docker/clone needed
#
# Two image modes, picked per task:
#   wheel  : tasks/pr_<n>/metadata.json has "wheel": "<vllm pypi version>".
#            Dockerfile.gpu-wheel installs that wheel AND clones/checks out the
#            PR's base_sha + merge_sha INSIDE the image (/base, /gold). No vLLM
#            clone on the host. Needed on GPUs the cu121 source build can't
#            target (sm_120). Pure-python / Triton PRs only.
#   source : (legacy) Dockerfile.gpu builds vLLM from source at base_sha from a
#            host clone at VLLM_GIT / paths.vllm_git.
#
# Pipeline (all in code):
#   1. checkout  : export base + gold trees (from the image in wheel mode)
#   2. build     : one image per task
#   3. isolate   : each arm = a .py-tree swap INSIDE that same image, so the
#                  environment is byte-identical across arms (FormulaCode-style)
#   4. run       : the hidden pytest oracle — base must FAIL, gold must PASS
#   5. solve     : each enabled solver edits a base tree, is re-verified
#   6. measure   : run the task's workload cmd per arm, `repeats` times, inside
#                  the same image -> ms per scenario, mean ± sd
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"; cd "$HERE"
PR="${1:?usage: run_one.sh <pr>}"
DRY_RUN="${DRY_RUN:-0}"

VLLM_GIT="${VLLM_GIT:-$(python3 conf.py get paths.vllm_git)}"
[[ "$VLLM_GIT" = /* ]] || VLLM_GIT="$HERE/$VLLM_GIT"
ENGINE="${CONTAINER_ENGINE:-docker}"
GPU="${GPU_FLAG:---gpus all}"
WORK="$HERE/.work/pr_${PR}"
TASK="$HERE/tasks/pr_${PR}"
REPEATS="${REPEATS:-$(python3 conf.py get measure.repeats)}"; REPEATS="${REPEATS:-5}"

say(){ echo "[$PR] $*"; }
die(){ echo "[$PR] ERROR: $*" >&2; exit 1; }

# ---- resolve shas from the vendored index (always present) ------------------
BASE=$(python3 -c "import json;print({x['pr']:x for x in json.load(open('index.json'))}[$PR]['base_sha'])") \
  || die "pr $PR not in index.json"
MERGE=$(python3 -c "import json;print({x['pr']:x for x in json.load(open('index.json'))}[$PR]['merge_sha'])")
[ -n "$BASE" ] && [ -n "$MERGE" ] || die "missing base/merge sha for $PR"

# ---- task metadata: tests, wheel mode, workload ----------------------------
meta(){ python3 -c "
import json,sys
try: m=json.load(open('$TASK/metadata.json'))
except Exception: m={}
v=m
for k in sys.argv[1].split('.'):
    v=v.get(k,'') if isinstance(v,dict) else ''
print(' '.join(v) if isinstance(v,list) else v)" "$1"; }
TESTS="$(meta test_files)"
WHEEL="$(meta wheel)"
WL_CMD="$(meta workload.cmd)"
OBJ="$(cat "$TASK/problem.md" 2>/dev/null || echo "Optimize PR #$PR; make the hidden tests pass.")"
IMG="vllm-task:${PR}"
MODE=$([ -n "$WHEEL" ] && echo wheel || echo source)

# ---- preflight: fail early with a clear message -----------------------------
preflight() {
  "$ENGINE" info >/dev/null 2>&1 || die "container engine '$ENGINE' not available/running"
  if [ "$MODE" = source ]; then
    [ -d "$VLLM_GIT/.git" ] || die "no vLLM clone at '$VLLM_GIT' — set VLLM_GIT or paths.vllm_git (or add 'wheel' to metadata.json)"
    git -C "$VLLM_GIT" cat-file -t "$BASE"  >/dev/null 2>&1 || die "base_sha $BASE not in clone (git fetch?)"
    git -C "$VLLM_GIT" cat-file -t "$MERGE" >/dev/null 2>&1 || die "merge_sha $MERGE not in clone"
  fi
}

VER=0.0.0
if [ "$MODE" = source ] && [ -d "$VLLM_GIT/.git" ]; then
  VER=$(git -C "$VLLM_GIT" describe --tags --abbrev=0 "$BASE" 2>/dev/null | sed 's/^v//;s/rc.*//' || echo 0.0.0)
  [ -n "$VER" ] || VER=0.0.0
fi

# =============================== DRY RUN =====================================
if [ "$DRY_RUN" = "1" ]; then
  say "DRY_RUN — validating pipeline (no docker, no clone) mode=$MODE"
  if [ "$MODE" = wheel ]; then
    DF=$([ -f "$TASK/workload/Dockerfile" ] && echo "$TASK/workload/Dockerfile" || echo Dockerfile.gpu-wheel)
    say "2. build    : $ENGINE build -t $IMG -f $DF --build-arg VLLM_VERSION=$WHEEL --build-arg BASE_SHA=$BASE --build-arg MERGE_SHA=$MERGE <ctx>"
    say "1. checkout : $ENGINE cp <container>:/base -> $WORK/base_full ; /gold -> $WORK/gold_full"
  else
    say "1. checkout : git -C $VLLM_GIT archive $BASE  -> $WORK/base_full"
    say "            : git -C $VLLM_GIT archive $MERGE -> $WORK/gold_full"
    say "2. build    : $ENGINE build -t $IMG -f Dockerfile.gpu --build-arg SRC_DIR=.work/pr_${PR}/base_full --build-arg PRETEND_VERSION=$VER ."
  fi
  say "3. isolate  : $ENGINE run --rm $GPU -v <tree>:/arm:ro -v <gold>:/gold:ro $IMG  (wipe site-packages/vllm/*.py, cp /arm/vllm -> site-packages, tests+benchmarks copied to /t)"
  say "4. run      : python3 -m pytest -q ${TESTS:-<none>}  (base expect FAIL, gold expect PASS)"
  [ -n "$TESTS" ] || say "   WARN: no test_files in $TASK/metadata.json"
  [ -f "$TASK/metadata.json" ] || say "   WARN: no vendored task at $TASK (oracle unknown)"
  n=0
  while IFS=$'\t' read -r HN MODEL CMD GUIDED; do
    [ -n "${HN:-}" ] || continue
    mode=$([ "${GUIDED:-0}" = "1" ] && echo guided || echo blind)
    say "5. solve    : [$HN/$MODEL $mode] (cd base_tree && $CMD \"<objective:$mode>\")"
    n=$((n+1))
  done < <(python3 conf.py solvers)
  say "   enabled solvers: $n"
  if [ -n "$WL_CMD" ]; then say "6. measure  : per arm x$REPEATS inside $IMG: $WL_CMD  -> parse_workload.py"
  elif [ -f "$TASK/results_measured.json" ]; then say "6. measure  : fold $TASK/results_measured.json (µs/op + variance)"
  else say "6. measure  : no bench shipped for this PR — latency omitted"; fi
  say "DRY_RUN OK"
  exit 0
fi

# =============================== REAL RUN ====================================
preflight
say "mode=$MODE base=$BASE merge=$MERGE wheel=${WHEEL:--} engine=$ENGINE repeats=$REPEATS"

rm -rf "$WORK"; mkdir -p "$WORK/base_full" "$WORK/gold_full"
if [ "$MODE" = wheel ]; then
  # 2. build (one image per task: wheel + both PR trees cloned inside). The
  # per-PR tasks/pr_<n>/workload/Dockerfile (gen_workloads.py) is the source of
  # truth; Dockerfile.gpu-wheel at the root is the same recipe as a fallback.
  if [ -f "$TASK/workload/Dockerfile" ]; then DF="$TASK/workload/Dockerfile"; CTX="$TASK/workload"
  else DF="Dockerfile.gpu-wheel"; CTX="."; fi
  say "building $IMG from $DF (wheel $WHEEL, trees cloned in-image) ..."
  $ENGINE build -t "$IMG" -f "$DF" \
    --build-arg VLLM_VERSION="$WHEEL" \
    --build-arg BASE_SHA="$BASE" --build-arg MERGE_SHA="$MERGE" "$CTX"
  # 1. checkout: export the trees from the image (host keeps no vLLM clone)
  CID=$($ENGINE create "$IMG")
  $ENGINE cp "$CID:/base/." "$WORK/base_full/"
  $ENGINE cp "$CID:/gold/." "$WORK/gold_full/"
  $ENGINE rm "$CID" >/dev/null
  [ "$(cat "$WORK/base_full/.sha")" = "$BASE" ] || die "image /base sha mismatch"
else
  git -C "$VLLM_GIT" archive "$BASE"  | tar -x -C "$WORK/base_full"
  git -C "$VLLM_GIT" archive "$MERGE" | tar -x -C "$WORK/gold_full"
  say "building $IMG ..."
  $ENGINE build -t "$IMG" -f Dockerfile.gpu \
    --build-arg SRC_DIR=".work/pr_${PR}/base_full" \
    --build-arg PRETEND_VERSION="$VER" .
fi

# ---- in-container prelude: swap the arm's .py tree over the installed vllm --
# Wipes the wheel's own .py files first so a file the arm tree lacks (e.g. a
# kernel the PR introduces) cannot leak into the base arm; compiled .so stay.
# tests/ + benchmarks/ are copied to /t WITHOUT a sibling vllm/ package, so
# `import vllm` always resolves to site-packages (= the swapped arm).
PRELUDE='
set -e
site=$(python3 -c "import vllm,os;print(os.path.dirname(os.path.dirname(vllm.__file__)))")
find "$site/vllm" -name "*.py" -delete
cp -a /arm/vllm/. "$site/vllm/"
rm -rf /t && mkdir -p /t && cd /gold
for d in tests benchmarks pyproject.toml conftest.py; do [ -e "$d" ] && cp -a "$d" /t/; done
cd /t
'

# Common mounts: /arm = the arm's tree, /gold = merge tree (tests/benchmarks),
# /task = tasks/pr_<n> (vendored bench.py etc.), HF cache shared across PRs at
# .weights/hf (model weights download once, never committed). --shm-size for
# vLLM engine workloads.
mkdir -p "$HERE/.weights/hf"
run_in_img() { # run_in_img <tree> <script>
  $ENGINE run --rm $GPU --shm-size=2g \
    -v "$1":/arm:ro -v "$WORK/gold_full":/gold:ro -v "$TASK":/task:ro \
    -v "$HERE/.weights/hf":/root/.cache/huggingface \
    -e HF_HUB_DISABLE_PROGRESS_BARS=1 \
    "$IMG" -lc "$2"
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

# 6. workload: run the author's bench cmd `REPEATS` times per arm, same image
run_workload() {
  local arm="$1" tree="$2" i
  [ -n "$WL_CMD" ] || return 0
  for i in $(seq 1 "$REPEATS"); do
    say "workload $arm run $i/$REPEATS"
    run_in_img "$tree" "$PRELUDE $WL_CMD" >"$WORK/wl_${arm}_${i}.log" 2>&1 \
      || { say "workload $arm run $i failed (see $WORK/wl_${arm}_${i}.log)"; break; }
  done
}

say "tests: ${TESTS:-<none-known>}"
BASE_RES=$(run_arm base "$WORK/base_full" || echo FAIL)
GOLD_RES=$(run_arm gold "$WORK/gold_full" || echo FAIL)
say "base=$BASE_RES gold=$GOLD_RES"
run_workload base "$WORK/base_full"
run_workload gold "$WORK/gold_full"

# blind objective = problem.md with the "## Expert solution" section removed
OBJ_BLIND="$(printf '%s\n' "$OBJ" | awk '
  /^## Expert solution/{skip=1; next}
  skip && /^## /{skip=0}
  !skip{print}')"

# 5. solvers: each starts from the base tree, edits ./vllm via `cmd + objective`.
SOLVERS=""; ARMS="base gold"
while IFS=$'\t' read -r HN MODEL CMD GUIDED; do
  [ -n "${HN:-}" ] || continue
  SLUG=$(printf '%s' "$HN" | tr -c 'A-Za-z0-9._-' '_')
  SDIR="$WORK/solutions/$SLUG"
  rm -rf "$SDIR"; mkdir -p "$SDIR"
  cp -a "$WORK/base_full/." "$SDIR/"
  if [ "${GUIDED:-0}" = "1" ]; then THIS_OBJ="$OBJ"; mode=guided; else THIS_OBJ="$OBJ_BLIND"; mode=blind; fi
  say "solver $HN ($MODEL $mode): $CMD"
  ( cd "$SDIR" && eval "$CMD \"\$THIS_OBJ\"" </dev/null ) >"$WORK/solve_$SLUG.log" 2>&1 \
    || say "solver $HN exited nonzero (see $WORK/solve_$SLUG.log)"
  ( cd "$SDIR" && diff -ru "$WORK/base_full/vllm" vllm >"$WORK/solve_$SLUG.patch" ) || true
  R=$(run_arm "$SLUG" "$SDIR" || echo FAIL)
  say "solver $HN tests=$R"
  run_workload "$SLUG" "$SDIR"
  SOLVERS="$SOLVERS$HN:$R:$SLUG"$'\t'; ARMS="$ARMS $SLUG"
done < <(python3 conf.py solvers)

# 6. record + fold measured latency
LAT_JSON=""
if [ -n "$WL_CMD" ]; then
  LAT_JSON=$(python3 parse_workload.py "$WORK" $ARMS)
elif [ -f "$TASK/results_measured.json" ]; then
  LAT_JSON=$(cat "$TASK/results_measured.json")
fi
LAT_JSON="$LAT_JSON" python3 - "$PR" "$BASE" "$MERGE" "$BASE_RES" "$GOLD_RES" "$SOLVERS" "$TASK" "$MODE" "$WHEEL" "$WL_CMD" <<'PY'
import json,sys,os,time
pr,base,merge,bres,gres,solvers,task,mode,wheel,wl=sys.argv[1:11]
idx={x['pr']:x for x in json.load(open('index.json'))}[int(pr)]
arms={"base":{"tests":bres,"slug":"base"},"gold":{"tests":gres,"slug":"gold"}}
for tok in solvers.split('\t'):
    tok=tok.strip()
    if tok.count(':')>=2:
        n,r,slug=tok.rsplit(':',2); arms[n]={"tests":r,"slug":slug}
lat=os.environ.get("LAT_JSON") or ""
latency=json.loads(lat) if lat.strip() else None
experiments=None
if latency and "scenarios" in latency:   # re-key arm slugs -> dashboard arm names
    slug2name={v["slug"]:k for k,v in arms.items()}
    for s,d in latency["scenarios"].items():
        latency["scenarios"][s]={slug2name.get(k,k):v for k,v in d.items()}   # keeps the "unit" key
    # experiments_v2 for web/index.html: one table per scenario. Scenarios the
    # task metadata names come first (with the author's reported numbers);
    # every other measured scenario follows with reported=null.
    try: meta=json.load(open(os.path.join(task,"metadata.json")))
    except Exception: meta={}
    listed=meta.get("experiments") or []
    names=[e["name"] for e in listed]+[s for s in latency["scenarios"] if s not in {e["name"] for e in listed}]
    rep_by={e["name"]:e.get("reported") for e in listed}
    pyref=latency.get("pytorch_ref") or {}
    experiments=[]
    for n in names:
        a=dict(latency["scenarios"].get(n,{}))
        unit=a.pop("unit",None) or (meta.get("workload") or {}).get("unit") or "ms"
        note=None
        if "base" not in a and n in pyref:   # base tree can't run the script (PR introduces it)
            a["base"]=pyref[n]; note=(meta.get("workload") or {}).get("base_note") or "base = script's PyTorch column"
        e={"name":n,"unit":unit,"workload":wl,"reported":rep_by.get(n),"arms":a}
        if note: e["note"]=note
        experiments.append(e)
out={
  "schema":"experiments_v2" if experiments is not None else None,
  "experiments":experiments,
  "pr":int(pr),
  "pr_url":f"https://github.com/vllm-project/vllm/pull/{pr}",
  "title":idx.get("title"),"field":idx.get("field"),
  "reported_gain":idx.get("reported_gain"),"gain_kind":idx.get("gain_kind"),
  "base_sha":base,"merge_sha":merge,
  "image":{"mode":mode,"wheel":wheel or None,"workload_cmd":wl or None,"repeats":int(os.environ.get("REPEATS","5"))},
  "arms":arms,
  "improved": gres=="PASS" and bres=="FAIL",
  "latency":latency,
  "ts":time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime()),
}
json.dump(out,open(f"results/{pr}.json","w"),indent=1)
print("wrote results/%s.json improved=%s"%(pr,out["improved"]))
PY
