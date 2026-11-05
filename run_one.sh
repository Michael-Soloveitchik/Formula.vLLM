#!/usr/bin/env bash
# Build + verify + measure ONE pr, self-contained. Dumps results/<pr>.json.
#   ./run_one.sh <pr>
#   DRY_RUN=1 ./run_one.sh <pr>     # validate every step, no docker/clone needed
#
# Pipeline (all in code):
#   1. checkout  : export base_sha + merge_sha trees from the vLLM clone
#   2. build     : one image per task at base_sha (Dockerfile.gpu)
#   3. isolate   : each arm = a .py-tree swap INSIDE that same image, so the
#                  environment is byte-identical across arms (FormulaCode-style)
#   4. run       : the hidden pytest oracle — base must FAIL, gold must PASS
#   5. solve     : each enabled solver edits a base tree, is re-verified
#   6. measure   : fold in µs/op latency + variance (if the task ships a bench)
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

say(){ echo "[$PR] $*"; }
die(){ echo "[$PR] ERROR: $*" >&2; exit 1; }

# ---- resolve shas from the vendored index (always present) ------------------
BASE=$(python3 -c "import json;print({x['pr']:x for x in json.load(open('index.json'))}[$PR]['base_sha'])") \
  || die "pr $PR not in index.json"
MERGE=$(python3 -c "import json;print({x['pr']:x for x in json.load(open('index.json'))}[$PR]['merge_sha'])")
[ -n "$BASE" ] && [ -n "$MERGE" ] || die "missing base/merge sha for $PR"

# ---- preflight: fail early with a clear message -----------------------------
preflight() {
  [ -d "$VLLM_GIT/.git" ] || die "no vLLM clone at '$VLLM_GIT' — set VLLM_GIT or paths.vllm_git"
  git -C "$VLLM_GIT" cat-file -t "$BASE"  >/dev/null 2>&1 || die "base_sha $BASE not in clone (git fetch?)"
  git -C "$VLLM_GIT" cat-file -t "$MERGE" >/dev/null 2>&1 || die "merge_sha $MERGE not in clone"
  "$ENGINE" info >/dev/null 2>&1 || die "container engine '$ENGINE' not available/running"
}

VER=$(git -C "$VLLM_GIT" describe --tags --abbrev=0 "$BASE" 2>/dev/null | sed 's/^v//;s/rc.*//' || echo 0.0.0)
[ -n "$VER" ] || VER=0.0.0

# ---- test files + objective from the vendored task --------------------------
TESTS=""
if [ -f "$TASK/metadata.json" ]; then
  TESTS=$(python3 -c "import json;print(' '.join(json.load(open('$TASK/metadata.json')).get('test_files',[])))")
fi
OBJ="$(cat "$TASK/problem.md" 2>/dev/null || echo "Optimize PR #$PR; make the hidden tests pass.")"
IMG="vllm-task:${PR}"

# =============================== DRY RUN =====================================
# Validate wiring without docker or a clone: confirm index/task/solver/tests all
# resolve and print the exact commands the real run would execute.
if [ "$DRY_RUN" = "1" ]; then
  say "DRY_RUN — validating pipeline (no docker, no clone)"
  say "1. checkout : git -C $VLLM_GIT archive $BASE  -> $WORK/base_full"
  say "            : git -C $VLLM_GIT archive $MERGE -> $WORK/gold_full"
  say "2. build    : $ENGINE build -t $IMG -f Dockerfile.gpu --build-arg SRC_DIR=.work/pr_${PR}/base_full --build-arg PRETEND_VERSION=$VER ."
  say "3. isolate  : $ENGINE run --rm $GPU -v <tree>:/arm:ro -v <gold>:/gold:ro $IMG  (cp /arm/vllm -> site-packages, cd /gold)"
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
  [ -f "$TASK/results_measured.json" ] \
    && say "6. measure  : fold $TASK/results_measured.json (µs/op + variance)" \
    || say "6. measure  : no bench shipped for this PR — latency omitted"
  say "DRY_RUN OK"
  exit 0
fi

# =============================== REAL RUN ====================================
preflight
say "base=$BASE merge=$MERGE ver=$VER vllm_git=$VLLM_GIT engine=$ENGINE"

# 1. checkout
rm -rf "$WORK"; mkdir -p "$WORK/base_full" "$WORK/gold_full"
git -C "$VLLM_GIT" archive "$BASE"  | tar -x -C "$WORK/base_full"
git -C "$VLLM_GIT" archive "$MERGE" | tar -x -C "$WORK/gold_full"

# 2. build (one image per task, at base_sha)
say "building $IMG ..."
$ENGINE build -t "$IMG" -f Dockerfile.gpu \
  --build-arg SRC_DIR=".work/pr_${PR}/base_full" \
  --build-arg PRETEND_VERSION="$VER" .

# 3+4. isolate an arm's .py tree in the same image, run the oracle -> PASS/FAIL
run_arm() {
  local arm="$1" tree="$2"
  $ENGINE run --rm $GPU -v "$tree":/arm:ro -v "$WORK/gold_full":/gold:ro "$IMG" -lc '
    set -e
    site=$(python3 -c "import vllm,os;print(os.path.dirname(os.path.dirname(vllm.__file__)))")
    cp -a /arm/vllm/. "$site/vllm/"
    cd /gold
    python3 -m pytest -q '"$TESTS"' >/tmp/t.log 2>&1 && echo PASS || echo FAIL
  '
}

say "tests: ${TESTS:-<none-known>}"
BASE_RES=$(run_arm base "$WORK/base_full" || echo FAIL)
GOLD_RES=$(run_arm gold "$WORK/gold_full" || echo FAIL)
say "base=$BASE_RES gold=$GOLD_RES"

# blind objective = problem.md with the "## Expert solution" section removed
# (keeps everything up to it, resumes at the next "## " heading). Guided arms
# get the full OBJ incl. the expert patch and the "beat it" ask.
OBJ_BLIND="$(printf '%s\n' "$OBJ" | awk '
  /^## Expert solution/{skip=1; next}
  skip && /^## /{skip=0}
  !skip{print}')"

# 5. solvers: each starts from the base tree, edits ./vllm via `cmd + objective`.
# Arm names carry spaces/"+", so slugify for dir + log paths; keep the spaced
# name as the JSON arm key (SOLVERS entries use TAB as the field separator).
SOLVERS=""
while IFS=$'\t' read -r HN MODEL CMD GUIDED; do
  [ -n "${HN:-}" ] || continue
  SLUG=$(printf '%s' "$HN" | tr -c 'A-Za-z0-9._-' '_')
  SDIR="$WORK/solutions/$SLUG"
  rm -rf "$SDIR"; mkdir -p "$SDIR"
  cp -a "$WORK/base_full/." "$SDIR/"
  if [ "${GUIDED:-0}" = "1" ]; then THIS_OBJ="$OBJ"; mode=guided; else THIS_OBJ="$OBJ_BLIND"; mode=blind; fi
  say "solver $HN ($MODEL $mode): $CMD"
  ( cd "$SDIR" && eval "$CMD \"\$THIS_OBJ\"" ) >"$WORK/solve_$SLUG.log" 2>&1 \
    || say "solver $HN exited nonzero (see $WORK/solve_$SLUG.log)"
  R=$(run_arm "$SLUG" "$SDIR" || echo FAIL)
  SOLVERS="$SOLVERS$HN:$R"$'\t'
done < <(python3 conf.py solvers)

# 6. record + fold measured latency
python3 - "$PR" "$BASE" "$MERGE" "$BASE_RES" "$GOLD_RES" "$SOLVERS" "$TASK" <<'PY'
import json,sys,os,time
pr,base,merge,bres,gres,solvers,task=sys.argv[1:8]
idx={x['pr']:x for x in json.load(open('index.json'))}[int(pr)]
arms={"base":{"tests":bres},"gold":{"tests":gres}}
for tok in solvers.split('\t'):
    tok=tok.strip()
    if ':' in tok:
        n,r=tok.rsplit(':',1); arms[n]={"tests":r}
meas=os.path.join(task,"results_measured.json")
latency=json.load(open(meas)) if os.path.exists(meas) else None
out={
  "pr":int(pr),
  "pr_url":f"https://github.com/vllm-project/vllm/pull/{pr}",
  "title":idx.get("title"),"field":idx.get("field"),
  "reported_gain":idx.get("reported_gain"),"gain_kind":idx.get("gain_kind"),
  "base_sha":base,"merge_sha":merge,
  "arms":arms,
  "improved": gres=="PASS" and bres=="FAIL",
  "latency":latency,
  "ts":time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime()),
}
json.dump(out,open(f"results/{pr}.json","w"),indent=1)
print("wrote results/%s.json improved=%s"%(pr,out["improved"]))
PY
