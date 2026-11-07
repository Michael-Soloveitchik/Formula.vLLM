#!/usr/bin/env bash
# Cluster-ordered workload runner. For each cluster in clusters.json["order"]:
#   1. download.sh ONCE (fetch-on-demand, gated, cached in .weights/<cluster>)
#   2. for each PR in the cluster: build tasks/pr_<n>/workload/Dockerfile,
#      run run_workload.sh for BOTH arms (baseline tree + PR's solution tree) with the
#      shared weights mounted at /weights, fold latency into results_workload/<pr>.json
#
# SAFE on a laptop: GPU clusters self-skip (run_workload.sh prints SKIP=no_cuda,
# exit 0) and download.sh never fetches until actually run on the target box.
# Mixtral is LAST and refuses unless ALLOW_MIXTRAL=1.
#
#   ./run_workloads.sh              # all clusters in order
#   ./run_workloads.sh opt125m      # a single cluster
#   ./run_workloads.sh --no-download   # skip weight fetch (assume cached)
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"; cd "$HERE"
mkdir -p .work results_workload .weights

DO_DOWNLOAD=1
WANT=""
for a in "$@"; do
  case "$a" in
    --no-download) DO_DOWNLOAD=0 ;;
    *) WANT="$WANT $a" ;;
  esac
done

have_cuda() { command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; }

ORDER=$(python3 -c "import json;print(' '.join(json.load(open('clusters.json'))['order']))")
[ -n "$WANT" ] && ORDER="$WANT"

for CK in $ORDER; do
  read -r NEEDS MODEL PRS < <(python3 - "$CK" <<'PY'
import json,sys
c=json.load(open('clusters.json'))['clusters'][sys.argv[1]]
print(c['needs'], c['model'] or '-', ' '.join(str(p) for p in c['prs']))
PY
)
  echo "================= cluster $CK (needs=$NEEDS model=$MODEL) ================="

  if [ "$NEEDS" = "gpu" ] && ! have_cuda; then
    echo "# cluster $CK: no CUDA on this box — SCAFFOLD ONLY, skipping run."
    continue
  fi

  # weights once per cluster (first PR owning this cluster's download.sh is fine)
  if [ "$DO_DOWNLOAD" = "1" ] && [ "$MODEL" != "-" ]; then
    FIRST=$(echo "$PRS" | awk '{print $1}')
    DL="tasks/pr_${FIRST}/workload/download.sh"
    [ -f "$DL" ] && { echo "# fetching weights via $DL"; bash "$DL" || echo "# download.sh non-fatal rc=$?"; }
  fi

  WCACHE="$HERE/.weights/$CK"

  for PR in $PRS; do
    WDIR="tasks/pr_${PR}/workload"
    [ -d "$WDIR" ] || { echo "# pr_$PR: no workload/ (not scaffolded), skip"; continue; }
    RES="results/${PR}.json"
    [ -f "$RES" ] || { echo "# pr_$PR: no results/${PR}.json (need baseline + PR commits), skip"; continue; }

    read -r BASE GOLD < <(python3 - "$RES" <<'PY'
import json,sys
r=json.load(open(sys.argv[1]))
print(r.get('base_sha','-'), r.get('merge_sha','-'))
PY
)
    IMG="vllm-wl:${PR}"
    echo "--- pr_$PR build $IMG ($WDIR/Dockerfile) ---"
    docker build -t "$IMG" -f "$WDIR/Dockerfile" "$WDIR" || { echo "# pr_$PR build failed, skip"; continue; }

    for ARM in base gold; do
      SHA=$([ "$ARM" = base ] && echo "$BASE" || echo "$GOLD")
      TREE="$HERE/.work/${PR}/${ARM}"
      if [ ! -d "$TREE" ]; then
        echo "# pr_$PR arm=$ARM: tree $TREE missing (run ./run_one.sh $PR to extract $SHA), skip arm"
        continue
      fi
      MOUNTS=(-v "$TREE:/src:ro" -v "$HERE/$WDIR/run_workload.sh:/bench/run_workload.sh:ro")
      [ -d "$WCACHE" ] && MOUNTS+=(-v "$WCACHE:/weights:ro")
      GPUFLAG=""; [ "$NEEDS" = gpu ] && GPUFLAG="--gpus all"
      echo "# pr_$PR arm=$ARM run (sha=$SHA)"
      docker run --rm $GPUFLAG -e ARM="$ARM" "${MOUNTS[@]}" "$IMG" \
        bash /bench/run_workload.sh 2>&1 | tee "results_workload/${PR}.${ARM}.log" || true
    done
  done
done

echo "# workloads done. parseable lines in results_workload/*.log"
