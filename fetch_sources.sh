#!/usr/bin/env bash
# Make sure the vLLM clone exists and contains this PR's baseline and PR-solution commits.
#   ./fetch_sources.sh <pr>
# - clones https://github.com/vllm-project/vllm (blobless: history only, file contents on demand)
#   to paths.vllm_git if it is missing
# - fetches the PR's baseline sha and merge sha if they are not in the clone yet
# run_one.sh then exports each tree with `git archive <sha>` (= exact checkout of that commit).
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"; cd "$HERE"
PR="${1:?usage: fetch_sources.sh <pr>}"
VLLM_GIT="${VLLM_GIT:-$(python3 conf.py get paths.vllm_git)}"
[[ "$VLLM_GIT" = /* ]] || VLLM_GIT="$HERE/$VLLM_GIT"
URL="${VLLM_URL:-$(python3 conf.py get paths.vllm_url)}"; URL="${URL:-https://github.com/vllm-project/vllm.git}"

sha(){ python3 -c "import json;print({x['pr']:x for x in json.load(open('index.json'))}[$PR]['$1'])"; }
BASE=$(sha base_sha); MERGE=$(sha merge_sha)

if [ ! -d "$VLLM_GIT/.git" ]; then
  echo "[$PR] cloning $URL -> $VLLM_GIT (blobless)"
  git clone --filter=blob:none --no-checkout "$URL" "$VLLM_GIT"
fi
for S in "$BASE" "$MERGE"; do
  if git -C "$VLLM_GIT" cat-file -e "$S^{commit}" 2>/dev/null; then echo "[$PR] have $S"
  else echo "[$PR] fetching $S"; git -C "$VLLM_GIT" fetch --filter=blob:none origin "$S"; fi
done
# verify: the exact commits resolve (a missing blob would be fetched lazily by git archive)
git -C "$VLLM_GIT" rev-parse --verify "$BASE^{commit}" "$MERGE^{commit}" >/dev/null
echo "[$PR] sources ready: baseline=$BASE  PR-solution=$MERGE"
