#!/usr/bin/env bash
# AUTO-GENERATED — PR #28280. Fetch-on-demand weights for cluster qwen05b.
# SAFE: nothing downloads unless you RUN this on the target box. Skips if cached.
# Shared cache: ../../../.weights/qwen05b (gitignored, reused across the cluster's PRs).
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
CACHE="$HERE/../../../.weights/qwen05b"
HF_ID="Qwen/Qwen2-0.5B"
SIZE="~1GB"

if [ -d "$CACHE" ] && [ -n "$(ls -A "$CACHE" 2>/dev/null)" ]; then
  echo "# qwen05b: cache HIT at $CACHE — skip download."; exit 0
fi
mkdir -p "$CACHE"
echo "# qwen05b: downloading $HF_ID ($SIZE) -> $CACHE"
python3 - "$HF_ID" "$CACHE" <<'PY'
import sys
from huggingface_hub import snapshot_download
snapshot_download(repo_id=sys.argv[1], local_dir=sys.argv[2],
                  local_dir_use_symlinks=False, resume_download=True)
print("done")
PY

echo "# qwen05b: ready at $CACHE"
