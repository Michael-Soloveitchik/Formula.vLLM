#!/usr/bin/env bash
# Serve the live dashboard. MUST serve from the benchmark root (not web/), because
# web/index.html fetches ../index.json and ../results/all.json — a parent escape
# that an http server rooted at web/ blocks (404). Open the printed URL.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"; cd "$HERE"
PORT="${1:-8000}"
echo "dashboard: http://localhost:${PORT}/web/"
python3 -m http.server "$PORT"
