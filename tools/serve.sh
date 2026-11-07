#!/usr/bin/env bash
# Serve the live dashboard. MUST serve from the benchmark root (not web/), because
# web/index.html fetches ../index.json, ../results/manifest.json and ../data/pr_<n>/... — a
# parent escape that an http server rooted at web/ blocks (404). serve.py adds gzip so the
# JSON payloads are small enough for a slow link (Tailscale Funnel).
#
# On the GPU box this runs as systemd `formula-dashboard.service` and is mounted
# publicly at https://<tailnet-host>:10000/formula/web/ via
#   tailscale funnel --bg --https=10000 --set-path /formula http://127.0.0.1:8000
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"; cd "$HERE/.."
PORT="${1:-8000}"
exec python3 "$HERE/serve.py" "$PORT" --bind "${BIND:-127.0.0.1}"
