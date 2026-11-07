#!/usr/bin/env bash
# thin entry point: the script lives in tools/ (see README "Repository layout")
exec bash "$(cd "$(dirname "$0")" && pwd)/tools/serve.sh" "$@"
