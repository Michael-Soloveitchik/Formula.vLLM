#!/usr/bin/env bash
# thin entry point: the script lives in pipeline/ (see README "Repository layout")
exec bash "$(cd "$(dirname "$0")" && pwd)/pipeline/run_one.sh" "$@"
