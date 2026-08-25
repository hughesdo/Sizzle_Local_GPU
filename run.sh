#!/usr/bin/env bash
# Sizzle - run LOCALLY at http://127.0.0.1:8000, rendering on this box's GPU.
# Nothing is exposed to the internet and nothing is uploaded anywhere.
# To share it with a friend over a Cloudflare tunnel, use ./host.sh instead.
# (Windows: use startup.bat / host.bat instead of these.)
set -euo pipefail

HOST="${SIZZLE_HOST:-127.0.0.1}"
PORT="${SIZZLE_PORT:-8000}"

# Prefer the project venv if one exists.
PY="python"
[ -x ".venv/bin/python" ] && PY=".venv/bin/python"

echo ">> starting Sizzle on http://${HOST}:${PORT}  (Ctrl-C to stop)"
exec "$PY" -m uvicorn backend.app:app --host "${HOST}" --port "${PORT}"
