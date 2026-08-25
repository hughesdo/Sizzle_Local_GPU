#!/usr/bin/env bash
# Sizzle - SERVE IT TO A FRIEND over a Cloudflare quick tunnel.
# Opens a temporary, random public URL (https://something.trycloudflare.com);
# hand it to your friend. Ctrl-C stops the app and the tunnel together.
#
# Every render runs on YOUR GPU, one at a time (the app holds a global
# single-render lock). It costs no API credits - just your card being busy.
#
# Requires cloudflared on PATH:
#   https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/downloads/
set -euo pipefail

HOST="${SIZZLE_HOST:-127.0.0.1}"
PORT="${SIZZLE_PORT:-8000}"

PY="python"
[ -x ".venv/bin/python" ] && PY=".venv/bin/python"

if ! command -v cloudflared >/dev/null; then
  echo "!! cloudflared not found. Install it, then re-run:"
  echo "   https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/downloads/"
  exit 1
fi

cleanup(){ kill 0 2>/dev/null || true; }
trap cleanup EXIT INT TERM

echo ">> starting Sizzle server on http://${HOST}:${PORT}"
"$PY" -m uvicorn backend.app:app --host "${HOST}" --port "${PORT}" &
sleep 2

echo ">> opening Cloudflare quick tunnel - share the https://...trycloudflare.com URL below"
cloudflared tunnel --url "http://${HOST}:${PORT}"
