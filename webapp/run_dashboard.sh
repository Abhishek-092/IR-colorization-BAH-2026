#!/usr/bin/env bash
# Launch the SUTRAM real-model dashboard. Run from anywhere.
set -e
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PY="./.venv/bin/python"; [ -x "$PY" ] || PY=python3
echo "SUTRAM dashboard  ->  http://127.0.0.1:${PORT:-8000}"
exec "$PY" webapp/server.py
