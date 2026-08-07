#!/usr/bin/env bash
# Project SUTRAM — one-click dashboard launcher (macOS / Linux)
set -e
cd "$(dirname "$0")"
if [ ! -d .venv ]; then
  echo "[sutram] first run — creating virtualenv and installing requirements…"
  python3 -m venv .venv
  ./.venv/bin/pip install --upgrade pip -q
  ./.venv/bin/pip install -r requirements.txt
fi
echo "[sutram] starting dashboard at http://127.0.0.1:8000"
exec ./.venv/bin/python webapp/server.py
