@echo off
REM Project SUTRAM - one-click dashboard launcher (Windows)
cd /d "%~dp0"
if not exist .venv (
  echo [sutram] first run - creating virtualenv and installing requirements...
  python -m venv .venv
  .venv\Scripts\python -m pip install --upgrade pip -q
  .venv\Scripts\pip install -r requirements.txt
)
echo [sutram] starting dashboard at http://127.0.0.1:8000
.venv\Scripts\python webapp\server.py
pause
