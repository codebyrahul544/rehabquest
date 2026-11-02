#!/usr/bin/env bash
# RehabQuest launcher for macOS / Linux: creates .venv on first run, then starts the game.
set -e
cd "$(dirname "$0")"
if [ ! -x ".venv/bin/python" ]; then
  echo "Creating virtual environment..."
  PY=$(command -v python3.11 || command -v python3.10 || command -v python3)
  "$PY" -m venv .venv
  .venv/bin/python -m pip install --upgrade pip
  .venv/bin/python -m pip install -r requirements.txt
fi
exec .venv/bin/python main.py "$@"
