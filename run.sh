#!/usr/bin/env bash
set -euo pipefail

echo "=== Installing dependencies ==="
pip install -r requirements.txt

echo "=== Starting Llama Trading Bot ==="
python bot.py
