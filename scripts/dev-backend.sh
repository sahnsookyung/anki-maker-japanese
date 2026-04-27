#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../backend"
uv sync --group dev --extra ocr
uv run uvicorn app.main:app --reload --port 8000
