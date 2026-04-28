#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../backend"
uv sync --no-build --group dev --extra ocr
uv run --no-build uvicorn app.main:app --reload --port 8000
