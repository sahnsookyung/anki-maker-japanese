#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../backend"

backend_python="${ANKI_MAKER_BACKEND_PYTHON:-3.12}"
backend_host="${BACKEND_HOST:-127.0.0.1}"
backend_port="${BACKEND_PORT:-8000}"
sync_log="$(mktemp)"
cleanup() {
  rm -f "$sync_log"
}
trap cleanup EXIT

sync_backend() {
  : > "$sync_log"
  uv sync --python "$backend_python" "$@" 2>&1 | tee "$sync_log"
}

if ! sync_backend --group dev --extra ocr --extra ocr-vl; then
  if grep -Eiq "paddlepaddle|no binary distribution|can't be installed because it is marked as.*no-build" "$sync_log"; then
    cat <<'EOF'

Full local OCR dependencies could not be installed for this Python/platform.
Retrying with the base PaddleOCR engine only.

Recommended fixes:
- Use the uv-managed backend Python from backend/.python-version, then rerun npm run dev.
- On macOS Intel, use Docker or another machine with a supported PaddlePaddle wheel.
- Set ANKI_MAKER_REQUIRE_LOCAL_OCR=1 to make this startup fail instead of falling back.

EOF
    if sync_backend --group dev --extra ocr; then
      cat <<'EOF'

Started with base PaddleOCR support only. OCR-VL actions will be unavailable until its optional dependencies install cleanly.

EOF
    elif grep -Eiq "paddlepaddle|no binary distribution|can't be installed because it is marked as.*no-build" "$sync_log"; then
      cat <<'EOF'

Base PaddleOCR dependencies also could not be installed for this Python/platform.
The web app will still start, but local OCR actions will be unavailable until PaddlePaddle is installed.

EOF
      if [[ "${ANKI_MAKER_REQUIRE_LOCAL_OCR:-}" == "1" ]]; then
        exit 1
      fi
      sync_backend --group dev
    else
      exit 1
    fi
  else
    exit 1
  fi
fi

uv run uvicorn app.main:app --reload --host "$backend_host" --port "$backend_port"
