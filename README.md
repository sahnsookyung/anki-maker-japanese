# Japanese Study Image to Anki

Local-first web app for turning Japanese study-book photos into reviewable Anki card candidates.

The app intentionally creates candidates, not trusted final decks. OCR output is shown with focused visual evidence, confidence, warnings, and edit/approve controls before CSV export.

## Quick Start

One-command local dev:

```bash
cd apps/web
npm install
npm run dev
```

`npm run dev` starts the FastAPI backend on `127.0.0.1:8000` when it is not already running, waits for `/api/health`, then starts Next.js on `127.0.0.1:3000`.

The backend dev script uses `uv` with the repo-pinned Python from `backend/.python-version` (`3.12`). It first tries to install local OCR plus OCR-VL extras, then falls back to base PaddleOCR if only OCR-VL is unavailable. If PaddlePaddle does not publish a wheel for your Python/platform, the backend still starts without local OCR so the UI can run; set `ANKI_MAKER_REQUIRE_LOCAL_OCR=1` if you want startup to fail instead.

Manual backend:

```bash
cd backend
uv sync --python 3.12 --group dev --extra ocr
uv run uvicorn app.main:app --reload --port 8000
```

Manual frontend:

```bash
cd apps/web
npm install
npm run dev
```

Open http://localhost:3000.

Docker fallback:

```bash
docker compose up
```

Docker is the recommended path for machines where local PaddlePaddle wheels are unavailable, especially macOS Intel laptops.

## Workflow

1. Upload one or more workbook photos from the browser.
2. Process pages with the default PaddleOCR button. This runs sequentially to avoid overrunning local memory.
3. Review candidates with the visual evidence panel. Edit fields, approve good cards, and leave red/blocked cards excluded.
4. Export approved cards to Anki CSV.

Page names can be renamed in the UI without renaming files on disk. Re-uploading the same filename replaces the existing page record instead of piling up duplicates.

Each processing attempt is saved as an OCR run. Tokens and generated candidates are scoped to that run, and the page shows one active successful run for review/export. Use the Advanced OCR diagnostics drawer to inspect recent runs or reactivate an older successful run if a rerun performs worse.

## Optional Local OCR / VLM

The backend now uses `uv` instead of handwritten `requirements.txt` files or manual virtualenv setup.

Local OCR is expected to run with PaddleOCR 3.5.0:

```bash
cd backend
uv sync --python 3.12 --group dev --extra ocr
```

PaddlePaddle wheel support is platform-specific. The current lockfile has wheels for common Linux x86_64, Windows x86_64, and macOS Apple Silicon Python versions, but not every Python/platform pair. If `uv sync` reports that `paddlepaddle` has no binary distribution, use the pinned Python 3.12 first, then use Docker if the platform itself is unsupported.

PaddleOCR-VL is available as an optional document parser for comparing Paddle's own VLM-style extraction against the lightweight OCR pipeline:

```bash
cd backend
uv sync --python 3.12 --group dev --extra ocr --extra ocr-vl
```

The backend exposes it at `POST /api/pages/{page_id}/document/parse`. Keep it opt-in while testing memory and output quality.

The UI also has a visually distinct OCR-VL processing path. It is memory-guarded and useful for comparison, but the default production workflow remains base PaddleOCR until OCR-VL performs better on the benchmark set.

Install Google Cloud Vision comparison support only if you want the optional comparator endpoint:

```bash
cd backend
uv sync --python 3.12 --group dev --extra ocr --extra cloud
```

Create a repo-level `.env` and place your Google service-account JSON at the path referenced by `GOOGLE_APPLICATION_CREDENTIALS`. This app uses the Google Cloud SDK credential-file flow rather than a raw `GOOGLE_API_KEY`.

Google Vision cloud calls are disabled by default even when credentials are installed. To use the no-cost free-tier comparison path intentionally, set:

```bash
export GOOGLE_VISION_ALLOW_CLOUD=true
export GOOGLE_VISION_CACHE_ENABLED=true
export GOOGLE_VISION_MONTHLY_CAP=1000
# Optional regional endpoint: us-vision.googleapis.com or eu-vision.googleapis.com
export GOOGLE_VISION_API_ENDPOINT=
```

Uncached successful Google Vision requests are tracked in `backend/usage/google_vision_usage.json`; cached image-hash results under `backend/ocr_cache/google_vision` do not increment the local ledger.

For local VLM cleanup, install the optional Ollama model:

```bash
ollama pull qwen3.5
```

Or run a multimodal `llama.cpp` server with OpenAI-compatible endpoints and set:

```bash
export VLM_CLEANUP_ENABLED=true
export VLM_PROVIDER=llama_cpp
export LLAMA_CPP_BASE_URL=http://localhost:8080
export LLAMA_CPP_MODEL=your-local-vision-model
```

The code defaults to local providers only. No cloud API key is required for the implemented MVP.

The lightweight PaddleOCR path is the default local workflow. It uses Japanese OCR, optional Korean OCR for gloss columns, local glossary normalization, and review-state warnings so output remains editable and auditable. Keep `VLM_CLEANUP_ENABLED=false` for the lowest-memory local workflow, and use either PaddleOCR-VL or the existing local VLM cleanup path only when you want a second-pass cleanup layer.

Optional dictionary validation reads `DICTIONARY_PATH`, a JSON array shaped like:

```json
[{"surface": "学校", "reading": "がっこう"}]
```

## Sample Images

The four `data/evaluation/new upload (category N page).jpg` files are canonical benchmark fixtures. Other uploaded images, processed images, crops, exports, and SQLite databases are local runtime artifacts and can be deleted safely.

The app accepts uploads from the browser. Uploaded pages can be renamed in the UI; renaming changes only display metadata and does not rename image files on disk.

## Export

Approved cards export as UTF-8 CSV for Anki text import. The file includes Anki import headers for comma separation, HTML fields, note type, and tags, followed by these columns:

```csv
notetype,front,back,source_page,source_bbox,confidence,tags
```

Only approved cards are exported by default, and red/blocked cards stay excluded.

The CSV is designed for Anki's text import flow. The app does not generate `.apkg` deck packages, and the target Anki collection must have matching note types such as `jp_vocab_reading` or `jp_spelling_mcq_recall` before import.

Optional local verification against Anki's Python importer:

```bash
cd backend
uv sync --python 3.12 --group dev --extra anki-import
uv run python scripts/verify_anki_csv_import.py exports/example.csv
```

## Verification

Run the fast regression suite after code changes:

```bash
cd backend
uv run pytest -q
uv run python scripts/evaluate_golden.py
uv run python scripts/evaluate_golden.py --from-db --json
uv run python scripts/evaluate_golden.py --from-db --run-id run_... --json
uv run python scripts/benchmark_ocr_modes.py --include-vl --include-google-vision --json
uv run python -m compileall app scripts

cd ../apps/web
npx playwright install chromium
npm run test:coverage
npm run test:e2e
npm run lint
npm run build
```

Run OCR mode benchmarks locally when comparing extraction quality or resource usage:

```bash
cd backend
uv run python scripts/benchmark_ocr_modes.py --json
uv run python scripts/benchmark_ocr_modes.py --engine all --vl-limit 1 --worker-max-rss-mb 4096 --json
```

The Playwright e2e smoke test fails on app-owned browser runtime errors, while ignoring external browser-extension noise such as `Extension context invalidated` from injected `content.js`.

## Runtime Cleanup

Generated uploads, processed images, crops, exports, and SQLite databases are disposable local state:

```bash
find backend/uploads backend/processed backend/crops backend/exports -type f ! -name .gitkeep -delete
find backend -maxdepth 1 -type f -name "*.db" -delete
```
