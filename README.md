# Japanese Study Image to Anki

Local-first web app for turning Japanese study-book photos into reviewable Anki card candidates.

The app intentionally creates candidates, not trusted final decks. OCR/VLM output is shown with evidence, confidence, warnings, and edit/approve controls before TSV export.

## Quick Start

Backend:

```bash
cd backend
uv sync --group dev --extra ocr
uv run uvicorn app.main:app --reload --port 8000
```

Frontend:

```bash
cd apps/web
npm install
npm run dev
```

Open http://localhost:3000.

## Optional Local OCR / VLM

The backend now uses `uv` instead of handwritten `requirements.txt` files or manual virtualenv setup.

Local OCR is expected to run with PaddleOCR 3.5.0:

```bash
cd backend
uv sync --group dev --extra ocr
```

PaddleOCR-VL is available as an optional document parser for comparing Paddle's own VLM-style extraction against the lightweight OCR pipeline:

```bash
cd backend
uv sync --group dev --extra ocr --extra ocr-vl
```

The backend exposes it at `POST /api/pages/{page_id}/document/parse`. Keep it opt-in while testing memory and output quality.

Install Google Cloud Vision comparison support only if you want the optional comparator endpoint:

```bash
cd backend
uv sync --group dev --extra ocr --extra cloud
```

Create a repo-level `.env` and place your Google service-account JSON at the path referenced by `GOOGLE_APPLICATION_CREDENTIALS`. This app uses the Google Cloud SDK credential-file flow rather than a raw `GOOGLE_API_KEY`.

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

At the moment, lightweight PaddleOCR-only mode is stable and sufficient for preprocessing, OCR overlays, and page classification, but it is not yet sufficient to reliably produce vocabulary cards from the supplied sample photos. The Korean gloss column is still degraded badly enough that OCR-only extraction returns zero vocab cards on those pages. Keep `VLM_CLEANUP_ENABLED=false` for the lowest-memory local workflow, and use either PaddleOCR-VL or the existing local VLM cleanup path only when you want a second-pass cleanup layer.

Optional dictionary validation reads `DICTIONARY_PATH`, a JSON array shaped like:

```json
[{"surface": "学校", "reading": "がっこう"}]
```

## Sample Images

The app accepts uploads from the browser. If macOS blocks terminal access to `~/Downloads`, move or copy sample images into a folder the app can read, or upload them through the web UI.

## Export

Approved cards export as UTF-8 TSV with columns:

```tsv
note_type	front	back	source_page	source_bbox	confidence	tags
```
