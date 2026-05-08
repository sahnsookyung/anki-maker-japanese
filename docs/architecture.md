# Architecture

This app is a local-first review pipeline for turning Japanese workbook photos into editable Anki CSV candidates. The core design choice is that OCR output is never treated as final truth: the backend creates candidates with evidence and warnings, and the frontend helps a human approve only the cards that are good enough to export.

## Core OCR Strategy

The repo does not fine-tune OCR models. The benchmarked path uses pretrained PaddleOCR models and gets usable structured output by separating raw recognition from deterministic post-processing:

```text
workbook image
  -> image preprocessing
  -> selected OCR engine
     -> default: PaddleOCR Japanese text detection/recognition
     -> optional: PaddleOCR-VL document blocks kept as block evidence
  -> optional PaddleOCR Korean recognition pass for vocab glosses
  -> page-type classifier
  -> vocab / MCQ extraction heuristics
  -> OCR evidence/provenance checks
  -> reviewable Anki card candidates with visual evidence
```

For the four-page golden benchmark, the OCR-plus-structure pipeline is measured without OCR fine-tuning. Vocab rows are scored only when the extracted surface, reading, and Korean meaning are backed by OCR field evidence, so glossary-filled values do not inflate the benchmark.

Important runtime settings:

```env
OCR_PROVIDER=paddle
OCR_PROVIDER_CACHE_ENABLED=true
PADDLE_OCR_MAX_SIDE_LEN=1600
PADDLE_OCR_USE_DOC_ORIENTATION_CLASSIFY=false
PADDLE_OCR_USE_DOC_UNWARPING=false
PADDLE_OCR_USE_TEXTLINE_ORIENTATION=false
PADDLE_OCR_TEXT_DETECTION_MODEL_NAME=PP-OCRv3_mobile_det
PADDLE_OCR_TEXT_RECOGNITION_MODEL_NAME=japan_PP-OCRv3_mobile_rec
PADDLE_OCR_KOREAN_TEXT_DETECTION_MODEL_NAME=PP-OCRv5_mobile_det
PADDLE_OCR_KOREAN_TEXT_RECOGNITION_MODEL_NAME=korean_PP-OCRv5_mobile_rec
VOCAB_DUAL_OCR_ENABLED=true
KOREAN_GLOSSARY_PATH=backend/data/dictionaries/jlpt_basic_ko.json
VLM_CLEANUP_ENABLED=false
```

Why this matters:

- Japanese and Korean text are recognized with separate OCR passes instead of expecting one recognizer to infer both roles.
- The measured production default uses Japanese-specific PP-OCRv3 recognition for Japanese workbook text and PP-OCRv5 Korean recognition for Korean glosses.
- PaddleOCR emits text, bounding boxes, confidence, and script classes; app code decides what is a row, question, target, choice, answer, or Korean gloss.
- Vocab extraction does not fill missing rows from the local glossary; each benchmarked vocab row must have OCR-backed evidence for surface, reading, and Korean meaning.
- Review state, warnings, and evidence overlays remain part of the product because benchmark accuracy is not a guarantee for arbitrary new pages.
- PaddleOCR-VL is optional and can be run as a card-generation engine, but it remains visually separated from the default PaddleOCR path and is scored honestly in benchmarks.
- OCR-VL document-block preview is separate from OCR-VL processing: previewing blocks does not create, approve, or export cards.
- Experimental OCR profiles and extraction variants are recorded as run metadata and benchmark diagnostics. `accuracy_recovery_v2` adds Japanese region, Korean residual glyph, MCQ prompt-line, and MCQ choice-glyph recovery for benchmark/review inspection only. These variants do not change the safe default unless holdout results pass the promotion gates.
- The backend caches OCR providers by default for UI responsiveness; benchmark scripts run pages in subprocesses so Paddle memory is released after each page.

## System Diagram

```text
┌──────────────────────────────┐
│ Browser / Next.js Web UI     │
│ apps/web                     │
│                              │
│ - Upload pages               │
│ - Rename pages               │
│ - Process with PaddleOCR     │
│ - Optional OCR-VL processing │
│ - Review focused evidence    │
│ - Edit / approve candidates  │
│ - Export approved CSV        │
└───────────────┬──────────────┘
                │ HTTP / JSON
                ▼
┌──────────────────────────────┐
│ FastAPI Backend              │
│ backend/app                  │
│                              │
│ API routes:                  │
│ - /api/pages                 │
│ - /api/pages/upload          │
│ - /api/pages/{id}/process    │
│   ?engine=paddleocr|paddleocr_vl
│ - /api/pages/dedupe          │
│ - /api/pages/{id}/ocr        │
│ - /api/pages/{id}/cards      │
│ - /api/cards/{id}            │
│ - /api/exports/csv           │
└───────┬──────────────┬───────┘
        │              │
        │              │ static files
        │              ▼
        │      ┌──────────────────────┐
        │      │ Runtime File Folders │
        │      │ backend/uploads      │
        │      │ backend/processed    │
        │      │ backend/crops        │
        │      │ backend/exports      │
        │      └──────────────────────┘
        │
        │ SQLite reads/writes
        ▼
┌──────────────────────────────┐
│ SQLite App DB                │
│ backend/app.db               │
│                              │
│ pages                        │
│ ocr_runs                     │
│ ocr_tokens                   │
│ cards                        │
└───────────────┬──────────────┘
                │
                │ processing pipeline
                ▼
┌──────────────────────────────┐
│ Extraction Pipeline          │
│ backend/app/extraction       │
│                              │
│ 1. preprocess image          │
│ 2. OCR engine normalization  │
│ 3. classify page type        │
│ 4. extract vocab / MCQ items │
│ 5. verify evidence provenance│
│ 6. generate card candidates  │
│ 7. store evidence metadata   │
└───────┬──────────────┬───────┘
        │              │
        ▼              ▼
┌──────────────┐   ┌────────────────────┐
│ OCR Engines  │   │ Diagnostics        │
│ PaddleOCR    │   │ Google Vision      │
│ PaddleOCR-VL │   │ VL block preview   │
│ Korean OCR   │   │ Ollama / llama.cpp │
└──────────────┘   └────────────────────┘
```

## Main Data Flow

```text
1. Upload
   Browser sends an image to POST /api/pages/upload.
   Backend writes the original image to backend/uploads.
   Backend creates a Page row with display_name defaulting to the filename stem.

2. Process
   Browser calls POST /api/pages/{page_id}/process.
   The default engine is `paddleocr`; passing `engine=paddleocr_vl` runs the heavier OCR-VL candidate-generation path.
   Backend preprocesses the image into backend/processed.
   Backend creates a new OCR run with engine, image hash, preprocessing metadata, provider config, metrics, warnings, and lifecycle status.
   The selected OCR engine emits normalized OCR tokens with text, bbox, confidence, script_class, and source.
   Vocab-table pages can run a second Korean PaddleOCR pass for gloss columns.
   The classifier decides whether the page is vocab_table, reading_mcq, spelling_mcq, or unknown_review_required.

3. Extract
   For vocabulary pages, the pipeline extracts one candidate per surface/reading/Korean meaning row, plus source_bbox and evidence_tokens.
   For MCQ pages, the pipeline extracts question_no, sentence, target, choices, correct_answer, correct_choice_no, answer_source, source_bbox, evidence_tokens, and token_roles.
   Vocab rows are not supplemented from glossary data; benchmark row accuracy only credits rows whose fields are backed by OCR evidence.
   Experimental recovery variants may add persisted crop/region/glyph OCR tokens before candidate mutation. V2 recovery is ordered after base extraction: v1 Korean recovery, Japanese vocab region recovery, Korean residual glyph recovery, v1 MCQ source rebuild/choice-band recovery, prompt-line recovery, then choice-glyph recovery.
   Residual diagnostics and focus-miss inventories are written after scoring and are never extraction inputs.
   The card generator turns each extracted item into reviewable CardCandidate rows scoped to the OCR run.

4. Review
   Browser fetches Page, OCR tokens, and CardCandidate rows.
   The UI defaults to Focused evidence mode.
   When a card is selected, the evidence stage zooms to evidence_tokens first, then falls back to source_bbox.
   Non-relevant OCR boxes are dimmed instead of removed, so the user keeps page context without seeing all OCR noise.

5. Approve
   The user edits front/back/tags as needed and approves cards.
   Green/yellow/red review state remains separate from approved/pending status.
   Red cards are blocked from normal export.

6. Export
   Browser calls POST /api/exports/csv.
   Backend filters to approved cards by default and excludes red cards.
   Backend writes an Anki-headered CSV into backend/exports and returns a download URL.
```

## Runtime State vs Fixtures

```text
Tracked benchmark fixtures:
  data/evaluation/golden_pages.example.json
  data/evaluation/new upload (category 1 page).jpg
  data/evaluation/new upload (category 2 page).jpg
  data/evaluation/new upload (category 3 page).jpg
  data/evaluation/new upload (category 4 page).jpg

Ignored disposable runtime state:
  backend/uploads/*
  backend/processed/*
  backend/crops/*
  backend/exports/*
  backend/*.db
```

The benchmark fixtures are the regression set. Runtime files are recreated by upload, process, benchmark, and export commands and can be safely deleted when resetting local state.

OCR review persistence treats SQLite as the source of truth for pages, OCR runs, OCR tokens, and card candidates. Processed images are a derived cache for display alignment: on `/api/pages/{page_id}/ocr`, the backend can lazily regenerate a missing processed image from the original upload and hydrate missing image dimensions so old OCR evidence can render after a server restart without rerunning OCR. If regenerated image geometry no longer matches the stored OCR geometry, the backend hides stale evidence boxes and asks for reprocessing rather than showing misleading overlays.

Each processing attempt creates a new `ocr_runs` row. Tokens and generated cards are run-scoped, and `pages.active_ocr_run_id` chooses which successful run the UI shows and exports. This makes reruns safe: a new PaddleOCR or OCR-VL run no longer destroys prior OCR evidence or approved candidates from an older run.

SQLite remains the preferred database for this repo because the workflow is local, single-user, and artifact-heavy. A Postgres migration should wait until there is a real multi-user/server deployment need, such as shared review queues, accounts, remote storage, or concurrent writers.

## Important Modules

```text
apps/web/components/StudyWorkbench.tsx
  Main review UI: upload, page rail, focused evidence, card editing, approval, export.

apps/web/lib/api.ts
  Typed frontend API helpers.

backend/app/api/routes.py
  FastAPI routes for pages, cards, OCR comparison, optional document parse, and export.

backend/app/db/database.py
  SQLite schema, compatibility migration, OCR run history, indexes, foreign-key enforcement, persistence helpers, page card-count summaries.

backend/app/extraction/pipeline.py
  Orchestrates preprocessing, OCR, classification, extraction, card generation, and persistence.

backend/app/extraction/vocab.py
  Vocabulary table extraction and Japanese/Korean field handling.

backend/app/extraction/mcq.py
  MCQ block extraction, answer attachment, evidence_tokens, and token_roles.

backend/app/extraction/cards.py
  Converts extracted source objects into editable Anki card candidates.

backend/app/ocr/*
  OCR provider abstraction and PaddleOCR / Google Vision / Tesseract integrations.

backend/app/vision/paddle_ocr_vl.py
  Optional PaddleOCR-VL document parser used by the OCR-VL processing and preview paths, not the default hot path.

backend/scripts/evaluate_golden.py
  Runs the golden benchmark against the four canonical fixture pages.
```

## Review Evidence Model

Every card candidate can carry visual evidence:

```text
CardCandidate
  source_bbox
    Region for the extracted source row/question.

  source.evidence_tokens
    OCR token ids most relevant to this card.

  source.token_roles
    Optional MCQ roles:
      sentence
      target
      choice
      answer

  source.field_evidence
    Field-level evidence for editable facts:
      surface / reading / meaning_ko
      question_no / sentence / target
      choice_1..choice_4 / correct_answer / answer_source
```

The UI highlight priority is:

```text
1. Highlight the selected field's source.field_evidence token ids or bbox.
2. Fall back to source.evidence_tokens for the selected candidate.
3. Fall back to source_bbox for the selected candidate.
4. If neither exists, show a no-focused-evidence message and allow All OCR mode for debugging.
```

## Field OCR Correction

Manual correction is field-first. When a user selects a candidate field in the review UI, the app focuses the matching evidence box. The user can drag or resize that box, preview OCR for only that field, then apply the suggestion to the editable source data.

```text
selected card field
  -> source.field_evidence bbox
  -> POST /api/cards/{card_id}/field-ocr/preview
  -> supervised CropOcrWorker process
  -> PaddleOCR on the crop
  -> suggested source patch
  -> user applies via PATCH /api/cards/{card_id}
```

The crop OCR worker is deliberately outside the FastAPI process. It starts lazily, stays warm for repeated crop previews, and offloads after an idle timeout or memory-limit breach. Full-page OCR, OCR comparison, PaddleOCR-VL parsing, and crop OCR share a runtime lock so the app does not accidentally run multiple heavy OCR jobs at once.

Default crop field providers:

```text
meaning_ko -> paddle_korean
all other fields -> OCR_PROVIDER
```

## OCR Engines And Diagnostics

The default local processing path is PaddleOCR plus deterministic extraction. OCR-VL is supported in two distinct ways:

```text
Process with OCR-VL
  Calls /api/pages/{page_id}/process?engine=paddleocr_vl.
  Keeps PaddleOCR-VL output as document-block evidence.
  Generates review candidates through the same extractor/card pipeline.

Preview OCR-VL document blocks
  Calls /api/pages/{page_id}/document/parse.
  Shows raw document-parser blocks for diagnostics.
  Does not create or approve Anki cards.
```

Other optional tools are intentionally separate:

```text
Google Cloud Vision
  Used through /api/pages/{page_id}/ocr/compare.
  Compares cloud OCR tokens against stored local OCR tokens.
  Cloud calls are disabled unless GOOGLE_VISION_ALLOW_CLOUD=true.
  Cached image-hash results do not consume the local monthly quota ledger.

Ollama / llama.cpp
  Used only when VLM_CLEANUP_ENABLED=true.
  Intended as a second-pass cleanup layer, not the primary OCR path.
```

This separation keeps the normal local workflow lower-memory and easier to debug.

## Benchmark Semantics

MCQ benchmark output now reports two useful scores:

```text
semantic_accuracy
  Whether the generated Anki card has the correct target, answer, and correct choice number.

source_field_accuracy
  Whether the OCR source details also match the human transcript:
  sentence, target, choices, correct answer, and correct choice number.
```

This is deliberate. A card can be semantically correct and exportable while still carrying noisy source fields that deserve review in the UI.

## Verification Loop

Use these checks after meaningful changes:

```bash
cd backend
uv run pytest -q
uv run python scripts/evaluate_golden.py
uv run python scripts/evaluate_golden.py --from-db --json
uv run python scripts/benchmark_ocr_modes.py --json
uv run python scripts/benchmark_ocr_modes.py --engine all --vl-limit 1 --worker-max-rss-mb 14336 --json
uv run python -m compileall app scripts

cd ../apps/web
npx playwright install chromium
npm run test:coverage
npm run test:e2e
npm run lint
npm run build
```

Optional Anki import verifier:

```bash
cd backend
uv sync --group dev --extra anki-import
uv run python scripts/verify_anki_csv_import.py backend/exports/example.csv
```

Clean disposable runtime state when needed:

```bash
find backend/uploads backend/processed backend/crops backend/exports -type f ! -name .gitkeep -delete
find backend -maxdepth 1 -type f -name "*.db" -delete
```
