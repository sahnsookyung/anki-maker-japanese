# Architecture

This app is a local-first review pipeline for turning Japanese workbook photos into editable Anki CSV candidates. The core design choice is that OCR output is never treated as final truth: the backend creates candidates with evidence and warnings, and the frontend helps a human approve only the cards that are good enough to export.

## Core OCR Strategy

The repo does not fine-tune OCR models. The benchmarked path uses pretrained PaddleOCR models and gets usable structured output by separating raw recognition from deterministic post-processing:

```text
workbook image
  -> image preprocessing
  -> selected OCR engine
     -> default: PaddleOCR Japanese text detection/recognition
     -> optional: PaddleOCR-VL document blocks converted to normalized OCR tokens
  -> optional PaddleOCR Korean recognition pass for vocab glosses
  -> page-type classifier
  -> vocab / MCQ extraction heuristics
  -> local glossary normalization
  -> reviewable Anki card candidates with visual evidence
```

For the four-page golden benchmark, the 100% match came from this OCR-plus-structure pipeline, not from PaddleOCR-VL and not from OCR fine-tuning.

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
- The local Korean glossary normalizes known workbook vocabulary when OCR is partial or noisy.
- Review state, warnings, and evidence overlays remain part of the product because benchmark accuracy is not a guarantee for arbitrary new pages.
- PaddleOCR-VL is optional and can be run as a card-generation engine, but it remains visually separated from the default PaddleOCR path and is scored honestly in benchmarks.
- OCR-VL diagnostics are still separate from processing: document-block preview does not create, approve, or export cards.
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
│ 5. normalize via glossary    │
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
   The selected OCR engine emits normalized OCR tokens with text, bbox, confidence, script_class, and source.
   Vocab-table pages can run a second Korean PaddleOCR pass for gloss columns.
   The classifier decides whether the page is vocab_table, reading_mcq, spelling_mcq, or unknown_review_required.

3. Extract
   For vocabulary pages, the pipeline extracts surface, reading, Korean meaning, source_bbox, and evidence_tokens.
   For MCQ pages, the pipeline extracts question_no, sentence, target, choices, correct_answer, correct_choice_no, answer_source, source_bbox, evidence_tokens, and token_roles.
   The Korean glossary can normalize known vocab/answers where OCR is noisy, with warnings when normalization affects the result.
   The card generator turns each extracted item into reviewable CardCandidate rows.

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

## Important Modules

```text
apps/web/components/StudyWorkbench.tsx
  Main review UI: upload, page rail, focused evidence, card editing, approval, export.

apps/web/lib/api.ts
  Typed frontend API helpers.

backend/app/api/routes.py
  FastAPI routes for pages, cards, OCR comparison, optional document parse, and export.

backend/app/db/database.py
  SQLite schema, compatibility migration, persistence helpers, page card-count summaries.

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
  Optional PaddleOCR-VL document parser endpoint for comparison, not the default hot path.

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
  Converts PaddleOCR-VL document blocks/markdown into normalized OCR tokens.
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
uv run python scripts/benchmark_ocr_modes.py --engine all --vl-limit 1 --worker-max-rss-mb 4096 --json
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
