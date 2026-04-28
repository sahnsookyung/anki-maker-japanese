# Architecture

This app is a local-first review pipeline for turning Japanese workbook photos into editable Anki TSV candidates. The core design choice is that OCR output is never treated as final truth: the backend creates candidates with evidence and warnings, and the frontend helps a human approve only the cards that are good enough to export.

## Core OCR Strategy

The repo does not fine-tune OCR models. The benchmarked path uses pretrained PaddleOCR models and gets usable structured output by separating raw recognition from deterministic post-processing:

```text
workbook image
  -> image preprocessing
  -> PaddleOCR Japanese text detection/recognition
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
- PaddleOCR-VL is intentionally optional and exposed as a document-parse comparison endpoint, not the default card-generation path.
- The backend caches OCR providers by default for UI responsiveness; benchmark scripts run pages in subprocesses so Paddle memory is released after each page.

## System Diagram

```text
┌──────────────────────────────┐
│ Browser / Next.js Web UI     │
│ apps/web                     │
│                              │
│ - Upload pages               │
│ - Rename pages               │
│ - Process pages              │
│ - Review focused evidence    │
│ - Edit / approve candidates  │
│ - Export approved TSV        │
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
│ - /api/pages/{id}/ocr        │
│ - /api/pages/{id}/cards      │
│ - /api/cards/{id}            │
│ - /api/exports/tsv           │
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
│ 2. PaddleOCR JP/KR OCR       │
│ 3. classify page type        │
│ 4. extract vocab / MCQ items │
│ 5. normalize via glossary    │
│ 6. generate card candidates  │
│ 7. store evidence metadata   │
└───────┬──────────────┬───────┘
        │              │
        ▼              ▼
┌──────────────┐   ┌────────────────────┐
│ Local OCR    │   │ Optional Comparers │
│ PaddleOCR    │   │ Google Vision      │
│ Japanese OCR │   │ PaddleOCR-VL       │
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
   Backend preprocesses the image into backend/processed.
   PaddleOCR reads the processed image and emits OCR tokens with text, bbox, confidence, script_class, and source.
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
   Browser calls POST /api/exports/tsv.
   Backend filters to approved cards by default and excludes red cards.
   Backend writes a TSV into backend/exports and returns a download URL.
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
```

The UI highlight priority is:

```text
1. Highlight token ids in source.evidence_tokens.
2. If token ids are unavailable, highlight source_bbox.
3. If neither exists, show a no-focused-evidence message and allow All OCR mode for debugging.
```

## Optional Comparison Paths

The default local processing path is PaddleOCR plus deterministic extraction. Optional tools are intentionally separate:

```text
Google Cloud Vision
  Used through /api/pages/{page_id}/ocr/compare.
  Compares cloud OCR tokens against stored local OCR tokens.

PaddleOCR-VL
  Used through /api/pages/{page_id}/document/parse.
  Gives a document-parser view for comparison and future experiments.

Ollama / llama.cpp
  Used only when VLM_CLEANUP_ENABLED=true.
  Intended as a second-pass cleanup layer, not the primary OCR path.
```

This separation keeps the normal local workflow lower-memory and easier to debug.

## Verification Loop

Use these checks after meaningful changes:

```bash
cd backend
uv run pytest -q
uv run python scripts/evaluate_golden.py
uv run python -m compileall app scripts

cd ../apps/web
npm run lint
npm run build
```

Clean disposable runtime state when needed:

```bash
find backend/uploads backend/processed backend/crops backend/exports -type f ! -name .gitkeep -delete
find backend -maxdepth 1 -type f -name "*.db" -delete
```
