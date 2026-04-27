# Implementation Review Against Plan

This is a living checklist for `japanese_study_image_to_anki_plan.md`.

## Implemented

- Monorepo skeleton with `apps/web` Next.js frontend and `backend` FastAPI backend.
- SQLite-backed uploads, pages, OCR tokens, and card candidates.
- Image upload, original/processed image serving, basic preprocessing, page contour crop, perspective correction, contrast normalization, and warnings.
- OCR provider interface with PaddleOCR, Tesseract fallback, and optional Google Cloud Vision provider.
- OCR comparison endpoint for comparing stored local OCR against Google Cloud Vision or another configured provider.
- OCR token bounding boxes, script classification, token storage, and frontend SVG overlay.
- Content-based page classifier with script/question/choice/checkbox features.
- Initial vocabulary row extraction and vocabulary card generation.
- Initial MCQ block extraction, answer-strip parser, answer attachment, and MCQ card generation.
- `uv`-managed backend project with `pyproject.toml`, lockfile workflow, and `PaddleOCR 3.5.0` as the explicit OCR target.
- Optional PaddleOCR-VL document parser endpoint for testing Paddle's own document VLM pipeline separately from the lightweight OCR hot path.
- Local VLM client adapters for Ollama and OpenAI-compatible `llama.cpp`.
- Official Ollama `qwen3.5` configured as an optional local VLM target.
- Optional VLM cleanup pipeline for row/question crops, gated by `VLM_CLEANUP_ENABLED`, with JSON/evidence-token enforcement.
- Dictionary validation hook through `DICTIONARY_PATH`.
- Review UI for editing card front/back/tags, approving candidates, warnings, and TSV export.
- Basic tests for script classification, answer parsing, and TSV escaping.

## Partially Implemented

- Page preprocessing does simple perspective correction; full curved-page dewarping and coordinate mapping back to original image are not done.
- Vocabulary extraction is heuristic and line-based; robust checkbox detection, horizontal-rule extraction, and true two-column segmentation need more work.
- PaddleOCR-only extraction is now memory-safe on the supplied sample images, but OCR-only vocab extraction still fails to recover Korean glosses reliably enough to emit card candidates from those pages.
- MCQ extraction is heuristic; underline detection is not yet OpenCV-based and target selection is guessed from script class.
- Answer strip parsing uses bottom-token text; gray band detection and OCR-only recropping of the answer strip are not fully implemented.
- Dictionary validation supports a small JSON dictionary path; full JMdict/KANJIDIC2 parsing and multiple-reading diagnostics are not implemented.
- Review UI supports editing/approval/export, but not token-click target selection, drag selection, merge/split rows, batch approve green, or filters.
- Deduplication is implemented inside vocab extraction only, not as global card/source dedupe across pages.

## Not Yet Implemented

- Golden image tests from the supplied samples.
- `.apkg` generation or AnkiConnect push.
- Media crop inclusion in exported cards.
- SQLAlchemy/SQLModel models; the current MVP uses direct SQLite.
- Tailwind/shadcn; the frontend uses plain CSS.

## Added Beyond Checked-In Plan

- Optional Google Cloud Vision OCR provider and `/api/pages/{page_id}/ocr/compare` endpoint. The checked-in plan did not mention Google Cloud Vision, but the app now supports using it as a comparator or explicit OCR provider.
