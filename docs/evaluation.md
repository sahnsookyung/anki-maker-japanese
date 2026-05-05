# Evaluation Set

Use a small golden set to measure whether OCR and extraction changes are actually improving semantic accuracy.

Start with 10-20 rows per page, not the whole page. Pick rows that are visually clear and cover both columns, multiple kana sections, simple Korean glosses, and at least a few multi-word meanings.

## What The Benchmark Measures

The strict benchmark path uses pretrained PaddleOCR plus deterministic extraction code. There is no OCR model fine-tuning, and vocab row accuracy no longer gives credit for rows filled from local glossary data.

The default evaluated path is:

```text
image preprocessing
  -> selected OCR engine
     -> default: PaddleOCR Japanese OCR
     -> optional: PaddleOCR-VL normalized document blocks
  -> optional PaddleOCR Korean OCR for vocab glosses when using base OCR
  -> page classification
  -> vocab / MCQ extraction
  -> OCR-evidence provenance checks
  -> golden JSON comparison
```

PaddleOCR is still the production default because it is faster and currently scores better overall locally. PaddleOCR-VL is now supported as a real processing engine through the same downstream card interfaces, but its visual evidence is document-block evidence rather than word-token evidence and it is benchmarked honestly. Google Cloud Vision, Ollama, and llama.cpp remain diagnostics or optional comparison paths; they do not create or approve cards.

Interpret the benchmark as regression coverage for the four canonical workbook pages, not as a claim that arbitrary new pages will be perfect. Vocab row accuracy is intentionally strict: surface, reading, and Korean meaning must come from live OCR-backed field evidence whose text supports the extracted value, so glossary-filled, stale, or manually inferred values do not inflate OCR accuracy. Vocab results also report layout recall and per-field surface/reading/meaning accuracy to show whether failures come from row detection or text recognition. New uploads still need focused evidence review, warnings, and approval before export.

## Format

The starter file is:

```text
data/evaluation/golden_pages.example.json
```

The current benchmark set covers four canonical fixture images:

- `data/evaluation/new upload (category 1 page).jpg`
- `data/evaluation/new upload (category 2 page).jpg`
- `data/evaluation/new upload (category 3 page).jpg`
- `data/evaluation/new upload (category 4 page).jpg`

Do not treat generated uploads, processed images, crops, exports, or SQLite databases as benchmark fixtures. Those files are disposable runtime state and are intentionally ignored by Git.

Each page entry has:

- `image_path`: local path relative to the repo root.
- `category`: source material category, such as `vocab_table`, `reading_mcq`, or `spelling_mcq`.
- `expected_page_type`: classifier target.
- `language_columns`: expected script/language role for each field.
- `expected_rows`: manually transcribed truth rows.
- `scoring`: matching policy for a lightweight evaluator.

Each vocab row should include:

```json
{
  "row_id": "new-cat1-a-001",
  "section": "あ",
  "column": "left",
  "surface": "間",
  "reading": "あいだ",
  "meaning_ko": "사이"
}
```

## What To Annotate

For vocabulary pages, annotate:

- `surface`: kanji/kana written form exactly as printed.
- `reading`: kana reading exactly as printed.
- `meaning_ko`: Korean gloss exactly enough to identify the meaning.
- `column`: `left` or `right`.
- `section`: visible kana section marker when available.

For MCQ pages, annotate:

- `question_no`
- `sentence`
- `target`
- `choices`
- `correct_answer`
- `correct_choice_no`
- `answer_source`

For category stubs, keep `stub: true` until the page has a real image and manually transcribed expected values. Stubs should not be counted as passing or failing benchmark rows.

## Runtime Cleanup

It is safe to delete local runtime artifacts when resetting the app:

```bash
find backend/uploads backend/processed backend/crops backend/exports -type f ! -name .gitkeep -delete
find backend -maxdepth 1 -type f -name "*.db" -delete
```

The browser upload flow and evaluator will recreate runtime files as needed.

## Metrics To Track

Track these separately:

- Page classification accuracy: did the page become `vocab_table`, `reading_mcq`, etc.
- Row recall: how many golden rows were found.
- Field exact match: `surface` and `reading`.
- Korean meaning match: exact or contains normalized Korean text.
- Script confusion: Japanese fields containing Hangul, Korean fields missing Hangul, or Korean fields dominated by kanji/kana.
- Card generation count: how many usable card candidates were emitted.
- MCQ semantic accuracy: did the generated card capture the expected `target`, `correct_answer`, and `correct_choice_no`.
- MCQ source-field accuracy: did extraction preserve the visible `sentence`, `target`, all four `choices`, answer, and answer number.

The most useful first score is row-level accuracy:

```text
row is correct if surface exact + reading exact + Korean meaning contains expected gloss
```

For MCQ pages, the most useful product score is semantic accuracy, while source-field accuracy is the stricter OCR/layout quality score. A page can produce correct Anki cards while still showing source-field misses that deserve UI review.

Run the current evaluator with:

```bash
cd backend
uv run python scripts/evaluate_golden.py
```

For machine-readable output:

```bash
cd backend
uv run python scripts/evaluate_golden.py --json
```

Run a specific OCR engine:

```bash
cd backend
uv run python scripts/evaluate_golden.py --engine paddleocr --json
uv run python scripts/evaluate_golden.py --engine paddleocr_vl --json
uv run python scripts/evaluate_golden.py --engine all --json
```

Score candidates that were created through the browser and persisted in SQLite:

```bash
cd backend
uv run python scripts/evaluate_golden.py --from-db --json
```

This uses the same golden evaluator as the CLI path, so UI-created cards and benchmark-created cards are compared with the same scoring rules.

## Why This Helps

This lets us compare:

- Lightweight PaddleOCR full-page OCR.
- Column/row-cropped PaddleOCR.
- PaddleOCR-VL document parsing.
- Google Cloud Vision comparison.
- Optional Qwen/Ollama cleanup.

The key is to score all approaches against the same rows, so we can tell whether the next bottleneck is preprocessing, OCR language recognition, layout grouping, or semantic cleanup.
