# OCR Benchmark Plan

This repo has two different OCR questions:

1. Can the default local OCR extraction pipeline produce correct Anki candidates?
2. Can PaddleOCR-VL recover more useful page text/layout than the base OCR pipeline?

The first question is measured by `backend/scripts/evaluate_golden.py`. The second is measured by `backend/scripts/benchmark_ocr_modes.py`.

## Model Choice

Use the measured production default for normal extraction:

```env
PADDLE_OCR_TEXT_DETECTION_MODEL_NAME=PP-OCRv3_mobile_det
PADDLE_OCR_TEXT_RECOGNITION_MODEL_NAME=japan_PP-OCRv3_mobile_rec
PADDLE_OCR_KOREAN_TEXT_DETECTION_MODEL_NAME=PP-OCRv5_mobile_det
PADDLE_OCR_KOREAN_TEXT_RECOGNITION_MODEL_NAME=korean_PP-OCRv5_mobile_rec
```

Use PP-OCRv5 mobile models as a candidate comparison before trying server models:

```env
PADDLE_OCR_TEXT_DETECTION_MODEL_NAME=PP-OCRv5_mobile_det
PADDLE_OCR_TEXT_RECOGNITION_MODEL_NAME=PP-OCRv5_mobile_rec
PADDLE_OCR_KOREAN_TEXT_DETECTION_MODEL_NAME=PP-OCRv5_mobile_det
PADDLE_OCR_KOREAN_TEXT_RECOGNITION_MODEL_NAME=korean_PP-OCRv5_mobile_rec
```

Rationale:

- PaddleOCR documents PP-OCRv5 as the newer recognition family for Chinese, Traditional Chinese, English, Japanese, handwriting, vertical text, pinyin, and rare characters.
- The mobile variant is the safer local default because it has lower expected memory use than the server variant.
- On the current golden pages, Japanese-specific PP-OCRv3 recognition scored higher than generic PP-OCRv5 recognition, so production should stay on the measured-better default until PP-OCRv5 extraction is improved.
- Korean glosses should remain a separate Korean recognition pass because the Korean PP-OCRv5 model explicitly supports Korean, English, and numeric text recognition.
- Server models are worth testing later only if mobile accuracy is insufficient and local memory headroom is comfortable.

## PaddleOCR-VL Comparison

PaddleOCR-VL is not a drop-in replacement for the current card extractor. It returns document blocks/markdown-style text, while the app needs structured `surface`, `reading`, `meaning_ko`, `question_no`, `target`, choices, and answer provenance.

The comparison benchmark therefore measures:

- Base OCR extraction accuracy against golden rows/questions.
- PaddleOCR-VL text coverage against the same golden expected fields.
- Wall time, user/system CPU time, CPU percent relative to one core, peak RSS, and RSS samples.
- NPU/GPU fields are included in the JSON output; they are marked unavailable unless a local collector is configured.

Run base-only comparison:

```bash
cd backend
uv run python scripts/benchmark_ocr_modes.py
```

By default, each page runs in a separate subprocess and the script reports memory samples for each page. This is intentional: PaddleOCR providers keep runtime state in memory, and process isolation gives the operating system a clean opportunity to reclaim it after every page.

Use JSON output when comparing runs:

```bash
cd backend
uv run python scripts/benchmark_ocr_modes.py --json
```

Run a cautious one-page PaddleOCR-VL comparison:

```bash
cd backend
uv run python scripts/benchmark_ocr_modes.py --include-vl --vl-limit 1
```

Keep `--vl-limit` low at first. PaddleOCR-VL is intentionally opt-in because it is heavier than the base OCR path, and this project has already seen local memory pressure from vision-model experiments.

For memory debugging in the app server, temporarily set:

```env
OCR_PROVIDER_CACHE_ENABLED=false
```

Leave caching enabled during normal UI use; disabling it trades memory observability for slower repeated processing.

## Latest Local Benchmark Notes

On April 28, 2026, subprocess-isolated local runs showed:

- PP-OCRv5 mobile Japanese/general + Korean PP-OCRv5 peaked around 2.7 GB RSS per page and produced golden accuracies of 86.1%, 60.0%, 100.0%, and 0.0%.
- Japanese PP-OCRv3 mobile + Korean PP-OCRv5 peaked around 2.8 GB RSS per page and produced golden accuracies of 100.0%, 50.0%, 100.0%, and 60.0%.
- PaddleOCR-VL dependencies now resolve with `paddlex[ocr]`, but first-run model download for `PaddleOCR-VL-1.5` did not complete during the local attempt, so VL inference metrics are still pending.
