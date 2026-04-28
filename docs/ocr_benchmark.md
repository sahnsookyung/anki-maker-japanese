# OCR Benchmark Plan

This repo has three different OCR questions:

1. Can the default local OCR extraction pipeline produce correct Anki candidates?
2. Can PaddleOCR-VL feed the same extraction/card-generation pipeline safely and accurately?
3. Can diagnostics explain OCR behavior without mutating review state?

The first two questions are measured by `backend/scripts/evaluate_golden.py` and `backend/scripts/benchmark_ocr_modes.py`. Diagnostics such as Google Vision comparison and OCR-VL block preview remain UI/API tools and do not create or approve Anki cards.

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

PaddleOCR-VL is supported as an optional processing engine, but it is not treated as automatically equivalent to the base OCR path. It returns document blocks/markdown-style text, so the backend normalizes those blocks into the same OCR token shape used by PaddleOCR: text, bbox, confidence, provider, reading order, and optional role.

The benchmark therefore measures:

- Base OCR extraction accuracy against golden rows/questions.
- PaddleOCR-VL extraction accuracy against the same golden rows/questions when the model can run within local resource limits.
- MCQ semantic accuracy for generated Anki cards.
- MCQ source-field accuracy for sentence, target, all four choices, answer, and answer number.
- Wall time, user/system CPU time, CPU percent relative to one core, peak RSS, RSS samples, and worker failures.
- NPU/GPU fields are included in the JSON output; they are marked unavailable unless a local collector is configured.

Run base-only comparison:

```bash
cd backend
uv run python scripts/benchmark_ocr_modes.py --engine paddleocr
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
uv run python scripts/benchmark_ocr_modes.py --engine all --vl-limit 1 --worker-timeout-seconds 120 --worker-max-rss-mb 4096
```

Keep `--vl-limit` low at first. PaddleOCR-VL is intentionally opt-in because it is heavier than the base OCR path, and this project has already seen local memory pressure from vision-model experiments. The worker memory cap is deliberate: a failed VL run should become a benchmark record, not a laptop crash.

Run golden scoring directly:

```bash
cd backend
uv run python scripts/evaluate_golden.py --engine paddleocr --json
uv run python scripts/evaluate_golden.py --engine paddleocr_vl --json
uv run python scripts/evaluate_golden.py --from-db --json
```

Use `--from-db` after processing pages from the browser to verify that UI-generated candidates score the same way as CLI-generated candidates.

For memory debugging in the app server, temporarily set:

```env
OCR_PROVIDER_CACHE_ENABLED=false
```

Leave caching enabled during normal UI use; disabling it trades memory observability for slower repeated processing.

## Latest Local Benchmark Notes

On April 28, 2026, guarded subprocess-isolated local runs showed:

- The measured default, Japanese PP-OCRv3 mobile + Korean PP-OCRv5, reached 100% vocab row accuracy and 100% MCQ semantic accuracy on all four golden pages.
- MCQ source-field accuracy is stricter than card accuracy and currently exposes remaining OCR/layout roughness on the MCQ pages.
- Base PaddleOCR page workers peaked around 2.8 GB RSS per page on the current machine.
- PaddleOCR-VL model loading exceeded a cautious 4096 MB worker cap during the local one-page comparison, so the benchmark reported a safe `worker_failed` result instead of continuing until the machine became unstable.

Interpret those notes as a local hardware/resource snapshot, not a universal model limit. If you raise `--worker-max-rss-mb`, run one page at a time and keep the JSON metrics so the result is comparable.
