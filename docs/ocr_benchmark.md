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
- Normalized OCR-token text coverage, which checks whether OCR text contains expected transcript fields before answer-strip or glossary heuristics can rescue the card.
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
uv run python scripts/benchmark_ocr_modes.py --engine all --vl-limit 1 --worker-timeout-seconds 120 --worker-max-rss-mb 14336
```

Keep `--vl-limit` low at first. PaddleOCR-VL is intentionally opt-in because it is heavier than the base OCR path, and this project has already seen local memory pressure from vision-model experiments. The worker memory cap is deliberate: a failed VL run should become a benchmark record, not a laptop crash.

In the app server, base PaddleOCR page workers use `OCR_PAGE_WORKER_MAX_RSS_MB` and OCR-VL page/document workers use `OCR_VL_PAGE_WORKER_MAX_RSS_MB`. The default OCR-VL cap is higher because local PaddleOCR-VL 1.5 loads above the base OCR cap on this machine and one smoke parse sampled around 11.8 GB RSS; lower it if you need stricter safety, or raise it only when you have enough RAM headroom.

Run golden scoring directly:

```bash
cd backend
uv run python scripts/evaluate_golden.py --engine paddleocr --json
uv run python scripts/evaluate_golden.py --engine paddleocr_vl --json
uv run python scripts/evaluate_golden.py --from-db --json
uv run python scripts/evaluate_golden.py --from-db --run-id run_... --json
uv run python scripts/benchmark_ocr_modes.py --include-vl --include-google-vision --json
```

Fresh evaluation now uses the same isolated per-page worker path for every engine, so PaddleOCR and OCR-VL are scored under the same subprocess/DB/processed-image isolation rules. Use `--from-db` after processing pages from the browser to verify that UI-generated candidates score the same way as CLI-generated candidates. Because `--from-db` evaluates whatever OCR run is currently active for a page, it labels results as `persisted_paddleocr`, `persisted_paddleocr_vl`, or `persisted_unknown` instead of pretending a fresh engine run happened. Use `--run-id` when you need to score a specific persisted run.

## Google Vision Guardrails

Google Vision comparison is available only as an explicit diagnostic. It does not create, approve, or export cards. Cloud calls are disabled by default:

```env
GOOGLE_VISION_ALLOW_CLOUD=false
GOOGLE_VISION_CACHE_ENABLED=true
GOOGLE_VISION_MONTHLY_CAP=1000
GOOGLE_VISION_API_ENDPOINT=
```

Set `GOOGLE_VISION_ALLOW_CLOUD=true` only when you have configured `GOOGLE_APPLICATION_CREDENTIALS` and intentionally want to spend a free-tier request. Leave `GOOGLE_VISION_API_ENDPOINT` empty for Google's default endpoint, or set it to a regional endpoint such as `us-vision.googleapis.com` / `eu-vision.googleapis.com` when data-location controls matter. Results are cached by processed-image hash under `backend/ocr_cache/google_vision`, and uncached successful requests are recorded in `backend/usage/google_vision_usage.json` so the app can enforce the local monthly cap.

`benchmark_ocr_modes.py --include-google-vision` reports Google Vision text coverage and resource timing alongside PaddleOCR/PaddleOCR-VL. It remains diagnostic-only: it does not generate Anki candidates, and with the default guardrail settings it returns a configuration warning instead of making an uncached cloud call.

For memory debugging in the app server, temporarily set:

```env
OCR_PROVIDER_CACHE_ENABLED=false
```

Leave caching enabled during normal UI use; disabling it trades memory observability for slower repeated processing.

## Latest Local Benchmark Notes

On May 3, 2026, guarded subprocess-isolated local runs showed:

- The measured default, Japanese PP-OCRv3 mobile + Korean PP-OCRv5, reached 100% vocab row accuracy and 100% MCQ semantic accuracy on all four golden pages.
- MCQ source-field accuracy is stricter than card accuracy and currently exposes remaining OCR/layout roughness on the MCQ pages; the two MCQ pages scored 84% source-field accuracy.
- Normalized OCR-token text coverage for PaddleOCR was 278/320 expected fields (86.9%), which is the clearest signal that 100% card accuracy is not raw OCR accuracy.
- Base PaddleOCR page workers peaked around 2.8 GB RSS per page on the current machine.
- PaddleOCR-VL generated no cards on the four golden pages through the current shared extraction path, with 24/320 normalized token-text fields matched (7.5%). It peaked around 12.1 GB RSS per page under the 14336 MB guardrail.

Interpret those notes as a local hardware/resource snapshot, not a universal model limit. If you raise `--worker-max-rss-mb`, run one page at a time and keep the JSON metrics so the result is comparable.
