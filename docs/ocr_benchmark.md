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

## Experimental OCR Profiles And Variants

Newer models are benchmarked as experimental profiles, not promoted by assumption. Each run records a profile manifest in `ocr_runs.provider_config` and benchmark JSON, including package versions, selected model names, language configuration, runtime platform/device environment, preprocessing metadata, model-cache paths and fingerprints when available, OCR cache status, and the extraction variant.

Candidate profiles:

- `jp_v3_mobile_current`: frozen production control, Japanese PP-OCRv3 mobile plus Korean PP-OCRv5.
- `jp_v3_det_v3_rec`: explicit model-pair alias for the frozen production control.
- `jp_v3_det_v5_rec`: PP-OCRv3 mobile detector with PP-OCRv5 mobile recognizer.
- `jp_v5_det_v3_rec`: PP-OCRv5 mobile detector with Japanese PP-OCRv3 mobile recognizer.
- `jp_v5_det_v5_rec`: explicit latest local model-pair alias for `jp_v5_mobile_general`.
- `jp_v5_mobile_general`: newer PP-OCRv5 mobile detector and recognizer.
- `jp_v5_server_general`: heavier PP-OCRv5 server detector and recognizer.
- `jp_lang_auto`: PaddleOCR `lang="japan"` profile when supported by the installed PaddleOCR package. This can resolve to server-class models, so default profile-matrix runs skip it unless `--include-heavy-profiles` is set.
- `ko_v5_current`: default Korean PP-OCRv5 diagnostic profile.
- `ko_v5_det_v5_rec`: explicit Korean PP-OCRv5 detector plus Korean PP-OCRv5 recognizer alias for the default Korean pass.
- `ko_lang_auto`: optional PaddleOCR `lang="korean"` diagnostic profile. This is heavy/experimental and should be run only as a Korean OCR comparison.
- `google_vision`: optional cloud diagnostic profile, not a default candidate-generation path.

Extraction variants:

- `baseline_current`: frozen current extractor.
- `line_graph_v1`: records reading-order line-graph diagnostics.
- `table_graph_v1`: records row/cell/selection/header diagnostics from the provider-neutral document graph.
- `ranked_rows_v1`: records ranked row-hypothesis scores for evidence quality, script compatibility, alignment, and completeness.
- `crop_confirm_v1`: runs bounded crop OCR on uncertain field boxes and records the result as review-only diagnostics; it does not silently fill benchmark fields.
- `provider_agreement_v1`: runs the configured comparison provider as a diagnostic-only agreement hook; it is never an automatic extraction decision.
- `v5_token_split_v1`: splits merged PP-OCRv5-style vocab tokens into surface/reading candidates with derived boxes.
- `v5_vocab_rows_v1`: records guarded v5-aware vocab row diagnostics.
- `ko_alignment_v1`: records Korean gloss pairing, raw recall, unpaired Hangul tokens, stale evidence, and bbox alignment diagnostics.
- `v5_mcq_v1`: enables PP-OCRv5 MCQ recovery, especially answer-strip and choice parsing recovery.
- `v5_full_adapted_v1`: combines token splitting, guarded row/Korean diagnostics, and MCQ recovery.
- `ko_crop_confirm_v1`: experimental Korean meaning recovery from bounded field crops. Unlike `crop_confirm_v1`, this variant may replace uncertain Korean meaning fields, but only with live `crop_ocr` tokens that persist with the run.
- `ko_region_columns_v1`: experimental Korean meaning-column recovery from derived row/column regions, with duplicate-token and cross-column guards.
- `ko_consensus_v1`: combines full-page, crop, and region OCR signals for Korean meaning recovery and records accepted/rejected alternatives.
- `mcq_source_rebuild_v1`: separates strict OCR-backed MCQ source fields from glossary/answer-strip-assisted semantic fields.
- `mcq_choice_band_ocr_v1`: runs bounded OCR on MCQ choice/answer-strip bands for strict source-field diagnostics.
- `accuracy_recovery_v1`: combines the Korean recovery and MCQ source-field recovery experiments. It remains benchmark-only.
- `residual_diagnostics_v1`: benchmark diagnostic mode only. It writes residual miss diagnostics and contact sheets after scoring and never mutates candidates.
- `jp_region_columns_v1`: benchmark-only Japanese vocab surface/reading recovery from clipped row/column regions. It may create a missing vocab row only when surface, reading, and Korean meaning all have live OCR/image evidence.
- `ko_residual_glyph_v1`: benchmark-only Korean residual glyph recovery for weak `meaning_ko` fields. It accepts only row-owned, meaning-column-owned Hangul or numeric-unit evidence and fails open when local glyph resources are unavailable.
- `mcq_prompt_line_ocr_v1`: benchmark-only clipped prompt-line OCR for strict MCQ `source_fields.sentence`; it updates source fields only and does not infer from semantic choices or glossary text.
- `mcq_choice_glyph_v1`: benchmark-only per-choice glyph recovery for strict spelling-MCQ `source_fields.choices`; it preserves semantic MCQ fields.
- `accuracy_recovery_v2`: combines all `accuracy_recovery_v1` components with Japanese region recovery, Korean residual glyph recovery, MCQ prompt-line OCR, and MCQ choice-glyph recovery. It does not change production defaults.

Run a cautious profile comparison:

```bash
cd backend
uv run python scripts/benchmark_ocr_modes.py --profile-matrix --json
```

`--profile-matrix` skips heavy server profiles by default. Add `--include-heavy-profiles` only when you intentionally want to spend the extra RAM/time.

Run the staged ablation protocol without a combinatorial blast:

```bash
cd backend
uv run python scripts/benchmark_ocr_modes.py --experiment-stage 0 --json
uv run python scripts/benchmark_ocr_modes.py --experiment-stage 1 --json
uv run python scripts/benchmark_ocr_modes.py --experiment-stage 2 --json
uv run python scripts/benchmark_ocr_modes.py --experiment-stage 3 --json
uv run python scripts/benchmark_ocr_modes.py --experiment-stage 4 --json
uv run python scripts/benchmark_ocr_modes.py --experiment-stage 5 --json
```

Stage 0/1 run the four safe local model pairs against `baseline_current`. Stage 2 runs atomic accuracy variants, including Korean crop/region recovery, MCQ source recovery, `jp_region_columns_v1`, `ko_residual_glyph_v1`, `mcq_prompt_line_ocr_v1`, and `mcq_choice_glyph_v1`. Stage 3 runs combined variants including `ko_consensus_v1`, `accuracy_recovery_v1`, and `accuracy_recovery_v2` on the current, hybrid, and v5/v5 candidates. Stage 4 checks `fresh_cli`, `persisted_db`, and `ui_api` parity. Stage 5 is optional heavy/external diagnostics. Unknown staged profile ids are skipped with a warning so long experiment runs can continue. Heavy profiles are still skipped unless `--include-heavy-profiles` is explicit.

Run one explicit latest-model adapted experiment:

```bash
cd backend
uv run python scripts/benchmark_ocr_modes.py \
  --model-profile jp_v5_det_v5_rec \
  --korean-profile ko_v5_current \
  --extraction-variant v5_full_adapted_v1 \
  --json
```

Non-default profiles run through subprocess workers so PaddleOCR imports with the requested model environment. Direct `--in-process` runs are only safe for the frozen default profile unless the profile environment is already active before Python starts.

Promotion rule: without a holdout set beyond the four canonical golden pages, profile/variant results are experimental only. A default change requires pre-registered gates: vocab row accuracy improves by at least 15 percentage points, no page regresses more than 3 points, MCQ does not regress, evidence alignment remains reviewable, and RSS/time stay within the chosen resource budget.

Benchmark modes are intentionally separate:

- `fresh_cli` runs OCR and scores the in-memory result.
- `persisted_db` runs OCR, reloads tokens/cards/document blocks from SQLite, and scores persisted state.
- `ui_api` creates a page record, calls the FastAPI processing route, reloads persisted state, and scores that output.

Use all three before trusting a change:

```bash
cd backend
uv run python scripts/benchmark_ocr_modes.py --benchmark-mode fresh_cli --json
uv run python scripts/benchmark_ocr_modes.py --benchmark-mode persisted_db --json
uv run python scripts/benchmark_ocr_modes.py --benchmark-mode ui_api --json
```

When `--work-dir` or `--keep-work-dir` is set, the benchmark writes visual audit artifacts under `audit/`: one deterministic overlay JSON and one PNG per page/profile/variant. The JSON is the reviewable source of truth; the PNG is a quick visual sanity check for token boxes, document blocks, and candidate regions.

The OCR cache key includes the original-image hash, preprocessing hash, OCR engine/provider, model profile, selected model/config fingerprint, and package versions. It intentionally excludes extraction variant, so graph/ranking/crop diagnostic variants can reuse the same OCR payload while rerunning extraction from cached tokens/document blocks. Benchmark cache hits therefore do not freeze stale candidate logic. Benchmark resource output marks result/model cache status and a `cache_phase`/`timing_bucket` of `cold_or_uncached`, `warm_ocr_cache`, or `unknown`; compare a cold run with a repeated warm run in the same `--work-dir` when timing cache behavior.

Recovery OCR uses a separate region/glyph cache under `backend/ocr_cache/region_ocr`. Its key includes the schema version, processed-image hash, preprocessing hash, clipped processed-image bbox, padding, strategy id, provider/profile fingerprints, Korean profile when relevant, OCR max-side settings, package/model versions, and template/font/glyph-scorer fingerprints when a local glyph strategy is used. Full-page OCR cache hits and repeated region/glyph cache hits are reported separately; the full-page OCR cache remains independent of extraction variant.

All overlay and document-graph boxes are stored in processed-image coordinates, because providers read the preprocessed image. The run transform metadata records both original and processed paths, original/processed dimensions, preprocessing steps, and whether the original-to-processed mapping is invertible for visual audit.

`crop_confirm_v1` is capped by `OCR_CROP_CONFIRM_MAX_FIELDS` per page so it can test uncertain-field crop OCR without turning a diagnostic run into a full second OCR pass.

The browser review UI also exposes the active run profile, variant, runtime, evidence-alignment score, blocked candidate count, and whether a strict benchmark score is available. Arbitrary uploads do not have a strict OCR score unless they are part of a golden evaluation set.

## PaddleOCR-VL Comparison

PaddleOCR-VL is supported as an optional processing engine, but it is not treated as automatically equivalent to the base OCR path. It returns document blocks/markdown-style text, so the backend keeps those blocks as document-block evidence instead of pretending they are word-level OCR tokens. Card extraction consumes the same downstream item/card interfaces, but visual evidence remains clearly labeled as OCR-VL block evidence.

The benchmark therefore measures:

- Base OCR extraction accuracy against golden rows/questions.
- PaddleOCR-VL extraction accuracy against the same golden rows/questions when the model can run within local resource limits.
- MCQ semantic accuracy for generated Anki cards.
- MCQ source-field accuracy for sentence, target, all four choices, answer, and answer number.
- Normalized OCR text coverage, which checks whether PaddleOCR tokens or OCR-VL document text contain expected transcript fields before answer-strip or glossary heuristics can rescue the card.
- Miss analysis for strict failures, grouped by cause such as Korean OCR error, surface OCR error, missing row, wrong pairing, or MCQ source-field OCR error.
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
uv run python scripts/benchmark_ocr_modes.py --json --dashboard-markdown ../benchmark-dashboard.md
```

The Markdown dashboard includes overall/strict accuracy, evidence alignment, raw Korean recall, per-field vocab accuracy, miss-cause counts, MCQ field-error counts, review-blocked counts, shadow row diagnostics, resource timing, RSS, and cache-hit status. Use the miss-cause and field-error columns to decide whether the next experiment should target OCR model coverage, Korean recognition, row pairing, or MCQ source-field recovery.

`--miss-inventory-json` writes a diagnostic-only inventory of current misses. `--focus-misses-from` can attach a prior inventory for focused tracing, but benchmark scoring still runs full pages and the inventory must never be used as an extraction oracle.

`--residual-diagnostics-dir` writes diagnostic-only residual artifacts after scoring: `residual-diagnostics.json`, one contact-sheet PNG per page, optional crops under `residual-crops/`, and a README that marks the directory as non-oracle output. The JSON includes page id, miss kind, failed fields, expected/current benchmark values, evidence, token ids, crop bboxes, OCR/recovery candidates, rejected candidates, rejection reasons, cache status, confidence, resource metrics, `diagnostic_only: true`, and `oracle_use_allowed: false`. Extraction code does not read this directory.

The dashboard includes executable gates. The v1 target remains overall at least `72/80`, strict OCR at least `142/160`, vocab meaning at least `52/60`, vocab surface/reading each at least `58/60`, MCQ semantic exactly `20/20`, MCQ source fields at least `90/100`, evidence alignment at least `88.5%`, and safe-local peak RSS under `3.2 GB`. The v2 gate raises the targets to overall at least `75/80`, strict OCR at least `150/160`, vocab meaning at least `55/60`, vocab surface at least `59/60`, vocab reading at least `58/60`, MCQ semantic exactly `20/20`, MCQ source fields at least `95/100`, evidence alignment at least `92%`, warm full-page OCR cache hits `4/4`, repeated region/glyph cache hits where recovery repeats, and safe-local peak RSS under `3.2 GB`.

The dashboard recovery column aggregates schema-v2 recovery payloads: attempts, accepted replacements, rejected buckets, resource caps, cache hits/misses, strict deltas, vocab surface/reading/meaning deltas, and MCQ sentence/choice/source deltas.

Run a cautious one-page PaddleOCR-VL comparison:

```bash
cd backend
uv run python scripts/benchmark_ocr_modes.py --engine all --vl-limit 1 --worker-timeout-seconds 300 --worker-max-rss-mb 14336
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

Run all four golden pages through PaddleOCR-VL explicitly:

```bash
cd backend
uv run python scripts/benchmark_ocr_modes.py --engine paddleocr_vl --json --worker-timeout-seconds 900 --worker-max-rss-mb 14336
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

On May 8, 2026, the safe-local `accuracy_recovery_v2` gate run was:

```text
jp_v3_det_v3_rec + ko_v5_current + accuracy_recovery_v2
80/80 overall
160/160 strict OCR fields
60/60 vocab meaning, surface, and reading
20/20 MCQ semantic
100/100 MCQ source fields
94.3% evidence alignment
2822.27 MB safe-local peak RSS
```

Primary artifacts:

```text
.benchmark-runs/2026-05-08-accuracy-recovery-v2/accuracy-recovery-v2-final.json
.benchmark-runs/2026-05-08-accuracy-recovery-v2/accuracy-recovery-v2-final-dashboard.md
.benchmark-runs/2026-05-08-accuracy-recovery-v2/residual-diagnostics/residual-diagnostics.json
```

Interpretation:

- `accuracy_recovery_v2` recovers the remaining Japanese surface/missing-row, Korean meaning, MCQ sentence, and MCQ choice strict-source failures from the May 7 run on the four-page golden set.
- The run is still experimental and benchmark-only. It is not a production default change, and a holdout set is required before promotion.
- Strict OCR source fields require OCR/image-derived evidence. Glossary-assisted values, golden expected values, and residual miss inventories cannot fill strict fields.
- Cloud/VL/heavy providers are outside the v2 promotion gate and remain diagnostic-only unless a separate pre-registered gate is created.

On May 6, 2026, the best safe-local adapted run was:

```text
jp_v3_det_v3_rec + ko_v5_current + v5_full_adapted_v1
68/80 overall, 85.0% accuracy
132/160 strict OCR fields, 82.5%
```

Dashboard artifact:

```text
.benchmark-runs/2026-05-06-latest-accuracy/miss-analysis-check-dashboard.md
```

Current interpretation:

- The adapted code beats the earlier measured `52/80` baseline while keeping MCQ semantic accuracy at 100% on both MCQ golden pages.
- The explicit Korean alias `ko_v5_det_v5_rec` produces the same score as `ko_v5_current` and now reuses warm OCR payloads through canonical cache/work-dir aliases.
- Vocab surface and reading extraction are no longer the main bottleneck: the latest summary shows 96.7% surface accuracy and 96.7% reading accuracy across vocab pages.
- Vocab meaning remains the highest-value safe-local target: latest raw Korean recall is 81.7%, meaning accuracy is 80.0%, and miss analysis reports 10 Korean OCR errors.
- MCQ candidates are semantically correct, but strict source-field scoring still reports 16 source-field OCR errors, mostly choice text plus some sentence text. Future MCQ work should target source-field token reconstruction and choice segmentation, not answer inference.
- Row-graph and Korean-alignment variants remain diagnostic because their shadow rows still show high risk on the vocab pages. Promoting row replacement before Korean OCR improves would mostly move errors around.
- Warm-cache runs across extraction variants should show OCR cache hits; if a graph variant goes cold against the same image/profile/preprocessing tuple, check profile alias canonicalization and cache-key fields first.

Worthwhile next experiments:

- Run `accuracy_recovery_v1` against the frozen miss inventory in `.benchmark-runs/2026-05-07-accuracy-recovery/miss-inventory.json`, then score the full golden set.
- Test Korean `ko_lang_auto` as a diagnostic-only comparison for the remaining meaning misses, with the same cache/resource guardrails as other heavy profiles.
- Continue MCQ choice/sentence source-field recovery around the category 4 failures while preserving 100% semantic MCQ accuracy.
- Add a holdout set before any production-default promotion. Without holdout, the current winner is experimental only.

Interpret these notes as a local hardware/resource snapshot, not a universal model limit. If you raise `--worker-max-rss-mb`, run one page at a time and keep the JSON metrics so the result is comparable.
