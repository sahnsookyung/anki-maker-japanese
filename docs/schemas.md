# Schemas

The backend stores canonical extraction results as JSON-backed rows in SQLite and exposes them as Pydantic models.
SQLite is intentionally the production store for this local-first app: the data is single-user, file-backed,
portable, and colocated with disposable OCR artifacts. Postgres would be a better fit only if this becomes a
multi-user service with concurrent reviewers, remote access, role-based permissions, or server-side job queues.

## Page

- `id`: stable page id.
- `original_image_path`: local upload path.
- `processed_image_path`: local preprocessed image path.
- `active_ocr_run_id`: the successful OCR run currently shown in review/export views.
- `page_type`: `uploaded`, `vocab_table`, `reading_mcq`, `spelling_mcq`, or `unknown_review_required`.
- `page_type_confidence`: classifier score.
- `warnings`: preprocessing/OCR/extraction issues visible in the UI.

## OCR Run

- `id`: stable run id.
- `page_id`: parent page.
- `engine`: `paddleocr`, `paddleocr_vl`, `google_vision`, or `legacy`.
- `status`: `queued`, `running`, `succeeded`, `failed`, or `cancelled`.
- `image_sha256`: original image checksum used for reproducibility and cache lookup.
- `processed_image_path`, `image_width`, `image_height`: processed-image geometry for evidence alignment.
- `preprocessing`: JSON object with preprocessing dimensions and warnings.
- `provider_config`: JSON object with OCR model names and stage flags.
- `provider_config.model_profile`: profile manifest with provider, model names, language configuration, package versions, runtime/device environment, preprocessing config, model-cache paths/fingerprints, cache key, and cache-hit status.
- `metrics`: JSON object with token/card counts, timing-adjacent metrics, script summaries, extraction variant metrics, evidence-alignment score, and review-blocked/exportable counts.
- `warnings` / `error`: run-level review and failure details.

Fresh processing creates a new OCR run instead of overwriting history. The UI reviews the active run by default, and previous successful runs can be reactivated for comparison.
When the OCR cache key matches a previous successful run, the backend reuses the OCR payload and reruns extraction from cached tokens or document blocks; it does not copy old review edits as the new extraction result.

## OCR Token

- `run_id`: OCR run that produced this token.
- `text`: recognized text.
- `bbox`: `[x1, y1, x2, y2]` in processed-image coordinates.
- `confidence`: provider confidence normalized to `0..1`.
- `script_class`: Unicode script class.
- `source`: OCR provider name.

Recovery variants can create additional OCR tokens before candidates are persisted. Accepted field evidence may reference these provenance values:

- `ocr`: full-page OCR token evidence.
- `crop_ocr`: bounded field crop OCR.
- `region_ocr`: bounded region OCR.
- `answer_strip_ocr`: MCQ answer-strip OCR/image evidence.
- `source_rebuild`: deterministic source-field rebuild from OCR-backed fields.
- `jp_region_ocr`: Japanese surface/reading row-region OCR.
- `ko_glyph_ocr`: Korean residual glyph/region evidence.
- `prompt_line_ocr`: clipped MCQ prompt-line OCR.
- `choice_glyph_ocr`: MCQ per-choice glyph evidence.

Any `token_ids` stored in field evidence must resolve to rows in `ocr_tokens` for the same active run and must appear in the API token payload. If a preview/apply path cannot persist new recovery tokens atomically, it must strip token ids before saving the evidence and record a warning instead of storing stale ids.

## Card Candidate

- `run_id`: OCR run that generated this candidate.
- `source`: source extraction object, such as a vocab row or MCQ question.
- `front` / `back`: editable Anki HTML.
- `review_state`: `green`, `yellow`, or `red`.
- `status`: `pending_review`, `approved`, or `skipped`.
- `warnings`: issues that should be inspected before export.

Vocabulary candidates use one active note type: `jp_vocab_entry`. Unsupported older vocab note shapes are deleted during DB initialization instead of being converted or preserved. MCQ candidates keep their current front/back note types.

Experimental MCQ source-recovery variants may add `source.source_fields` and `source.semantic_fields`. `source_fields` contains strict OCR-backed `sentence`, `target`, `choices`, `correct_answer`, and `correct_choice_no` for benchmark source-field scoring. `semantic_fields` contains glossary/answer-strip-assisted values for Anki semantics and export compatibility. When `source_fields` exists, strict MCQ evaluation uses it instead of the legacy top-level fields; glossary-derived values must stay out of `source_fields`. `mcq_prompt_line_ocr_v1` may update only `source_fields.sentence`, and `mcq_choice_glyph_v1` may update only `source_fields.choices` plus mirrored source `correct_answer` when `correct_choice_no` already exists.

Only approved candidates are exported by default. Red candidates are blocked from export unless explicitly included through the API. The Anki export format is CSV with Anki text-import headers; TSV support was removed to keep the import contract singular.

## Document Graph

Experimental OCR variants persist a provider-neutral graph inside `ocr_runs.metrics.document_graph`:

- `text_nodes`: OCR tokens or document blocks with processed-image coordinate bounding boxes.
- `line_nodes`: reading-order line groupings with token relationships.
- `region_nodes`: page, line, and document-block regions.
- `table_cells`: token-level cell candidates for table-style pages.
- `selection_marks`: checkbox/selection mark detections when OCR exposes them.
- `field_hypotheses`: candidate field evidence linked back to tokens or document blocks.
- `row_hypotheses`: candidate row regions and review state.
- `relationships`: containment/support edges between regions, lines, cells, fields, rows, tokens, and blocks.
- `transform`: schema-versioned coordinate provenance with original/processed image dimensions, original and processed image paths, preprocessing steps, and whether the original-to-processed mapping is invertible. Current OCR boxes are stored against `processed_image_path`.

The graph is diagnostic infrastructure. `baseline_current` remains the safe default extractor, and variants add metrics without silently filling benchmarked OCR fields from glossary or provider agreement.

Recovery variants may add live `crop_ocr`, `region_ocr`, `jp_region_ocr`, `ko_glyph_ocr`, `prompt_line_ocr`, or `choice_glyph_ocr` tokens to the same OCR run before candidates are persisted. Any field evidence with recovery provenance must reference token ids that exist in the persisted `ocr_tokens` table for the active run.

`ocr_runs.metrics.extraction_variant_metrics.recovery` uses schema version 2 for `accuracy_recovery_v2` and combined recovery runs:

```json
{
  "schema_version": 2,
  "kind": "accuracy_recovery_v2",
  "attempted": 0,
  "accepted": 0,
  "counts": {},
  "components": {
    "korean_recovery": {},
    "japanese_vocab_recovery": {},
    "mcq_prompt_line_recovery": {},
    "mcq_choice_glyph_recovery": {}
  },
  "resource_caps": {},
  "cache": {}
}
```

Component payloads may include attempts, OCR candidates, rejected candidates, rejection reasons, confidence, crop bboxes, token ids, and cache hit/miss metadata. Dashboard summaries aggregate attempts, accepted replacements, rejected buckets, resource caps, cache hits, strict deltas, vocab surface/reading/meaning deltas, and MCQ sentence/choice/source deltas.

## Benchmark Artifact

`backend/scripts/benchmark_ocr_modes.py --json` emits one schema-versioned result per page. The `base` payload includes the strict score, raw OCR field coverage, review/exportability counts, failure taxonomy, and `miss_analysis`.

`miss_analysis` is diagnostic only. For vocab pages it lists each strict-missed golden row with expected surface/reading/Korean meaning, raw OCR presence booleans, the closest generated candidate, and a cause such as `korean_ocr_error`, `surface_ocr_error`, `reading_ocr_error`, `wrong_pairing`, or `missing_row`. For MCQ pages it reports source-field OCR mismatch counts. These fields are for benchmark interpretation and do not approve, mutate, or fill card candidates.

`residual-diagnostics.json` is also diagnostic only. It is written after benchmark scoring, carries `diagnostic_only: true` and `oracle_use_allowed: false`, and is never read by extraction code. It may include golden expected values for reporting, but those values are forbidden as extraction inputs.

## Export Response

- `export_id`: stable id for one export action.
- `files`: CSV files produced by schema, usually `vocab` and/or `mcq`.
- `note_count`: number of exported Anki notes.
- `estimated_generated_card_count`: estimated Anki cards after the single vocab template and MCQ cards are applied.

Mixed vocab/MCQ exports return multiple CSV file entries instead of a ZIP. Vocab entries missing `Surface`, `Reading`, `MeaningKo`, or explicitly disabled for all study directions do not produce CSV rows. Duplicate vocab candidates with the same `Surface` and `Reading` export as one `jp_vocab_entry` note.

## SQLite Schema Notes

- Foreign keys are enabled on each connection, so tokens/cards must belong to a page.
- `ocr_runs` keeps OCR history per page; `pages.active_ocr_run_id` selects the review/export view.
- `replace_tokens` and `replace_cards` attach inserted rows to the target page id to prevent stale page references.
- Lookup indexes exist for upload replacement, page token/card loading, run loading, card summaries, and source lookup.
- `source_json`, `tags_json`, and `warnings_json` remain JSON text because the app reads/writes whole candidate objects rather than querying inside individual fields.
