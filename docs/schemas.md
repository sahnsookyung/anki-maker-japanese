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

## Card Candidate

- `run_id`: OCR run that generated this candidate.
- `source`: source extraction object, such as a vocab row or MCQ question.
- `front` / `back`: editable Anki HTML.
- `review_state`: `green`, `yellow`, or `red`.
- `status`: `pending_review`, `approved`, or `skipped`.
- `warnings`: issues that should be inspected before export.

Vocabulary candidates use one active note type: `jp_vocab_entry`. Unsupported older vocab note shapes are deleted during DB initialization instead of being converted or preserved. MCQ candidates keep their current front/back note types.

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

## Export Response

- `export_id`: stable id for one export action.
- `files`: CSV files produced by schema, usually `vocab` and/or `mcq`.
- `note_count`: number of exported Anki notes.
- `estimated_generated_card_count`: estimated Anki cards after the single vocab template and MCQ cards are applied.

Mixed vocab/MCQ exports return multiple CSV file entries instead of a ZIP. Vocab entries missing `Surface`, `Reading`, `MeaningKo`, or explicitly disabled for all legacy study directions do not produce CSV rows.

## SQLite Schema Notes

- Foreign keys are enabled on each connection, so tokens/cards must belong to a page.
- `ocr_runs` keeps OCR history per page; `pages.active_ocr_run_id` selects the review/export view.
- `replace_tokens` and `replace_cards` attach inserted rows to the target page id to prevent stale page references.
- Lookup indexes exist for upload replacement, page token/card loading, run loading, card summaries, and source lookup.
- `source_json`, `tags_json`, and `warnings_json` remain JSON text because the app reads/writes whole candidate objects rather than querying inside individual fields.
