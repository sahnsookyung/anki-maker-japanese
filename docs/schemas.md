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
- `metrics`: JSON object with token/card counts, timing-adjacent metrics, and script summaries.
- `warnings` / `error`: run-level review and failure details.

Fresh processing creates a new OCR run instead of overwriting history. The UI reviews the active run by default, and previous successful runs can be reactivated for comparison.

## OCR Token

- `run_id`: OCR run that produced this token.
- `text`: recognized text.
- `bbox`: `[x1, y1, x2, y2]` on the processed image.
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

Only approved cards are exported by default. Red cards are blocked from export unless explicitly included through the API.
The Anki export format is CSV with Anki text-import headers; TSV support was removed to keep the import contract singular.

## SQLite Schema Notes

- Foreign keys are enabled on each connection, so tokens/cards must belong to a page.
- `ocr_runs` keeps OCR history per page; `pages.active_ocr_run_id` selects the review/export view.
- `replace_tokens` and `replace_cards` attach inserted rows to the target page id to prevent stale page references.
- Lookup indexes exist for upload replacement, page token/card loading, run loading, card summaries, and source lookup.
- `source_json`, `tags_json`, and `warnings_json` remain JSON text because the app reads/writes whole candidate objects rather than querying inside individual fields.
