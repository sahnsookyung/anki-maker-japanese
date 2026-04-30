# Schemas

The backend stores canonical extraction results as JSON-backed rows in SQLite and exposes them as Pydantic models.

## Page

- `id`: stable page id.
- `original_image_path`: local upload path.
- `processed_image_path`: local preprocessed image path.
- `page_type`: `uploaded`, `vocab_table`, `reading_mcq`, `spelling_mcq`, or `unknown_review_required`.
- `page_type_confidence`: classifier score.
- `warnings`: preprocessing/OCR/extraction issues visible in the UI.

## OCR Token

- `text`: recognized text.
- `bbox`: `[x1, y1, x2, y2]` on the processed image.
- `confidence`: provider confidence normalized to `0..1`.
- `script_class`: Unicode script class.
- `source`: OCR provider name.

## Card Candidate

- `source`: source extraction object, such as a vocab row or MCQ question.
- `front` / `back`: editable Anki HTML.
- `review_state`: `green`, `yellow`, or `red`.
- `status`: `pending_review`, `approved`, or `skipped`.
- `warnings`: issues that should be inspected before export.

Only approved cards are exported by default. Red cards are blocked from export unless explicitly included through the API. The primary Anki export is CSV with Anki text-import headers; TSV remains a legacy compatibility format.
