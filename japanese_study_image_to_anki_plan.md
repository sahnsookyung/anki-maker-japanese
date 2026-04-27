# Japanese Study Image to Anki Card Generator — Implementation Plan

## 1. Goal

Build a local-first web app that accepts phone photos or scanned images of Japanese study-book pages and converts them into reviewable Anki card candidates.

The app must support these initial material types:

1. Vocabulary enumeration pages: Japanese written form, reading, and Korean meaning.
2. Reading multiple-choice pages: Japanese sentence with an underlined word; choose the correct pronunciation from four options.
3. Spelling / 표기 enumeration pages: Japanese phonetic form, usually hiragana, mapped to the correct written form, usually kanji or katakana.
4. Spelling / 표기 multiple-choice pages: Japanese sentence with an underlined kana word; choose the correct written form from four options.

The app should not silently produce final decks. It should produce card candidates, score confidence, show the original evidence, allow review/editing, and then export to Anki.

---

## 2. Product Principle

The central rule:

> The app creates reviewable Anki card candidates, not blindly trusted final cards.

This matters because OCR and vision-language models can make small but damaging mistakes in Japanese readings, kanji, kana, and Korean definitions.

The MVP should prioritize:

- source-grounded extraction,
- bounding boxes,
- confidence labels,
- human review,
- traceability back to the source image,
- TSV export before `.apkg` export.

---

## 3. Target User Flow

1. User uploads one or more study-book page images.
2. App detects and crops the page area.
3. App runs OCR with bounding boxes.
4. App classifies the page content type.
5. App extracts vocabulary rows or question blocks.
6. App uses a local vision-language model for cleanup and contextual extraction.
7. App validates readings/spellings against Japanese dictionary data.
8. App shows extracted card candidates in a review UI.
9. User approves, edits, skips, or marks uncertain cards.
10. App exports approved cards to TSV first.
11. Later versions can export `.apkg` or push directly to Anki.

---

## 4. Non-Goals for MVP

Do not implement these first:

- perfect automatic deck generation with no review,
- full `.apkg` generation before TSV works,
- cloud-only extraction,
- support for every possible workbook layout,
- automatic answer guessing without answer keys or validation,
- public deck sharing.

The app should be designed for personal study material processing. Avoid features that encourage redistribution of copyrighted book content.

---

## 5. Reference Page Patterns

The supplied sample images show three important layout families.

### 5.1 Vocabulary Enumeration Pages

Observed structure:

- kana section markers such as あ, か, さ, た, な, は, ま, や;
- two-column layout;
- checkbox before each row;
- Japanese written form;
- reading in hiragana/katakana;
- Korean gloss;
- horizontal row dividers;
- sometimes the written form comes first;
- sometimes the reading-like item appears first depending on section type.

Detection should be content-based, not header-based.

Useful signals:

- many repeated rows,
- checkboxes,
- Japanese + Hangul mixed on the same row,
- kana group headings,
- two-column structure,
- horizontal table lines.

### 5.2 Reading Multiple-Choice Pages

Observed structure:

- numbered questions, usually 1 to 10;
- Japanese sentence;
- underlined target word;
- four answer options;
- options are mostly hiragana/katakana;
- bottom answer strip with correct choices.

Useful signals:

- question numbers;
- sentence followed by four options;
- choice numbers 1, 2, 3, 4;
- short underline inside the sentence area;
- bottom answer band.

### 5.3 Spelling / 표기 Multiple-Choice Pages

Observed structure:

- numbered questions, usually 1 to 10;
- Japanese sentence;
- underlined kana target word;
- four answer options;
- options often contain kanji, katakana, or visually similar distractors;
- bottom answer strip with correct choices.

Useful signals:

- target is often kana;
- answer options contain kanji/katakana;
- answer strip exists;
- content resembles exam-style question blocks.

---

## 6. Core Architecture

```text
Next.js frontend
  ↓
FastAPI backend
  ↓
Image upload
  ↓
Page crop / deskew / dewarp
  ↓
OCR with bounding boxes
  ↓
Content-based page classifier
  ↓
Region extraction
  ↓
Local VLM cleanup / structured extraction
  ↓
Dictionary validation
  ↓
Review UI
  ↓
TSV export
  ↓
Later: .apkg export / AnkiConnect push
```

---

## 7. Recommended Technology Stack

### 7.1 Frontend

Use:

- Next.js;
- React;
- TypeScript;
- Tailwind CSS;
- shadcn/ui if useful;
- Canvas or SVG overlay for OCR boxes and selection.

Frontend responsibilities:

- image upload;
- crop preview;
- OCR overlay;
- extracted item review;
- manual correction;
- export controls.

### 7.2 Backend

Use:

- Python;
- FastAPI;
- Pydantic;
- SQLite for MVP;
- SQLAlchemy or SQLModel;
- OpenCV;
- Pillow;
- PaddleOCR;
- local VLM integration through MLX-VLM or Ollama;
- dictionary validation with JMdict/KANJIDIC2 data.

Backend responsibilities:

- preprocessing;
- OCR;
- page classification;
- region extraction;
- VLM calls;
- validation;
- card generation;
- export generation.

### 7.3 OCR

Use PaddleOCR as the local-first OCR engine.

The OCR output must include:

```json
{
  "text": "学校",
  "bbox": [100, 200, 160, 230],
  "confidence": 0.97,
  "source": "paddleocr"
}
```

Plain text OCR is not enough. Bounding boxes are required for:

- row grouping;
- underline matching;
- answer strip parsing;
- review UI overlays;
- source traceability.

Implement the OCR adapter so that additional providers can be plugged in later.

```python
class OcrProvider(Protocol):
    def recognize(self, image: Image.Image) -> list[OcrToken]:
        ...
```

### 7.4 Local Vision-Language Model

Use a local VLM as a contextual extraction assistant.

Default target:

- Qwen3-VL-8B-Instruct.

Alternative fallback:

- Qwen2.5-VL-7B-Instruct.

Runtime options:

- MLX-VLM for Apple Silicon-native inference;
- Ollama if the local serving workflow is simpler.

Use the VLM for:

- row crop interpretation;
- question crop interpretation;
- OCR cleanup;
- deciding whether grouping looks correct;
- emitting structured JSON from visual context.

Do not use the VLM for:

- silently inventing missing answers;
- guessing answer keys without marking uncertainty;
- inventing Korean meanings;
- overriding dictionary validation without evidence.

Every VLM response should be source-grounded.

Required VLM output style:

```json
{
  "items": [
    {
      "type": "vocab_item",
      "surface": "学校",
      "reading": "がっこう",
      "meaning_ko": "학교",
      "evidence_tokens": ["tok_12", "tok_13", "tok_14"],
      "bbox": [100, 200, 420, 235],
      "confidence": 0.91,
      "needs_review": false,
      "warnings": []
    }
  ]
}
```

### 7.5 Anki Export

MVP export format:

- TSV.

Later formats:

- `.apkg` through `genanki`;
- direct desktop Anki insertion through AnkiConnect.

---

## 8. Data Model

### 8.1 Page

```json
{
  "id": "page_001",
  "original_image_path": "uploads/page_001.jpg",
  "processed_image_path": "processed/page_001.png",
  "page_type": "vocab_table",
  "page_type_confidence": 0.92,
  "created_at": "..."
}
```

### 8.2 OCR Token

```json
{
  "id": "tok_001",
  "page_id": "page_001",
  "text": "学校",
  "bbox": [100, 200, 160, 230],
  "confidence": 0.97,
  "script_class": "kanji"
}
```

### 8.3 Region

```json
{
  "id": "region_001",
  "page_id": "page_001",
  "type": "vocab_row",
  "bbox": [90, 190, 520, 245],
  "token_ids": ["tok_001", "tok_002", "tok_003"],
  "confidence": 0.89
}
```

### 8.4 Vocabulary Item

```json
{
  "id": "vocab_001",
  "page_id": "page_001",
  "surface": "学校",
  "reading": "がっこう",
  "meaning_ko": "학교",
  "bbox": [100, 200, 420, 235],
  "confidence": 0.93,
  "validation_status": "valid",
  "warnings": []
}
```

### 8.5 Question Item

```json
{
  "id": "q_001",
  "page_id": "page_001",
  "question_no": 1,
  "question_type": "reading_mcq",
  "sentence": "にわの ある 家が ほしいけど、お金が たりない。",
  "target": "お金",
  "target_bbox": [300, 220, 350, 238],
  "choices": ["おきん", "おかね", "おがね", "おきん"],
  "correct_choice_no": 2,
  "correct_answer": "おかね",
  "answer_source": "answer_strip",
  "confidence": 0.88,
  "warnings": []
}
```

### 8.6 Card Candidate

```json
{
  "id": "card_001",
  "source_type": "vocab_item",
  "source_id": "vocab_001",
  "note_type": "jp_vocab_reading",
  "front": "学校<br>뜻: 학교<br><br>읽는 법?",
  "back": "がっこう",
  "tags": ["jlpt", "vocab", "reading"],
  "confidence": 0.93,
  "status": "pending_review"
}
```

---

## 9. Image Preprocessing

The sample photos are phone photos of curved pages, with shadows, hands, table background, and neighboring pages. Preprocessing is mandatory.

Implement:

1. Load original image.
2. Detect likely page polygon.
3. Crop page area.
4. Correct perspective.
5. Deskew text orientation.
6. Normalize contrast.
7. Reduce shadows where possible.
8. Save processed image.
9. Preserve coordinate mapping from processed image back to original image.

MVP can use simple perspective correction. Full dewarping for curved pages can come later.

Quality warnings:

- if page crop confidence is low;
- if blur is high;
- if OCR token count is unexpectedly low;
- if answer strip is cut off;
- if neighboring page text leaks into crop.

---

## 10. Script Detection

Add Unicode script classification for each OCR token.

Classes:

- hiragana;
- katakana;
- kanji;
- hangul;
- latin;
- number;
- punctuation;
- mixed.

Example:

```json
{
  "text": "がっこう",
  "script_class": "hiragana"
}
```

This enables content-based classification:

- kanji + hiragana + hangul → likely vocabulary row;
- kanji target + kana choices → likely reading MCQ;
- kana target + kanji choices → likely spelling MCQ.

---

## 11. Content-Based Page Classification

Do not rely only on headers. Headers are hints only.

Implement a scoring classifier.

### 11.1 Features

Compute:

- number of checkbox-like boxes;
- number of question numbers;
- number of choice markers 1, 2, 3, 4;
- presence of bottom answer strip;
- amount of Hangul;
- amount of hiragana;
- amount of kanji;
- repeated row spacing;
- two-column table likelihood;
- underline segment count;
- average choice script type.

### 11.2 Classifier Logic

Pseudo-code:

```python
if has_many_questions and has_four_choices_per_question:
    if choices_are_mostly_kana:
        return "reading_mcq"
    if choices_contain_kanji_or_katakana:
        return "spelling_mcq"
    return "mcq_unknown"

if has_many_checkbox_rows and has_japanese_hangul_pairs:
    return "vocab_table"

return "unknown_review_required"
```

Every page type should have a confidence score. Unknown pages still show OCR output for manual review.

---

## 12. Vocabulary Table Extraction

### 12.1 Detection

Detect row groups using:

- checkbox position;
- horizontal rules;
- repeated y-spacing;
- two-column layout;
- Japanese/Hangul token proximity;
- kana group headings.

### 12.2 Extraction Target

Each vocabulary item should become:

```json
{
  "surface": "学校",
  "reading": "がっこう",
  "meaning_ko": "학교"
}
```

However, not all rows are pure kanji vocabulary. Some written forms are kana-only or katakana. Use `surface`, not `kanji`, as the canonical field name.

### 12.3 Row Interpretation Rules

Within each row:

- Japanese written form is usually kanji/kana/katakana;
- reading is usually hiragana/katakana;
- Korean meaning is Hangul;
- small Korean text may be adjacent to Japanese text;
- group headings are not vocabulary items;
- checkboxes are not content.

### 12.4 VLM Cleanup

For each row crop, pass:

- row image crop;
- OCR tokens in that crop;
- expected JSON schema.

Ask the VLM to return only grounded fields.

If row-level OCR is messy, rerun OCR on the row crop at higher resolution.

---

## 13. Multiple-Choice Extraction

### 13.1 Question Block Detection

A question block contains:

- question number;
- sentence;
- underlined target;
- four choices;
- optional answer from answer strip.

Detect blocks by y-position from each question number to the next question number.

### 13.2 Reading MCQ

Expected output:

```json
{
  "question_type": "reading_mcq",
  "question_no": 1,
  "sentence": "...",
  "target": "お金",
  "choices": ["おきん", "おかね", "おがね", "おきん"],
  "correct_choice_no": 2,
  "correct_answer": "おかね",
  "answer_source": "answer_strip"
}
```

The answer choices should be mostly kana.

### 13.3 Spelling MCQ

Expected output:

```json
{
  "question_type": "spelling_mcq",
  "question_no": 1,
  "sentence": "...",
  "target": "やま",
  "choices": ["ヨ", "山", "川", "由"],
  "correct_choice_no": 2,
  "correct_answer": "山",
  "answer_source": "answer_strip"
}
```

The answer choices often contain kanji, katakana, or visually similar distractors.

---

## 14. Underline Detection

OCR alone is not enough to identify the underlined word.

Use OpenCV to detect short horizontal underline segments inside sentence regions.

Algorithm:

1. Detect horizontal line segments.
2. Ignore long table rules and section dividers.
3. Restrict search to question sentence regions.
4. Match underline y-position to nearby OCR token baselines.
5. Select token or span whose bbox overlaps the underline.
6. Return target text and target bbox.
7. If confidence is low, require manual review.

Confidence should decrease when:

- underline overlaps multiple possible tokens;
- underline is too long;
- underline is near a table rule;
- OCR tokenization is poor;
- sentence crop contains neighboring question text.

Manual fallback:

- user can click OCR tokens to set the target;
- user can drag-select a target span on the image.

---

## 15. Answer Strip Parsing

The sample MCQ pages include a bottom answer strip. This should be the preferred answer source.

Algorithm:

1. Crop bottom 10-15% of the processed page.
2. Detect gray horizontal answer band.
3. OCR only the answer strip crop.
4. Parse pairs of question number and answer number.
5. Attach correct choice to each question block.

Expected result:

```json
{
  "1": 2,
  "2": 3,
  "3": 1,
  "4": 4,
  "5": 1,
  "6": 3,
  "7": 1,
  "8": 3,
  "9": 3,
  "10": 2
}
```

Answer source priority:

1. `answer_strip`
2. `manual`
3. `dictionary_verified`
4. `model_inferred`
5. `unknown`

Only the first three should be exportable without warning.

If answer strip parsing fails, keep the question but mark it as needing review.

---

## 16. Dictionary Validation

Use validation to catch extraction mistakes.

Validation goals:

- confirm surface + reading pair;
- detect mismatched kanji/kana;
- detect multiple valid readings;
- detect OCR confusion;
- flag suspicious model corrections.

Recommended dictionary sources:

- JMdict for word-level Japanese readings and meanings;
- KANJIDIC2 for kanji-level readings;
- optional MeCab/UniDic or MeCab/IPADIC for sentence tokenization.

Validation examples:

```text
学校 → がっこう = valid
午前 → ごぜん = valid
右 → みぎ = valid
大きい → おおきい = valid
```

Warning examples:

```text
surface and reading not found in dictionary
multiple readings possible
reading is kana but selected answer is kanji
target span missing
correct choice missing
OCR confidence below threshold
```

Dictionary validation should not automatically delete items. It should add warnings and affect confidence.

---

## 17. Card Generation

Generate cards from canonical extracted objects.

### 17.1 Vocabulary Cards

For each vocabulary item:

```json
{
  "surface": "学校",
  "reading": "がっこう",
  "meaning_ko": "학교"
}
```

Generate:

#### Card A: Reading Recall

Front:

```html
学校<br>
뜻: 학교<br><br>
읽는 법?
```

Back:

```html
がっこう
```

#### Card B: Meaning Recall

Front:

```html
学校<br>
がっこう<br><br>
뜻?
```

Back:

```html
학교
```

#### Card C: Writing Recall

Front:

```html
がっこう<br>
학교<br><br>
올바른 표기는?
```

Back:

```html
学校
```

For kana-only words, still use `surface`. Do not call this field `kanji`.

### 17.2 Reading MCQ Cards

Generate active recall first.

Front:

```html
にわの ある 家が ほしいけど、お金が たりない。<br><br>
밑줄: お金<br>
읽는 법?
```

Back:

```html
おかね
```

Optional exam-style card:

Front:

```html
にわの ある 家が ほしいけど、お金が たりない。<br><br>
1. おきん<br>
2. おかね<br>
3. おがね<br>
4. おきん
```

Back:

```html
정답: 2. おかね
```

### 17.3 Spelling MCQ Cards

Front:

```html
あの やまは 3,000メートルいじょうです。<br><br>
밑줄: やま<br>
올바른 표기는?
```

Back:

```html
山
```

Optional exam-style card:

Front:

```html
あの やまは 3,000メートルいじょうです。<br><br>
1. ヨ<br>
2. 山<br>
3. 川<br>
4. 由
```

Back:

```html
정답: 2. 山
```

---

## 18. Confidence and Review Policy

Use three states:

```text
green = safe candidate
yellow = review recommended
red = blocked from export until fixed
```

### 18.1 Green

Requirements:

- OCR confidence acceptable;
- required fields present;
- source bbox exists;
- dictionary validation passes or is not required;
- answer source is `answer_strip`, `manual`, or `dictionary_verified`.

### 18.2 Yellow

Examples:

- low OCR confidence;
- possible dictionary mismatch;
- multiple valid readings;
- VLM corrected OCR;
- underline detected but confidence is below threshold;
- Korean meaning seems incomplete.

### 18.3 Red

Examples:

- missing target;
- missing answer;
- fewer than four choices for MCQ;
- answer is only `model_inferred`;
- no source bbox;
- OCR block is badly grouped;
- duplicate with conflicting fields.

Red cards should not export unless manually approved.

---

## 19. Review UI Requirements

Each extracted item should show:

- cropped source image;
- OCR token overlay;
- extracted fields;
- generated card preview;
- confidence;
- warnings;
- answer source;
- approve/edit/skip controls.

### 19.1 Vocabulary Review

Fields:

- surface;
- reading;
- Korean meaning;
- source crop;
- generated cards.

### 19.2 MCQ Review

Fields:

- sentence;
- target;
- target bbox;
- choices;
- correct choice;
- answer source;
- generated active recall card;
- generated exam-style card.

### 19.3 Manual Correction Tools

Implement:

- click token to set target;
- drag box to select target;
- edit extracted text fields;
- change correct answer;
- approve all green;
- filter yellow/red;
- skip item;
- merge/split rows if needed later.

---

## 20. Deduplication

The same words may appear across multiple pages.

Create normalized keys.

Vocabulary key:

```text
normalize(surface) + "|" + normalize(reading) + "|" + normalize(meaning_ko)
```

Question key:

```text
normalize(sentence) + "|" + normalize(target) + "|" + normalize(correct_answer)
```

If duplicate:

- do not create duplicate cards by default;
- preserve additional source references;
- allow user to export duplicates only if explicitly enabled.

---

## 21. API Endpoints

### 21.1 Upload Image

```http
POST /api/pages/upload
```

Returns:

```json
{
  "page_id": "page_001",
  "status": "uploaded"
}
```

### 21.2 Process Page

```http
POST /api/pages/{page_id}/process
```

Runs:

- preprocessing;
- OCR;
- classification;
- extraction;
- validation;
- card generation.

### 21.3 Get OCR Overlay

```http
GET /api/pages/{page_id}/ocr
```

### 21.4 Get Card Candidates

```http
GET /api/pages/{page_id}/cards
```

### 21.5 Update Candidate

```http
PATCH /api/cards/{card_id}
```

### 21.6 Approve Candidate

```http
POST /api/cards/{card_id}/approve
```

### 21.7 Export TSV

```http
POST /api/exports/tsv
```

Body:

```json
{
  "page_ids": ["page_001", "page_002"],
  "include_yellow": true,
  "include_red": false
}
```

---

## 22. TSV Export Format

Use UTF-8 TSV.

Recommended columns:

```tsv
note_type	front	back	source_page	source_bbox	confidence	tags
```

Example:

```tsv
jp_vocab_reading	学校<br>뜻: 학교<br><br>읽는 법?	がっこう	page_001	[100,200,420,235]	0.93	jlpt vocab reading
jp_vocab_meaning	学校<br>がっこう<br><br>뜻?	학교	page_001	[100,200,420,235]	0.93	jlpt vocab meaning
jp_vocab_writing	がっこう<br>학교<br><br>올바른 표기는?	学校	page_001	[100,200,420,235]	0.93	jlpt vocab writing
```

Do not include unescaped tabs or newlines inside fields.

---

## 23. Local Model Prompting

### 23.1 Vocab Row Prompt

System:

```text
You extract Japanese study vocabulary from a cropped workbook row. Return only JSON matching the schema. Do not invent missing data. Every field must be supported by visible text or OCR tokens. If uncertain, set needs_review=true and add a warning.
```

User payload:

```json
{
  "task": "extract_vocab_row",
  "ocr_tokens": [...],
  "expected_fields": ["surface", "reading", "meaning_ko"]
}
```

Expected response:

```json
{
  "surface": "学校",
  "reading": "がっこう",
  "meaning_ko": "학교",
  "evidence_tokens": ["tok_1", "tok_2", "tok_3"],
  "needs_review": false,
  "warnings": []
}
```

### 23.2 MCQ Prompt

System:

```text
You extract a Japanese multiple-choice vocabulary question from a cropped workbook region. Return only JSON. Do not guess the correct answer unless it is present in the supplied answer map. If the target underline is unclear, set needs_review=true.
```

User payload:

```json
{
  "task": "extract_mcq_question",
  "ocr_tokens": [...],
  "answer_map": {"1": 2},
  "expected_fields": ["sentence", "target", "choices", "correct_choice_no"]
}
```

---

## 24. Implementation Phases

### Phase 1: App Skeleton

Build:

- Next.js app;
- FastAPI backend;
- image upload;
- local file storage;
- SQLite database;
- basic page list.

Done when:

- user can upload an image;
- backend stores original image;
- frontend displays uploaded image.

### Phase 2: Preprocessing and OCR Overlay

Build:

- page crop;
- deskew;
- contrast normalization;
- PaddleOCR integration;
- OCR token storage;
- OCR overlay in frontend.

Done when:

- OCR boxes appear on the uploaded image;
- Japanese, kana, kanji, and Korean tokens are visible;
- user can inspect OCR output.

### Phase 3: Script Classification

Build:

- Unicode script classifier;
- token script labels;
- script distribution summary per page.

Done when:

- tokens are tagged as hiragana, katakana, kanji, hangul, number, punctuation, mixed;
- page-level script ratios are available.

### Phase 4: Vocabulary Table Extraction

Build:

- checkbox detection;
- row grouping;
- two-column segmentation;
- vocab item extraction;
- row crop review;
- vocab card generation.

Done when:

- supplied vocabulary pages produce editable vocab items;
- approved items export to TSV.

### Phase 5: Answer Strip Parser

Build:

- bottom answer strip crop;
- gray band detection;
- answer OCR;
- answer map parser.

Done when:

- MCQ pages produce `question_no → correct_choice_no` maps;
- failures are marked as review-needed.

### Phase 6: MCQ Block Extraction

Build:

- question number detection;
- question block segmentation;
- choice extraction;
- underline detection;
- answer attachment;
- MCQ card generation.

Done when:

- reading MCQ and spelling MCQ pages produce editable question items;
- answer source is tracked;
- missing targets/answers are blocked from export.

### Phase 7: Local VLM Cleanup

Build:

- Qwen3-VL-8B integration through MLX-VLM or Ollama;
- row crop cleanup;
- question crop cleanup;
- strict JSON validation;
- evidence-token enforcement.

Done when:

- VLM can improve OCR grouping;
- VLM output cannot bypass validation;
- uncertain VLM output is marked yellow/red.

### Phase 8: Dictionary Validation

Build:

- local JMdict lookup;
- surface-reading validation;
- warning generation;
- duplicate detection.

Done when:

- obvious valid pairs pass;
- suspicious pairs are flagged;
- duplicates are detected before export.

### Phase 9: Review UI Hardening

Build:

- approve/edit/skip;
- filter green/yellow/red;
- manual target selection;
- manual answer correction;
- batch approve green;
- export selected cards.

Done when:

- user can correct all extraction errors without touching the database directly.

### Phase 10: Export Improvements

Build later:

- `.apkg` export via genanki;
- optional AnkiConnect push;
- deck/note-type configuration;
- media crop inclusion.

---

## 25. Testing Plan

### 25.1 Unit Tests

Test:

- script classifier;
- bbox normalization;
- row grouping;
- question block segmentation;
- answer strip parser;
- TSV escaping;
- dedupe keys;
- confidence scoring rules.

### 25.2 Golden Image Tests

Create a small test set from representative pages:

- clean vocabulary page;
- angled vocabulary page;
- reading MCQ page;
- spelling MCQ page;
- page with answer strip cut off;
- page with shadows;
- page with neighboring page visible.

For each image, store expected structured JSON.

Test tolerance:

- exact fields for manually verified items;
- bbox IoU threshold for target detection;
- answer map accuracy;
- card count accuracy.

### 25.3 Manual Acceptance Tests

For each uploaded sample image:

- OCR overlay is readable;
- page type is correct or review-required;
- rows/questions are not badly merged;
- answer strip is parsed if visible;
- target underline is detected or manually selectable;
- TSV imports into Anki.

---

## 26. Risk Register

### Risk: Phone-photo distortion hurts OCR

Mitigation:

- crop/dewarp;
- OCR overlay;
- retake warning;
- manual crop override.

### Risk: OCR confuses similar kanji

Mitigation:

- dictionary validation;
- review UI;
- source crop display;
- confidence scoring.

### Risk: Underline detection fails

Mitigation:

- OpenCV line detection;
- search only inside sentence regions;
- manual target selection.

### Risk: Answer strip unreadable

Mitigation:

- mark answers as review-needed;
- allow manual answer entry;
- never silently guess.

### Risk: VLM hallucinates corrections

Mitigation:

- strict JSON schema;
- evidence tokens;
- answer source tracking;
- validation layer;
- block model-only answers by default.

### Risk: Duplicate cards

Mitigation:

- normalized dedupe keys;
- preserve multiple source references;
- export duplicates only when enabled.

---

## 27. Suggested Repository Structure

```text
anki-card-generator/
  apps/
    web/
      app/
      components/
      lib/
  backend/
    app/
      api/
      core/
      db/
      models/
      ocr/
      vision/
      extraction/
      validation/
      export/
      tests/
    data/
      dictionaries/
    uploads/
    processed/
    exports/
  docs/
    plan.md
    schemas.md
    prompts.md
  docker-compose.yml
  README.md
```

---

## 28. First Codex Task List

Ask Codex to implement in this order.

### Task 1

Create the monorepo structure with:

- `apps/web` Next.js app;
- `backend` FastAPI app;
- SQLite database;
- upload endpoint;
- uploaded image preview.

### Task 2

Add image preprocessing:

- load image;
- detect page crop;
- deskew;
- save processed image;
- expose original and processed images through API.

### Task 3

Add OCR adapter:

- PaddleOCR provider;
- normalized OCR token schema;
- OCR token storage;
- frontend OCR overlay.

### Task 4

Add script classifier:

- classify OCR tokens by Unicode script;
- expose script summaries;
- show token class in frontend debug panel.

### Task 5

Add vocabulary extraction:

- checkbox/row detection;
- row grouping;
- vocab item model;
- card candidate generation;
- review/edit UI;
- TSV export.

### Task 6

Add answer strip parser:

- detect bottom answer band;
- OCR answer band;
- parse question-answer pairs;
- expose answer map.

### Task 7

Add MCQ extraction:

- question block segmentation;
- choice extraction;
- underline detection;
- answer attachment;
- MCQ card generation.

### Task 8

Add local VLM integration:

- adapter interface;
- Qwen3-VL call path;
- strict JSON schema validation;
- VLM cleanup for row/question crops.

### Task 9

Add dictionary validation:

- JMdict loading;
- surface-reading validation;
- confidence/warning updates;
- duplicate detection.

---

## 29. MVP Definition of Done

The MVP is done when:

1. User can upload one of the supplied sample images.
2. App crops and preprocesses the page.
3. OCR boxes are visible on the page.
4. Vocabulary pages produce editable vocabulary card candidates.
5. MCQ pages produce editable question card candidate