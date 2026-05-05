# CSV Vocab Export Adversarial Review

This review covers the CSV-only vocabulary-note change after shifting vocab pages from three duplicate review cards to one `jp_vocab_entry` note candidate per `(Surface, Reading, MeaningKo)` row.

## 1. Adversarial Critique

- The implementation correctly changes the Anki shape, but the benchmark still shows weak vocab extraction quality: category 1 scored 22/36 rows and category 3 scored 10/24 rows with strict OCR-only PaddleOCR scoring. Calling this a solved vocab pipeline would be misleading.
- The first pass over-preserved old vocab card types by trying to recover `jp_vocab_reading`, `jp_vocab_meaning`, and `jp_vocab_writing`. That contradicts the product decision: those shapes should disappear, not become a second supported migration pathway.
- The generic CSV helper silently chose a schema when given mixed cards. That is dangerous because mixed vocab/MCQ export is now intentionally multi-file; single-file helpers must reject mixed schemas rather than drop one side.
- A vocab note with every legacy study direction disabled generates no useful Anki card. The UI should no longer expose those legacy toggles, and the export path should omit explicit zero-output notes.
- Opening every generated CSV in a new browser tab is fragile. Popup blockers can suppress downloads, and users need persistent links they can retry after export.

## 2. Defense Of The Current Direction

- One `jp_vocab_entry` note per row is the right Anki model. It removes duplicate review-state noise in the UI while letting one Anki template generate the pronunciation-to-kanji card and hide support fields such as Korean meaning.
- CSV-only is the correct boundary for this repo right now. It keeps import behavior inspectable, cheap to test, and compatible with Anki's built-in text import flow.
- Keeping SQLite remains appropriate for a local-first single-user OCR review app. The active data model is page/run/token/card oriented and does not need Postgres until there is multi-user concurrency, remote sync, or large shared datasets.
- The lower vocab benchmark scores are not caused by the new note model. They expose the existing OCR/extraction limits more honestly because glossary supplementation is not allowed to inflate row accuracy.

## 3. Unbiased Conclusion

The strongest argument is that the note-model change is correct, but the implementation should be stricter about the visible/generated card shape. The code should delete unsupported vocab card shapes, reject ambiguous single-file CSV exports, collapse legacy enabled directions into the one kana-to-kanji card, prevent zero-card vocab notes, and make exported CSV files available through stable UI links. The benchmark should continue reporting low vocab accuracy honestly; improving OCR extraction is a separate pipeline task, not something the export layer should hide.

## 4. Fixes Applied From This Review

- Unsupported vocab note types are deleted from SQLite instead of being converted or preserved.
- Single-file CSV helpers now reject empty or mixed-schema input.
- Export only writes `jp_vocab_entry` notes with required facts and a generated kana-to-kanji card.
- The vocab editor shows one generated card preview instead of legacy writing/reading/meaning toggles.
- Export results are shown as persistent CSV download links instead of relying on multiple popup tabs.
- Vocab evaluation reports `generated_notes`; MCQ evaluation keeps `generated_cards`.
