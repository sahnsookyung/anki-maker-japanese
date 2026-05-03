# UX Heuristic Review

This review uses practical Nielsen-style heuristics for a local OCR review tool: visibility of system status, match to the user's task language, user control and recovery, consistency, error prevention, recognition over recall, efficient workflows, minimalist presentation, useful diagnostics, and persistence/reliability.

## Current Score

Overall: 7.8 / 10 after the May 3 UI pass.

| Heuristic | Score | Notes |
| --- | ---: | --- |
| System status visibility | 8 | Current job state, latest batch stats, and per-page timings are visible. The compact sticky batch strip keeps total, avg, min, p25, p75, and max accessible while scrolling. |
| Match to user workflow | 8 | The app is organized around Upload -> Process -> Review -> Export, and page/candidate language mostly matches the workbook review task. |
| User control and recovery | 8 | Rename, delete, reprocess, approval toggles, and field edits are available. Destructive delete still requires confirmation. |
| Consistency and standards | 7 | PaddleOCR vs OCR-VL actions are visually distinct, but the page-level action cluster remains dense on small screens. |
| Error prevention | 7 | Runtime guards, sequential processing, isolated benchmarks, and stale-evidence hiding reduce failure risk. OCR-VL can still fail because the local model is heavy. |
| Recognition over recall | 8 | Evidence colors, badges, timing strips, and field-linked controls reduce memorization. Advanced diagnostics remain tucked away. |
| Efficiency | 8 | Sequential batch processing releases page results as they finish; persisted OCR evidence survives reloads. |
| Minimalism | 7 | The interface is clearer than the earlier all-OCR-first workbench, but the hero and page actions still consume a lot of vertical space. |
| Diagnosis and recovery | 8 | Warnings, benchmark source-field accuracy, browser-error e2e tests, and OCR runtime status expose failures more honestly. |
| Persistence and data integrity | 9 | Reloaded OCR tokens/cards are retrieved from SQLite, same-filename uploads replace state, deletion clears runtime artifacts, and fresh benchmarks no longer write into app state. |

## Improvements Completed In This Pass

- Added a sticky command/status bar so the latest batch timing summary remains available while reviewing.
- Added a compact timing strip with total, average, min, p25, p75, and max processing times.
- Made token-only OCR pages open in `All OCR` mode after run/reload so OCR-VL pages with no generated cards still show visual evidence.
- Added visible token-only OCR box styling and included `paddleocr_vl` in the OCR filter list.
- Added an e2e reload test proving persisted token-only OCR evidence remains visible after browser reload.
- Made `evaluate_golden.py` use an isolated temp DB and processed-image directory for fresh runs, leaving `--from-db` as the explicit UI-persistence evaluator.
- Updated benchmark page metadata to match UI uploads more closely.

## Remaining Optional Improvements

- Compress the hero area after the first upload so the review workspace appears sooner on smaller screens.
- Group page-level actions into a primary `Process` action plus an overflow/advanced menu once the page card gets narrow.
- Add a small benchmark comparison panel in the UI that explains semantic accuracy vs source-field accuracy without requiring CLI output.
- Add visual diffing for screenshots at narrow and desktop widths if the UI stabilizes enough to make pixel thresholds useful.
