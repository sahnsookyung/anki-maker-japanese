import { describe, expect, it } from "vitest";

import {
  cardForToken,
  confidenceClass,
  warningKey
} from "../../components/StudyWorkbench";
import type { CardCandidate, OcrToken } from "../../lib/api";

describe("StudyWorkbench evidence helpers", () => {
  it("treats 92% candidate confidence as high-confidence visual evidence", () => {
    expect(confidenceClass(candidate({ confidence: 0.92, review_state: "red" }))).toBe("confidence-high");
    expect(confidenceClass(candidate({ confidence: 0.89 }))).toBe("confidence-medium");
    expect(confidenceClass(candidate({ confidence: 0.74 }))).toBe("confidence-low");
  });

  it("links OCR tokens to candidates by source bbox when evidence token ids are missing", () => {
    const card = candidate({ source_bbox: [10, 10, 100, 100], source: { evidence_tokens: [] } });
    const token = ocrToken({ id: "ocr-outside-explicit-evidence", bbox: [25, 25, 45, 45] });

    expect(cardForToken([card], token)?.id).toBe(card.id);
  });

  it("keeps duplicate warning React keys unique", () => {
    const warning = "Page contour confidence was low; using full image.";

    expect(warningKey(warning, 0)).not.toBe(warningKey(warning, 1));
  });
});

function candidate(overrides: Partial<CardCandidate> = {}): CardCandidate {
  return {
    id: "card-1",
    page_id: "page-1",
    source_type: "question_item",
    source_id: "q-1",
    source: { evidence_tokens: ["token-1"] },
    note_type: "jp_reading_mcq_recall",
    front: "front",
    back: "back",
    tags: ["jlpt"],
    confidence: 0.92,
    status: "pending_review",
    review_state: "green",
    source_bbox: [0, 0, 50, 50],
    warnings: [],
    ...overrides
  };
}

function ocrToken(overrides: Partial<OcrToken> = {}): OcrToken {
  return {
    id: "token-1",
    page_id: "page-1",
    text: "その",
    bbox: [5, 5, 20, 20],
    confidence: 0.95,
    script_class: "hiragana",
    source: "paddleocr",
    ...overrides
  };
}
