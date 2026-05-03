import { describe, expect, it } from "vitest";

import {
  allFieldTokenIds,
  bboxFromSource,
  applyFieldOcrPreview,
  applyQuestionChoicesEdit,
  applyQuestionSourceEdit,
  applyVocabSourceEdit,
  candidateSubtitle,
  candidateTitle,
  cardForToken,
  cardMatchesFilter,
  choicesText,
  choicesFromText,
  clamp,
  confidenceClass,
  editableFieldBbox,
  evidenceOverlayModeForLoad,
  evidenceSourceLabel,
  evidenceSummary,
  evidenceTokensForSource,
  evidenceViewBox,
  fieldEvidence,
  fieldBbox,
  fieldLabel,
  fieldNamesForCard,
  focusBbox,
  focusedTokenIds,
  focusedTokenIdsForField,
  initialReviewCardId,
  isAnswerSupportToken,
  comparisonEvidenceAvailable,
  isHighConfidenceCard,
  normalizeBbox,
  nextReviewCardId,
  numberOrEmpty,
  provenanceLabel,
  questionFront,
  reviewReasonBadges,
  reviewQualityClass,
  sourceBbox,
  sourceChoices,
  summarizeCards,
  syncQuestionAnswerBack,
  syncVocabCardText,
  textValue,
  tokenConfidenceClass,
  tokenDisplayClass,
  tokenOnlyEvidenceClass,
  tokenInside,
  tokenTitle,
  unionBoxes,
  uniqueStrings,
  warningKey,
  workflowClass
} from "../../lib/review";
import type { CardCandidate, OcrComparison, OcrToken, Page } from "../../lib/api";

describe("StudyWorkbench evidence helpers", () => {
  it("treats 92% candidate confidence as high-confidence visual evidence", () => {
    expect(confidenceClass(candidate({ confidence: 0.92, review_state: "red" }))).toBe("confidence-high");
    expect(confidenceClass(candidate({ confidence: 0.89 }))).toBe("confidence-medium");
    expect(confidenceClass(candidate({ confidence: 0.74 }))).toBe("confidence-low");
    expect(confidenceClass(null)).toBe("confidence-unknown");
    expect(reviewQualityClass(null)).toBe("confidence-unknown");
    expect(reviewQualityClass(candidate({ confidence: 1, review_state: "red" }))).toBe("confidence-low");
    expect(reviewQualityClass(candidate({ confidence: 0.95, warnings: ["Expected exactly four choices."] }))).toBe("confidence-medium");
  });

  it("links OCR tokens to candidates by source bbox when evidence token ids are missing", () => {
    const card = candidate({ source_bbox: [10, 10, 100, 100], source: { evidence_tokens: [] } });
    const token = ocrToken({ id: "ocr-outside-explicit-evidence", bbox: [25, 25, 45, 45] });

    expect(cardForToken([card], token)?.id).toBe(card.id);
  });

  it("links and focuses field-level evidence before falling back to source evidence", () => {
    const card = candidate({
      source: {
        evidence_tokens: ["source-token"],
        field_evidence: {
          target: { bbox: [30, 40, 80, 90], token_ids: ["target-token"], text: "上" }
        }
      },
      source_bbox: [0, 0, 200, 200]
    });
    const target = ocrToken({ id: "target-token", bbox: [32, 42, 78, 88] });
    const source = ocrToken({ id: "source-token", bbox: [5, 5, 15, 15] });
    const fieldOnlyCard = candidate({
      source: {
        field_evidence: {
          target: { token_ids: ["field-only-token"], text: "上" }
        }
      },
      source_bbox: null
    });

    expect(fieldNamesForCard(card)).toContain("target");
    expect(fieldNamesForCard(candidate({ source_type: "vocab_item" }))).toEqual(["surface", "reading", "meaning_ko"]);
    expect(fieldLabel("choice_3")).toBe("Choice 3");
    expect(fieldLabel("answer_source")).toBe("answer source");
    expect(fieldBbox(card, "target")).toEqual([30, 40, 80, 90]);
    expect(editableFieldBbox(card, "target")).toEqual([30, 40, 80, 90]);
    expect(editableFieldBbox(card, "target", { cardId: card.id, field: "target", bbox: [1, 2, 3, 4] })).toEqual([
      1, 2, 3, 4
    ]);
    expect(editableFieldBbox(candidate({ source: {}, source_bbox: [9, 10, 11, 12] }), "target")).toEqual([
      9, 10, 11, 12
    ]);
    expect([...focusedTokenIdsForField(card, "target")]).toEqual(["target-token"]);
    expect(focusBbox(card, [target, source], "target")).toEqual([32, 42, 78, 88]);
    expect(cardForToken([card], target)?.id).toBe(card.id);
    expect(cardForToken([fieldOnlyCard], ocrToken({ id: "field-only-token" }))?.id).toBe(fieldOnlyCard.id);
    expect(cardForToken([candidate({ source_bbox: null, source: {} })], ocrToken())).toBeNull();
  });

  it("keeps duplicate warning React keys unique", () => {
    const warning = "Page contour confidence was low; using full image.";

    expect(warningKey(warning, 0)).not.toBe(warningKey(warning, 1));
  });

  it("summarizes and filters candidates for review lists", () => {
    const green = candidate({ id: "green", status: "approved", review_state: "green", confidence: 0.95 });
    const yellow = candidate({ id: "yellow", review_state: "yellow", warnings: ["Check answer"] });
    const red = candidate({ id: "red", review_state: "red" });

    expect(summarizeCards([green, yellow, red])).toEqual({ total: 3, approved: 1, needsReview: 1, red: 1 });
    expect(cardMatchesFilter(green, "approved")).toBe(true);
    expect(cardMatchesFilter(yellow, "approved")).toBe(false);
    expect(cardMatchesFilter(yellow, "needs_review")).toBe(true);
    expect(cardMatchesFilter(candidate({ review_state: "green", warnings: ["manual review"] }), "needs_review")).toBe(true);
    expect(cardMatchesFilter(red, "red")).toBe(true);
    expect(cardMatchesFilter(green, "yellow")).toBe(false);
    expect(cardMatchesFilter(green, "all")).toBe(true);
    expect(isHighConfidenceCard(green)).toBe(true);
    expect(isHighConfidenceCard({ ...green, warnings: ["manual review"] })).toBe(false);
    expect(initialReviewCardId([green, yellow])).toBe("yellow");
    expect(initialReviewCardId([green])).toBe("green");
    expect(initialReviewCardId([])).toBeNull();
    expect(nextReviewCardId([yellow, green, red], "yellow")).toBeNull();
    expect(nextReviewCardId([{ ...yellow, status: "approved" }], "yellow")).toBeNull();
    expect(nextReviewCardId([yellow, candidate({ id: "next", review_state: "yellow" })], "yellow")).toBe("next");
    expect(nextReviewCardId([candidate({ id: "first", review_state: "yellow" }), green], "missing")).toBe("first");
  });

  it("marks workflow steps complete only when their prerequisites exist", () => {
    const page = appPage({ page_type: "uploaded" });
    const processedPage = appPage({ page_type: "reading_mcq" });
    const cards = [candidate()];

    expect(workflowClass(0, page, [], 0)).toBe("step complete");
    expect(workflowClass(0, null, [], 0)).toBe("step");
    expect(workflowClass(1, page, [], 0)).toBe("step");
    expect(workflowClass(1, processedPage, [], 0)).toBe("step complete");
    expect(workflowClass(2, processedPage, cards, 0)).toBe("step complete");
    expect(workflowClass(3, processedPage, cards, 1)).toBe("step complete");
    expect(workflowClass(3, processedPage, cards, 0)).toBe("step");
  });

  it("builds confidence-aware token display metadata", () => {
    const page = appPage({ page_type: "reading_mcq", image_height: 1000 });
    const selected = candidate({ confidence: 0.95, source: { question_no: 1, target: "うえ" } });
    const token = ocrToken({ text: "1①", bbox: [10, 900, 20, 920], confidence: 0.72 });

    expect(tokenDisplayClass(token, page, selected, selected, true)).toContain("selected-evidence confidence-high");
    expect(tokenDisplayClass(token, page, null, candidate({ confidence: 1, review_state: "red" }), false)).toBe("candidate-evidence confidence-low");
    expect(tokenDisplayClass(token, page, null, candidate({ confidence: 0.95, review_state: "yellow" }), false)).toBe(
      "candidate-evidence confidence-medium"
    );
    expect(tokenDisplayClass(token, page, null, null, false)).toBe("used-context answer-support");
    expect(tokenDisplayClass(ocrToken({ text: "abc", confidence: 0.5 }), appPage(), null, null, false)).toBe(
      "scanned-unused token-confidence-low"
    );
    expect(evidenceOverlayModeForLoad([], [token])).toBe("all");
    expect(evidenceOverlayModeForLoad([selected], [token])).toBe("focused");
    expect(evidenceOverlayModeForLoad([], [])).toBe("focused");
    expect(tokenOnlyEvidenceClass(token)).toBe("token-review token-confidence-low");
    expect(tokenConfidenceClass(0.95)).toBe("high");
    expect(tokenConfidenceClass(0.8)).toBe("medium");
    expect(tokenConfidenceClass(0.2)).toBe("low");
    expect(tokenTitle(token, selected, false)).toContain("used by Question 1");
    expect(tokenTitle(token, null, true)).toContain("selected candidate evidence");
    expect(tokenTitle(token, null, false)).toContain("scanned but unused");
    expect(isAnswerSupportToken(token, page)).toBe(true);
    expect(isAnswerSupportToken(ocrToken({ text: "答", bbox: [10, 900, 20, 920] }), page)).toBe(false);
    expect(isAnswerSupportToken(token, appPage({ image_height: 0, page_type: "reading_mcq" }))).toBe(false);
    expect(isAnswerSupportToken(token, appPage({ page_type: "vocab_table" }))).toBe(false);
    expect(isAnswerSupportToken(ocrToken({ text: "1①", bbox: [10, 100, 20, 120] }), page)).toBe(false);
  });

  it("switches visual evidence tokens to Google Vision when comparison boxes exist", () => {
    const local = [ocrToken({ id: "local-token", source: "paddleocr" })];
    const google = [ocrToken({ id: "google-token", source: "google_vision" })];
    const comparison: OcrComparison = {
      primary_provider: "paddleocr",
      compare_provider: "google_vision",
      primary_token_count: 1,
      compare_token_count: 1,
      agreement: 0.5,
      missing_from_primary: [],
      missing_from_comparison: [],
      compare_tokens: google,
      warnings: []
    };

    expect(comparisonEvidenceAvailable(comparison)).toBe(true);
    expect(evidenceTokensForSource("comparison", local, comparison)).toBe(google);
    expect(evidenceSourceLabel("comparison", comparison)).toBe("Google Vision");
    expect(evidenceTokensForSource("comparison", local, { ...comparison, compare_tokens: [] })).toBe(local);
    expect(evidenceSourceLabel("comparison", { ...comparison, compare_tokens: [] })).toBe("Local OCR");
    expect(comparisonEvidenceAvailable(null)).toBe(false);
  });

  it("computes focused evidence geometry", () => {
    const card = candidate({ source_bbox: [0, 0, 100, 100], source: { evidence_tokens: ["token-a", 3, "token-b"] } });
    const tokens = [
      ocrToken({ id: "token-a", bbox: [5, 10, 30, 35] }),
      ocrToken({ id: "token-b", bbox: [40, 20, 70, 60] }),
      ocrToken({ id: "other", bbox: [200, 200, 250, 250] })
    ];

    expect([...focusedTokenIds(card)]).toEqual(["token-a", "token-b"]);
    expect([...focusedTokenIdsForField(card, "missing")]).toEqual(["token-a", "token-b"]);
    expect([...focusedTokenIdsForField(candidate({ source: {} }), null)]).toEqual([]);
    expect(unionBoxes(tokens.slice(0, 2).map((token) => token.bbox))).toEqual([5, 10, 70, 60]);
    expect(unionBoxes([[1, 2, 3]])).toBeNull();
    expect(focusBbox(card, tokens)).toEqual([5, 10, 70, 60]);
    expect(focusBbox(candidate({ source_bbox: [8, 9, 10, 11], source: {} }), [])).toEqual([8, 9, 10, 11]);
    expect(sourceBbox(candidate({ source_bbox: null, source: { bbox: [1, 2, 3, 4] } }))).toEqual([1, 2, 3, 4]);
    expect(sourceBbox(candidate({ source_bbox: null, source: {} }))).toBeNull();
    expect(bboxFromSource({ bbox: [1, 2, "bad", 4] })).toBeNull();
    expect(bboxFromSource({ bbox: [1, 2, 3] })).toBeNull();
    expect(tokenInside(ocrToken({ bbox: [20, 20, 30, 30] }), [10, 10, 40, 40])).toBe(true);
    expect(tokenInside(ocrToken({ bbox: [80, 80, 90, 90] }), [10, 10, 40, 40])).toBe(false);
    expect(tokenInside(ocrToken({ bbox: [20, 20, 20, 30] }), [10, 10, 40, 40])).toBe(false);
  });

  it("normalizes reversed OCR-VL bounding boxes before geometry checks", () => {
    expect(normalizeBbox([40, 30, 10, 5])).toEqual([10, 5, 40, 30]);
    expect(normalizeBbox([10, 5, 10, 30])).toBeNull();
    expect(normalizeBbox(["1", "2", "3", "4"])).toEqual([1, 2, 3, 4]);
    expect(normalizeBbox(["1", "bad", "3", "4"])).toBeNull();
    expect(normalizeBbox([1, 2, 3])).toBeNull();
    expect(unionBoxes([[40, 30, 10, 5], [45, 35, 50, 40]])).toEqual([10, 5, 50, 40]);
    expect(tokenInside(ocrToken({ bbox: [40, 30, 10, 5] }), [0, 0, 50, 50])).toBe(true);
  });

  it("computes evidence viewport with clamped bounds", () => {
    const page = appPage({ image_width: 1000, image_height: 2000 });

    expect(evidenceViewBox(page, null)).toBe("0 0 1000 2000");
    expect(evidenceViewBox(appPage({ image_width: undefined, image_height: undefined }), [1, 2, 3, 4])).toBe("0 0 1 1");
    expect(evidenceViewBox(appPage({ image_width: 0, image_height: 0 }), [1, 2, 3, 4])).toBe("0 0 0 0");
    expect(evidenceViewBox(page, [900, 1800, 980, 1900])).toBe("456 1316 544 684");
    expect(clamp(12, 0, 10)).toBe(10);
    expect(clamp(-2, 0, 10)).toBe(0);
  });

  it("formats candidate and source text helpers", () => {
    const vocab = candidate({
      source_type: "vocab_item",
      source: { surface: "学校", reading: "がっこう" },
      note_type: "jp_vocab_reading"
    });
    const question = candidate({
      note_type: "jp_spelling_mcq_recall",
      source: { question_no: 6, target: "なまえ", correct_answer: "名前", correct_choice_no: 2 }
    });

    expect(candidateTitle(vocab)).toBe("学校 · がっこう");
    expect(candidateTitle(candidate({ source_type: "vocab_item", source: {} }))).toBe("Vocab · ");
    expect(candidateTitle(question)).toBe("Question 6 · なまえ");
    expect(candidateTitle(candidate({ source: {} }))).toBe("Question ? · target");
    expect(candidateSubtitle(question)).toBe("jp_spelling_mcq_recall · OCR 92%");
    expect(provenanceLabel("answer_strip")).toBe("Answer key");
    expect(provenanceLabel("local_glossary")).toBe("Glossary inferred");
    expect(provenanceLabel("manual")).toBe("Manual edit");
    expect(provenanceLabel("model_inferred")).toBe("model inferred");
    expect(textValue(true)).toBe("true");
    expect(textValue(null, "fallback")).toBe("fallback");
    expect(numberOrEmpty("")).toBe("");
    expect(numberOrEmpty("7")).toBe(7);
    expect(numberOrEmpty("oops")).toBe("oops");
    expect(choicesFromText(" 天気 \n\n夫気\n夫气 ")).toEqual(["天気", "夫気", "夫气"]);
    expect(choicesText({ choices: ["上", "止", "午", "下"] })).toBe("上\n止\n午\n下");
    expect(uniqueStrings(["a", "a", "b"])).toEqual(["a", "b"]);
    expect(syncQuestionAnswerBack({ ...question, note_type: "jp_spelling_mcq_exam" }, question.source).back).toBe("정답: 2. 名前");
    expect(syncQuestionAnswerBack({ ...question, note_type: "jp_spelling_mcq_exam" }, { correct_answer: "上" }).back).toBe("정답: ?. 上");
    expect(syncQuestionAnswerBack(question, question.source).back).toBe("名前");
    expect(syncQuestionAnswerBack(question, { correct_answer: "" }).back).toBe(question.back);
    expect(questionFront(question)).toContain("올바른 표기는?");
    expect(questionFront({ ...question, note_type: "jp_reading_mcq_recall" })).toContain("읽는 법?");
    expect(sourceChoices({ choices: ["1", 2] })).toEqual(["1", "2"]);
    expect(sourceChoices({ choices: "上" })).toEqual([]);
  });

  it("updates question choices and derived answer summaries together", () => {
    const card = candidate({
      review_state: "red",
      warnings: ["Expected exactly four choices."],
      source: { question_no: 1, target: "うえ", correct_choice_no: 1, correct_answer: "old", choices: ["old"] }
    });

    const updatedChoices = applyQuestionChoicesEdit(card, "上\n止\n午\n下\n");

    expect(updatedChoices.source.choices).toEqual(["上", "止", "午", "下"]);
    expect(updatedChoices.source.correct_answer).toBe("上");
    expect(updatedChoices.back).toBe("上");
    expect(updatedChoices.review_state).toBe("yellow");
    expect(updatedChoices.warnings).toEqual([]);

    const updatedChoiceNo = applyQuestionSourceEdit(updatedChoices, "correct_choice_no", "3");
    expect(updatedChoiceNo.source.correct_answer).toBe("午");
    expect(updatedChoiceNo.back).toBe("午");

    const updatedTarget = applyQuestionSourceEdit(updatedChoices, "target", "した");
    expect(updatedTarget.source.target).toBe("した");

    const missingChoice = applyQuestionChoicesEdit(updatedChoices, "上\n止");
    expect(missingChoice.source.correct_answer).toBe("上");
    expect(missingChoice.warnings).toContain("Expected exactly four choices.");

    const unmatchedChoiceNo = applyQuestionSourceEdit(updatedChoices, "correct_choice_no", "9");
    expect(unmatchedChoiceNo.source.correct_answer).toBe("上");

    const unmatchedChoices = applyQuestionChoicesEdit({ ...updatedChoices, source: { ...updatedChoices.source, correct_choice_no: 9 } }, "上\n止\n午\n下");
    expect(unmatchedChoices.source.correct_answer).toBe("上");
  });

  it("separates provenance labels from review reason badges", () => {
    expect(reviewReasonBadges(candidate({ warnings: ["Expected exactly four choices."] }))).toEqual(["Missing choice"]);
    expect(reviewReasonBadges(candidate({ warnings: ["Target text unclear."] }))).toEqual(["Target unclear"]);
    expect(reviewReasonBadges(candidate({ warnings: ["Answer source not export-safe."] }))).toEqual(["Needs answer"]);
    expect(reviewReasonBadges(candidate({ confidence: 0.72 }))).toEqual(["Low OCR confidence"]);
    expect(reviewReasonBadges(candidate({ review_state: "red", warnings: [], confidence: 0.92 }))).toEqual(["Blocked"]);
    expect(reviewReasonBadges(candidate({ review_state: "yellow", warnings: [], confidence: 0.92 }))).toEqual(["Needs review"]);
    expect(reviewReasonBadges(candidate({ source: { answer_source: "local_glossary" }, warnings: [] }))).toEqual([]);
  });

  it("applies field OCR previews to source, evidence, and derived answer text", () => {
    const card = candidate({
      source: { choices: ["天気", "夫気"], correct_choice_no: 2, correct_answer: "夫気" },
      warnings: ["Could not extract exactly four choices."],
      review_state: "red"
    });

    const updated = applyFieldOcrPreview(card, {
      card_id: card.id,
      page_id: card.page_id,
      field: "choice_2",
      bbox: [10, 20, 30, 40],
      provider: "paddle",
      text: "天気",
      confidence: 0.97,
      tokens: [],
      suggested_source: { choices: ["天気", "天気", "", ""], correct_answer: "天気" },
      field_evidence: { bbox: [10, 20, 30, 40], text: "天気" },
      worker: { state: "running" },
      warnings: []
    });

    expect(updated.source.choices).toEqual(["天気", "天気", "", ""]);
    expect(updated.source.correct_answer).toBe("天気");
    expect(updated.source.field_evidence).toMatchObject({ choice_2: { text: "天気" } });
    expect(choicesText(updated.source)).toBe("天気\n天気\n\n");
    expect(updated.back).toBe("天気");
    expect(updated.warnings).toEqual([]);

    const sentenceUpdated = applyFieldOcrPreview(card, {
      card_id: card.id,
      page_id: card.page_id,
      field: "sentence",
      bbox: [1, 2, 3, 4],
      provider: "paddle",
      text: "にわに しろい はなが さきました。",
      confidence: 0,
      tokens: [],
      suggested_source: { sentence: "にわに しろい はなが さきました。" },
      field_evidence: { bbox: [1, 2, 3, 4], text: "にわに しろい はなが さきました." },
      worker: { state: "running" },
      warnings: []
    });
    expect(sentenceUpdated.front).toContain("にわに しろい はなが さきました。");
    expect(sentenceUpdated.confidence).toBe(card.confidence);

    const withExistingEvidence = applyFieldOcrPreview(
      candidate({ source: { field_evidence: { sentence: { text: "old" } }, choices: ["上"], correct_choice_no: 1 } }),
      {
        card_id: card.id,
        page_id: card.page_id,
        field: "target",
        bbox: [5, 6, 7, 8],
        provider: "paddle",
        text: "上",
        confidence: 0.91,
        tokens: [],
        suggested_source: { target: "上" },
        field_evidence: { bbox: [5, 6, 7, 8], text: "上" },
        worker: { state: "running" },
        warnings: []
      }
    );
    expect(withExistingEvidence.source.field_evidence).toMatchObject({ sentence: { text: "old" }, target: { text: "上" } });
  });

  it("updates vocabulary facts and derived semantic cards together", () => {
    const card = candidate({
      source_type: "vocab_item",
      note_type: "jp_vocab_writing",
      source: { surface: "学校", reading: "がっこう", meaning_ko: "학교" },
      front: "old front",
      back: "old back"
    });

    const edited = applyVocabSourceEdit(card, "surface", "先生");

    expect(edited.source.surface).toBe("先生");
    expect(edited.front).toContain("がっこう");
    expect(edited.back).toBe("先生");

    const previewed = applyFieldOcrPreview(edited, {
      card_id: card.id,
      page_id: card.page_id,
      field: "reading",
      bbox: [10, 20, 30, 40],
      provider: "paddle",
      text: "せんせい",
      confidence: 0.94,
      tokens: [],
      suggested_source: { reading: "せんせい" },
      field_evidence: { bbox: [10, 20, 30, 40], text: "せんせい" },
      worker: { state: "running" },
      warnings: []
    });

    expect(previewed.source.reading).toBe("せんせい");
    expect(previewed.front).toContain("せんせい");
    expect(previewed.back).toBe("先生");

    const readingCard = { ...card, note_type: "jp_vocab_reading" };
    const meaningCard = { ...card, note_type: "jp_vocab_meaning" };
    const unknownCard = { ...card, note_type: "unknown_vocab" };
    expect(syncVocabCardText(readingCard, { surface: "<語>", reading: "ご", meaning_ko: "뜻&의미" }).front).toContain("&lt;語&gt;");
    expect(syncVocabCardText(readingCard, { surface: "<語>", reading: "ご", meaning_ko: "뜻&의미" }).back).toBe("ご");
    expect(syncVocabCardText(meaningCard, meaningCard.source).back).toBe("학교");
    expect(syncVocabCardText(unknownCard, unknownCard.source)).toBe(unknownCard);
    expect(syncVocabCardText(candidate({ source_type: "question_item" }), {})).toEqual(candidate({ source_type: "question_item" }));
  });

  it("summarizes evidence availability", () => {
    expect(evidenceSummary(candidate({ source: { evidence_tokens: ["token-1"] } }))).toContain("1 evidence tokens");
    expect(evidenceSummary(candidate({ source: {}, source_bbox: [1, 2, 3, 4] }))).toBe(
      "Source region is highlighted for the selected candidate."
    );
    expect(evidenceSummary(candidate({ source: {}, source_bbox: null }))).toContain("switch to All OCR");
  });

  it("guards field evidence extraction against malformed source data", () => {
    const fieldCard = candidate({
      source: {
        field_evidence: {
          target: { bbox: [1, 2, 3, 4], token_ids: ["a", 3, "b"] },
          bad: ["not-object"],
          empty: null
        }
      }
    });

    expect(fieldEvidence(null, "target")).toBeNull();
    expect(fieldEvidence(fieldCard, null)).toBeNull();
    expect(fieldEvidence(candidate({ source: { field_evidence: [] } }), "target")).toBeNull();
    expect(fieldEvidence(fieldCard, "bad")).toBeNull();
    expect(fieldEvidence(fieldCard, "empty")).toBeNull();
    expect(fieldEvidence(fieldCard, "target")).toMatchObject({ bbox: [1, 2, 3, 4] });
    expect([...allFieldTokenIds(fieldCard)]).toEqual(["a", "b"]);
    expect([...allFieldTokenIds(candidate({ source: { field_evidence: { bad: [], noTokens: {} } } }))]).toEqual([]);
    expect(editableFieldBbox(fieldCard, "target", { cardId: fieldCard.id, field: "target", bbox: [9, 9, 9, 10] })).toEqual([1, 2, 3, 4]);
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

function appPage(overrides: Partial<Page> = {}): Page {
  return {
    id: "page-1",
    original_image_path: "/uploads/page.jpg",
    processed_image_path: null,
    page_type: "uploaded",
    page_type_confidence: 0.9,
    image_width: 1000,
    image_height: 1000,
    warnings: [],
    created_at: "2026-04-28T00:00:00Z",
    ...overrides
  };
}
