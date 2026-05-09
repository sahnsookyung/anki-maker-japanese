import type { CardCandidate, DocumentParseBlock, FieldOcrPreview, OcrComparison, OcrToken, Page } from "./api";

export type ReviewFilter = "all" | "needs_review" | "approved" | "green" | "yellow" | "red";
export type EvidenceOverlayMode = "focused" | "region" | "all" | "off";
export type EvidenceTokenSource = "local" | "comparison";

export const HIGH_CONFIDENCE_THRESHOLD = 0.9;
export const REVIEW_CONFIDENCE_THRESHOLD = 0.75;

export function workflowClass(index: number, page: Page | null, cards: CardCandidate[], exportableCount: number): string {
  const complete =
    (index === 0 && page) ||
    (index === 1 && page?.page_type && page.page_type !== "uploaded") ||
    (index === 2 && cards.length > 0) ||
    (index === 3 && exportableCount > 0);
  return complete ? "step complete" : "step";
}

export function summarizeCards(cards: CardCandidate[]) {
  return {
    total: cards.length,
    approved: cards.filter((card) => card.status === "approved").length,
    needsReview: cards.filter((card) => card.review_state === "yellow" || card.warnings.length).length,
    red: cards.filter((card) => card.review_state === "red").length
  };
}

export function cardMatchesFilter(card: CardCandidate, filter: ReviewFilter): boolean {
  if (filter === "all") return true;
  if (filter === "needs_review") return card.review_state === "yellow" || card.warnings.length > 0;
  if (filter === "approved") return card.status === "approved";
  return card.review_state === filter;
}

export function isHighConfidenceCard(card: CardCandidate): boolean {
  return card.review_state === "green" && card.warnings.length === 0 && card.confidence >= HIGH_CONFIDENCE_THRESHOLD;
}

export function initialReviewCardId(cards: CardCandidate[]): string | null {
  return cards.find((card) => !isHighConfidenceCard(card))?.id ?? cards[0]?.id ?? null;
}

export function nextReviewCardId(cards: CardCandidate[], currentCardId: string): string | null {
  const currentIndex = cards.findIndex((card) => card.id === currentCardId);
  const afterCurrent = currentIndex >= 0 ? cards.slice(currentIndex + 1) : cards;
  return (
    afterCurrent.find((card) => card.status !== "approved" && card.review_state !== "red")?.id ??
    cards.find((card) => card.status !== "approved" && card.review_state !== "red" && card.id !== currentCardId)?.id ??
    null
  );
}

export function reviewQualityClass(card: CardCandidate | null): string {
  if (!card) return "confidence-unknown";
  if (card.review_state === "red") return "confidence-low";
  if (card.review_state === "yellow" || card.warnings.length > 0) return "confidence-medium";
  return confidenceClass(card);
}

export function cardForToken(cards: CardCandidate[], token: OcrToken): CardCandidate | null {
  return (
    cards.find((card) => focusedTokenIds(card).has(token.id)) ??
    cards.find((card) => allFieldTokenIds(card).has(token.id)) ??
    cards.find((card) => {
      const bbox = sourceBbox(card);
      return bbox ? tokenInside(token, bbox) : false;
    }) ??
    null
  );
}

export function cardForDocumentBlock(cards: CardCandidate[], block: DocumentParseBlock): CardCandidate | null {
  return (
    cards.find((card) => focusedDocumentBlockIds(card).has(block.id ?? "")) ??
    cards.find((card) => allFieldDocumentBlockIds(card).has(block.id ?? "")) ??
    cards.find((card) => {
      const bbox = sourceBbox(card);
      const blockBox = normalizeBbox(block.bbox);
      return bbox && blockBox ? boxCentersOverlap(blockBox, bbox) : false;
    }) ??
    null
  );
}

export function warningKey(warning: string, index: number): string {
  return `${warning}-${index}`;
}

export function tokenDisplayClass(
  token: OcrToken,
  page: Page,
  selectedCard: CardCandidate | null,
  linkedCard: CardCandidate | null,
  relevant: boolean,
): string {
  if (relevant) return `candidate-evidence selected-evidence ${reviewQualityClass(selectedCard)}`;
  if (linkedCard) return `candidate-evidence ${reviewQualityClass(linkedCard)}`;
  if (isAnswerSupportToken(token, page)) return "used-context answer-support";
  return `scanned-unused token-confidence-${tokenConfidenceClass(token.confidence)}`;
}

export function evidenceOverlayModeForLoad(cards: CardCandidate[], tokens: OcrToken[]): EvidenceOverlayMode {
  return cards.length === 0 && tokens.length > 0 ? "all" : "focused";
}

export function shouldShowDocumentBlocks(page: Page | null, tokens: OcrToken[], hasDocumentBlocks: boolean): boolean {
  if (!page || !hasDocumentBlocks) return false;
  return page.active_ocr_engine === "paddleocr_vl" || tokens.length === 0;
}

export function tokenOnlyEvidenceClass(token: OcrToken): string {
  return `token-review token-confidence-${tokenConfidenceClass(token.confidence)}`;
}

export function comparisonEvidenceAvailable(comparison: OcrComparison | null | undefined): boolean {
  return Boolean(comparison?.compare_tokens.length);
}

export function evidenceTokensForSource(
  source: EvidenceTokenSource,
  localTokens: OcrToken[],
  comparison: OcrComparison | null | undefined
): OcrToken[] {
  if (source === "comparison" && comparisonEvidenceAvailable(comparison)) return comparison?.compare_tokens ?? [];
  return localTokens;
}

export function evidenceSourceLabel(source: EvidenceTokenSource, comparison: OcrComparison | null | undefined): string {
  if (source === "comparison" && comparisonEvidenceAvailable(comparison)) {
    return comparison?.compare_provider === "google_vision" ? "Google Vision" : comparison?.compare_provider ?? "Comparison OCR";
  }
  return "Local OCR";
}

export function confidenceClass(card: CardCandidate | null): string {
  if (!card) return "confidence-unknown";
  if (card.confidence < REVIEW_CONFIDENCE_THRESHOLD) return "confidence-low";
  if (card.confidence < HIGH_CONFIDENCE_THRESHOLD) return "confidence-medium";
  return "confidence-high";
}

export function tokenConfidenceClass(confidence: number): string {
  if (confidence >= HIGH_CONFIDENCE_THRESHOLD) return "high";
  if (confidence >= REVIEW_CONFIDENCE_THRESHOLD) return "medium";
  return "low";
}

export function tokenTitle(token: OcrToken, linkedCard: CardCandidate | null, relevant: boolean): string {
  let use = "scanned but unused";
  if (relevant) {
    use = "selected candidate evidence";
  } else if (linkedCard) {
    use = `used by ${candidateTitle(linkedCard)}`;
  }
  return `${token.text} (${token.source}, ${token.script_class}, OCR ${Math.round(token.confidence * 100)}%, ${use})`;
}

export function documentBlockDisplayClass(
  block: DocumentParseBlock,
  selectedCard: CardCandidate | null,
  linkedCard: CardCandidate | null,
  relevant: boolean
): string {
  if (relevant) return `document-block selected-evidence ${reviewQualityClass(selectedCard)}`;
  if (linkedCard) return `document-block linked-document-block ${reviewQualityClass(linkedCard)}`;
  const confidence = typeof block.confidence === "number" ? block.confidence : 0.68;
  return `document-block scanned-unused token-confidence-${tokenConfidenceClass(confidence)}`;
}

export function documentBlockTitle(block: DocumentParseBlock, linkedCard: CardCandidate | null, relevant: boolean): string {
  let use = "unused PaddleOCR-VL document block";
  if (relevant) {
    use = "selected candidate document block";
  } else if (linkedCard) {
    use = `used by ${candidateTitle(linkedCard)}`;
  }
  const confidence = typeof block.confidence === "number" ? `, ${Math.round(block.confidence * 100)}%` : "";
  return `${block.label}${confidence}: ${block.content.slice(0, 120)} (${use})`;
}

export function isAnswerSupportToken(token: OcrToken, page: Page): boolean {
  if (!page.image_height || !page.page_type.endsWith("_mcq")) return false;
  if (token.bbox[1] < page.image_height * 0.82) return false;
  return /[①②③④⑤⑥⑦⑧⑨⑩0-9]/.test(token.text);
}

export function focusedTokenIds(card: CardCandidate | null): Set<string> {
  return focusedTokenIdsForField(card, null);
}

export function focusedDocumentBlockIds(card: CardCandidate | null): Set<string> {
  return focusedDocumentBlockIdsForField(card, null);
}

export function focusedTokenIdsForField(card: CardCandidate | null, field: string | null): Set<string> {
  if (field) {
    const tokens = fieldEvidence(card, field)?.token_ids;
    if (Array.isArray(tokens)) return new Set(tokens.filter((token): token is string => typeof token === "string"));
  }
  const tokens = card?.source.evidence_tokens;
  if (!Array.isArray(tokens)) return new Set();
  return new Set(tokens.filter((token): token is string => typeof token === "string"));
}

export function focusedDocumentBlockIdsForField(card: CardCandidate | null, field: string | null): Set<string> {
  if (field) {
    const blocks = fieldEvidence(card, field)?.block_ids;
    if (Array.isArray(blocks)) return new Set(blocks.filter((block): block is string => typeof block === "string"));
  }
  const blocks = card?.source.evidence_blocks;
  if (Array.isArray(blocks)) return new Set(blocks.filter((block): block is string => typeof block === "string"));
  return new Set();
}

export function focusBbox(card: CardCandidate | null, tokens: OcrToken[], field: string | null = null): number[] | null {
  const relevantIds = focusedTokenIdsForField(card, field);
  const relevantBoxes = tokens.filter((token) => relevantIds.has(token.id)).map((token) => token.bbox);
  return unionBoxes(relevantBoxes) ?? fieldBbox(card, field) ?? sourceBbox(card);
}

export function sourceBbox(card: CardCandidate | null): number[] | null {
  return normalizeBbox(card?.source_bbox) ?? bboxFromSource(card?.source);
}

export function unionBoxes(boxes: number[][]): number[] | null {
  const valid = boxes.map((box) => normalizeBbox(box)).filter((box): box is number[] => Boolean(box));
  if (!valid.length) return null;
  return [
    Math.min(...valid.map((box) => box[0])),
    Math.min(...valid.map((box) => box[1])),
    Math.max(...valid.map((box) => box[2])),
    Math.max(...valid.map((box) => box[3]))
  ];
}

export function evidenceViewBox(page: Page, bbox: number[] | null): string {
  const normalized = normalizeBbox(bbox);
  if (!normalized || !page.image_width || !page.image_height) {
    return `0 0 ${page.image_width ?? 1} ${page.image_height ?? 1}`;
  }
  const [x1, y1, x2, y2] = normalized;
  const width = Math.max(x2 - x1, page.image_width * 0.22);
  const height = Math.max(y2 - y1, page.image_height * 0.18);
  const padding = Math.max(width, height) * 0.45;
  const focusWidth = Math.min(page.image_width, width + padding * 2);
  const focusHeight = Math.min(page.image_height, height + padding * 2);
  const centerX = (x1 + x2) / 2;
  const centerY = (y1 + y2) / 2;
  const viewX = clamp(centerX - focusWidth / 2, 0, page.image_width - focusWidth);
  const viewY = clamp(centerY - focusHeight / 2, 0, page.image_height - focusHeight);
  return `${viewX} ${viewY} ${focusWidth} ${focusHeight}`;
}

export function clamp(value: number, min: number, max: number): number {
  return Math.min(Math.max(value, min), Math.max(min, max));
}

export function bboxFromSource(source: Record<string, unknown> | undefined): number[] | null {
  const bbox = source?.bbox;
  if (!Array.isArray(bbox) || bbox.length !== 4) return null;
  return normalizeBbox(bbox);
}

export function fieldEvidence(card: CardCandidate | null, field: string | null): Record<string, unknown> | null {
  if (!card || !field) return null;
  const evidence = card.source.field_evidence;
  if (!evidence || typeof evidence !== "object" || Array.isArray(evidence)) return null;
  const value = (evidence as Record<string, unknown>)[field];
  return value && typeof value === "object" && !Array.isArray(value) ? (value as Record<string, unknown>) : null;
}

export function fieldBbox(card: CardCandidate | null, field: string | null): number[] | null {
  const bbox = fieldEvidence(card, field)?.bbox;
  return normalizeBbox(bbox);
}

export function editableFieldBbox(
  card: CardCandidate | null,
  field: string | null,
  draft?: { cardId?: string; field?: string; bbox?: unknown } | null
): number[] | null {
  if (card && field && draft?.cardId === card.id && draft.field === field) {
    const bbox = normalizeBbox(draft.bbox);
    if (bbox) return bbox;
  }
  return fieldBbox(card, field) ?? sourceBbox(card);
}

export function allFieldTokenIds(card: CardCandidate | null): Set<string> {
  const ids = new Set<string>();
  const evidence = card?.source.field_evidence;
  if (!evidence || typeof evidence !== "object" || Array.isArray(evidence)) return ids;
  Object.values(evidence).forEach((item) => {
    if (!item || typeof item !== "object" || Array.isArray(item)) return;
    const tokenIds = (item as Record<string, unknown>).token_ids;
    if (!Array.isArray(tokenIds)) return;
    tokenIds.forEach((token) => {
      if (typeof token === "string") ids.add(token);
    });
  });
  return ids;
}

export function allFieldDocumentBlockIds(card: CardCandidate | null): Set<string> {
  const ids = new Set<string>();
  const evidence = card?.source.field_evidence;
  if (!evidence || typeof evidence !== "object" || Array.isArray(evidence)) return ids;
  Object.values(evidence).forEach((item) => {
    if (!item || typeof item !== "object" || Array.isArray(item)) return;
    const blockIds = (item as Record<string, unknown>).block_ids;
    if (!Array.isArray(blockIds)) return;
    blockIds.forEach((block) => {
      if (typeof block === "string") ids.add(block);
    });
  });
  return ids;
}

export function tokenInside(token: OcrToken, bbox: number[]): boolean {
  const normalizedToken = normalizeBbox(token.bbox);
  const normalizedBbox = normalizeBbox(bbox);
  if (!normalizedToken || !normalizedBbox) return false;
  const [x1, y1, x2, y2] = normalizedBbox;
  const cx = (normalizedToken[0] + normalizedToken[2]) / 2;
  const cy = (normalizedToken[1] + normalizedToken[3]) / 2;
  return cx >= x1 && cx <= x2 && cy >= y1 && cy <= y2;
}

export function boxCentersOverlap(inner: number[], outer: number[]): boolean {
  const normalizedInner = normalizeBbox(inner);
  const normalizedOuter = normalizeBbox(outer);
  if (!normalizedInner || !normalizedOuter) return false;
  const [x1, y1, x2, y2] = normalizedOuter;
  const cx = (normalizedInner[0] + normalizedInner[2]) / 2;
  const cy = (normalizedInner[1] + normalizedInner[3]) / 2;
  return cx >= x1 && cx <= x2 && cy >= y1 && cy <= y2;
}

export function normalizeBbox(value: unknown): number[] | null {
  if (!Array.isArray(value) || value.length !== 4) return null;
  const numbers = value.map((item) => (typeof item === "number" ? item : Number(item)));
  if (!numbers.every((item) => Number.isFinite(item))) return null;
  const [x1, y1, x2, y2] = numbers;
  const left = Math.min(x1, x2);
  const right = Math.max(x1, x2);
  const top = Math.min(y1, y2);
  const bottom = Math.max(y1, y2);
  if (right <= left || bottom <= top) return null;
  return [left, top, right, bottom];
}

export function evidenceSummary(card: CardCandidate): string {
  const count = focusedTokenIds(card).size;
  if (count) return `${count} evidence tokens are highlighted for the selected candidate.`;
  const blockCount = focusedDocumentBlockIds(card).size;
  if (blockCount) return `${blockCount} PaddleOCR-VL document block${blockCount === 1 ? " is" : "s are"} highlighted for the selected candidate.`;
  if (card.source_bbox) return "Source region is highlighted for the selected candidate.";
  return "Select another candidate or switch to All OCR for debugging.";
}

export function fieldNamesForCard(card: CardCandidate): string[] {
  if (card.source_type === "vocab_item") return ["surface", "reading", "meaning_ko"];
  return [
    "sentence",
    "target",
    "choice_1",
    "choice_2",
    "choice_3",
    "choice_4",
    "correct_answer",
    "question_no",
    "answer_source"
  ];
}

export function fieldLabel(field: string): string {
  if (field.startsWith("choice_")) return `Choice ${field.replace("choice_", "")}`;
  return field.replaceAll("_", " ");
}

export function applyFieldOcrPreview(card: CardCandidate, preview: FieldOcrPreview): CardCandidate {
  const source = { ...card.source, ...preview.suggested_source };
  const currentEvidence = source.field_evidence;
  const fieldEvidenceMap =
    currentEvidence && typeof currentEvidence === "object" && !Array.isArray(currentEvidence)
      ? { ...(currentEvidence as Record<string, unknown>) }
      : {};
  fieldEvidenceMap[preview.field] = preview.field_evidence;
  source.field_evidence = fieldEvidenceMap;
  const nextWarnings = uniqueStrings([
    ...card.warnings.filter((warning) => !(preview.field.startsWith("choice_") && warning.includes("four choices"))),
    ...preview.warnings
  ]);
  const nextBase = { ...card, source, warnings: nextWarnings, confidence: preview.confidence || card.confidence };
  if (card.source_type === "vocab_item") {
    return syncVocabCardText(nextBase, source);
  }
  const next = syncQuestionAnswerBack(nextBase, source);
  if (preview.field === "sentence" || preview.field === "target") {
    return { ...next, front: questionFront(next) };
  }
  return next;
}

export function questionFront(card: CardCandidate): string {
  const prompt = card.note_type.includes("reading_mcq") ? "읽는 법?" : "올바른 표기는?";
  return `${textValue(card.source.sentence)}<br><br>밑줄: ${textValue(card.source.target)}<br>${prompt}`;
}

export function candidateTitle(card: CardCandidate): string {
  if (card.source_type === "vocab_item") {
    return `${textValue(card.source.surface, "Vocab")} · ${textValue(card.source.reading)}`;
  }
  return `Question ${textValue(card.source.question_no, "?")} · ${textValue(card.source.target, "target")}`;
}

export function candidateSubtitle(card: CardCandidate): string {
  if (card.source_type === "vocab_item") {
    return `Vocabulary entry · OCR ${Math.round(card.confidence * 100)}%`;
  }
  return `${card.note_type} · OCR ${Math.round(card.confidence * 100)}%`;
}

export function provenanceLabel(value: string): string {
  if (value === "answer_strip") return "Answer key";
  if (value === "answer_strip_ocr") return "Answer-strip OCR";
  if (value === "crop_ocr") return "Crop OCR";
  if (value === "region_ocr") return "Region OCR";
  if (value === "jp_region_ocr") return "Japanese region OCR";
  if (value === "ko_glyph_ocr") return "Korean glyph OCR";
  if (value === "prompt_line_ocr") return "Prompt-line OCR";
  if (value === "choice_glyph_ocr") return "Choice glyph OCR";
  if (value === "source_rebuild") return "Source rebuild";
  if (value === "local_glossary") return "Glossary inferred";
  if (value === "manual") return "Manual edit";
  return value.replaceAll("_", " ");
}

export function reviewReasonBadges(card: CardCandidate): string[] {
  const badges: string[] = [];
  const warningCodes = Array.isArray(card.source.warning_codes) ? card.source.warning_codes.map(String) : [];
  if (warningCodes.includes("MISSING_SURFACE")) badges.push("Missing surface");
  if (warningCodes.includes("MISSING_READING")) badges.push("Missing reading");
  if (warningCodes.includes("MISSING_KOREAN_MEANING")) badges.push("Missing Korean meaning");
  if (warningCodes.includes("WEAK_OCR_EVIDENCE")) badges.push("Weak OCR evidence");
  if (warningCodes.includes("SCRIPT_MISMATCH")) badges.push("Script mismatch");
  if (warningCodes.includes("GLOSSARY_SUGGESTION")) badges.push("Glossary suggestion");
  const warningText = card.warnings.join(" ");
  if (/Missing surface/i.test(warningText)) badges.push("Missing surface");
  if (/Missing reading/i.test(warningText)) badges.push("Missing reading");
  if (/Missing Korean meaning/i.test(warningText)) badges.push("Missing Korean meaning");
  if (/Weak OCR evidence/i.test(warningText)) badges.push("Weak OCR evidence");
  if (/script/i.test(warningText)) badges.push("Script mismatch");
  if (/four choices|exactly four choices|Missing choice/i.test(warningText)) badges.push("Missing choice");
  if (/target/i.test(warningText)) badges.push("Target unclear");
  if (/Answer source|Correct choice is missing|not export-safe/i.test(warningText)) badges.push("Needs answer");
  if (card.confidence < REVIEW_CONFIDENCE_THRESHOLD) badges.push("Low OCR confidence");
  if (card.review_state === "red" && !badges.length) badges.push("Blocked");
  if (card.review_state === "yellow" && !badges.length) badges.push("Needs review");
  return uniqueStrings(badges);
}

export function textValue(value: unknown, fallback = ""): string {
  if (typeof value === "string") return value;
  if (typeof value === "number" || typeof value === "boolean") return String(value);
  return fallback;
}

export function numberOrEmpty(value: string): number | string {
  const trimmed = value.trim();
  if (!trimmed) return "";
  const parsed = Number(trimmed);
  return Number.isFinite(parsed) ? parsed : trimmed;
}

export function syncQuestionAnswerBack(card: CardCandidate, source: Record<string, unknown>): CardCandidate {
  const answer = textValue(source.correct_answer);
  if (!answer) return card;
  if (card.note_type.endsWith("_exam")) {
    const choiceNo = textValue(source.correct_choice_no, "?");
    return { ...card, back: `정답: ${choiceNo}. ${answer}` };
  }
  return { ...card, back: answer };
}

export function applyQuestionSourceEdit(card: CardCandidate, field: string, value: string): CardCandidate {
  const nextSource = { ...card.source, [field]: field === "question_no" || field === "correct_choice_no" ? numberOrEmpty(value) : value };
  if (field === "correct_choice_no") {
    const choices = sourceChoices(nextSource);
    const choiceNo = Number(nextSource.correct_choice_no);
    if (Number.isInteger(choiceNo) && choices[choiceNo - 1]) {
      nextSource.correct_answer = choices[choiceNo - 1];
    }
  }
  return syncQuestionAnswerBack({ ...card, source: nextSource }, nextSource);
}

export function applyQuestionChoicesEdit(card: CardCandidate, value: string): CardCandidate {
  const choices = choicesFromText(value);
  const choiceNo = Number(card.source.correct_choice_no);
  const correctAnswer = Number.isInteger(choiceNo) && choices[choiceNo - 1] ? choices[choiceNo - 1] : card.source.correct_answer;
  const nextSource = { ...card.source, choices, correct_answer: correctAnswer };
  const nextWarnings = choices.length === 4
    ? card.warnings.filter((warning) => !/exactly four choices|four choices/i.test(warning))
    : uniqueStrings([...card.warnings, "Expected exactly four choices."]);
  const nextState = choices.length === 4 && card.review_state === "red" ? "yellow" : card.review_state;
  return syncQuestionAnswerBack({ ...card, source: nextSource, warnings: nextWarnings, review_state: nextState }, nextSource);
}

export function applyVocabSourceEdit(card: CardCandidate, field: string, value: string): CardCandidate {
  const nextSource = { ...card.source, [field]: value };
  return syncVocabCardText({ ...card, source: nextSource }, nextSource);
}

export function applyVocabStudyToggle(card: CardCandidate, field: string, enabled: boolean): CardCandidate {
  const nextSource = { ...card.source, [field]: enabled };
  return syncVocabCardText({ ...card, source: normalizeVocabStudySource(nextSource) }, nextSource);
}

export function syncVocabCardText(card: CardCandidate, source: Record<string, unknown>): CardCandidate {
  if (card.source_type !== "vocab_item") return card;
  const surface = textValue(source.surface);
  const reading = textValue(source.reading);
  const meaning = textValue(source.meaning_ko);
  const meaningOnly = source.vocab_type === "jp_ko_meaning";
  if (card.note_type === "jp_vocab_entry") {
    return {
      ...card,
      source: normalizeVocabStudySource(source),
      front: escapeHtml(meaningOnly ? surface : reading),
      back: escapeHtml(meaningOnly ? meaning : surface)
    };
  }
  return card;
}

function normalizeVocabStudySource(source: Record<string, unknown>): Record<string, unknown> {
  const meaningOnly = source.vocab_type === "jp_ko_meaning";
  const legacyMeaningOnlyStudy = meaningOnly &&
    source.study_japanese_to_korean === undefined &&
    studyDirectionEnabled(source.study_meaning, false);
  const studySource = {
    ...source,
    study_writing: studyDirectionEnabled(source.study_writing, !meaningOnly),
    study_reading: studyDirectionEnabled(source.study_reading, false),
    study_meaning: legacyMeaningOnlyStudy ? false : studyDirectionEnabled(source.study_meaning, false),
    study_japanese_to_korean: studyDirectionEnabled(source.study_japanese_to_korean, meaningOnly)
  };
  if (
    !studySource.study_writing &&
    !studySource.study_reading &&
    !studySource.study_meaning &&
    !studySource.study_japanese_to_korean
  ) {
    studySource.study_japanese_to_korean = meaningOnly;
    studySource.study_writing = !meaningOnly;
  }
  return {
    ...studySource
  };
}

export function sourceChoices(source: Record<string, unknown>): string[] {
  return Array.isArray(source.choices) ? source.choices.map(String) : [];
}

export function choicesText(source: Record<string, unknown>): string {
  return sourceChoices(source).join("\n");
}

export function choicesFromText(value: string): string[] {
  return value
    .split(/\r?\n/)
    .map((choice) => choice.trim())
    .filter(Boolean);
}

export function uniqueStrings(values: string[]): string[] {
  return [...new Set(values)];
}

function escapeHtml(value: string): string {
  return value
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#x27;");
}

export function studyDirectionEnabled(value: unknown, defaultEnabled = true): boolean {
  if (value === undefined) return defaultEnabled;
  if (value === false || value === null) return false;
  if (typeof value === "number") return value !== 0;
  if (typeof value === "string") return !["", "0", "false", "no"].includes(value.trim().toLowerCase());
  return true;
}
