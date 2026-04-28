import type { CardCandidate, FieldOcrPreview, OcrToken, Page } from "./api";

export type ReviewFilter = "all" | "needs_review" | "approved" | "green" | "yellow" | "red";

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
  const use = relevant ? "selected candidate evidence" : linkedCard ? `used by ${candidateTitle(linkedCard)}` : "scanned but unused";
  return `${token.text} (${token.source}, ${token.script_class}, OCR ${Math.round(token.confidence * 100)}%, ${use})`;
}

export function isAnswerSupportToken(token: OcrToken, page: Page): boolean {
  if (!page.image_height || !page.page_type.endsWith("_mcq")) return false;
  if (token.bbox[1] < page.image_height * 0.82) return false;
  return /[①②③④⑤⑥⑦⑧⑨⑩0-9]/.test(token.text);
}

export function focusedTokenIds(card: CardCandidate | null): Set<string> {
  return focusedTokenIdsForField(card, null);
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

export function focusBbox(card: CardCandidate | null, tokens: OcrToken[], field: string | null = null): number[] | null {
  const relevantIds = focusedTokenIdsForField(card, field);
  const relevantBoxes = tokens.filter((token) => relevantIds.has(token.id)).map((token) => token.bbox);
  return unionBoxes(relevantBoxes) ?? fieldBbox(card, field) ?? sourceBbox(card);
}

export function sourceBbox(card: CardCandidate | null): number[] | null {
  return card?.source_bbox ?? bboxFromSource(card?.source);
}

export function unionBoxes(boxes: number[][]): number[] | null {
  const valid = boxes.filter((box) => box.length === 4 && box.every((value) => Number.isFinite(value)));
  if (!valid.length) return null;
  return [
    Math.min(...valid.map((box) => box[0])),
    Math.min(...valid.map((box) => box[1])),
    Math.max(...valid.map((box) => box[2])),
    Math.max(...valid.map((box) => box[3]))
  ];
}

export function evidenceViewBox(page: Page, bbox: number[] | null): string {
  if (!bbox || !page.image_width || !page.image_height) {
    return `0 0 ${page.image_width ?? 1} ${page.image_height ?? 1}`;
  }
  const [x1, y1, x2, y2] = bbox;
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
  return bbox.every((value) => typeof value === "number") ? bbox : null;
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
  return Array.isArray(bbox) && bbox.length === 4 && bbox.every((value) => typeof value === "number") ? bbox : null;
}

export function editableFieldBbox(
  card: CardCandidate | null,
  field: string | null,
  draft?: { cardId?: string; field?: string; bbox?: unknown } | null
): number[] | null {
  if (card && field && draft?.cardId === card.id && draft.field === field) {
    const bbox = draft.bbox;
    if (Array.isArray(bbox) && bbox.length === 4 && bbox.every((value) => typeof value === "number")) return bbox;
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

export function tokenInside(token: OcrToken, bbox: number[]): boolean {
  const [x1, y1, x2, y2] = bbox;
  const cx = (token.bbox[0] + token.bbox[2]) / 2;
  const cy = (token.bbox[1] + token.bbox[3]) / 2;
  return cx >= x1 && cx <= x2 && cy >= y1 && cy <= y2;
}

export function evidenceSummary(card: CardCandidate): string {
  const count = focusedTokenIds(card).size;
  if (count) return `${count} evidence tokens are highlighted for the selected candidate.`;
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
  const next = syncQuestionAnswerBack({ ...card, source, warnings: nextWarnings, confidence: preview.confidence || card.confidence }, source);
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
  return `${card.note_type} · OCR ${Math.round(card.confidence * 100)}%`;
}

export function provenanceLabel(value: string): string {
  if (value === "answer_strip") return "Answer key";
  if (value === "local_glossary") return "Glossary inferred";
  if (value === "manual") return "Manual edit";
  return value.replaceAll("_", " ");
}

export function reviewReasonBadges(card: CardCandidate): string[] {
  const badges: string[] = [];
  const warningText = card.warnings.join(" ");
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

export function choicesFromText(value: string): string[] {
  return value
    .split(/\r?\n/)
    .map((choice) => choice.trim())
    .filter(Boolean);
}

export function uniqueStrings(values: string[]): string[] {
  return [...new Set(values)];
}
