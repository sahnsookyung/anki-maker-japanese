"use client";

import { type PointerEvent, type RefObject, type KeyboardEvent, type ReactNode, useEffect, useRef, useState } from "react";
import {
  API_BASE,
  type CardCandidate,
  type DocumentParseResult,
  type FieldOcrPreview,
  type OcrComparison,
  type OcrEngine,
  type OcrRuntimeStatus,
  type OcrToken,
  type Page,
  type ProcessResult,
  apiGet,
  apiErrorMessage,
  approveCard,
  compareOcr,
  dedupePages,
  deletePage,
  exportCsv,
  getOcrRuntime,
  imageUrl,
  parseDocument,
  previewFieldOcr,
  processPage,
  updateCard,
  updatePage,
  uploadImages
} from "../lib/api";
import { batchTimingReport, durationMs, formatDuration, type BatchTimingReport, type PageTiming } from "../lib/batchTiming";
import {
  type ReviewFilter,
  applyFieldOcrPreview,
  applyQuestionChoicesEdit,
  applyQuestionSourceEdit,
  applyVocabSourceEdit,
  cardForToken,
  cardMatchesFilter,
  candidateSubtitle,
  candidateTitle,
  choicesText,
  editableFieldBbox,
  evidenceSummary,
  evidenceViewBox,
  fieldBbox,
  fieldLabel,
  fieldNamesForCard,
  focusBbox,
  focusedTokenIdsForField,
  initialReviewCardId,
  isHighConfidenceCard,
  normalizeBbox,
  nextReviewCardId,
  provenanceLabel,
  reviewReasonBadges,
  reviewQualityClass,
  sourceBbox,
  summarizeCards,
  textValue,
  tokenDisplayClass,
  tokenInside,
  tokenTitle,
  uniqueStrings,
  warningKey,
  workflowClass
} from "../lib/review";

type OverlayMode = "focused" | "region" | "all" | "off";
type CardScrollTarget = "evidence" | "card" | "none";
type FieldRegionDraft = { cardId: string; field: string; bbox: number[] };
type CardRefs = RefObject<Record<string, HTMLElement | null>>;
type PageTimingStatus = PageTiming & { engineLabel: string };

const PAGE_TYPE_LABELS: Record<string, string> = {
  uploaded: "Uploaded",
  vocab_table: "Vocabulary table",
  spelling_vocab_table: "Spelling vocabulary",
  reading_mcq: "Reading MCQ",
  spelling_mcq: "Spelling MCQ",
  unknown_review_required: "Needs manual review"
};

const SCRIPT_LABELS = ["paddleocr", "paddleocr_korean", "hiragana", "katakana", "kanji", "hangul", "mixed", "number"];
const BATCH_REPORT_STORAGE_KEY = "anki-maker-latest-batch-report";
const INITIAL_STATUS_MESSAGE = "Upload a study-book photo to begin.";

function scrollTargetElement(
  target: CardScrollTarget,
  evidenceElement: HTMLElement | null,
  cardElement: HTMLElement | null
): HTMLElement | null {
  if (target === "evidence") return evidenceElement;
  if (target === "card") return cardElement;
  return null;
}

function svgPoint(event: PointerEvent<SVGElement>): [number, number] {
  const svg = event.currentTarget.ownerSVGElement ?? (event.currentTarget as SVGSVGElement);
  const point = svg.createSVGPoint();
  point.x = event.clientX;
  point.y = event.clientY;
  const matrix = svg.getScreenCTM();
  if (!matrix) return [0, 0];
  const transformed = point.matrixTransform(matrix.inverse());
  return [transformed.x, transformed.y];
}

function keyActivatesCard(event: KeyboardEvent<HTMLElement>): boolean {
  return event.key === "Enter" || event.key === " ";
}

function loadStoredBatchReport(): BatchTimingReport | null {
  if (typeof globalThis.window === "undefined") return null;
  try {
    const raw = globalThis.localStorage.getItem(BATCH_REPORT_STORAGE_KEY);
    if (!raw) return null;
    const parsed = JSON.parse(raw) as unknown;
    return isBatchTimingReport(parsed) ? parsed : null;
  } catch {
    return null;
  }
}

function storeBatchReport(report: BatchTimingReport): void {
  try {
    globalThis.localStorage.setItem(BATCH_REPORT_STORAGE_KEY, JSON.stringify(report));
  } catch {
    // A failed persistence write should not interrupt OCR review.
  }
}

function isBatchTimingReport(value: unknown): value is BatchTimingReport {
  if (!value || typeof value !== "object") return false;
  const report = value as Partial<BatchTimingReport>;
  return (
    typeof report.text === "string" &&
    typeof report.engineLabel === "string" &&
    typeof report.totalPages === "number" &&
    typeof report.processedPages === "number" &&
    typeof report.successfulPages === "number" &&
    typeof report.failedPages === "number" &&
    Array.isArray(report.failures) &&
    Boolean(report.stats)
  );
}

export function StudyWorkbench() { // NOSONAR: orchestration root delegates rendering/editing to focused child components below.
  const [pages, setPages] = useState<Page[]>([]);
  const [selectedPage, setSelectedPage] = useState<Page | null>(null);
  const [tokens, setTokens] = useState<OcrToken[]>([]);
  const [cards, setCards] = useState<CardCandidate[]>([]);
  const [selectedCardId, setSelectedCardId] = useState<string | null>(null);
  const [overlayMode, setOverlayMode] = useState<OverlayMode>("focused");
  const [reviewFilter, setReviewFilter] = useState<ReviewFilter>("all");
  const [activeTokenFilters, setActiveTokenFilters] = useState<Set<string>>(new Set());
  const [comparison, setComparison] = useState<OcrComparison | null>(null);
  const [documentParse, setDocumentParse] = useState<DocumentParseResult | null>(null);
  const [runtimeStatus, setRuntimeStatus] = useState<OcrRuntimeStatus | null>(null);
  const [selectedField, setSelectedField] = useState<string | null>(null);
  const [regionDraft, setRegionDraft] = useState<FieldRegionDraft | null>(null);
  const [fieldPreview, setFieldPreview] = useState<FieldOcrPreview | null>(null);
  const [message, setMessage] = useState(INITIAL_STATUS_MESSAGE);
  const [isBatchProcessing, setIsBatchProcessing] = useState(false);
  const [processingPageId, setProcessingPageId] = useState<string | null>(null);
  const [vlScanningPageId, setVlScanningPageId] = useState<string | null>(null);
  const [vlProcessingPageId, setVlProcessingPageId] = useState<string | null>(null);
  const [isComparingOcr, setIsComparingOcr] = useState(false);
  const [isPreviewingField, setIsPreviewingField] = useState(false);
  const [pageTimingById, setPageTimingById] = useState<Record<string, PageTimingStatus>>({});
  const [latestBatchReport, setLatestBatchReport] = useState<BatchTimingReport | null>(null);
  const evidenceRef = useRef<HTMLElement | null>(null);
  const cardRefs = useRef<Record<string, HTMLElement | null>>({});
  const ocrActionInFlightRef = useRef(false);

  useEffect(() => {
    void refreshPages();
  }, []);

  useEffect(() => {
    setLatestBatchReport(loadStoredBatchReport());
  }, []);

  const selectedCard = selectedCardId ? cards.find((card) => card.id === selectedCardId) ?? null : null;
  const cardStats = summarizeCards(cards);
  const filteredCards = cards.filter((card) => cardMatchesFilter(card, reviewFilter));
  const exportableCount = cards.filter((card) => card.status === "approved" && card.review_state !== "red").length;
  const anyOcrJobRunning =
    isBatchProcessing || Boolean(processingPageId || vlProcessingPageId || vlScanningPageId || isComparingOcr || isPreviewingField);

  useEffect(() => {
    if (!selectedCard) {
      setSelectedField(null);
      setRegionDraft(null);
      setFieldPreview(null);
      return;
    }
    const fields = fieldNamesForCard(selectedCard);
    if (!selectedField || !fields.includes(selectedField)) {
      setSelectedField(fields[0] ?? null);
    }
  }, [selectedCard, selectedField]);

  async function refreshPages(preferredPageId?: string) {
    try {
      const nextPages = await apiGet<Page[]>("/api/pages");
      setPages(nextPages);
      const nextSelected =
        nextPages.find((page) => page.id === preferredPageId) ??
        nextPages.find((page) => page.id === selectedPage?.id) ??
        nextPages[0];
      if (nextSelected) {
        await selectPage(nextSelected, false);
        setMessage((current) =>
          current === INITIAL_STATUS_MESSAGE
            ? `Loaded ${pageCountLabel(nextPages.length)}. Process pages or review existing candidates.`
            : current
        );
      } else {
        setMessage(INITIAL_STATUS_MESSAGE);
      }
    } catch (error) {
      setPages([]);
      setSelectedPage(null);
      setTokens([]);
      setCards([]);
      setSelectedCardId(null);
      setMessage(apiErrorMessage(error, "Could not load pages."));
    }
  }

  async function selectPage(page: Page, clearMessage = true) {
    setSelectedPage(page);
    const [ocr, pageCards] = await Promise.all([
      apiGet<{ page: Page; tokens: OcrToken[] }>(`/api/pages/${page.id}/ocr`).catch(() => ({ page, tokens: [] })),
      apiGet<CardCandidate[]>(`/api/pages/${page.id}/cards`).catch(() => [])
    ]);
    setSelectedPage(ocr.page);
    setTokens(ocr.tokens);
    setCards(pageCards);
    syncPageCardSummary(page.id, pageCards);
    setSelectedCardId(initialReviewCardId(pageCards));
    setComparison(null);
    setDocumentParse(null);
    if (clearMessage) setMessage(`Selected ${pageTitle(ocr.page)}.`);
  }

  async function onUpload(files: FileList | null) {
    const selectedFiles = Array.from(files ?? []);
    if (!selectedFiles.length) return;
    setMessage(`Uploading ${selectedFiles.length} page${selectedFiles.length === 1 ? "" : "s"}...`);
    const result = await uploadImages(selectedFiles);
    const lastPageId = result.uploaded.at(-1)?.pageId;
    if (lastPageId) {
      await refreshPages(lastPageId);
    }
    if (result.failed.length) {
      const failedNames = result.failed.map((failure) => `${failure.fileName} (${failure.message})`).join(", ");
      setMessage(`Uploaded ${result.uploaded.length}/${selectedFiles.length} pages. Failed: ${failedNames}.`);
    } else {
      setMessage(`Uploaded ${result.uploaded.length} page${result.uploaded.length === 1 ? "" : "s"}. Run processing to create review candidates.`);
    }
  }

  async function onProcessAllPages(engine: OcrEngine = "paddleocr") {
    if (!pages.length || !beginOcrAction()) return;
    setIsBatchProcessing(true);
    const failures: string[] = [];
    const timings: PageTiming[] = [];
    const engineLabel = engine === "paddleocr_vl" ? "PaddleOCR-VL" : "PaddleOCR";
    try {
      for (const [index, page] of pages.entries()) {
        if (engine === "paddleocr_vl") setVlProcessingPageId(page.id);
        else setProcessingPageId(page.id);
        setMessage(`Processing page ${index + 1}/${pages.length} with ${engineLabel}: ${pageTitle(page)}...`);
        const startedAt = performance.now();
        try {
          const result = await processPage(page.id, engine);
          const elapsedMs = durationMs(startedAt, performance.now());
          const timing = { pageId: page.id, pageTitle: pageTitle(result.page), ms: elapsedMs, success: true };
          timings.push(timing);
          recordPageTiming(timing, engineLabel);
          applyProcessResult(result, true);
          const report = batchTimingReport(timings, pages.length, failures, engineLabel, index + 1);
          rememberBatchReport(report);
          setMessage(batchProgressMessage(report));
          await yieldToBrowser();
        } catch (error) {
          const elapsedMs = durationMs(startedAt, performance.now());
          const timing = { pageId: page.id, pageTitle: pageTitle(page), ms: elapsedMs, success: false };
          timings.push(timing);
          recordPageTiming(timing, engineLabel);
          const failedPage = `${pageTitle(page)} (${apiErrorMessage(error, "Processing failed.")})`;
          failures.push(failedPage);
          const report = batchTimingReport(timings, pages.length, failures, engineLabel, index + 1);
          rememberBatchReport(report);
          setMessage(batchProgressMessage(report));
          await yieldToBrowser();
        } finally {
          if (engine === "paddleocr_vl") setVlProcessingPageId(null);
          else setProcessingPageId(null);
        }
      }
      const report = batchTimingReport(timings, pages.length, failures, engineLabel);
      rememberBatchReport(report);
      setMessage(batchFinishedMessage(report));
    } finally {
      setIsBatchProcessing(false);
      finishOcrAction();
    }
  }

  async function onProcessPage(page: Page, engine: OcrEngine = "paddleocr") {
    if (!beginOcrAction()) return;
    if (engine === "paddleocr_vl") setVlProcessingPageId(page.id);
    else setProcessingPageId(page.id);
    const engineLabel = engine === "paddleocr_vl" ? "PaddleOCR-VL" : "PaddleOCR";
    setMessage(`Processing ${pageTitle(page)} with ${engineLabel}...`);
    const startedAt = performance.now();
    try {
      const result = await processPage(page.id, engine);
      applyProcessResult(result, true);
      const elapsedMs = durationMs(startedAt, performance.now());
      recordPageTiming({ pageId: result.page.id, pageTitle: pageTitle(result.page), ms: elapsedMs, success: true }, engineLabel);
      setMessage(`Processed ${pageTitle(result.page)} with ${engineLabel} in ${formatDuration(elapsedMs)}.`);
    } catch (error) {
      const elapsedMs = durationMs(startedAt, performance.now());
      recordPageTiming({ pageId: page.id, pageTitle: pageTitle(page), ms: elapsedMs, success: false }, engineLabel);
      setMessage(apiErrorMessage(error, "Processing page failed."));
    } finally {
      if (engine === "paddleocr_vl") setVlProcessingPageId(null);
      else setProcessingPageId(null);
      finishOcrAction();
    }
  }

  async function cleanupDuplicatePages() {
    const confirmed = globalThis.confirm("Remove duplicate local page records with the same upload name, keeping the newest copy?");
    if (!confirmed) return;
    try {
      const result = await dedupePages();
      await refreshPages(selectedPage?.id);
      setMessage(`Removed ${result.removed_count} duplicate page${result.removed_count === 1 ? "" : "s"}.`);
    } catch (error) {
      setMessage(apiErrorMessage(error, "Duplicate cleanup failed."));
    }
  }

  async function renamePage(page: Page, displayName: string) {
    try {
      const updated = await updatePage(page.id, displayName);
      setPages((current) => current.map((item) => (item.id === updated.id ? updated : item)));
      if (selectedPage?.id === updated.id) setSelectedPage(updated);
      setMessage(`Renamed page to ${pageTitle(updated)}.`);
    } catch (error) {
      setMessage(apiErrorMessage(error, "Rename failed."));
    }
  }

  async function removePage(page: Page) {
    const title = pageTitle(page);
    const confirmed = globalThis.confirm(`Delete "${title}" and its generated OCR/cards? This only removes local app state.`);
    if (!confirmed) return;
    try {
      await deletePage(page.id);
      const remainingPages = pages.filter((item) => item.id !== page.id);
      setPages(remainingPages);
      if (selectedPage?.id === page.id) {
        const nextPage = remainingPages[0];
        setSelectedPage(null);
        setTokens([]);
        setCards([]);
        setSelectedCardId(null);
        setComparison(null);
        setDocumentParse(null);
        if (nextPage) {
          await selectPage(nextPage, false);
        }
      }
      setMessage(`Deleted ${title}.`);
    } catch (error) {
      setMessage(apiErrorMessage(error, "Delete failed."));
    }
  }

  async function saveCard(card: CardCandidate) {
    try {
      const updated = await updateCard(card);
      const nextCards = cards.map((item) => (item.id === updated.id ? updated : item));
      setCards(nextCards);
      syncPageCardSummary(updated.page_id, nextCards);
      setMessage("Saved card edits.");
    } catch (error) {
      setMessage(apiErrorMessage(error, "Saving card failed."));
    }
  }

  async function approve(cardId: string) {
    try {
      const updated = await approveCard(cardId);
      const nextCards = cards.map((item) => (item.id === updated.id ? updated : item));
      setCards(nextCards);
      syncPageCardSummary(updated.page_id, nextCards);
      setSelectedCardId(nextReviewCardId(nextCards, updated.id));
      setRegionDraft(null);
      setFieldPreview(null);
      setMessage("Approved card.");
    } catch (error) {
      setMessage(apiErrorMessage(error, "Approving card failed."));
    }
  }

  async function toggleApproval(card: CardCandidate) {
    if (card.status !== "approved") {
      await approve(card.id);
      return;
    }
    const updatedCard = { ...card, status: "pending_review" as const };
    try {
      const updated = await updateCard(updatedCard);
      const nextCards = cards.map((item) => (item.id === updated.id ? updated : item));
      setCards(nextCards);
      syncPageCardSummary(updated.page_id, nextCards);
      setSelectedCardId(updated.id);
      setMessage("Moved card back to pending review.");
    } catch (error) {
      setMessage(apiErrorMessage(error, "Unapproving card failed."));
    }
  }

  async function onExport() {
    if (!selectedPage) return;
    try {
      const result = await exportCsv([selectedPage.id], { approved_only: true, include_yellow: true, include_red: false });
      setMessage(`Exported ${result.card_count} approved cards.`);
      globalThis.open(`${API_BASE}${result.download_url}`, "_blank");
    } catch (error) {
      setMessage(apiErrorMessage(error, "Export failed."));
    }
  }

  async function onCompareOcr() {
    if (!selectedPage) return;
    if (!beginOcrAction()) return;
    setIsComparingOcr(true);
    setMessage("Comparing local OCR with Google Cloud Vision...");
    try {
      const result = await compareOcr(selectedPage.id);
      setComparison(result);
      setMessage(`OCR comparison complete: ${Math.round(result.agreement * 100)}% token agreement.`);
    } catch (error) {
      setMessage(apiErrorMessage(error, "OCR comparison failed."));
    } finally {
      setIsComparingOcr(false);
      finishOcrAction();
    }
  }

  async function onParseDocumentForPage(page: Page | null = selectedPage) {
    if (!page || !beginOcrAction()) return;
    setVlScanningPageId(page.id);
    setMessage(`Scanning ${pageTitle(page)} with PaddleOCR-VL...`);
    try {
      if (selectedPage?.id !== page.id) {
        await selectPage(page, false);
      }
      const result = await parseDocument(page.id);
      setDocumentParse(result);
      setMessage(`PaddleOCR-VL scan returned ${result.block_count} document blocks for ${pageTitle(page)}.`);
    } catch (error) {
      setMessage(apiErrorMessage(error, "PaddleOCR-VL parsing failed."));
    } finally {
      setVlScanningPageId(null);
      finishOcrAction();
    }
  }

  function selectCard(card: CardCandidate, scrollTarget: CardScrollTarget = "evidence") {
    setSelectedCardId(card.id);
    const fields = fieldNamesForCard(card);
    setSelectedField((current) => (current && fields.includes(current) ? current : fields[0] ?? null));
    setFieldPreview(null);
    setRegionDraft(null);
    const target = scrollTargetElement(scrollTarget, evidenceRef.current, cardRefs.current[card.id]);
    target?.scrollIntoView({ behavior: "smooth", block: "nearest" });
  }

  function selectCardForToken(token: OcrToken) {
    const card = cardForToken(cards, token);
    if (!card) return;
    selectCard(card, "card");
    setMessage(`Selected ${candidateTitle(card)} from OCR evidence.`);
  }

  function selectField(field: string, card: CardCandidate | null = selectedCard) {
    if (!card) return;
    setSelectedCardId(card.id);
    setSelectedField(field);
    setFieldPreview(null);
    const bbox = fieldBbox(card, field) ?? sourceBbox(card);
    setRegionDraft(bbox ? { cardId: card.id, field, bbox } : null);
  }

  async function previewSelectedField(field = selectedField, card = selectedCard) {
    if (!card || !field) return;
    const activeDraft = regionDraft?.cardId === card.id && regionDraft.field === field ? regionDraft : null;
    const bbox = editableFieldBbox(card, field, activeDraft);
    if (!bbox) return;
    if (!beginOcrAction()) return;
    setSelectedCardId(card.id);
    setSelectedField(field);
    setRegionDraft({ cardId: card.id, field, bbox });
    setIsPreviewingField(true);
    setMessage(`Previewing OCR for ${fieldLabel(field)}...`);
    try {
      const preview = await previewFieldOcr(card.id, field, bbox);
      setFieldPreview(preview);
      setMessage(preview.text ? `OCR preview for ${fieldLabel(field)}: ${preview.text}` : "OCR preview returned no text.");
    } catch (error) {
      setMessage(apiErrorMessage(error, "Field OCR preview failed."));
    } finally {
      setIsPreviewingField(false);
      finishOcrAction();
    }
  }

  function beginOcrAction(): boolean {
    if (ocrActionInFlightRef.current || anyOcrJobRunning) {
      setMessage("Another OCR job is already running. Wait for it to finish before starting a new scan.");
      return false;
    }
    ocrActionInFlightRef.current = true;
    return true;
  }

  function finishOcrAction() {
    ocrActionInFlightRef.current = false;
  }

  async function applySelectedFieldPreview() {
    if (!selectedCard || !fieldPreview) return;
    const updatedCard = applyFieldOcrPreview(selectedCard, fieldPreview);
    await saveCard(updatedCard);
    setFieldPreview(null);
    setRegionDraft({ cardId: updatedCard.id, field: fieldPreview.field, bbox: fieldPreview.bbox });
  }

  async function refreshRuntimeStatus() {
    try {
      setRuntimeStatus(await getOcrRuntime());
    } catch (error) {
      setMessage(apiErrorMessage(error, "Could not load OCR runtime status."));
    }
  }

  function toggleTokenFilter(value: string) {
    setActiveTokenFilters((current) => {
      const next = new Set(current);
      if (next.has(value)) next.delete(value);
      else next.add(value);
      return next;
    });
  }

  function syncPageCardSummary(pageId: string, pageCards: CardCandidate[]) {
    const summary = summarizeCards(pageCards);
    setPages((current) =>
      current.map((page) =>
        page.id === pageId
          ? {
              ...page,
              card_count: summary.total,
              approved_card_count: summary.approved,
              red_card_count: summary.red
            }
          : page
      )
    );
  }

  function recordPageTiming(timing: PageTiming, engineLabel: string) {
    setPageTimingById((current) => ({
      ...current,
      [timing.pageId]: { ...timing, engineLabel }
    }));
  }

  function rememberBatchReport(report: BatchTimingReport) {
    setLatestBatchReport(report);
    storeBatchReport(report);
  }

  function applyProcessResult(result: ProcessResult, selectResult: boolean) {
    const summary = summarizeCards(result.cards);
    setPages((current) =>
      current.map((page) =>
        page.id === result.page.id
          ? {
              ...result.page,
              card_count: summary.total,
              approved_card_count: summary.approved,
              red_card_count: summary.red
            }
          : page
      )
    );
    if (!selectResult) return;
    setSelectedPage({
      ...result.page,
      card_count: summary.total,
      approved_card_count: summary.approved,
      red_card_count: summary.red
    });
    setTokens(result.tokens);
    setCards(result.cards);
    setSelectedCardId(initialReviewCardId(result.cards));
    setComparison(null);
    setDocumentParse(null);
  }

  const processedUrl = imageUrl(selectedPage?.processed_image_path);
  const originalUrl = imageUrl(selectedPage?.original_image_path);
  const visibleUrl = versionedImageUrl(processedUrl ?? originalUrl, selectedPage, tokens.length, cards.length);

  return (
    <main className="app-shell">
      <header className="hero">
        <div>
          <p className="eyebrow">Local-first Japanese OCR review</p>
          <h1>Turn workbook photos into Anki candidates you can actually trust.</h1>
          <p className="lede">
            Upload, process, inspect the exact evidence for each card, approve only what looks right, then export Anki CSV.
          </p>
        </div>
        <label className="upload-card">
          <span>Upload pages</span>
          <small>Multi-select JPG, PNG, WEBP, TIFF</small>
          <input
            type="file"
            accept="image/*"
            multiple
            onChange={(event) => {
              void onUpload(event.target.files);
              event.target.value = "";
            }}
          />
        </label>
      </header>

      <section className="workflow">
        {["Upload", "Process", "Review", "Export"].map((step, index) => (
          <div className={workflowClass(index, selectedPage, cards, exportableCount)} key={step}>
            <span>{index + 1}</span>
            <strong>{step}</strong>
          </div>
        ))}
      </section>

      <section className="command-bar">
        <p>{message}</p>
        <div className="command-actions">
          {pages.length ? (
            <button
              onClick={() => void onProcessAllPages("paddleocr")}
              disabled={anyOcrJobRunning}
              title="Create Anki review candidates by running local PaddleOCR sequentially across all pages."
            >
              Process pages with PaddleOCR
            </button>
          ) : null}
          {pages.length ? (
            <button
              className="vl-scan"
              onClick={() => void onProcessAllPages("paddleocr_vl")}
              disabled={anyOcrJobRunning}
              title="Create Anki review candidates by running PaddleOCR-VL sequentially across all pages. This is slower and memory-guarded."
            >
              Process pages with OCR-VL
            </button>
          ) : null}
          {pages.length > 1 ? (
            <button
              className="secondary"
              onClick={() => void cleanupDuplicatePages()}
              disabled={anyOcrJobRunning}
              title="Remove duplicate local page records that have the same uploaded filename, keeping the newest copy."
            >
              Clean duplicates
            </button>
          ) : null}
          {cards.length ? (
            <button
              onClick={onExport}
              disabled={exportableCount === 0}
              title={`Export ${exportableCount} approved ${exportableCount === 1 ? "entity" : "entities"} to Anki CSV.`}
            >
              Export to Anki CSV
            </button>
          ) : null}
        </div>
      </section>

      <div className="workspace">
        <aside className="panel page-rail">
          <div className="panel-title">
            <h2>Pages</h2>
            <span>{pages.length}</span>
          </div>
          <BatchSummaryCard report={latestBatchReport} />
          {pages.length === 0 ? <p className="muted">No uploads yet.</p> : null}
          {pages.map((page) => (
            <PageCard
              key={page.id}
              page={page}
              active={page.id === selectedPage?.id}
              candidateCount={page.id === selectedPage?.id ? cards.length : page.card_count ?? 0}
              onSelect={() => void selectPage(page)}
              onRename={(name) => renamePage(page, name)}
              onDelete={() => removePage(page)}
              onProcess={() => onProcessPage(page, "paddleocr")}
              onVlProcess={() => onProcessPage(page, "paddleocr_vl")}
              processing={processingPageId === page.id}
              vlProcessing={vlProcessingPageId === page.id}
              disabled={anyOcrJobRunning}
              timing={pageTimingById[page.id]}
            />
          ))}
        </aside>

        <section className="review-stack">
          <section className="panel evidence-panel" ref={evidenceRef}>
            <EvidenceHeader
              page={selectedPage}
              tokenCount={tokens.length}
              selectedCard={selectedCard}
              overlayMode={overlayMode}
              setOverlayMode={setOverlayMode}
            />
            {visibleUrl ? (
              <EvidenceStage
                imageUrl={visibleUrl}
                page={selectedPage}
                tokens={tokens}
                cards={cards}
                card={selectedCard}
                selectedField={selectedField}
                regionDraft={regionDraft}
                mode={overlayMode}
                activeFilters={activeTokenFilters}
                onSelectCard={selectCardForToken}
                onSelectEvidenceCard={(card) => selectCard(card, "card")}
                onRegionDraftChange={setRegionDraft}
              />
            ) : (
              <div className="empty">Upload a page to see review evidence.</div>
            )}
            {selectedCard &&
            overlayMode === "focused" &&
            !focusedTokenIdsForField(selectedCard, selectedField).size &&
            !fieldBbox(selectedCard, selectedField) &&
            !selectedCard.source_bbox ? (
              <p className="focus-note">No focused evidence available for this candidate yet.</p>
            ) : null}
            <details className="advanced-ocr">
              <summary>Advanced OCR filters</summary>
              <div className="filter-pills">
                {SCRIPT_LABELS.map((label) => (
                  <button
                    key={label}
                    className={activeTokenFilters.has(label) ? "pill active" : "pill"}
                    onClick={() => toggleTokenFilter(label)}
                  >
                    {label}
                  </button>
                ))}
                {activeTokenFilters.size ? <button className="pill" onClick={() => setActiveTokenFilters(new Set())}>Clear</button> : null}
              </div>
            </details>
            <details className="advanced-ocr">
              <summary>Advanced OCR diagnostics</summary>
              <p className="muted">
                Diagnostics help explain OCR behavior. These tools do not create, approve, or export Anki cards.
              </p>
              <div className="actions">
                {selectedPage ? (
                  <button
                    className="secondary vl-secondary"
                    onClick={() => void onParseDocumentForPage(selectedPage)}
                    disabled={anyOcrJobRunning}
                    title="Preview raw PaddleOCR-VL document blocks and markdown without changing cards."
                  >
                    {vlScanningPageId === selectedPage.id ? "Previewing OCR-VL blocks" : "Preview OCR-VL document blocks"}
                  </button>
                ) : null}
                {tokens.length ? (
                  <button
                    className="secondary"
                    onClick={onCompareOcr}
                    disabled={anyOcrJobRunning}
                    title="Compare stored local OCR tokens with Google Cloud Vision. Requires Google credentials and does not change cards."
                  >
                    {isComparingOcr ? "Comparing Google Vision" : "Compare Google Vision"}
                  </button>
                ) : null}
                <button
                  className="secondary"
                  onClick={() => void refreshRuntimeStatus()}
                  title="Show crop OCR worker lifecycle, memory, and provider status."
                >
                  OCR runtime status
                </button>
              </div>
              {runtimeStatus ? (
                <p className="muted">
                  Crop OCR worker: {runtimeStatus.state}
                  {runtimeStatus.current_rss_mb ? ` · ${runtimeStatus.current_rss_mb} MB RSS` : ""}
                  {runtimeStatus.loaded_provider ? ` · ${runtimeStatus.loaded_provider}` : ""}
                </p>
              ) : null}
            </details>
            {selectedPage?.warnings.length ? <WarningList warnings={selectedPage.warnings} /> : null}
            {comparison ? <OcrComparisonPanel comparison={comparison} /> : null}
            {documentParse ? <DocumentParsePanel result={documentParse} /> : null}
          </section>

          <section className="panel cards-panel">
            <div className="panel-title">
              <div>
                <h2>Review candidates</h2>
                <p className="muted">
                  {cardStats.approved} approved, {cardStats.needsReview} need review, {cardStats.red} blocked.
                </p>
              </div>
              <select value={reviewFilter} onChange={(event) => setReviewFilter(event.target.value as ReviewFilter)}>
                <option value="all">All candidates</option>
                <option value="needs_review">Needs review</option>
                <option value="approved">Approved</option>
                <option value="green">Green</option>
                <option value="yellow">Yellow</option>
                <option value="red">Red</option>
              </select>
            </div>
            {cards.length === 0 ? <div className="empty">Process the selected page to generate editable candidates.</div> : null}
            <CandidateList
              cards={filteredCards}
              selectedCard={selectedCard}
              cardRefs={cardRefs}
              onSelect={selectCard}
              onChange={saveCard}
              onToggleApproval={toggleApproval}
              selectedField={selectedField}
              fieldPreview={fieldPreview}
              isPreviewingField={isPreviewingField}
              onSelectField={selectField}
              onPreviewField={(field, card) => void previewSelectedField(field, card)}
              onApplyFieldPreview={() => void applySelectedFieldPreview()}
              onCancelFieldPreview={() => {
                setFieldPreview(null);
                setRegionDraft(null);
              }}
              grouped={reviewFilter === "all"}
            />
          </section>
        </section>
      </div>
    </main>
  );
}

function PageCard({
  page,
  active,
  candidateCount,
  timing,
  onSelect,
  onRename,
  onDelete,
  onProcess,
  onVlProcess,
  processing,
  vlProcessing,
  disabled
}: Readonly<{
  page: Page;
  active: boolean;
  candidateCount: number;
  timing?: PageTimingStatus;
  onSelect: () => void;
  onRename: (name: string) => Promise<void>;
  onDelete: () => Promise<void>;
  onProcess: () => Promise<void>;
  onVlProcess: () => Promise<void>;
  processing: boolean;
  vlProcessing: boolean;
  disabled: boolean;
}>) {
  const [editing, setEditing] = useState(false);
  const [draft, setDraft] = useState(pageTitle(page));
  useEffect(() => setDraft(pageTitle(page)), [page]);
  return (
    <article className={pageCardClass(active)}>
      {editing ? (
        <PageRenameForm
          draft={draft}
          onDraftChange={setDraft}
          onCancel={() => setEditing(false)}
          onSave={() => void onRename(draft).then(() => setEditing(false))}
        />
      ) : (
        <PageSelectButton page={page} candidateCount={candidateCount} timing={timing} onSelect={onSelect} />
      )}
      {editing ? null : (
        <PageActions
          disabled={disabled}
          processing={processing}
          vlProcessing={vlProcessing}
          onEdit={() => setEditing(true)}
          onDelete={onDelete}
          onProcess={onProcess}
          onVlProcess={onVlProcess}
        />
      )}
    </article>
  );
}

function BatchSummaryCard({ report }: Readonly<{ report: BatchTimingReport | null }>) {
  if (!report) return null;
  const statItems = [
    ["Total", report.stats.total],
    ["Avg", report.stats.average],
    ["Min", report.stats.min],
    ["P25", report.stats.p25],
    ["P75", report.stats.p75],
    ["Max", report.stats.max]
  ];
  return (
    <section className="batch-summary-card" aria-label="Latest batch timing summary">
      <div>
        <p className="eyebrow">Latest batch</p>
        <strong>{report.engineLabel}</strong>
        <span>
          {report.successfulPages}/{report.totalPages} processed
          {report.failedPages ? ` · ${report.failedPages} failed` : ""}
        </span>
      </div>
      <dl>
        {statItems.map(([label, value]) => (
          <div key={label}>
            <dt>{label}</dt>
            <dd>{value}</dd>
          </div>
        ))}
      </dl>
      {report.failures.length ? <p className="batch-failures">{report.failures.join(", ")}</p> : null}
    </section>
  );
}

function batchProgressMessage(report: BatchTimingReport): string {
  const issueText = report.failedPages ? `, ${report.failedPages} failed` : "";
  return `${report.engineLabel} batch: ${report.processedPages}/${report.totalPages} pages checked${issueText}.`;
}

function batchFinishedMessage(report: BatchTimingReport): string {
  const issueText = report.failedPages ? ` with ${pageCountLabel(report.failedPages)} failed` : "";
  return `Finished ${report.engineLabel} batch${issueText}. See Latest batch for timing details.`;
}

function pageCountLabel(count: number): string {
  return `${count} ${count === 1 ? "page" : "pages"}`;
}

function PageRenameForm({
  draft,
  onDraftChange,
  onCancel,
  onSave
}: Readonly<{
  draft: string;
  onDraftChange: (value: string) => void;
  onCancel: () => void;
  onSave: () => void;
}>) {
  return (
    <form
      onSubmit={(event) => {
        event.preventDefault();
        onSave();
      }}
    >
      <input value={draft} onChange={(event) => onDraftChange(event.target.value)} autoFocus />
      <div className="mini-actions">
        <button type="submit">Save</button>
        <button type="button" className="ghost" onClick={onCancel}>Cancel</button>
      </div>
    </form>
  );
}

function PageSelectButton({
  page,
  candidateCount,
  timing,
  onSelect
}: Readonly<{
  page: Page;
  candidateCount: number;
  timing?: PageTimingStatus;
  onSelect: () => void;
}>) {
  return (
    <button className="page-select" onClick={onSelect}>
      <strong>{pageTitle(page)}</strong>
      <span>{pageTypeLabel(page.page_type)} · {Math.round(page.page_type_confidence * 100)}%</span>
      <small>{page.warnings.length} warnings · {pageCardSummary(page, candidateCount)}</small>
      {timing ? <PageTimingLine timing={timing} /> : null}
    </button>
  );
}

function PageTimingLine({ timing }: Readonly<{ timing: PageTimingStatus }>) {
  return (
    <small className={pageTimingClass(timing)}>
      {timing.engineLabel}: {pageTimingStatus(timing)} in {formatDuration(timing.ms)}
    </small>
  );
}

function PageActions({
  disabled,
  processing,
  vlProcessing,
  onEdit,
  onDelete,
  onProcess,
  onVlProcess
}: Readonly<{
  disabled: boolean;
  processing: boolean;
  vlProcessing: boolean;
  onEdit: () => void;
  onDelete: () => Promise<void>;
  onProcess: () => Promise<void>;
  onVlProcess: () => Promise<void>;
}>) {
  return (
    <div className="page-actions">
      <button className="rename" disabled={disabled} onClick={onEdit}>Rename</button>
      <button className="process-page" disabled={disabled || processing} onClick={() => void onProcess()}>
        {processing ? "Processing" : "Process"}
      </button>
      <button
        className="vl-page"
        disabled={disabled || vlProcessing}
        onClick={() => void onVlProcess()}
        title="Create candidates for only this page with PaddleOCR-VL."
      >
        {vlProcessing ? "VL processing" : "Process VL"}
      </button>
      <button className="delete-page" disabled={disabled} onClick={() => void onDelete()}>Delete</button>
    </div>
  );
}

function pageCardClass(active: boolean): string {
  return active ? "page-card active" : "page-card";
}

function pageCardSummary(page: Page, candidateCount: number): string {
  if (!candidateCount) return "No candidates";
  const approvedCount = page.approved_card_count ?? 0;
  const redCount = page.red_card_count ?? 0;
  return `${approvedCount}/${candidateCount} approved · ${redCount} blocked`;
}

function pageTimingClass(timing: PageTimingStatus): string {
  return timing.success ? "page-timing" : "page-timing failed";
}

function pageTimingStatus(timing: PageTimingStatus): string {
  return timing.success ? "processed" : "failed";
}

function EvidenceHeader({
  page,
  tokenCount,
  selectedCard,
  overlayMode,
  setOverlayMode
}: Readonly<{
  page: Page | null;
  tokenCount: number;
  selectedCard: CardCandidate | null;
  overlayMode: OverlayMode;
  setOverlayMode: (mode: OverlayMode) => void;
}>) {
  const overlayLabels: Record<OverlayMode, string> = {
    focused: "Focused",
    region: "Source region",
    all: "All OCR",
    off: "Off"
  };
  return (
    <div className="evidence-header">
      <div>
        <p className="eyebrow">Visual evidence</p>
        <h2>{page ? pageTitle(page) : "No page selected"}</h2>
        <p className="muted">
          {selectedCard ? evidenceSummary(selectedCard) : `${tokenCount} OCR tokens available after processing.`}
        </p>
      </div>
      <div className="segmented">
        {(["focused", "region", "all", "off"] as OverlayMode[]).map((mode) => (
          <button className={overlayMode === mode ? "active" : ""} key={mode} onClick={() => setOverlayMode(mode)}>
            {overlayLabels[mode]}
          </button>
        ))}
        <div className="overlay-legend" aria-label="OCR evidence color legend">
          <span><i className="legend-dot selected-high" /> high confidence</span>
          <span><i className="legend-dot selected-medium" /> review confidence</span>
          <span><i className="legend-dot selected-low" /> low confidence</span>
          <span><i className="legend-dot used-context" /> answer key/support</span>
          <span><i className="legend-dot scanned-unused" /> unused scan</span>
        </div>
      </div>
    </div>
  );
}

function EvidenceStage({
  imageUrl,
  page,
  tokens,
  cards,
  card,
  selectedField,
  regionDraft,
  mode,
  activeFilters,
  onSelectCard,
  onSelectEvidenceCard,
  onRegionDraftChange
}: Readonly<{
  imageUrl: string;
  page: Page | null;
  tokens: OcrToken[];
  cards: CardCandidate[];
  card: CardCandidate | null;
  selectedField: string | null;
  regionDraft: FieldRegionDraft | null;
  mode: OverlayMode;
  activeFilters: Set<string>;
  onSelectCard: (token: OcrToken) => void;
  onSelectEvidenceCard: (card: CardCandidate) => void;
  onRegionDraftChange: (draft: FieldRegionDraft) => void;
}>) {
  if (!page?.image_width || !page.image_height) {
    return (
      <div className="image-stage">
        <img src={imageUrl} alt="Uploaded study page" />
      </div>
    );
  }
  const focusBox = stageFocusBox(mode, card, tokens, selectedField);
  const viewBox = evidenceViewBox(page, focusBox);
  return (
    <div className={focusBox ? "image-stage zoomed" : "image-stage"}>
      <svg
        className="evidence-svg"
        viewBox={viewBox}
        width={page.image_width}
        height={page.image_height}
        aria-label="Uploaded study page with OCR evidence overlay"
        preserveAspectRatio="xMidYMid meet"
      >
        <image href={imageUrl} width={page.image_width} height={page.image_height} />
        <TokenOverlay
          page={page}
          tokens={tokens}
          cards={cards}
          card={card}
          selectedField={selectedField}
          mode={mode}
          activeFilters={activeFilters}
          onSelectCard={onSelectCard}
          onSelectEvidenceCard={onSelectEvidenceCard}
        />
        {regionDraft && card?.id === regionDraft.cardId ? (
          <EditableEvidenceBox
            bbox={regionDraft.bbox}
            page={page}
            onChange={(bbox) => onRegionDraftChange({ ...regionDraft, bbox })}
          />
        ) : null}
      </svg>
    </div>
  );
}

function stageFocusBox(
  mode: OverlayMode,
  card: CardCandidate | null,
  tokens: OcrToken[],
  selectedField: string | null
): number[] | null {
  if (mode === "focused") return focusBbox(card, tokens, selectedField);
  if (mode === "region") return sourceBbox(card);
  return null;
}

function TokenOverlay({
  page,
  tokens,
  cards,
  card,
  selectedField,
  mode,
  activeFilters,
  onSelectCard,
  onSelectEvidenceCard
}: Readonly<{
  page: Page | null;
  tokens: OcrToken[];
  cards: CardCandidate[];
  card: CardCandidate | null;
  selectedField: string | null;
  mode: OverlayMode;
  activeFilters: Set<string>;
  onSelectCard: (token: OcrToken) => void;
  onSelectEvidenceCard: (card: CardCandidate) => void;
}>) {
  if (!page?.image_width || !page.image_height || mode === "off") return null;
  const relevantIds = focusedTokenIdsForField(card, selectedField);
  const sourceBox = mode === "region" ? sourceBbox(card) : fieldBbox(card, selectedField) ?? sourceBbox(card);
  const selectedConfidenceClass = reviewQualityClass(card);
  const filteredTokens = activeFilters.size
    ? tokens.filter((token) => activeFilters.has(token.script_class) || activeFilters.has(token.source))
    : tokens;
  const shouldRenderTokens = mode === "all" || mode === "focused";
  return (
    <g className="overlay">
      {mode === "all"
        ? cards.map((candidate) => {
            const bbox = sourceBbox(candidate);
            return bbox ? (
              <EvidenceBox
                key={`source-${candidate.id}`}
                bbox={bbox}
                className={`source-region linked-region ${reviewQualityClass(candidate)}`}
                onClick={() => onSelectEvidenceCard(candidate)}
              />
            ) : null;
          })
        : null}
      {mode !== "all" && sourceBox ? (
        <EvidenceBox
          bbox={sourceBox}
          className={`source-region linked-region ${selectedConfidenceClass}`}
          onClick={card ? () => onSelectEvidenceCard(card) : undefined}
        />
      ) : null}
      {shouldRenderTokens
        ? filteredTokens.map((token) => {
            const relevant = relevantIds.has(token.id) || (!relevantIds.size && sourceBox ? tokenInside(token, sourceBox) : false);
            const bbox = clampBbox(token.bbox, page);
            if (!bbox) return null;
            const [x1, y1, x2, y2] = bbox;
            const linkedCard = cardForToken(cards, token);
            const displayClass = tokenDisplayClass(token, page, card, linkedCard, relevant);
            return (
              <g key={token.id}>
                <rect
                  x={x1}
                  y={y1}
                  width={x2 - x1}
                  height={y2 - y1}
                  className={`box ${token.script_class} ${displayClass} ${linkedCard ? "clickable" : ""}`}
                  onClick={
                    linkedCard
                      ? (event) => {
                          event.stopPropagation();
                          onSelectCard(token);
                        }
                      : undefined
                  }
                />
                <title>{tokenTitle(token, linkedCard, relevant)}</title>
              </g>
            );
          })
        : null}
    </g>
  );
}

function EvidenceBox({
  bbox,
  className,
  onClick
}: Readonly<{ bbox: number[]; className: string; onClick?: () => void }>) {
  const normalized = normalizeBbox(bbox);
  if (!normalized) return null;
  const [x1, y1, x2, y2] = normalized;
  return (
    <rect
      x={x1}
      y={y1}
      width={x2 - x1}
      height={y2 - y1}
      className={`${className} ${onClick ? "clickable-region" : ""}`}
      onClick={
        onClick
          ? (event) => {
              event.stopPropagation();
              onClick();
            }
          : undefined
      }
    />
  );
}

function EditableEvidenceBox({
  bbox,
  page,
  onChange
}: Readonly<{ bbox: number[]; page: Page; onChange: (bbox: number[]) => void }>) {
  const [drag, setDrag] = useState<{ mode: string; start: [number, number]; bbox: number[] } | null>(null);
  const normalized = clampBbox(bbox, page);
  if (!normalized) return null;
  const activeBbox = normalized;
  const [x1, y1, x2, y2] = activeBbox;

  function beginDrag(event: PointerEvent<SVGElement>, mode: string) {
    event.stopPropagation();
    event.currentTarget.setPointerCapture(event.pointerId);
    setDrag({ mode, start: svgPoint(event), bbox: activeBbox });
  }

  function moveDrag(event: PointerEvent<SVGElement>) {
    if (!drag) return;
    const [x, y] = svgPoint(event);
    const dx = x - drag.start[0];
    const dy = y - drag.start[1];
    let [left, top, right, bottom] = drag.bbox;
    if (drag.mode === "move") {
      left += dx;
      right += dx;
      top += dy;
      bottom += dy;
    } else {
      if (drag.mode.includes("w")) left += dx;
      if (drag.mode.includes("e")) right += dx;
      if (drag.mode.includes("n")) top += dy;
      if (drag.mode.includes("s")) bottom += dy;
    }
    const next = clampBbox([left, top, right, bottom], page);
    if (next) onChange(next);
  }

  return (
    <g className="editable-region" onPointerMove={moveDrag} onPointerUp={() => setDrag(null)} onPointerCancel={() => setDrag(null)}>
      <rect
        x={x1}
        y={y1}
        width={x2 - x1}
        height={y2 - y1}
        className="editable-region-box"
        onPointerDown={(event) => beginDrag(event, "move")}
      />
      {[
        ["nw", x1, y1],
        ["ne", x2, y1],
        ["sw", x1, y2],
        ["se", x2, y2]
      ].map(([mode, x, y]) => (
        <rect
          key={String(mode)}
          x={Number(x) - 8}
          y={Number(y) - 8}
          width={16}
          height={16}
          className="editable-region-handle"
          onPointerDown={(event) => beginDrag(event, String(mode))}
        />
      ))}
    </g>
  );
}

function clampBbox(bbox: number[], page: Page): number[] | null {
  const normalized = normalizeBbox(bbox);
  if (!normalized) return null;
  const width = page.image_width ?? 1;
  const height = page.image_height ?? 1;
  const left = Math.max(0, Math.min(width, normalized[0]));
  const right = Math.max(0, Math.min(width, normalized[2]));
  const top = Math.max(0, Math.min(height, normalized[1]));
  const bottom = Math.max(0, Math.min(height, normalized[3]));
  return normalizeBbox([left, top, right, bottom]);
}

function CandidateList({
  cards,
  selectedCard,
  cardRefs,
  onSelect,
  onChange,
  onToggleApproval,
  selectedField,
  fieldPreview,
  isPreviewingField,
  onSelectField,
  onPreviewField,
  onApplyFieldPreview,
  onCancelFieldPreview,
  grouped
}: Readonly<{
  cards: CardCandidate[];
  selectedCard: CardCandidate | null;
  cardRefs: CardRefs;
  onSelect: (card: CardCandidate, scrollTarget?: CardScrollTarget) => void;
  onChange: (card: CardCandidate) => Promise<void>;
  onToggleApproval: (card: CardCandidate) => Promise<void>;
  selectedField: string | null;
  fieldPreview: FieldOcrPreview | null;
  isPreviewingField: boolean;
  onSelectField: (field: string, card?: CardCandidate | null) => void;
  onPreviewField: (field?: string, card?: CardCandidate) => void;
  onApplyFieldPreview: () => void;
  onCancelFieldPreview: () => void;
  grouped: boolean;
}>) {
  if (!grouped) {
    return (
      <div className="candidate-list">
        <CandidateGroups
          cards={cards}
          selectedCard={selectedCard}
          cardRefs={cardRefs}
          onSelect={onSelect}
          onChange={onChange}
          onToggleApproval={onToggleApproval}
          selectedField={selectedField}
          fieldPreview={fieldPreview}
          isPreviewingField={isPreviewingField}
          onSelectField={onSelectField}
          onPreviewField={onPreviewField}
          onApplyFieldPreview={onApplyFieldPreview}
          onCancelFieldPreview={onCancelFieldPreview}
        />
      </div>
    );
  }

  const needsReview = cards.filter((card) => !isHighConfidenceCard(card));
  const highConfidence = cards.filter(isHighConfidenceCard);
  return (
    <div className="candidate-list">
      <CandidateSection
        title="Needs review"
        cards={needsReview}
        selectedCard={selectedCard}
        cardRefs={cardRefs}
        onSelect={onSelect}
        onChange={onChange}
        onToggleApproval={onToggleApproval}
        selectedField={selectedField}
        fieldPreview={fieldPreview}
        isPreviewingField={isPreviewingField}
        onSelectField={onSelectField}
        onPreviewField={onPreviewField}
        onApplyFieldPreview={onApplyFieldPreview}
        onCancelFieldPreview={onCancelFieldPreview}
        open
      />
      <CandidateSection
        title="High confidence"
        cards={highConfidence}
        selectedCard={selectedCard}
        cardRefs={cardRefs}
        onSelect={onSelect}
        onChange={onChange}
        onToggleApproval={onToggleApproval}
        selectedField={selectedField}
        fieldPreview={fieldPreview}
        isPreviewingField={isPreviewingField}
        onSelectField={onSelectField}
        onPreviewField={onPreviewField}
        onApplyFieldPreview={onApplyFieldPreview}
        onCancelFieldPreview={onCancelFieldPreview}
        open={false}
      />
    </div>
  );
}

function CandidateSection({
  title,
  cards,
  selectedCard,
  cardRefs,
  onSelect,
  onChange,
  onToggleApproval,
  selectedField,
  fieldPreview,
  isPreviewingField,
  onSelectField,
  onPreviewField,
  onApplyFieldPreview,
  onCancelFieldPreview,
  open
}: Readonly<{
  title: string;
  cards: CardCandidate[];
  selectedCard: CardCandidate | null;
  cardRefs: CardRefs;
  onSelect: (card: CardCandidate, scrollTarget?: CardScrollTarget) => void;
  onChange: (card: CardCandidate) => Promise<void>;
  onToggleApproval: (card: CardCandidate) => Promise<void>;
  selectedField: string | null;
  fieldPreview: FieldOcrPreview | null;
  isPreviewingField: boolean;
  onSelectField: (field: string, card?: CardCandidate | null) => void;
  onPreviewField: (field?: string, card?: CardCandidate) => void;
  onApplyFieldPreview: () => void;
  onCancelFieldPreview: () => void;
  open?: boolean;
}>) {
  if (!cards.length) return null;
  const sectionOpen = open || cards.some((card) => card.id === selectedCard?.id);
  return (
    <details className="candidate-section" open={sectionOpen}>
      <summary>
        <span>{title}</span>
        <b>{cards.length}</b>
      </summary>
      <div className="candidate-section-body">
        <CandidateGroups
          cards={cards}
          selectedCard={selectedCard}
          cardRefs={cardRefs}
          onSelect={onSelect}
          onChange={onChange}
          onToggleApproval={onToggleApproval}
          selectedField={selectedField}
          fieldPreview={fieldPreview}
          isPreviewingField={isPreviewingField}
          onSelectField={onSelectField}
          onPreviewField={onPreviewField}
          onApplyFieldPreview={onApplyFieldPreview}
          onCancelFieldPreview={onCancelFieldPreview}
        />
      </div>
    </details>
  );
}

function CandidateGroups({
  cards,
  selectedCard,
  cardRefs,
  onSelect,
  onChange,
  onToggleApproval,
  selectedField,
  fieldPreview,
  isPreviewingField,
  onSelectField,
  onPreviewField,
  onApplyFieldPreview,
  onCancelFieldPreview
}: Readonly<{
  cards: CardCandidate[];
  selectedCard: CardCandidate | null;
  cardRefs: CardRefs;
  onSelect: (card: CardCandidate, scrollTarget?: CardScrollTarget) => void;
  onChange: (card: CardCandidate) => Promise<void>;
  onToggleApproval: (card: CardCandidate) => Promise<void>;
  selectedField: string | null;
  fieldPreview: FieldOcrPreview | null;
  isPreviewingField: boolean;
  onSelectField: (field: string, card?: CardCandidate | null) => void;
  onPreviewField: (field?: string, card?: CardCandidate) => void;
  onApplyFieldPreview: () => void;
  onCancelFieldPreview: () => void;
}>) {
  return sourceGroups(cards).map((group) => (
    <div className="candidate-group" key={group.key}>
      {group.cards.length > 1 ? (
        <div className="candidate-group-title">
          <strong>{candidateTitle(group.cards[0])}</strong>
          <span>{group.cards.length} semantic Anki cards</span>
        </div>
      ) : null}
      {group.cards.map((card) => (
        <CardEditor
          key={card.id}
          card={card}
          selected={card.id === selectedCard?.id}
          cardRef={(element) => {
            cardRefs.current[card.id] = element;
          }}
          onSelect={() => onSelect(card)}
          onChange={onChange}
          onToggleApproval={onToggleApproval}
          selectedField={selectedField}
          fieldPreview={fieldPreview}
          isPreviewingField={isPreviewingField}
          onSelectField={onSelectField}
          onPreviewField={onPreviewField}
          onApplyFieldPreview={onApplyFieldPreview}
          onCancelFieldPreview={onCancelFieldPreview}
        />
      ))}
    </div>
  ));
}

function CardEditor({
  card,
  selected,
  cardRef,
  onSelect,
  onChange,
  onToggleApproval,
  selectedField,
  fieldPreview,
  isPreviewingField,
  onSelectField,
  onPreviewField,
  onApplyFieldPreview,
  onCancelFieldPreview
}: Readonly<{
  card: CardCandidate;
  selected: boolean;
  cardRef: (element: HTMLElement | null) => void;
  onSelect: () => void;
  onChange: (card: CardCandidate) => Promise<void>;
  onToggleApproval: (card: CardCandidate) => Promise<void>;
  selectedField: string | null;
  fieldPreview: FieldOcrPreview | null;
  isPreviewingField: boolean;
  onSelectField: (field: string, card?: CardCandidate | null) => void;
  onPreviewField: (field?: string, card?: CardCandidate) => void;
  onApplyFieldPreview: () => void;
  onCancelFieldPreview: () => void;
}>) {
  const [draft, setDraft] = useState(card);
  useEffect(() => setDraft(card), [card]);
  const source = card.source;
  const provenance = textValue(source.answer_source);
  const warnings = uniqueStrings(draft.warnings);
  const reviewBadges = reviewReasonBadges(draft);
  return (
    <article
      ref={cardRef}
      className={`candidate ${draft.review_state} ${reviewQualityClass(draft)} ${selected ? "selected" : "collapsed"}`}
      onClick={selected ? undefined : onSelect}
      onKeyDown={
        selected
          ? undefined
          : (event) => {
              if (!keyActivatesCard(event)) return;
              event.preventDefault();
              onSelect();
            }
      }
      role={selected ? undefined : "button"}
      tabIndex={selected ? undefined : 0}
    >
      <CandidateHeader
        card={card}
        draft={draft}
        provenance={provenance}
        reviewBadges={reviewBadges}
        warnings={warnings}
        onSelect={onSelect}
      />
      <SourceSummary card={draft} />

      {selected ? (
        <CandidateBody
          card={card}
          draft={draft}
          selectedField={selectedField}
          fieldPreview={fieldPreview}
          isPreviewingField={isPreviewingField}
          onDraftChange={setDraft}
          onSelectField={onSelectField}
          onPreviewField={onPreviewField}
          onApplyFieldPreview={onApplyFieldPreview}
          onCancelFieldPreview={onCancelFieldPreview}
          onChange={onChange}
          onToggleApproval={onToggleApproval}
        />
      ) : (
        <p className="candidate-fold-note">Click anywhere on this candidate to review and approve it.</p>
      )}
    </article>
  );
}

function CandidateHeader({
  card,
  draft,
  provenance,
  reviewBadges,
  warnings,
  onSelect
}: Readonly<{
  card: CardCandidate;
  draft: CardCandidate;
  provenance: string;
  reviewBadges: string[];
  warnings: string[];
  onSelect: () => void;
}>) {
  return (
    <>
      <button
        className="candidate-select"
        onClick={(event) => {
          event.stopPropagation();
          onSelect();
        }}
        type="button"
      >
        <div className="candidate-head">
          <div>
            <strong>{candidateTitle(card)}</strong>
            <p>{candidateSubtitle(card)}</p>
          </div>
          <div className="badges">
            <span className={`badge ${draft.review_state}`}>{draft.review_state}</span>
            {draft.status === "approved" ? <span className="badge approved">approved</span> : null}
            {reviewBadges.map((badge) => (
              <span className="badge review-reason" key={badge}>{badge}</span>
            ))}
            {provenance ? <span className={`badge ${provenance}`}>{provenanceLabel(provenance)}</span> : null}
          </div>
        </div>
      </button>
      {warnings.length ? <WarningList warnings={warnings} compact /> : null}
    </>
  );
}

function CandidateBody({
  card,
  draft,
  selectedField,
  fieldPreview,
  isPreviewingField,
  onDraftChange,
  onSelectField,
  onPreviewField,
  onApplyFieldPreview,
  onCancelFieldPreview,
  onChange,
  onToggleApproval
}: Readonly<{
  card: CardCandidate;
  draft: CardCandidate;
  selectedField: string | null;
  fieldPreview: FieldOcrPreview | null;
  isPreviewingField: boolean;
  onDraftChange: (card: CardCandidate) => void;
  onSelectField: (field: string, card?: CardCandidate | null) => void;
  onPreviewField: (field?: string, card?: CardCandidate) => void;
  onApplyFieldPreview: () => void;
  onCancelFieldPreview: () => void;
  onChange: (card: CardCandidate) => Promise<void>;
  onToggleApproval: (card: CardCandidate) => Promise<void>;
}>) {
  return (
    <div className="candidate-body">
      {card.source_type === "question_item" ? (
        <QuestionSourceEditor
          card={draft}
          selectedField={selectedField}
          fieldPreview={fieldPreview}
          isPreviewingField={isPreviewingField}
          onChange={onDraftChange}
          onSelectField={(field) => onSelectField(field, draft)}
          onPreviewField={(field) => onPreviewField(field, draft)}
          onApplyFieldPreview={onApplyFieldPreview}
          onCancelFieldPreview={onCancelFieldPreview}
        />
      ) : null}
      {card.source_type === "vocab_item" ? (
        <VocabSourceEditor
          card={draft}
          selectedField={selectedField}
          fieldPreview={fieldPreview}
          isPreviewingField={isPreviewingField}
          onChange={onDraftChange}
          onSelectField={(field) => onSelectField(field, draft)}
          onPreviewField={(field) => onPreviewField(field, draft)}
          onApplyFieldPreview={onApplyFieldPreview}
          onCancelFieldPreview={onCancelFieldPreview}
        />
      ) : null}
      <label>
        <span>Front</span>
        <textarea value={draft.front} onChange={(event) => onDraftChange({ ...draft, front: event.target.value })} />
      </label>
      <label>
        <span>Back</span>
        <textarea value={draft.back} onChange={(event) => onDraftChange({ ...draft, back: event.target.value })} />
      </label>
      <label>
        <span>Tags</span>
        <input
          value={draft.tags.join(" ")}
          onChange={(event) => onDraftChange({ ...draft, tags: event.target.value.split(/\s+/).filter(Boolean) })}
        />
      </label>
      <label>
        <span>Review state</span>
        <select
          value={draft.review_state}
          onChange={(event) => onDraftChange({ ...draft, review_state: event.target.value as CardCandidate["review_state"] })}
        >
          <option value="green">Green: exportable after approval</option>
          <option value="yellow">Yellow: needs review</option>
          <option value="red">Red: block from export</option>
        </select>
      </label>
      <div className="actions candidate-actions">
        <button className="secondary" onClick={() => void onChange(draft)}>Save edits</button>
        <button onClick={() => void onToggleApproval(draft)} disabled={draft.review_state === "red"}>
          {draft.status === "approved" ? "Unapprove" : "Approve"}
        </button>
      </div>
    </div>
  );
}

function QuestionSourceEditor({
  card,
  selectedField,
  fieldPreview,
  isPreviewingField,
  onChange,
  onSelectField,
  onPreviewField,
  onApplyFieldPreview,
  onCancelFieldPreview
}: Readonly<{
  card: CardCandidate;
  selectedField: string | null;
  fieldPreview: FieldOcrPreview | null;
  isPreviewingField: boolean;
  onChange: (card: CardCandidate) => void;
  onSelectField: (field: string) => void;
  onPreviewField: (field: string) => void;
  onApplyFieldPreview: () => void;
  onCancelFieldPreview: () => void;
}>) {
  const source = card.source;
  const [choicesDraft, setChoicesDraft] = useState(choicesText(source));
  const savedChoicesText = choicesText(source);
  useEffect(() => setChoicesDraft(savedChoicesText), [savedChoicesText]);

  function updateSource(field: string, value: string) {
    onChange(applyQuestionSourceEdit(card, field, value));
  }

  function updateChoices(value: string) {
    setChoicesDraft(value);
    onChange(applyQuestionChoicesEdit(card, value));
  }

  return (
    <fieldset className="question-source-editor">
      <legend>Question facts</legend>
      <QuestionFieldLabel field="question_no" selectedField={selectedField} onSelectField={onSelectField} onPreviewField={onPreviewField}>
        Question no.
      </QuestionFieldLabel>
      <label className="question-field">
        <span className="sr-only">Question no.</span>
        <input
          value={textValue(source.question_no)}
          onFocus={() => onSelectField("question_no")}
          onChange={(event) => updateSource("question_no", event.target.value)}
        />
        <InlineFieldPreview
          card={card}
          field="question_no"
          fieldPreview={fieldPreview}
          isPreviewingField={isPreviewingField}
          onApplyFieldPreview={onApplyFieldPreview}
          onCancelFieldPreview={onCancelFieldPreview}
        />
      </label>
      <QuestionFieldLabel field="sentence" selectedField={selectedField} onSelectField={onSelectField} onPreviewField={onPreviewField}>
        Sentence
      </QuestionFieldLabel>
      <label className="question-field wide">
        <span className="sr-only">Sentence</span>
        <textarea
          value={textValue(source.sentence)}
          onFocus={() => onSelectField("sentence")}
          onChange={(event) => updateSource("sentence", event.target.value)}
        />
        <InlineFieldPreview
          card={card}
          field="sentence"
          fieldPreview={fieldPreview}
          isPreviewingField={isPreviewingField}
          onApplyFieldPreview={onApplyFieldPreview}
          onCancelFieldPreview={onCancelFieldPreview}
        />
      </label>
      <QuestionFieldLabel field="target" selectedField={selectedField} onSelectField={onSelectField} onPreviewField={onPreviewField}>
        Target
      </QuestionFieldLabel>
      <label className="question-field">
        <span className="sr-only">Target</span>
        <input
          value={textValue(source.target)}
          onFocus={() => onSelectField("target")}
          onChange={(event) => updateSource("target", event.target.value)}
        />
        <InlineFieldPreview
          card={card}
          field="target"
          fieldPreview={fieldPreview}
          isPreviewingField={isPreviewingField}
          onApplyFieldPreview={onApplyFieldPreview}
          onCancelFieldPreview={onCancelFieldPreview}
        />
      </label>
      <div className="question-field-label">
        <span>Correct choice no.</span>
      </div>
      <label className="question-field">
        <span className="sr-only">Correct choice no.</span>
        <input
          inputMode="numeric"
          value={textValue(source.correct_choice_no)}
          onFocus={() => onSelectField("correct_answer")}
          onChange={(event) => updateSource("correct_choice_no", event.target.value)}
        />
      </label>
      <QuestionFieldLabel field="correct_answer" selectedField={selectedField} onSelectField={onSelectField} onPreviewField={onPreviewField}>
        Correct answer
      </QuestionFieldLabel>
      <label className="question-field">
        <span className="sr-only">Correct answer</span>
        <input
          value={textValue(source.correct_answer)}
          onFocus={() => onSelectField("correct_answer")}
          onChange={(event) => updateSource("correct_answer", event.target.value)}
        />
        <InlineFieldPreview
          card={card}
          field="correct_answer"
          fieldPreview={fieldPreview}
          isPreviewingField={isPreviewingField}
          onApplyFieldPreview={onApplyFieldPreview}
          onCancelFieldPreview={onCancelFieldPreview}
        />
      </label>
      <QuestionFieldLabel field="choice_1" selectedField={selectedField} onSelectField={onSelectField} onPreviewField={onPreviewField}>
        Choices (one per line)
      </QuestionFieldLabel>
      <label className="choice-editor question-field wide">
        <span className="sr-only">Choices, one per line</span>
        <div className="choice-field-actions" aria-label="Choice OCR evidence actions">
          {["choice_1", "choice_2", "choice_3", "choice_4"].map((field) => (
            <button
              key={field}
              type="button"
              className={selectedField === field ? "choice-eye active" : "choice-eye"}
              onClick={() => onSelectField(field)}
              title={`Highlight OCR evidence for ${fieldLabel(field)}`}
            >
              {fieldLabel(field)}
            </button>
          ))}
        </div>
        <textarea
          value={choicesDraft}
          onFocus={() => onSelectField(selectedField?.startsWith("choice_") ? selectedField : "choice_1")}
          onChange={(event) => updateChoices(event.target.value)}
          placeholder={"1st choice\n2nd choice\n3rd choice\n4th choice"}
        />
        {["choice_1", "choice_2", "choice_3", "choice_4"].map((field) => (
          <InlineFieldPreview
            key={field}
            card={card}
            field={field}
            fieldPreview={fieldPreview}
            isPreviewingField={isPreviewingField}
            onApplyFieldPreview={onApplyFieldPreview}
            onCancelFieldPreview={onCancelFieldPreview}
          />
        ))}
      </label>
      <QuestionFieldLabel field="answer_source" selectedField={selectedField} onSelectField={onSelectField} onPreviewField={onPreviewField}>
        Answer source
      </QuestionFieldLabel>
      <label className="question-field">
        <span className="sr-only">Answer source</span>
        <select
          value={textValue(source.answer_source, "manual")}
          onFocus={() => onSelectField("answer_source")}
          onChange={(event) => updateSource("answer_source", event.target.value)}
        >
          <option value="manual">Manual correction</option>
          <option value="answer_strip">Answer strip</option>
          <option value="local_glossary">Local glossary</option>
          <option value="unknown">Unknown</option>
        </select>
      </label>
    </fieldset>
  );
}

function VocabSourceEditor({
  card,
  selectedField,
  fieldPreview,
  isPreviewingField,
  onChange,
  onSelectField,
  onPreviewField,
  onApplyFieldPreview,
  onCancelFieldPreview
}: Readonly<{
  card: CardCandidate;
  selectedField: string | null;
  fieldPreview: FieldOcrPreview | null;
  isPreviewingField: boolean;
  onChange: (card: CardCandidate) => void;
  onSelectField: (field: string) => void;
  onPreviewField: (field: string) => void;
  onApplyFieldPreview: () => void;
  onCancelFieldPreview: () => void;
}>) {
  const source = card.source;

  function fieldInput(field: "surface" | "reading" | "meaning_ko", label: string) {
    return (
      <>
        <QuestionFieldLabel field={field} selectedField={selectedField} onSelectField={onSelectField} onPreviewField={onPreviewField}>
          {label}
        </QuestionFieldLabel>
        <label className="question-field">
          <span className="sr-only">{label}</span>
          <input
            value={textValue(source[field])}
            onFocus={() => onSelectField(field)}
            onChange={(event) => onChange(applyVocabSourceEdit(card, field, event.target.value))}
          />
          <InlineFieldPreview
            card={card}
            field={field}
            fieldPreview={fieldPreview}
            isPreviewingField={isPreviewingField}
            onApplyFieldPreview={onApplyFieldPreview}
            onCancelFieldPreview={onCancelFieldPreview}
          />
        </label>
      </>
    );
  }

  return (
    <fieldset className="question-source-editor">
      <legend>Vocabulary facts</legend>
      {fieldInput("surface", "Surface")}
      {fieldInput("reading", "Reading")}
      {fieldInput("meaning_ko", "Korean meaning")}
    </fieldset>
  );
}

function QuestionFieldLabel({
  field,
  selectedField,
  onSelectField,
  onPreviewField,
  children
}: Readonly<{
  field: string;
  selectedField: string | null;
  onSelectField: (field: string) => void;
  onPreviewField: (field: string) => void;
  children: ReactNode;
}>) {
  return (
    <div className={selectedField === field ? "question-field-label active" : "question-field-label"}>
      <button
        type="button"
        className="field-focus-button"
        onClick={() => onSelectField(field)}
        title={`Highlight OCR evidence for ${fieldLabel(field)}`}
      >
        {children}
      </button>
      <div className="field-mini-actions">
        <button type="button" className="field-eye preview" onClick={() => onPreviewField(field)} title={`Preview OCR for ${fieldLabel(field)}`}>
          Scan
        </button>
      </div>
    </div>
  );
}

function InlineFieldPreview({
  card,
  field,
  fieldPreview,
  isPreviewingField,
  onApplyFieldPreview,
  onCancelFieldPreview
}: Readonly<{
  card: CardCandidate;
  field: string;
  fieldPreview: FieldOcrPreview | null;
  isPreviewingField: boolean;
  onApplyFieldPreview: () => void;
  onCancelFieldPreview: () => void;
}>) {
  const matchingPreview = fieldPreview?.card_id === card.id && fieldPreview.field === field ? fieldPreview : null;
  if (!matchingPreview && (!isPreviewingField || fieldPreview?.field !== field)) return null;
  return (
    <div className="inline-field-preview">
      {matchingPreview ? (
        <>
          <strong>Suggested {fieldLabel(matchingPreview.field)}</strong>
          <p>{matchingPreview.text || "No text detected."}</p>
          {matchingPreview.warnings.length ? <WarningList warnings={matchingPreview.warnings} compact /> : null}
          <div className="actions candidate-actions">
            <button type="button" onClick={onApplyFieldPreview}>Apply suggestion</button>
            <button className="ghost" type="button" onClick={onCancelFieldPreview}>Discard</button>
          </div>
        </>
      ) : (
        <p className="muted">Previewing OCR...</p>
      )}
    </div>
  );
}

function SourceSummary({ card }: Readonly<{ card: CardCandidate }>) {
  const source = card.source;
  if (card.source_type === "vocab_item") {
    return (
      <div className="source-summary vocab">
        <span><b>Surface</b>{textValue(source.surface)}</span>
        <span><b>Reading</b>{textValue(source.reading)}</span>
        <span><b>Korean</b>{textValue(source.meaning_ko)}</span>
      </div>
    );
  }
  const choices = Array.isArray(source.choices) ? source.choices.map(String) : [];
  return (
    <div className="source-summary mcq">
      <span><b>Q</b>{textValue(source.question_no)}</span>
      <span><b>Target</b>{textValue(source.target)}</span>
      <span><b>Answer</b>{textValue(source.correct_choice_no, "?")}. {textValue(source.correct_answer)}</span>
      {choices.length ? <p>{choices.map((choice, index) => `${index + 1}. ${choice}`).join("  ·  ")}</p> : null}
    </div>
  );
}

function WarningList({ warnings, compact = false }: Readonly<{ warnings: string[]; compact?: boolean }>) {
  return (
    <div className={compact ? "warnings compact" : "warnings"}>
      {uniqueStrings(warnings).map((warning, index) => (
        <p key={warningKey(warning, index)}>{warning}</p>
      ))}
    </div>
  );
}

function DocumentParsePanel({ result }: Readonly<{ result: DocumentParseResult }>) {
  return (
    <div className="comparison">
      <h3>{result.provider}: {result.block_count} blocks via {result.backend}</h3>
      {result.warnings.length ? <WarningList warnings={result.warnings} compact /> : null}
      {result.markdown_text ? (
        <details>
          <summary>Markdown</summary>
          <pre>{result.markdown_text}</pre>
        </details>
      ) : null}
    </div>
  );
}

function OcrComparisonPanel({ comparison }: Readonly<{ comparison: OcrComparison }>) {
  return (
    <div className="comparison">
      <h3>OCR comparison</h3>
      <p>
        {comparison.primary_token_count} local tokens, {comparison.compare_token_count} comparison tokens,{" "}
        {Math.round(comparison.agreement * 100)}% overlap.
      </p>
      {comparison.warnings.length ? <WarningList warnings={comparison.warnings} compact /> : null}
    </div>
  );
}

function pageTitle(page: Page): string {
  return page.display_name?.trim() || page.original_image_path.split("/").pop()?.replace(/\.[^.]+$/, "") || page.id;
}

function versionedImageUrl(url: string | null, page: Page | null, tokenCount: number, cardCount: number): string | null {
  if (!url || !page) return url;
  const revision = [
    page.id,
    page.processed_image_path ?? page.original_image_path,
    page.page_type,
    page.page_type_confidence,
    tokenCount,
    cardCount
  ].join(":");
  const separator = url.includes("?") ? "&" : "?";
  return `${url}${separator}v=${encodeURIComponent(revision)}`;
}

function yieldToBrowser(): Promise<void> {
  return new Promise((resolve) => {
    globalThis.requestAnimationFrame?.(() => resolve()) ?? setTimeout(resolve, 0);
  });
}

function pageTypeLabel(type: string): string {
  return PAGE_TYPE_LABELS[type] ?? type.replaceAll("_", " ");
}

function sourceGroups(cards: CardCandidate[]): Array<{ key: string; cards: CardCandidate[] }> {
  const groups = new Map<string, CardCandidate[]>();
  for (const card of cards) {
    const key = `${card.source_type}:${card.source_id}`;
    groups.set(key, [...(groups.get(key) ?? []), card]);
  }
  return [...groups.entries()].map(([key, groupCards]) => ({ key, cards: groupCards }));
}
