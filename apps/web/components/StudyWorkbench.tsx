"use client";

import { type MutableRefObject, useEffect, useRef, useState, useTransition } from "react";
import {
  API_BASE,
  type CardCandidate,
  type DocumentParseResult,
  type OcrComparison,
  type OcrToken,
  type Page,
  apiGet,
  apiErrorMessage,
  approveCard,
  compareOcr,
  deletePage,
  exportTsv,
  imageUrl,
  parseDocument,
  processPage,
  updateCard,
  updatePage,
  uploadImages
} from "../lib/api";

type OverlayMode = "focused" | "region" | "all" | "off";
type ReviewFilter = "all" | "needs_review" | "approved" | "green" | "yellow" | "red";
type CardScrollTarget = "evidence" | "card" | "none";
const HIGH_CONFIDENCE_THRESHOLD = 0.9;
const REVIEW_CONFIDENCE_THRESHOLD = 0.75;

const PAGE_TYPE_LABELS: Record<string, string> = {
  uploaded: "Uploaded",
  vocab_table: "Vocabulary table",
  spelling_vocab_table: "Spelling vocabulary",
  reading_mcq: "Reading MCQ",
  spelling_mcq: "Spelling MCQ",
  unknown_review_required: "Needs manual review"
};

const SCRIPT_LABELS = ["paddleocr", "paddleocr_korean", "hiragana", "katakana", "kanji", "hangul", "mixed", "number"];

export function StudyWorkbench() {
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
  const [message, setMessage] = useState("Upload a study-book photo to begin.");
  const [isBatchProcessing, setIsBatchProcessing] = useState(false);
  const [isPending, startTransition] = useTransition();
  const evidenceRef = useRef<HTMLElement | null>(null);
  const cardRefs = useRef<Record<string, HTMLElement | null>>({});

  useEffect(() => {
    void refreshPages();
  }, []);

  const selectedCard = cards.find((card) => card.id === selectedCardId) ?? cards[0] ?? null;
  const cardStats = summarizeCards(cards);
  const filteredCards = cards.filter((card) => cardMatchesFilter(card, reviewFilter));
  const exportableCount = cards.filter((card) => card.status === "approved" && card.review_state !== "red").length;

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
    setSelectedCardId(pageCards[0]?.id ?? null);
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

  function onProcess() {
    if (!selectedPage) return;
    setMessage("Processing page locally. OCR may take a moment.");
    startTransition(async () => {
      try {
        const result = await processPage(selectedPage.id);
        setSelectedPage(result.page);
        setTokens(result.tokens);
        setCards(result.cards);
        setSelectedCardId(result.cards[0]?.id ?? null);
        setComparison(null);
        setDocumentParse(null);
        setMessage(`Processed as ${pageTypeLabel(result.page.page_type)}. Generated ${result.cards.length} candidates.`);
        await refreshPages(result.page.id);
      } catch (error) {
        setMessage(apiErrorMessage(error, "Processing failed."));
      }
    });
  }

  async function onProcessAllPages() {
    if (!pages.length || isBatchProcessing) return;
    setIsBatchProcessing(true);
    const failures: string[] = [];
    let lastSelectedResult: Awaited<ReturnType<typeof processPage>> | null = null;
    try {
      for (const [index, page] of pages.entries()) {
        setMessage(`Processing page ${index + 1}/${pages.length}: ${pageTitle(page)}...`);
        try {
          const result = await processPage(page.id);
          if (page.id === selectedPage?.id) {
            lastSelectedResult = result;
          }
        } catch (error) {
          failures.push(`${pageTitle(page)} (${apiErrorMessage(error, "Processing failed.")})`);
        }
      }
      if (lastSelectedResult) {
        setSelectedPage(lastSelectedResult.page);
        setTokens(lastSelectedResult.tokens);
        setCards(lastSelectedResult.cards);
        setSelectedCardId(lastSelectedResult.cards[0]?.id ?? null);
      }
      await refreshPages(selectedPage?.id);
      setMessage(
        failures.length
          ? `Processed ${pages.length - failures.length}/${pages.length} pages. Failed: ${failures.join(", ")}.`
          : `Processed ${pages.length} page${pages.length === 1 ? "" : "s"} sequentially.`
      );
    } finally {
      setIsBatchProcessing(false);
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
    const confirmed = window.confirm(`Delete "${title}" and its generated OCR/cards? This only removes local app state.`);
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
      setMessage("Moved card back to pending review.");
    } catch (error) {
      setMessage(apiErrorMessage(error, "Unapproving card failed."));
    }
  }

  async function onExport() {
    if (!selectedPage) return;
    try {
      const result = await exportTsv([selectedPage.id], { approved_only: true, include_yellow: true, include_red: false });
      setMessage(`Exported ${result.card_count} approved cards.`);
      window.open(`${API_BASE}${result.download_url}`, "_blank");
    } catch (error) {
      setMessage(apiErrorMessage(error, "Export failed."));
    }
  }

  async function onCompareOcr() {
    if (!selectedPage) return;
    setMessage("Comparing local OCR with Google Cloud Vision...");
    try {
      const result = await compareOcr(selectedPage.id);
      setComparison(result);
      setMessage(`OCR comparison complete: ${Math.round(result.agreement * 100)}% token agreement.`);
    } catch (error) {
      setMessage(apiErrorMessage(error, "OCR comparison failed."));
    }
  }

  async function onParseDocument() {
    if (!selectedPage) return;
    setMessage("Parsing page with PaddleOCR-VL...");
    try {
      const result = await parseDocument(selectedPage.id);
      setDocumentParse(result);
      setMessage(`PaddleOCR-VL returned ${result.block_count} document blocks.`);
    } catch (error) {
      setMessage(apiErrorMessage(error, "PaddleOCR-VL parsing failed."));
    }
  }

  function selectCard(card: CardCandidate, scrollTarget: CardScrollTarget = "evidence") {
    setSelectedCardId(card.id);
    const target =
      scrollTarget === "evidence" ? evidenceRef.current : scrollTarget === "card" ? cardRefs.current[card.id] : null;
    target?.scrollIntoView({ behavior: "smooth", block: "nearest" });
  }

  function selectCardForToken(token: OcrToken) {
    const card = cardForToken(cards, token);
    if (!card) return;
    selectCard(card, "card");
    setMessage(`Selected ${candidateTitle(card)} from OCR evidence.`);
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

  const processedUrl = imageUrl(selectedPage?.processed_image_path);
  const originalUrl = imageUrl(selectedPage?.original_image_path);
  const visibleUrl = processedUrl ?? originalUrl;

  return (
    <main className="app-shell">
      <header className="hero">
        <div>
          <p className="eyebrow">Local-first Japanese OCR review</p>
          <h1>Turn workbook photos into Anki candidates you can actually trust.</h1>
          <p className="lede">
            Upload, process, inspect the exact evidence for each card, approve only what looks right, then export TSV.
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
        <p>{isPending ? "Working..." : message}</p>
        <div>
          {selectedPage ? <button onClick={onProcess} disabled={isBatchProcessing}>Process page</button> : null}
          {pages.length ? (
            <button className="secondary" onClick={() => void onProcessAllPages()} disabled={isBatchProcessing}>
              Process all pages
            </button>
          ) : null}
          {selectedPage ? <button className="secondary" onClick={() => void onParseDocument()}>PaddleOCR-VL</button> : null}
          {tokens.length ? <button className="secondary" onClick={onCompareOcr}>Compare GCV</button> : null}
          {cards.length ? (
            <button onClick={onExport} disabled={exportableCount === 0}>
              Export {exportableCount} approved
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
                mode={overlayMode}
                activeFilters={activeTokenFilters}
                onSelectCard={selectCardForToken}
              />
            ) : (
              <div className="empty">Upload a page to see review evidence.</div>
            )}
            {selectedCard && overlayMode === "focused" && !focusedTokenIds(selectedCard).size && !selectedCard.source_bbox ? (
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
  onSelect,
  onRename,
  onDelete
}: Readonly<{
  page: Page;
  active: boolean;
  candidateCount: number;
  onSelect: () => void;
  onRename: (name: string) => Promise<void>;
  onDelete: () => Promise<void>;
}>) {
  const [editing, setEditing] = useState(false);
  const [draft, setDraft] = useState(pageTitle(page));
  useEffect(() => setDraft(pageTitle(page)), [page]);
  const approvedCount = page.approved_card_count ?? 0;
  const redCount = page.red_card_count ?? 0;
  const cardSummary = candidateCount ? `${approvedCount}/${candidateCount} approved · ${redCount} blocked` : "No candidates";
  return (
    <article className={active ? "page-card active" : "page-card"}>
      {editing ? (
        <form
          onSubmit={(event) => {
            event.preventDefault();
            void onRename(draft).then(() => setEditing(false));
          }}
        >
          <input value={draft} onChange={(event) => setDraft(event.target.value)} autoFocus />
          <div className="mini-actions">
            <button type="submit">Save</button>
            <button type="button" className="ghost" onClick={() => setEditing(false)}>Cancel</button>
          </div>
        </form>
      ) : (
        <button className="page-select" onClick={onSelect}>
          <strong>{pageTitle(page)}</strong>
          <span>{pageTypeLabel(page.page_type)} · {Math.round(page.page_type_confidence * 100)}%</span>
          <small>{page.warnings.length} warnings · {cardSummary}</small>
        </button>
      )}
      {editing ? null : (
        <div className="page-actions">
          <button className="rename" onClick={() => setEditing(true)}>Rename</button>
          <button className="delete-page" onClick={() => void onDelete()}>Delete</button>
        </div>
      )}
    </article>
  );
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
  mode,
  activeFilters,
  onSelectCard
}: Readonly<{
  imageUrl: string;
  page: Page | null;
  tokens: OcrToken[];
  cards: CardCandidate[];
  card: CardCandidate | null;
  mode: OverlayMode;
  activeFilters: Set<string>;
  onSelectCard: (token: OcrToken) => void;
}>) {
  if (!page?.image_width || !page.image_height) {
    return (
      <div className="image-stage">
        <img src={imageUrl} alt="Uploaded study page" />
      </div>
    );
  }
  const focusBox = mode === "focused" || mode === "region" ? focusBbox(card, tokens) : null;
  const viewBox = evidenceViewBox(page, focusBox);
  return (
    <div className={focusBox ? "image-stage zoomed" : "image-stage"}>
      <svg
        className="evidence-svg"
        viewBox={viewBox}
        aria-label="Uploaded study page with OCR evidence overlay"
        preserveAspectRatio="xMidYMid meet"
      >
        <image href={imageUrl} width={page.image_width} height={page.image_height} />
        <TokenOverlay
          page={page}
          tokens={tokens}
          cards={cards}
          card={card}
          mode={mode}
          activeFilters={activeFilters}
          onSelectCard={onSelectCard}
        />
      </svg>
    </div>
  );
}

function TokenOverlay({
  page,
  tokens,
  cards,
  card,
  mode,
  activeFilters,
  onSelectCard
}: Readonly<{
  page: Page | null;
  tokens: OcrToken[];
  cards: CardCandidate[];
  card: CardCandidate | null;
  mode: OverlayMode;
  activeFilters: Set<string>;
  onSelectCard: (token: OcrToken) => void;
}>) {
  if (!page?.image_width || !page.image_height || mode === "off") return null;
  const relevantIds = focusedTokenIds(card);
  const sourceBox = sourceBbox(card);
  const selectedConfidenceClass = confidenceClass(card);
  const filteredTokens = activeFilters.size
    ? tokens.filter((token) => activeFilters.has(token.script_class) || activeFilters.has(token.source))
    : tokens;
  const shouldRenderTokens = mode === "all" || mode === "focused";
  return (
    <g className="overlay">
      {mode !== "all" && sourceBox ? <EvidenceBox bbox={sourceBox} className={`source-region ${selectedConfidenceClass}`} /> : null}
      {shouldRenderTokens
        ? filteredTokens.map((token) => {
            const relevant = relevantIds.has(token.id) || (!relevantIds.size && sourceBox ? tokenInside(token, sourceBox) : false);
            const [x1, y1, x2, y2] = token.bbox;
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

function EvidenceBox({ bbox, className }: Readonly<{ bbox: number[]; className: string }>) {
  const [x1, y1, x2, y2] = bbox;
  return <rect x={x1} y={y1} width={x2 - x1} height={y2 - y1} className={className} />;
}

function CandidateList({
  cards,
  selectedCard,
  cardRefs,
  onSelect,
  onChange,
  onToggleApproval,
  grouped
}: Readonly<{
  cards: CardCandidate[];
  selectedCard: CardCandidate | null;
  cardRefs: MutableRefObject<Record<string, HTMLElement | null>>;
  onSelect: (card: CardCandidate, scrollTarget?: CardScrollTarget) => void;
  onChange: (card: CardCandidate) => Promise<void>;
  onToggleApproval: (card: CardCandidate) => Promise<void>;
  grouped: boolean;
}>) {
  if (!grouped) {
    return (
      <div className="candidate-list">
        {cards.map((card) => (
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
          />
        ))}
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
        open
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
  open
}: Readonly<{
  title: string;
  cards: CardCandidate[];
  selectedCard: CardCandidate | null;
  cardRefs: MutableRefObject<Record<string, HTMLElement | null>>;
  onSelect: (card: CardCandidate, scrollTarget?: CardScrollTarget) => void;
  onChange: (card: CardCandidate) => Promise<void>;
  onToggleApproval: (card: CardCandidate) => Promise<void>;
  open?: boolean;
}>) {
  if (!cards.length) return null;
  return (
    <details className="candidate-section" open={open}>
      <summary>
        <span>{title}</span>
        <b>{cards.length}</b>
      </summary>
      <div className="candidate-section-body">
        {cards.map((card) => (
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
          />
        ))}
      </div>
    </details>
  );
}

function CardEditor({
  card,
  selected,
  cardRef,
  onSelect,
  onChange,
  onToggleApproval
}: Readonly<{
  card: CardCandidate;
  selected: boolean;
  cardRef: (element: HTMLElement | null) => void;
  onSelect: () => void;
  onChange: (card: CardCandidate) => Promise<void>;
  onToggleApproval: (card: CardCandidate) => Promise<void>;
}>) {
  const [draft, setDraft] = useState(card);
  useEffect(() => setDraft(card), [card]);
  const source = card.source;
  const provenance = textValue(source.answer_source);
  return (
    <article
      ref={cardRef}
      className={`candidate ${draft.review_state} ${selected ? "selected" : "collapsed"}`}
      onClick={onSelect}
    >
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
            {provenance ? <span className={`badge ${provenance}`}>{provenanceLabel(provenance)}</span> : null}
          </div>
        </div>
      </button>
      <SourceSummary card={card} />

      {selected ? (
        <div className="candidate-body" onClick={(event) => event.stopPropagation()}>
          {card.source_type === "question_item" ? (
            <QuestionSourceEditor card={draft} onChange={setDraft} />
          ) : null}
          <label>
            <span>Front</span>
            <textarea value={draft.front} onChange={(event) => setDraft({ ...draft, front: event.target.value })} />
          </label>
          <label>
            <span>Back</span>
            <textarea value={draft.back} onChange={(event) => setDraft({ ...draft, back: event.target.value })} />
          </label>
          <label>
            <span>Tags</span>
            <input
              value={draft.tags.join(" ")}
              onChange={(event) => setDraft({ ...draft, tags: event.target.value.split(/\s+/).filter(Boolean) })}
            />
          </label>
          <label>
            <span>Review state</span>
            <select
              value={draft.review_state}
              onChange={(event) =>
                setDraft({ ...draft, review_state: event.target.value as CardCandidate["review_state"] })
              }
            >
              <option value="green">Green: exportable after approval</option>
              <option value="yellow">Yellow: review carefully</option>
              <option value="red">Red: block from export</option>
            </select>
          </label>
          {draft.warnings.length ? <WarningList warnings={draft.warnings} compact /> : null}
          <div className="actions candidate-actions">
            <button className="secondary" onClick={() => void onChange(draft)}>Save edits</button>
            <button onClick={() => void onToggleApproval(draft)} disabled={draft.review_state === "red"}>
              {draft.status === "approved" ? "Unapprove" : "Approve"}
            </button>
          </div>
        </div>
      ) : (
        <p className="candidate-fold-note">Click anywhere on this candidate to review and approve it.</p>
      )}
    </article>
  );
}

function QuestionSourceEditor({
  card,
  onChange
}: Readonly<{
  card: CardCandidate;
  onChange: (card: CardCandidate) => void;
}>) {
  const source = card.source;
  function updateSource(field: string, value: string) {
    const nextSource = { ...source, [field]: field === "question_no" || field === "correct_choice_no" ? numberOrEmpty(value) : value };
    const nextCard = syncQuestionAnswerBack({ ...card, source: nextSource }, nextSource);
    onChange(nextCard);
  }

  return (
    <fieldset className="question-source-editor">
      <legend>Question facts</legend>
      <label>
        <span>Question no.</span>
        <input value={textValue(source.question_no)} onChange={(event) => updateSource("question_no", event.target.value)} />
      </label>
      <label>
        <span>Sentence</span>
        <textarea value={textValue(source.sentence)} onChange={(event) => updateSource("sentence", event.target.value)} />
      </label>
      <label>
        <span>Target</span>
        <input value={textValue(source.target)} onChange={(event) => updateSource("target", event.target.value)} />
      </label>
      <label>
        <span>Correct choice no.</span>
        <input
          inputMode="numeric"
          value={textValue(source.correct_choice_no)}
          onChange={(event) => updateSource("correct_choice_no", event.target.value)}
        />
      </label>
      <label>
        <span>Correct answer</span>
        <input value={textValue(source.correct_answer)} onChange={(event) => updateSource("correct_answer", event.target.value)} />
      </label>
      <label>
        <span>Answer source</span>
        <select value={textValue(source.answer_source, "manual")} onChange={(event) => updateSource("answer_source", event.target.value)}>
          <option value="manual">Manual correction</option>
          <option value="answer_strip">Answer strip</option>
          <option value="local_glossary">Local glossary</option>
          <option value="unknown">Unknown</option>
        </select>
      </label>
    </fieldset>
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
      {warnings.map((warning, index) => (
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

function pageTypeLabel(type: string): string {
  return PAGE_TYPE_LABELS[type] ?? type.replaceAll("_", " ");
}

function workflowClass(index: number, page: Page | null, cards: CardCandidate[], exportableCount: number): string {
  const complete =
    (index === 0 && page) ||
    (index === 1 && page?.page_type && page.page_type !== "uploaded") ||
    (index === 2 && cards.length > 0) ||
    (index === 3 && exportableCount > 0);
  return complete ? "step complete" : "step";
}

function summarizeCards(cards: CardCandidate[]) {
  return {
    total: cards.length,
    approved: cards.filter((card) => card.status === "approved").length,
    needsReview: cards.filter((card) => card.review_state === "yellow" || card.warnings.length).length,
    red: cards.filter((card) => card.review_state === "red").length
  };
}

function cardMatchesFilter(card: CardCandidate, filter: ReviewFilter): boolean {
  if (filter === "all") return true;
  if (filter === "needs_review") return card.review_state === "yellow" || card.warnings.length > 0;
  if (filter === "approved") return card.status === "approved";
  return card.review_state === filter;
}

function isHighConfidenceCard(card: CardCandidate): boolean {
  return card.review_state === "green" && card.warnings.length === 0 && card.confidence >= HIGH_CONFIDENCE_THRESHOLD;
}

export function cardForToken(cards: CardCandidate[], token: OcrToken): CardCandidate | null {
  return (
    cards.find((card) => focusedTokenIds(card).has(token.id)) ??
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

function tokenDisplayClass(
  token: OcrToken,
  page: Page,
  selectedCard: CardCandidate | null,
  linkedCard: CardCandidate | null,
  relevant: boolean,
): string {
  if (relevant) return `candidate-evidence selected-evidence ${confidenceClass(selectedCard)}`;
  if (linkedCard) return `candidate-evidence ${confidenceClass(linkedCard)}`;
  if (isAnswerSupportToken(token, page)) return "used-context answer-support";
  return `scanned-unused token-confidence-${tokenConfidenceClass(token.confidence)}`;
}

export function confidenceClass(card: CardCandidate | null): string {
  if (!card) return "confidence-unknown";
  if (card.confidence < REVIEW_CONFIDENCE_THRESHOLD) return "confidence-low";
  if (card.confidence < HIGH_CONFIDENCE_THRESHOLD) return "confidence-medium";
  return "confidence-high";
}

function tokenConfidenceClass(confidence: number): string {
  if (confidence >= HIGH_CONFIDENCE_THRESHOLD) return "high";
  if (confidence >= REVIEW_CONFIDENCE_THRESHOLD) return "medium";
  return "low";
}

function tokenTitle(token: OcrToken, linkedCard: CardCandidate | null, relevant: boolean): string {
  const use = relevant ? "selected candidate evidence" : linkedCard ? `used by ${candidateTitle(linkedCard)}` : "scanned but unused";
  return `${token.text} (${token.source}, ${token.script_class}, OCR ${Math.round(token.confidence * 100)}%, ${use})`;
}

function isAnswerSupportToken(token: OcrToken, page: Page): boolean {
  if (!page.image_height || !page.page_type.endsWith("_mcq")) return false;
  if (token.bbox[1] < page.image_height * 0.82) return false;
  return /[①②③④⑤⑥⑦⑧⑨⑩0-9]/.test(token.text);
}

function focusedTokenIds(card: CardCandidate | null): Set<string> {
  const tokens = card?.source.evidence_tokens;
  if (!Array.isArray(tokens)) return new Set();
  return new Set(tokens.filter((token): token is string => typeof token === "string"));
}

function focusBbox(card: CardCandidate | null, tokens: OcrToken[]): number[] | null {
  const relevantIds = focusedTokenIds(card);
  const relevantBoxes = tokens.filter((token) => relevantIds.has(token.id)).map((token) => token.bbox);
  return unionBoxes(relevantBoxes) ?? sourceBbox(card);
}

function sourceBbox(card: CardCandidate | null): number[] | null {
  return card?.source_bbox ?? bboxFromSource(card?.source);
}

function unionBoxes(boxes: number[][]): number[] | null {
  const valid = boxes.filter((box) => box.length === 4 && box.every((value) => Number.isFinite(value)));
  if (!valid.length) return null;
  return [
    Math.min(...valid.map((box) => box[0])),
    Math.min(...valid.map((box) => box[1])),
    Math.max(...valid.map((box) => box[2])),
    Math.max(...valid.map((box) => box[3]))
  ];
}

function evidenceViewBox(page: Page, bbox: number[] | null): string {
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

function clamp(value: number, min: number, max: number): number {
  return Math.min(Math.max(value, min), Math.max(min, max));
}

function bboxFromSource(source: Record<string, unknown> | undefined): number[] | null {
  const bbox = source?.bbox;
  if (!Array.isArray(bbox) || bbox.length !== 4) return null;
  return bbox.every((value) => typeof value === "number") ? bbox : null;
}

function tokenInside(token: OcrToken, bbox: number[]): boolean {
  const [x1, y1, x2, y2] = bbox;
  const cx = (token.bbox[0] + token.bbox[2]) / 2;
  const cy = (token.bbox[1] + token.bbox[3]) / 2;
  return cx >= x1 && cx <= x2 && cy >= y1 && cy <= y2;
}

function evidenceSummary(card: CardCandidate): string {
  const count = focusedTokenIds(card).size;
  if (count) return `${count} evidence tokens are highlighted for the selected candidate.`;
  if (card.source_bbox) return "Source region is highlighted for the selected candidate.";
  return "Select another candidate or switch to All OCR for debugging.";
}

function candidateTitle(card: CardCandidate): string {
  if (card.source_type === "vocab_item") {
    return `${textValue(card.source.surface, "Vocab")} · ${textValue(card.source.reading)}`;
  }
  return `Question ${textValue(card.source.question_no, "?")} · ${textValue(card.source.target, "target")}`;
}

function candidateSubtitle(card: CardCandidate): string {
  return `${card.note_type} · ${Math.round(card.confidence * 100)}% confidence`;
}

function provenanceLabel(value: string): string {
  if (value === "answer_strip") return "answer strip";
  if (value === "local_glossary") return "review carefully";
  if (value === "manual") return "manual";
  return value.replaceAll("_", " ");
}

function textValue(value: unknown, fallback = ""): string {
  if (typeof value === "string") return value;
  if (typeof value === "number" || typeof value === "boolean") return String(value);
  return fallback;
}

function numberOrEmpty(value: string): number | string {
  const trimmed = value.trim();
  if (!trimmed) return "";
  const parsed = Number(trimmed);
  return Number.isFinite(parsed) ? parsed : trimmed;
}

function syncQuestionAnswerBack(card: CardCandidate, source: Record<string, unknown>): CardCandidate {
  const answer = textValue(source.correct_answer);
  if (!answer) return card;
  if (card.note_type.endsWith("_exam")) {
    const choiceNo = textValue(source.correct_choice_no, "?");
    return { ...card, back: `정답: ${choiceNo}. ${answer}` };
  }
  return { ...card, back: answer };
}
