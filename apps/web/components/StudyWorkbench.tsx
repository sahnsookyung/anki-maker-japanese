"use client";

import { useEffect, useRef, useState, useTransition } from "react";
import {
  API_BASE,
  type CardCandidate,
  type DocumentParseResult,
  type OcrComparison,
  type OcrToken,
  type Page,
  apiGet,
  approveCard,
  compareOcr,
  exportTsv,
  imageUrl,
  parseDocument,
  processPage,
  updateCard,
  updatePage,
  uploadImage
} from "../lib/api";

type OverlayMode = "focused" | "region" | "all" | "off";
type ReviewFilter = "all" | "needs_review" | "approved" | "green" | "yellow" | "red";

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
  const [pageCardCounts, setPageCardCounts] = useState<Record<string, number>>({});
  const [selectedCardId, setSelectedCardId] = useState<string | null>(null);
  const [overlayMode, setOverlayMode] = useState<OverlayMode>("focused");
  const [reviewFilter, setReviewFilter] = useState<ReviewFilter>("all");
  const [activeTokenFilters, setActiveTokenFilters] = useState<Set<string>>(new Set());
  const [comparison, setComparison] = useState<OcrComparison | null>(null);
  const [documentParse, setDocumentParse] = useState<DocumentParseResult | null>(null);
  const [message, setMessage] = useState("Upload a study-book photo to begin.");
  const [isPending, startTransition] = useTransition();
  const evidenceRef = useRef<HTMLElement | null>(null);

  useEffect(() => {
    void refreshPages();
  }, []);

  const selectedCard = cards.find((card) => card.id === selectedCardId) ?? cards[0] ?? null;
  const cardStats = summarizeCards(cards);
  const filteredCards = cards.filter((card) => cardMatchesFilter(card, reviewFilter));
  const exportableCount = cards.filter((card) => card.status === "approved" && card.review_state !== "red").length;

  async function refreshPages(preferredPageId?: string) {
    const nextPages = await apiGet<Page[]>("/api/pages");
    const counts = await pageCounts(nextPages);
    setPages(nextPages);
    setPageCardCounts(counts);
    const nextSelected =
      nextPages.find((page) => page.id === preferredPageId) ??
      nextPages.find((page) => page.id === selectedPage?.id) ??
      nextPages[0];
    if (nextSelected) {
      await selectPage(nextSelected, false);
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
    setPageCardCounts((current) => ({ ...current, [page.id]: pageCards.length }));
    setSelectedCardId(pageCards[0]?.id ?? null);
    setComparison(null);
    setDocumentParse(null);
    if (clearMessage) setMessage(`Selected ${pageTitle(ocr.page)}.`);
  }

  async function onUpload(file: File | undefined) {
    if (!file) return;
    setMessage("Uploading image...");
    const result = await uploadImage(file);
    setMessage("Image uploaded. Run processing to create review candidates.");
    await refreshPages(result.page_id);
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
        setMessage(error instanceof Error ? error.message : "Processing failed.");
      }
    });
  }

  async function renamePage(page: Page, displayName: string) {
    const updated = await updatePage(page.id, displayName);
    setPages((current) => current.map((item) => (item.id === updated.id ? updated : item)));
    if (selectedPage?.id === updated.id) setSelectedPage(updated);
    setMessage(`Renamed page to ${pageTitle(updated)}.`);
  }

  async function saveCard(card: CardCandidate) {
    const updated = await updateCard(card);
    setCards((current) => current.map((item) => (item.id === updated.id ? updated : item)));
  }

  async function approve(cardId: string) {
    const updated = await approveCard(cardId);
    setCards((current) => current.map((item) => (item.id === updated.id ? updated : item)));
  }

  async function onExport() {
    if (!selectedPage) return;
    const result = await exportTsv([selectedPage.id], { approved_only: true, include_yellow: true, include_red: false });
    setMessage(`Exported ${result.card_count} approved cards.`);
    window.open(`${API_BASE}${result.download_url}`, "_blank");
  }

  async function onCompareOcr() {
    if (!selectedPage) return;
    setMessage("Comparing local OCR with Google Cloud Vision...");
    try {
      const result = await compareOcr(selectedPage.id);
      setComparison(result);
      setMessage(`OCR comparison complete: ${Math.round(result.agreement * 100)}% token agreement.`);
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "OCR comparison failed.");
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
      setMessage(error instanceof Error ? error.message : "PaddleOCR-VL parsing failed.");
    }
  }

  function selectCard(card: CardCandidate) {
    setSelectedCardId(card.id);
    evidenceRef.current?.scrollIntoView({ behavior: "smooth", block: "start" });
  }

  function toggleTokenFilter(value: string) {
    setActiveTokenFilters((current) => {
      const next = new Set(current);
      if (next.has(value)) next.delete(value);
      else next.add(value);
      return next;
    });
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
          <span>Upload page</span>
          <small>JPG, PNG, WEBP, TIFF</small>
          <input type="file" accept="image/*" onChange={(event) => void onUpload(event.target.files?.[0])} />
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
          {selectedPage ? <button onClick={onProcess}>Process page</button> : null}
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
              candidateCount={page.id === selectedPage?.id ? cards.length : pageCardCounts[page.id] ?? 0}
              onSelect={() => void selectPage(page)}
              onRename={(name) => renamePage(page, name)}
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
              <div className="image-stage">
                <img src={visibleUrl} alt="Uploaded study page" />
                <TokenOverlay
                  page={selectedPage}
                  tokens={tokens}
                  card={selectedCard}
                  mode={overlayMode}
                  activeFilters={activeTokenFilters}
                />
              </div>
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
            <div className="candidate-list">
              {filteredCards.map((card) => (
                <CardEditor
                  key={card.id}
                  card={card}
                  selected={card.id === selectedCard?.id}
                  onSelect={() => selectCard(card)}
                  onChange={saveCard}
                  onApprove={approve}
                />
              ))}
            </div>
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
  onRename
}: {
  page: Page;
  active: boolean;
  candidateCount: number;
  onSelect: () => void;
  onRename: (name: string) => Promise<void>;
}) {
  const [editing, setEditing] = useState(false);
  const [draft, setDraft] = useState(pageTitle(page));
  useEffect(() => setDraft(pageTitle(page)), [page]);
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
          <small>{page.warnings.length} warnings · {candidateCount || "No"} candidates</small>
        </button>
      )}
      {!editing ? <button className="rename" onClick={() => setEditing(true)}>Rename</button> : null}
    </article>
  );
}

function EvidenceHeader({
  page,
  tokenCount,
  selectedCard,
  overlayMode,
  setOverlayMode
}: {
  page: Page | null;
  tokenCount: number;
  selectedCard: CardCandidate | null;
  overlayMode: OverlayMode;
  setOverlayMode: (mode: OverlayMode) => void;
}) {
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
            {mode === "focused" ? "Focused" : mode === "region" ? "Source region" : mode === "all" ? "All OCR" : "Off"}
          </button>
        ))}
      </div>
    </div>
  );
}

function TokenOverlay({
  page,
  tokens,
  card,
  mode,
  activeFilters
}: {
  page: Page | null;
  tokens: OcrToken[];
  card: CardCandidate | null;
  mode: OverlayMode;
  activeFilters: Set<string>;
}) {
  if (!page?.image_width || !page.image_height || mode === "off") return null;
  const relevantIds = focusedTokenIds(card);
  const sourceBox = card?.source_bbox ?? bboxFromSource(card?.source);
  const filteredTokens = activeFilters.size
    ? tokens.filter((token) => activeFilters.has(token.script_class) || activeFilters.has(token.source))
    : tokens;
  const shouldRenderTokens = mode === "all" || mode === "focused";
  return (
    <svg viewBox={`0 0 ${page.image_width} ${page.image_height}`} className="overlay">
      {mode !== "all" && sourceBox ? <EvidenceBox bbox={sourceBox} className="source-region" /> : null}
      {shouldRenderTokens
        ? filteredTokens.map((token) => {
            const relevant = relevantIds.has(token.id) || (!relevantIds.size && sourceBox ? tokenInside(token, sourceBox) : false);
            const dimmed = mode === "focused" && !relevant;
            const [x1, y1, x2, y2] = token.bbox;
            return (
              <g key={token.id}>
                <rect
                  x={x1}
                  y={y1}
                  width={x2 - x1}
                  height={y2 - y1}
                  className={`box ${token.script_class} ${relevant ? "relevant" : ""} ${dimmed ? "dimmed" : ""}`}
                />
                <title>{`${token.text} (${token.source}, ${token.script_class}, ${Math.round(token.confidence * 100)}%)`}</title>
              </g>
            );
          })
        : null}
    </svg>
  );
}

function EvidenceBox({ bbox, className }: { bbox: number[]; className: string }) {
  const [x1, y1, x2, y2] = bbox;
  return <rect x={x1} y={y1} width={x2 - x1} height={y2 - y1} className={className} />;
}

function CardEditor({
  card,
  selected,
  onSelect,
  onChange,
  onApprove
}: {
  card: CardCandidate;
  selected: boolean;
  onSelect: () => void;
  onChange: (card: CardCandidate) => Promise<void>;
  onApprove: (cardId: string) => Promise<void>;
}) {
  const [draft, setDraft] = useState(card);
  useEffect(() => setDraft(card), [card]);
  const source = card.source;
  const provenance = String(source.answer_source ?? "");
  return (
    <article className={`candidate ${draft.review_state} ${selected ? "selected" : ""}`} onClick={onSelect}>
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

      <SourceSummary card={card} />

      <label>
        Front
        <textarea value={draft.front} onChange={(event) => setDraft({ ...draft, front: event.target.value })} />
      </label>
      <label>
        Back
        <textarea value={draft.back} onChange={(event) => setDraft({ ...draft, back: event.target.value })} />
      </label>
      <label>
        Tags
        <input
          value={draft.tags.join(" ")}
          onChange={(event) => setDraft({ ...draft, tags: event.target.value.split(/\s+/).filter(Boolean) })}
        />
      </label>
      {draft.warnings.length ? <WarningList warnings={draft.warnings} compact /> : null}
      <div className="actions">
        <button className="secondary" onClick={() => void onChange(draft)}>Save edits</button>
        <button onClick={() => void onApprove(draft.id)} disabled={draft.review_state === "red"}>
          Approve
        </button>
      </div>
    </article>
  );
}

function SourceSummary({ card }: { card: CardCandidate }) {
  const source = card.source;
  if (card.source_type === "vocab_item") {
    return (
      <div className="source-summary vocab">
        <span><b>Surface</b>{String(source.surface ?? "")}</span>
        <span><b>Reading</b>{String(source.reading ?? "")}</span>
        <span><b>Korean</b>{String(source.meaning_ko ?? "")}</span>
      </div>
    );
  }
  const choices = Array.isArray(source.choices) ? source.choices.map(String) : [];
  return (
    <div className="source-summary mcq">
      <span><b>Q</b>{String(source.question_no ?? "")}</span>
      <span><b>Target</b>{String(source.target ?? "")}</span>
      <span><b>Answer</b>{String(source.correct_choice_no ?? "?")}. {String(source.correct_answer ?? "")}</span>
      {choices.length ? <p>{choices.map((choice, index) => `${index + 1}. ${choice}`).join("  ·  ")}</p> : null}
    </div>
  );
}

function WarningList({ warnings, compact = false }: { warnings: string[]; compact?: boolean }) {
  return (
    <div className={compact ? "warnings compact" : "warnings"}>
      {warnings.map((warning) => (
        <p key={warning}>{warning}</p>
      ))}
    </div>
  );
}

function DocumentParsePanel({ result }: { result: DocumentParseResult }) {
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

function OcrComparisonPanel({ comparison }: { comparison: OcrComparison }) {
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

async function pageCounts(pages: Page[]): Promise<Record<string, number>> {
  const entries = await Promise.all(
    pages.map(async (page) => {
      const pageCards = await apiGet<CardCandidate[]>(`/api/pages/${page.id}/cards`).catch(() => []);
      return [page.id, pageCards.length] as const;
    })
  );
  return Object.fromEntries(entries);
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

function focusedTokenIds(card: CardCandidate | null): Set<string> {
  const tokens = card?.source.evidence_tokens;
  if (!Array.isArray(tokens)) return new Set();
  return new Set(tokens.filter((token): token is string => typeof token === "string"));
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
    return `${String(card.source.surface ?? "Vocab")} · ${String(card.source.reading ?? "")}`;
  }
  return `Question ${String(card.source.question_no ?? "?")} · ${String(card.source.target ?? "target")}`;
}

function candidateSubtitle(card: CardCandidate): string {
  return `${card.note_type} · ${Math.round(card.confidence * 100)}% confidence`;
}

function provenanceLabel(value: string): string {
  if (value === "answer_strip") return "answer strip";
  if (value === "local_glossary") return "review carefully";
  return value.replaceAll("_", " ");
}
