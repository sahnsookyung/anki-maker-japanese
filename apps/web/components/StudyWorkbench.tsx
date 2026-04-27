"use client";

import { useEffect, useState, useTransition } from "react";
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
  uploadImage
} from "../lib/api";

export function StudyWorkbench() {
  const [pages, setPages] = useState<Page[]>([]);
  const [selectedPage, setSelectedPage] = useState<Page | null>(null);
  const [tokens, setTokens] = useState<OcrToken[]>([]);
  const [cards, setCards] = useState<CardCandidate[]>([]);
  const [comparison, setComparison] = useState<OcrComparison | null>(null);
  const [documentParse, setDocumentParse] = useState<DocumentParseResult | null>(null);
  const [message, setMessage] = useState("Upload a study-book photo to begin.");
  const [isPending, startTransition] = useTransition();

  useEffect(() => {
    void refreshPages();
  }, []);

  async function refreshPages() {
    const nextPages = await apiGet<Page[]>("/api/pages");
    setPages(nextPages);
    if (!selectedPage && nextPages.length) {
      await selectPage(nextPages[0]);
    }
  }

  async function selectPage(page: Page) {
    setSelectedPage(page);
    const [ocr, pageCards] = await Promise.all([
      apiGet<{ page: Page; tokens: OcrToken[] }>(`/api/pages/${page.id}/ocr`).catch(() => ({ page, tokens: [] })),
      apiGet<CardCandidate[]>(`/api/pages/${page.id}/cards`).catch(() => [])
    ]);
    setSelectedPage(ocr.page);
    setTokens(ocr.tokens);
    setCards(pageCards);
    setComparison(null);
    setDocumentParse(null);
  }

  async function onUpload(file: File | undefined) {
    if (!file) return;
    setMessage("Uploading image...");
    const result = await uploadImage(file);
    const page = (await apiGet<Page[]>("/api/pages")).find((candidate) => candidate.id === result.page_id);
    if (page) {
      setMessage("Image uploaded. Run processing to create candidates.");
      await selectPage(page);
      await refreshPages();
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
      setComparison(null);
      setDocumentParse(null);
      setMessage(`Processed as ${result.page.page_type}. Generated ${result.cards.length} card candidates.`);
      await refreshPages();
      } catch (error) {
        setMessage(error instanceof Error ? error.message : "Processing failed.");
      }
    });
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
    const result = await exportTsv([selectedPage.id]);
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

  const processedUrl = imageUrl(selectedPage?.processed_image_path);
  const originalUrl = imageUrl(selectedPage?.original_image_path);
  const visibleUrl = processedUrl ?? originalUrl;

  return (
    <main className="shell">
      <section className="hero">
        <div>
          <p className="eyebrow">Local-first Anki candidate generator</p>
          <h1>Japanese workbook photos, turned into reviewable cards.</h1>
          <p className="lede">
            Upload, OCR, inspect evidence, approve cards, then export TSV. The app stays deliberately skeptical so tiny
            Japanese reading mistakes do not sneak into Anki.
          </p>
        </div>
        <label className="upload">
          <input type="file" accept="image/*" onChange={(event) => void onUpload(event.target.files?.[0])} />
          Choose image
        </label>
      </section>

      <section className="status">
        <span>{isPending ? "Working..." : message}</span>
        {selectedPage ? <button onClick={onProcess}>Process selected page</button> : null}
        {tokens.length ? <button onClick={onCompareOcr}>Compare GCV OCR</button> : null}
        {selectedPage ? <button onClick={() => void onParseDocument()}>Parse PaddleOCR-VL</button> : null}
        {cards.length ? <button onClick={onExport}>Export approved TSV</button> : null}
      </section>

      <div className="grid">
        <aside className="panel page-list">
          <h2>Pages</h2>
          {pages.length === 0 ? <p>No uploads yet.</p> : null}
          {pages.map((page) => (
            <button
              className={page.id === selectedPage?.id ? "page active" : "page"}
              key={page.id}
              onClick={() => void selectPage(page)}
            >
              <span>{page.page_type}</span>
              <small>{page.id}</small>
            </button>
          ))}
        </aside>

        <section className="panel image-panel">
          <div className="panel-heading">
            <h2>Evidence</h2>
            <span>{tokens.length} OCR tokens</span>
          </div>
          {visibleUrl ? (
            <div className="image-wrap">
              <img src={visibleUrl} alt="Uploaded study page" />
              <TokenOverlay page={selectedPage} tokens={tokens} />
            </div>
          ) : (
            <div className="empty">Upload a sample image to see OCR boxes here.</div>
          )}
          {selectedPage?.warnings.length ? (
            <div className="warnings">
              {selectedPage.warnings.map((warning) => (
                <p key={warning}>{warning}</p>
              ))}
            </div>
          ) : null}
          {comparison ? <OcrComparisonPanel comparison={comparison} /> : null}
          {documentParse ? <DocumentParsePanel result={documentParse} /> : null}
        </section>

        <section className="panel cards">
          <div className="panel-heading">
            <h2>Card Candidates</h2>
            <span>{cards.length}</span>
          </div>
          {cards.length === 0 ? <div className="empty">Process the selected page to generate editable candidates.</div> : null}
          {cards.map((card) => (
            <CardEditor key={card.id} card={card} onChange={saveCard} onApprove={approve} />
          ))}
        </section>
      </div>
    </main>
  );
}

function DocumentParsePanel({ result }: { result: DocumentParseResult }) {
  return (
    <div className="comparison">
      <h3>
        {result.provider}: {result.block_count} blocks via {result.backend}
      </h3>
      {result.warnings.length ? (
        <div className="warnings compact">
          {result.warnings.map((warning) => (
            <p key={warning}>{warning}</p>
          ))}
        </div>
      ) : null}
      {result.markdown_text ? (
        <details open>
          <summary>Markdown</summary>
          <pre>{result.markdown_text}</pre>
        </details>
      ) : null}
      {result.blocks.length ? (
        <details>
          <summary>Blocks</summary>
          <div className="block-list">
            {result.blocks.map((block, index) => (
              <article key={`${block.label}-${index}`} className="parse-block">
                <strong>{block.order ?? index + 1}. {block.label}</strong>
                <p>{block.content || "(empty)"}</p>
              </article>
            ))}
          </div>
        </details>
      ) : null}
    </div>
  );
}

function OcrComparisonPanel({ comparison }: { comparison: OcrComparison }) {
  return (
    <div className="comparison">
      <h3>
        OCR comparison: {comparison.primary_provider} vs {comparison.compare_provider}
      </h3>
      <p>
        {comparison.primary_token_count} local tokens, {comparison.compare_token_count} comparison tokens,{" "}
        {Math.round(comparison.agreement * 100)}% overlap.
      </p>
      {comparison.warnings.length ? (
        <div className="warnings compact">
          {comparison.warnings.map((warning) => (
            <p key={warning}>{warning}</p>
          ))}
        </div>
      ) : null}
      <details>
        <summary>Text only in Google Cloud Vision</summary>
        <p>{comparison.missing_from_primary.join(" ") || "None"}</p>
      </details>
      <details>
        <summary>Text only in local OCR</summary>
        <p>{comparison.missing_from_comparison.join(" ") || "None"}</p>
      </details>
    </div>
  );
}

function TokenOverlay({ page, tokens }: { page: Page | null; tokens: OcrToken[] }) {
  if (!page?.image_width || !page.image_height) return null;
  return (
    <svg viewBox={`0 0 ${page.image_width} ${page.image_height}`} className="overlay">
      {tokens.map((token) => {
        const [x1, y1, x2, y2] = token.bbox;
        return (
          <g key={token.id}>
            <rect x={x1} y={y1} width={x2 - x1} height={y2 - y1} className={`box ${token.script_class}`} />
            <title>{`${token.text} (${token.script_class}, ${Math.round(token.confidence * 100)}%)`}</title>
          </g>
        );
      })}
    </svg>
  );
}

function CardEditor({
  card,
  onChange,
  onApprove
}: {
  card: CardCandidate;
  onChange: (card: CardCandidate) => Promise<void>;
  onApprove: (cardId: string) => Promise<void>;
}) {
  const [draft, setDraft] = useState(card);
  useEffect(() => setDraft(card), [card]);

  return (
    <article className={`card ${draft.review_state}`}>
      <div className="card-top">
        <strong>{draft.note_type}</strong>
        <span>{draft.review_state}</span>
      </div>
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
      {draft.warnings.length ? (
        <div className="warnings compact">
          {draft.warnings.map((warning) => (
            <p key={warning}>{warning}</p>
          ))}
        </div>
      ) : null}
      <div className="actions">
        <button onClick={() => void onChange(draft)}>Save edits</button>
        <button onClick={() => void onApprove(draft.id)} disabled={draft.review_state === "red"}>
          Approve
        </button>
      </div>
    </article>
  );
}
