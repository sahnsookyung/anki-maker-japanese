export const API_BASE = process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8000";

export type Page = {
  id: string;
  original_image_path: string;
  processed_image_path?: string | null;
  page_type: string;
  page_type_confidence: number;
  image_width?: number | null;
  image_height?: number | null;
  warnings: string[];
  created_at: string;
};

export type OcrToken = {
  id: string;
  page_id: string;
  text: string;
  bbox: number[];
  confidence: number;
  script_class: string;
  source: string;
};

export type CardCandidate = {
  id: string;
  page_id: string;
  source_type: string;
  source_id: string;
  source: Record<string, unknown>;
  note_type: string;
  front: string;
  back: string;
  tags: string[];
  confidence: number;
  status: "pending_review" | "approved" | "skipped";
  review_state: "green" | "yellow" | "red";
  source_bbox?: number[] | null;
  warnings: string[];
};

export type ProcessResult = {
  page: Page;
  tokens: OcrToken[];
  cards: CardCandidate[];
  script_summary: Record<string, number>;
  answer_map: Record<string, number>;
};

export type OcrComparison = {
  primary_provider: string;
  compare_provider: string;
  primary_token_count: number;
  compare_token_count: number;
  agreement: number;
  missing_from_primary: string[];
  missing_from_comparison: string[];
  compare_tokens: OcrToken[];
  warnings: string[];
};

export type DocumentParseBlock = {
  label: string;
  content: string;
  bbox?: number[] | null;
  order?: number | null;
};

export type DocumentParseResult = {
  page_id: string;
  provider: string;
  source_image_path: string;
  backend: string;
  block_count: number;
  blocks: DocumentParseBlock[];
  markdown_text: string;
  warnings: string[];
};

export async function apiGet<T>(path: string): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`, { cache: "no-store" });
  if (!response.ok) throw new Error(await response.text());
  return response.json() as Promise<T>;
}

export async function uploadImage(file: File): Promise<{ page_id: string; status: string }> {
  const form = new FormData();
  form.append("file", file);
  const response = await fetch(`${API_BASE}/api/pages/upload`, { method: "POST", body: form });
  if (!response.ok) throw new Error(await response.text());
  return response.json();
}

export async function processPage(pageId: string): Promise<ProcessResult> {
  const response = await fetch(`${API_BASE}/api/pages/${pageId}/process`, { method: "POST" });
  if (!response.ok) throw new Error(await response.text());
  return response.json();
}

export async function approveCard(cardId: string): Promise<CardCandidate> {
  const response = await fetch(`${API_BASE}/api/cards/${cardId}/approve`, { method: "POST" });
  if (!response.ok) throw new Error(await response.text());
  return response.json();
}

export async function updateCard(card: CardCandidate): Promise<CardCandidate> {
  const response = await fetch(`${API_BASE}/api/cards/${card.id}`, {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      front: card.front,
      back: card.back,
      tags: card.tags,
      status: card.status,
      review_state: card.review_state,
      source: card.source
    })
  });
  if (!response.ok) throw new Error(await response.text());
  return response.json();
}

export async function exportTsv(pageIds: string[]): Promise<{ card_count: number; download_url: string }> {
  const response = await fetch(`${API_BASE}/api/exports/tsv`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ page_ids: pageIds, include_yellow: true, include_red: false, approved_only: true })
  });
  if (!response.ok) throw new Error(await response.text());
  return response.json();
}

export async function compareOcr(pageId: string, provider = "google_vision"): Promise<OcrComparison> {
  return apiGet<OcrComparison>(`/api/pages/${pageId}/ocr/compare?provider=${encodeURIComponent(provider)}`);
}

export async function parseDocument(pageId: string): Promise<DocumentParseResult> {
  const response = await fetch(`${API_BASE}/api/pages/${pageId}/document/parse`, { method: "POST" });
  if (!response.ok) throw new Error(await response.text());
  return response.json();
}

export function imageUrl(path?: string | null): string | null {
  if (!path) return null;
  const filename = path.split("/").pop();
  if (!filename) return null;
  if (path.includes("/processed/")) return `${API_BASE}/files/processed/${filename}`;
  if (path.includes("/uploads/")) return `${API_BASE}/files/uploads/${filename}`;
  return null;
}
