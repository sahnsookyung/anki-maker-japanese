export const API_BASE = resolveApiBase(process.env.NEXT_PUBLIC_API_BASE_URL);

export function resolveApiBase(configuredBase?: string): string {
  const browserHost = typeof globalThis.window === "object" ? globalThis.window.location.hostname : "";
  const localBrowserHost = browserHost === "localhost" || browserHost === "127.0.0.1";
  const fallbackHost = localBrowserHost ? browserHost : "127.0.0.1";
  const fallbackBase = `http://${fallbackHost}:8000`;
  if (!configuredBase) return fallbackBase;

  try {
    const url = new URL(configuredBase);
    const localApiHost = url.hostname === "localhost" || url.hostname === "127.0.0.1";
    if (localBrowserHost && localApiHost) {
      url.hostname = browserHost;
    }
    return url.toString().replace(/\/$/, "");
  } catch {
    return configuredBase.replace(/\/$/, "");
  }
}

export type Page = {
  id: string;
  original_image_path: string;
  display_name?: string | null;
  processed_image_path?: string | null;
  page_type: string;
  page_type_confidence: number;
  image_width?: number | null;
  image_height?: number | null;
  warnings: string[];
  created_at: string;
  card_count?: number;
  approved_card_count?: number;
  red_card_count?: number;
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

export function apiErrorMessage(error: unknown, fallback = "Request failed."): string {
  return error instanceof Error ? error.message : fallback;
}

async function requestJson<T>(path: string, init?: RequestInit): Promise<T> {
  let response: Response;
  try {
    response = await fetch(`${API_BASE}${path}`, init);
  } catch {
    throw new Error(`Could not reach the backend at ${API_BASE}. Start FastAPI and try again.`);
  }
  if (!response.ok) {
    const detail = await response.text().catch(() => "");
    throw new Error(detail || `Backend request failed with HTTP ${response.status}.`);
  }
  return response.json() as Promise<T>;
}

export async function apiGet<T>(path: string): Promise<T> {
  return requestJson<T>(path, { cache: "no-store" });
}

export async function uploadImage(file: File): Promise<{ page_id: string; status: string }> {
  const form = new FormData();
  form.append("file", file);
  return requestJson<{ page_id: string; status: string }>("/api/pages/upload", { method: "POST", body: form });
}

export type BatchUploadResult = {
  uploaded: Array<{ fileName: string; pageId: string }>;
  failed: Array<{ fileName: string; message: string }>;
};

export async function uploadImages(files: File[]): Promise<BatchUploadResult> {
  const uploaded: BatchUploadResult["uploaded"] = [];
  const failed: BatchUploadResult["failed"] = [];
  for (const file of files) {
    try {
      const result = await uploadImage(file);
      uploaded.push({ fileName: file.name, pageId: result.page_id });
    } catch (error) {
      failed.push({ fileName: file.name, message: apiErrorMessage(error, "Upload failed.") });
    }
  }
  return { uploaded, failed };
}

export async function processPage(pageId: string): Promise<ProcessResult> {
  return requestJson<ProcessResult>(`/api/pages/${pageId}/process`, { method: "POST" });
}

export async function updatePage(pageId: string, displayName: string): Promise<Page> {
  return requestJson<Page>(`/api/pages/${pageId}`, {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ display_name: displayName })
  });
}

export async function deletePage(pageId: string): Promise<{ page_id: string; status: string }> {
  return requestJson<{ page_id: string; status: string }>(`/api/pages/${pageId}`, { method: "DELETE" });
}

export async function approveCard(cardId: string): Promise<CardCandidate> {
  return requestJson<CardCandidate>(`/api/cards/${cardId}/approve`, { method: "POST" });
}

export async function updateCard(card: CardCandidate): Promise<CardCandidate> {
  return requestJson<CardCandidate>(`/api/cards/${card.id}`, {
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
}

export type ExportOptions = {
  approved_only?: boolean;
  include_yellow?: boolean;
  include_red?: boolean;
};

export async function exportTsv(
  pageIds: string[],
  options: ExportOptions = {}
): Promise<{ card_count: number; download_url: string }> {
  return requestJson<{ card_count: number; download_url: string }>("/api/exports/tsv", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      page_ids: pageIds,
      approved_only: options.approved_only ?? true,
      include_yellow: options.include_yellow ?? true,
      include_red: options.include_red ?? false
    })
  });
}

export async function compareOcr(pageId: string, provider = "google_vision"): Promise<OcrComparison> {
  return apiGet<OcrComparison>(`/api/pages/${pageId}/ocr/compare?provider=${encodeURIComponent(provider)}`);
}

export async function parseDocument(pageId: string): Promise<DocumentParseResult> {
  return requestJson<DocumentParseResult>(`/api/pages/${pageId}/document/parse`, { method: "POST" });
}

export function imageUrl(path?: string | null): string | null {
  if (!path) return null;
  const filename = path.split("/").pop();
  if (!filename) return null;
  if (path.includes("/processed/")) return `${API_BASE}/files/processed/${filename}`;
  if (path.includes("/uploads/")) return `${API_BASE}/files/uploads/${filename}`;
  return null;
}
