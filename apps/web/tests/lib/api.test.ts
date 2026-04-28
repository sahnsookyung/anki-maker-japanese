import { afterEach, describe, expect, it, vi } from "vitest";

import {
  apiErrorMessage,
  apiGet,
  approveCard,
  compareOcr,
  deletePage,
  exportTsv,
  imageUrl,
  parseDocument,
  processPage,
  resolveApiBase,
  updateCard,
  updatePage,
  uploadImages,
  type CardCandidate
} from "../../lib/api";

afterEach(() => {
  vi.unstubAllGlobals();
});

describe("apiGet", () => {
  it("returns parsed JSON on a successful response", async () => {
    const fetchMock = vi.fn().mockResolvedValue(new Response(JSON.stringify([{ id: "page_1" }])));
    vi.stubGlobal("fetch", fetchMock);

    await expect(apiGet("/api/pages")).resolves.toEqual([{ id: "page_1" }]);
    expect(fetchMock).toHaveBeenCalledWith("http://127.0.0.1:8000/api/pages", { cache: "no-store" });
  });

  it("turns network failures into an actionable backend message", async () => {
    vi.stubGlobal("fetch", vi.fn().mockRejectedValue(new TypeError("fetch failed")));

    await expect(apiGet("/api/pages")).rejects.toThrow(
      "Could not reach the backend at http://127.0.0.1:8000. Start FastAPI and try again."
    );
  });

  it("uses backend response text for non-2xx failures", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(new Response("Page not found.", { status: 404 })));

    await expect(apiGet("/api/pages/missing")).rejects.toThrow("Page not found.");
  });
});

describe("resolveApiBase", () => {
  it("matches localhost API host to the current 127.0.0.1 dev origin", () => {
    vi.stubGlobal("window", { location: { hostname: "127.0.0.1" } });

    expect(resolveApiBase("http://localhost:8000")).toBe("http://127.0.0.1:8000");
  });

  it("matches 127.0.0.1 API host to the current localhost dev origin", () => {
    vi.stubGlobal("window", { location: { hostname: "localhost" } });

    expect(resolveApiBase("http://127.0.0.1:8000")).toBe("http://localhost:8000");
  });

  it("trims trailing slashes from non-URL fallback values", () => {
    expect(resolveApiBase("api/backend/")).toBe("api/backend");
  });
});

describe("apiErrorMessage", () => {
  it("keeps Error messages and falls back for unknown thrown values", () => {
    expect(apiErrorMessage(new Error("Specific failure"), "Fallback")).toBe("Specific failure");
    expect(apiErrorMessage("not an error", "Fallback")).toBe("Fallback");
  });
});

describe("uploadImages", () => {
  it("uploads multiple files sequentially", async () => {
    const fetchMock = vi
      .fn()
      .mockResolvedValueOnce(new Response(JSON.stringify({ page_id: "page_1", status: "uploaded" })))
      .mockResolvedValueOnce(new Response(JSON.stringify({ page_id: "page_2", status: "uploaded" })));
    vi.stubGlobal("fetch", fetchMock);

    const files = [new File(["a"], "page-a.jpg"), new File(["b"], "page-b.jpg")];

    await expect(uploadImages(files)).resolves.toEqual({
      uploaded: [
        { fileName: "page-a.jpg", pageId: "page_1" },
        { fileName: "page-b.jpg", pageId: "page_2" }
      ],
      failed: []
    });
    expect(fetchMock).toHaveBeenCalledTimes(2);
    expect(fetchMock.mock.invocationCallOrder[0]).toBeLessThan(fetchMock.mock.invocationCallOrder[1]);
  });

  it("continues after one file fails", async () => {
    const fetchMock = vi
      .fn()
      .mockResolvedValueOnce(new Response(JSON.stringify({ page_id: "page_1", status: "uploaded" })))
      .mockResolvedValueOnce(new Response("Unsupported image type.", { status: 400 }))
      .mockResolvedValueOnce(new Response(JSON.stringify({ page_id: "page_3", status: "uploaded" })));
    vi.stubGlobal("fetch", fetchMock);

    const files = [new File(["a"], "page-a.jpg"), new File(["b"], "notes.txt"), new File(["c"], "page-c.jpg")];

    await expect(uploadImages(files)).resolves.toEqual({
      uploaded: [
        { fileName: "page-a.jpg", pageId: "page_1" },
        { fileName: "page-c.jpg", pageId: "page_3" }
      ],
      failed: [{ fileName: "notes.txt", message: "Unsupported image type." }]
    });
    expect(fetchMock).toHaveBeenCalledTimes(3);
  });

  it("returns the backend connectivity message for upload network failures", async () => {
    vi.stubGlobal("fetch", vi.fn().mockRejectedValue(new TypeError("connection refused")));

    const files = [new File(["a"], "page-a.jpg")];

    await expect(uploadImages(files)).resolves.toEqual({
      uploaded: [],
      failed: [
        {
          fileName: "page-a.jpg",
          message: "Could not reach the backend at http://127.0.0.1:8000. Start FastAPI and try again."
        }
      ]
    });
  });
});

describe("API helpers", () => {
  it("calls page and card mutation endpoints", async () => {
    const card = candidate();
    const fetchMock = vi
      .fn()
      .mockResolvedValueOnce(new Response(JSON.stringify({ page_type: "uploaded" })))
      .mockResolvedValueOnce(new Response(JSON.stringify({ id: "page-1", display_name: "Renamed" })))
      .mockResolvedValueOnce(new Response(JSON.stringify({ page_id: "page-1", status: "deleted" })))
      .mockResolvedValueOnce(new Response(JSON.stringify({ ...card, status: "approved" })))
      .mockResolvedValueOnce(new Response(JSON.stringify({ ...card, front: "updated" })))
      .mockResolvedValueOnce(new Response(JSON.stringify({ card_count: 1, download_url: "/api/exports/export.tsv" })))
      .mockResolvedValueOnce(new Response(JSON.stringify({ agreement: 0.75 })))
      .mockResolvedValueOnce(new Response(JSON.stringify({ block_count: 2 })));
    vi.stubGlobal("fetch", fetchMock);

    await expect(processPage("page-1")).resolves.toEqual({ page_type: "uploaded" });
    await expect(updatePage("page-1", "Renamed")).resolves.toEqual({ id: "page-1", display_name: "Renamed" });
    await expect(deletePage("page-1")).resolves.toEqual({ page_id: "page-1", status: "deleted" });
    await expect(approveCard("card-1")).resolves.toMatchObject({ status: "approved" });
    await expect(updateCard(card)).resolves.toMatchObject({ front: "updated" });
    await expect(exportTsv(["page-1"], { approved_only: false, include_yellow: false, include_red: true })).resolves.toEqual({
      card_count: 1,
      download_url: "/api/exports/export.tsv"
    });
    await expect(compareOcr("page-1", "google vision")).resolves.toEqual({ agreement: 0.75 });
    await expect(parseDocument("page-1")).resolves.toEqual({ block_count: 2 });

    expect(fetchMock.mock.calls.map(([url]) => url)).toEqual([
      "http://127.0.0.1:8000/api/pages/page-1/process",
      "http://127.0.0.1:8000/api/pages/page-1",
      "http://127.0.0.1:8000/api/pages/page-1",
      "http://127.0.0.1:8000/api/cards/card-1/approve",
      "http://127.0.0.1:8000/api/cards/card-1",
      "http://127.0.0.1:8000/api/exports/tsv",
      "http://127.0.0.1:8000/api/pages/page-1/ocr/compare?provider=google%20vision",
      "http://127.0.0.1:8000/api/pages/page-1/document/parse"
    ]);
    expect(fetchMock.mock.calls[2]?.[1]).toMatchObject({ method: "DELETE" });
  });

  it("builds image URLs only for known backend image directories", () => {
    expect(imageUrl("/tmp/backend/processed/page.png")).toBe("http://127.0.0.1:8000/files/processed/page.png");
    expect(imageUrl("/tmp/backend/uploads/page.jpg")).toBe("http://127.0.0.1:8000/files/uploads/page.jpg");
    expect(imageUrl("/tmp/backend/exports/page.tsv")).toBeNull();
    expect(imageUrl(null)).toBeNull();
  });
});

function candidate(): CardCandidate {
  return {
    id: "card-1",
    page_id: "page-1",
    source_type: "vocab_item",
    source_id: "source-1",
    source: { surface: "学校" },
    note_type: "jp_vocab_reading",
    front: "学校",
    back: "がっこう",
    tags: ["jlpt"],
    confidence: 0.9,
    status: "pending_review",
    review_state: "yellow",
    warnings: []
  };
}
