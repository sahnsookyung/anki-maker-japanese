import { afterEach, describe, expect, it, vi } from "vitest";

import {
  apiErrorMessage,
  apiGet,
  activateOcrRun,
  approveCard,
  compareOcr,
  dedupePages,
  deletePage,
  exportCsv,
  getOcrProfiles,
  getOcrRuntime,
  imageUrl,
  listOcrRuns,
  parseDocument,
  preferredExperimentalOcrSelection,
  previewFieldOcr,
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

  it("falls back to an HTTP status message when error detail text is empty", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(new Response("", { status: 500 })));

    await expect(apiGet("/api/pages/broken")).rejects.toThrow("Backend request failed with HTTP 500.");
  });

  it("falls back to HTTP status when reading error text fails", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue({
        ok: false,
        status: 502,
        text: vi.fn().mockRejectedValue(new Error("stream failed"))
      })
    );

    await expect(apiGet("/api/pages/broken-stream")).rejects.toThrow("Backend request failed with HTTP 502.");
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

describe("preferredExperimentalOcrSelection", () => {
  it("defaults experiments to the strongest adapted local hypothesis when available", () => {
    expect(
      preferredExperimentalOcrSelection({
        profiles: [
          {
            id: "jp_v3_mobile_current",
            label: "Current",
            budget: "safe_local",
            provider: "paddle",
            creates_candidates: true,
            description: "Control"
          },
          {
            id: "jp_v5_det_v5_rec",
            label: "v5 det + v5 rec",
            budget: "safe_local",
            provider: "paddle",
            creates_candidates: true,
            description: "Latest local pair"
          }
        ],
        korean_profiles: [
          {
            id: "ko_v5_current",
            label: "Korean v5",
            budget: "safe_local",
            provider: "paddle_korean",
            creates_candidates: false,
            description: "Default Korean pass"
          }
        ],
        variants: [
          { id: "baseline_current", label: "Baseline", description: "Control" },
          { id: "v5_full_adapted_v1", label: "v5 full adapted", description: "Adapted hypothesis" }
        ],
        default_profile: "jp_v3_mobile_current",
        default_korean_profile: "ko_v5_current",
        default_variant: "baseline_current"
      })
    ).toEqual({
      modelProfile: "jp_v5_det_v5_rec",
      koreanProfile: "ko_v5_current",
      extractionVariant: "v5_full_adapted_v1"
    });
  });

  it("falls back to default safe controls when adapted options are absent", () => {
    expect(
      preferredExperimentalOcrSelection({
        profiles: [
          {
            id: "jp_v3_mobile_current",
            label: "Current",
            budget: "safe_local",
            provider: "paddle",
            creates_candidates: true,
            description: "Control"
          }
        ],
        variants: [{ id: "baseline_current", label: "Baseline", description: "Control" }],
        default_profile: "jp_v3_mobile_current",
        default_variant: "baseline_current"
      })
    ).toEqual({
      modelProfile: "jp_v3_mobile_current",
      koreanProfile: "ko_v5_current",
      extractionVariant: "baseline_current"
    });
  });

  it("uses an available Korean profile when the default Korean id is absent", () => {
    expect(
      preferredExperimentalOcrSelection({
        profiles: [
          {
            id: "jp_v3_mobile_current",
            label: "Current",
            budget: "safe_local",
            provider: "paddle",
            creates_candidates: true,
            description: "Control"
          }
        ],
        korean_profiles: [
          {
            id: "ko_v5_det_v5_rec",
            label: "Korean v5 pair",
            budget: "safe_local",
            provider: "paddle_korean",
            creates_candidates: false,
            description: "Alias"
          }
        ],
        variants: [{ id: "baseline_current", label: "Baseline", description: "Control" }],
        default_profile: "jp_v3_mobile_current",
        default_korean_profile: "missing_default",
        default_variant: "baseline_current"
      })
    ).toMatchObject({
      koreanProfile: "ko_v5_det_v5_rec"
    });
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
      .mockResolvedValueOnce(new Response(JSON.stringify({ page_type: "reading_mcq" })))
      .mockResolvedValueOnce(new Response(JSON.stringify({ id: "page-1", display_name: "Renamed" })))
      .mockResolvedValueOnce(new Response(JSON.stringify({ page_id: "page-1", status: "deleted" })))
      .mockResolvedValueOnce(new Response(JSON.stringify({ removed_count: 2, removed: [] })))
      .mockResolvedValueOnce(new Response(JSON.stringify({ ...card, status: "approved" })))
      .mockResolvedValueOnce(new Response(JSON.stringify({ ...card, front: "updated" })))
      .mockResolvedValueOnce(new Response(JSON.stringify({
        export_id: "export-1",
        files: [{ kind: "vocab", filename: "export_vocab.csv", path: "/tmp/export_vocab.csv", download_url: "/api/exports/export_vocab.csv", row_count: 1 }],
        note_count: 1,
        estimated_generated_card_count: 1
      })))
      .mockResolvedValueOnce(new Response(JSON.stringify({ agreement: 0.75 })))
      .mockResolvedValueOnce(new Response(JSON.stringify({ block_count: 2 })))
      .mockResolvedValueOnce(new Response(JSON.stringify({ text: "うえ", field: "target" })))
      .mockResolvedValueOnce(new Response(JSON.stringify({ state: "running", jobs_handled: 1 })))
      .mockResolvedValueOnce(new Response(JSON.stringify({
        profiles: [{ id: "jp_v5_mobile_general" }],
        variants: [{ id: "baseline_current", label: "Baseline", description: "Control" }],
        default_profile: "jp_v3_mobile_current",
        default_variant: "baseline_current"
      })))
      .mockResolvedValueOnce(new Response(JSON.stringify([{ id: "run-1", engine: "paddleocr" }])))
      .mockResolvedValueOnce(new Response(JSON.stringify({ id: "run-1", status: "succeeded" })));
    vi.stubGlobal("fetch", fetchMock);

    await expect(processPage("page-1")).resolves.toEqual({ page_type: "uploaded" });
    await expect(processPage("page-1", "paddleocr_vl")).resolves.toEqual({ page_type: "reading_mcq" });
    await expect(updatePage("page-1", "Renamed")).resolves.toEqual({ id: "page-1", display_name: "Renamed" });
    await expect(deletePage("page-1")).resolves.toEqual({ page_id: "page-1", status: "deleted" });
    await expect(dedupePages()).resolves.toEqual({ removed_count: 2, removed: [] });
    await expect(approveCard("card-1")).resolves.toMatchObject({ status: "approved" });
    await expect(updateCard(card)).resolves.toMatchObject({ front: "updated" });
    await expect(exportCsv(["page-1"], { approved_only: false, include_yellow: false, include_red: true })).resolves.toEqual({
      export_id: "export-1",
      files: [{ kind: "vocab", filename: "export_vocab.csv", path: "/tmp/export_vocab.csv", download_url: "/api/exports/export_vocab.csv", row_count: 1 }],
      note_count: 1,
      estimated_generated_card_count: 1
    });
    await expect(compareOcr("page-1", "google vision")).resolves.toEqual({ agreement: 0.75 });
    await expect(parseDocument("page-1")).resolves.toEqual({ block_count: 2 });
    await expect(previewFieldOcr("card-1", "target", [1, 2, 3, 4])).resolves.toEqual({ text: "うえ", field: "target" });
    await expect(getOcrRuntime()).resolves.toEqual({ state: "running", jobs_handled: 1 });
    await expect(getOcrProfiles()).resolves.toEqual({
      profiles: [{ id: "jp_v5_mobile_general" }],
      variants: [{ id: "baseline_current", label: "Baseline", description: "Control" }],
      default_profile: "jp_v3_mobile_current",
      default_variant: "baseline_current"
    });
    await expect(listOcrRuns("page-1")).resolves.toEqual([{ id: "run-1", engine: "paddleocr" }]);
    await expect(activateOcrRun("page-1", "run-1")).resolves.toEqual({ id: "run-1", status: "succeeded" });

    expect(fetchMock.mock.calls.map(([url]) => url)).toEqual([
      "http://127.0.0.1:8000/api/pages/page-1/process",
      "http://127.0.0.1:8000/api/pages/page-1/process?engine=paddleocr_vl",
      "http://127.0.0.1:8000/api/pages/page-1",
      "http://127.0.0.1:8000/api/pages/page-1",
      "http://127.0.0.1:8000/api/pages/dedupe",
      "http://127.0.0.1:8000/api/cards/card-1/approve",
      "http://127.0.0.1:8000/api/cards/card-1",
      "http://127.0.0.1:8000/api/exports/csv",
      "http://127.0.0.1:8000/api/pages/page-1/ocr/compare?provider=google%20vision",
      "http://127.0.0.1:8000/api/pages/page-1/document/parse",
      "http://127.0.0.1:8000/api/cards/card-1/field-ocr/preview",
      "http://127.0.0.1:8000/api/ocr/runtime",
      "http://127.0.0.1:8000/api/ocr/profiles",
      "http://127.0.0.1:8000/api/pages/page-1/ocr/runs",
      "http://127.0.0.1:8000/api/pages/page-1/ocr/runs/run-1/activate"
    ]);
    expect(fetchMock.mock.calls[3]?.[1]).toMatchObject({ method: "DELETE" });
    expect(JSON.parse(String(fetchMock.mock.calls[6]?.[1]?.body))).toMatchObject({
      confidence: 0.9,
      source_bbox: [0, 0, 10, 10],
      warnings: []
    });
    expect(JSON.parse(String(fetchMock.mock.calls[10]?.[1]?.body))).toEqual({ field: "target", bbox: [1, 2, 3, 4] });
  });

  it("retries page processing while the OCR runtime is busy", async () => {
    const fetchMock = vi
      .fn()
      .mockResolvedValueOnce(new Response("Another OCR job is already running.", { status: 409 }))
      .mockResolvedValueOnce(new Response(JSON.stringify({ page_type: "reading_mcq" })));
    vi.stubGlobal("fetch", fetchMock);

    await expect(processPage("page-queued", "paddleocr", { busyRetryAttempts: 1, busyRetryDelayMs: 0 })).resolves.toEqual({
      page_type: "reading_mcq"
    });
    expect(fetchMock).toHaveBeenCalledTimes(2);
    expect(fetchMock.mock.calls.map(([url]) => url)).toEqual([
      "http://127.0.0.1:8000/api/pages/page-queued/process",
      "http://127.0.0.1:8000/api/pages/page-queued/process"
    ]);
  });

  it("passes experimental OCR profile and extraction variant query params", async () => {
    const fetchMock = vi.fn().mockResolvedValue(new Response(JSON.stringify({ page_type: "vocab_table" })));
    vi.stubGlobal("fetch", fetchMock);

    await expect(
      processPage("page-exp", "paddleocr", {
        modelProfile: "jp_v5_mobile_general",
        koreanProfile: "ko_v5_current",
        extractionVariant: "table_graph_v1"
      })
    ).resolves.toEqual({ page_type: "vocab_table" });

    expect(fetchMock).toHaveBeenCalledWith(
      "http://127.0.0.1:8000/api/pages/page-exp/process?model_profile=jp_v5_mobile_general&korean_profile=ko_v5_current&extraction_variant=table_graph_v1",
      { method: "POST" }
    );
  });

  it("builds image URLs only for known backend image directories", () => {
    expect(imageUrl("/tmp/backend/processed/page.png")).toBe("http://127.0.0.1:8000/files/processed/page.png");
    expect(imageUrl("/tmp/backend/uploads/page.jpg")).toBe("http://127.0.0.1:8000/files/uploads/page.jpg");
    expect(imageUrl("/tmp/backend/exports/page.tsv")).toBeNull();
    expect(imageUrl("/tmp/backend/exports/page.csv")).toBeNull();
    expect(imageUrl("/tmp/backend/uploads/")).toBeNull();
    expect(imageUrl(null)).toBeNull();
  });
});

function candidate(): CardCandidate {
  return {
    id: "card-1",
    page_id: "page-1",
    source_type: "vocab_item",
    source_id: "source-1",
    source: { surface: "学校", reading: "がっこう", meaning_ko: "학교" },
    note_type: "jp_vocab_entry",
    front: "学校",
    back: "がっこう",
    tags: ["jlpt"],
    confidence: 0.9,
    status: "pending_review",
    review_state: "yellow",
    source_bbox: [0, 0, 10, 10],
    warnings: []
  };
}
