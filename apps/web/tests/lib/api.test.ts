import { afterEach, describe, expect, it, vi } from "vitest";

import { apiErrorMessage, apiGet, resolveApiBase, uploadImages } from "../../lib/api";

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
});
