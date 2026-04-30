import { describe, expect, it } from "vitest";

import { batchTimingSummary, durationMs, formatDuration, percentile } from "../../lib/batchTiming";

describe("batch timing helpers", () => {
  it("formats frontend processing durations compactly", () => {
    expect(formatDuration(250)).toBe("250ms");
    expect(formatDuration(1250)).toBe("1.3s");
    expect(formatDuration(12_500)).toBe("13s");
    expect(formatDuration(72_000)).toBe("1m 12s");
  });

  it("computes non-negative elapsed durations and percentile samples", () => {
    expect(durationMs(120.4, 150.9)).toBe(31);
    expect(durationMs(200, 150)).toBe(0);
    expect(percentile([400, 100, 300, 200], 0.25)).toBe(200);
    expect(percentile([400, 100, 300, 200], 0.75)).toBe(300);
    expect(percentile([], 0.75)).toBe(0);
  });

  it("summarizes total, average, min, max, and quartile processing times", () => {
    const message = batchTimingSummary(
      [
        { pageId: "page-1", pageTitle: "Page 1", ms: 1000, success: true },
        { pageId: "page-2", pageTitle: "Page 2", ms: 2000, success: true },
        { pageId: "page-3", pageTitle: "Page 3", ms: 3000, success: true },
        { pageId: "page-4", pageTitle: "Page 4", ms: 4000, success: true }
      ],
      4,
      [],
      "PaddleOCR"
    );

    expect(message).toBe("Processed 4/4 pages with PaddleOCR in frontend time: total 10s, avg 2.5s, min 1.0s, p25 2.0s, p75 3.0s, max 4.0s.");
  });

  it("includes failed pages in the batch timing summary", () => {
    const message = batchTimingSummary(
      [
        { pageId: "page-1", pageTitle: "Page 1", ms: 1000, success: true },
        { pageId: "page-2", pageTitle: "Page 2", ms: 500, success: false }
      ],
      2,
      ["Page 2 (OCR failed.)"],
      "PaddleOCR-VL"
    );

    expect(message).toContain("Processed 1/2 pages with PaddleOCR-VL");
    expect(message).toContain("total 1.5s");
    expect(message).toContain("Failed: Page 2 (OCR failed.).");
  });

  it("summarizes partial page processing during sequential batches", () => {
    const message = batchTimingSummary(
      [{ pageId: "page-1", pageTitle: "Page 1", ms: 1000, success: true }],
      4,
      [],
      "PaddleOCR-VL",
      1
    );

    expect(message).toContain("Processed 1/4 page with PaddleOCR-VL");
  });

  it("handles empty timing samples without dividing by missing data", () => {
    expect(percentile([100, 200], -1)).toBe(100);
    expect(percentile([100, 200], 2)).toBe(200);

    const message = batchTimingSummary([], 1, [], "PaddleOCR");

    expect(message).toBe(
      "Processed 1/1 page with PaddleOCR in frontend time: total 0ms, avg 0ms, min 0ms, p25 0ms, p75 0ms, max 0ms."
    );
  });
});
