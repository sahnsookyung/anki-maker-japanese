import { expect, test } from "@playwright/test";

const transparentPng = Buffer.from(
  "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMB/az4F7QAAAAASUVORK5CYII=",
  "base64"
);

test("reload keeps token-only persisted OCR evidence visible", async ({ page }) => {
  const persistedPage = {
    id: "page-vl",
    original_image_path: "/tmp/uploads/page-vl.jpg",
    upload_name: "page-vl.jpg",
    display_name: "Persisted VL page",
    processed_image_path: "/tmp/processed/page-vl.png",
    page_type: "unknown_review_required",
    page_type_confidence: 0.4,
    image_width: 100,
    image_height: 100,
    warnings: ["No card candidates were generated; inspect OCR overlay."],
    created_at: "2026-05-03T00:00:00+00:00",
    card_count: 0,
    approved_card_count: 0,
    red_card_count: 0
  };
  const tokens = [
    {
      id: "tok-vl-1",
      page_id: "page-vl",
      text: "その",
      bbox: [10, 10, 40, 20],
      confidence: 0.92,
      script_class: "hiragana",
      source: "paddleocr_vl"
    },
    {
      id: "tok-vl-2",
      page_id: "page-vl",
      text: "学校",
      bbox: [45, 30, 80, 42],
      confidence: 0.68,
      script_class: "kanji",
      source: "paddleocr_vl"
    },
    {
      id: "tok-base-retained",
      page_id: "page-vl",
      text: "retained geometry",
      bbox: [12, 60, 88, 75],
      confidence: 0.96,
      script_class: "latin",
      source: "paddleocr"
    }
  ];

  await page.route("**/api/pages", async (route) => {
    await route.fulfill({ contentType: "application/json", body: JSON.stringify([persistedPage]) });
  });
  await page.route("**/api/pages/page-vl/ocr", async (route) => {
    await route.fulfill({ contentType: "application/json", body: JSON.stringify({ page: persistedPage, tokens }) });
  });
  await page.route("**/api/pages/page-vl/cards", async (route) => {
    await route.fulfill({ contentType: "application/json", body: "[]" });
  });
  await page.route("**/files/processed/page-vl.png**", async (route) => {
    await route.fulfill({ contentType: "image/png", body: transparentPng });
  });

  await page.goto("/");

  await expect(page.getByText("3 OCR tokens are shown because no card candidates were generated.")).toBeVisible();
  await expect(page.getByRole("button", { name: "All OCR" })).toHaveClass(/active/);
  await expect(page.locator("rect.token-review")).toHaveCount(3);

  await page.reload();

  await expect(page.getByText("3 OCR tokens are shown because no card candidates were generated.")).toBeVisible();
  await expect(page.getByRole("button", { name: "All OCR" })).toHaveClass(/active/);
  await expect(page.locator("rect.token-review")).toHaveCount(3);
});
