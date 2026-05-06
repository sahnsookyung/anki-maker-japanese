import type { Page } from "@playwright/test";

const ocrProfilePayload = {
  profiles: [
    {
      id: "jp_v3_mobile_current",
      label: "Japanese PP-OCRv3 mobile + Korean PP-OCRv5",
      budget: "safe_local",
      provider: "paddle",
      creates_candidates: true,
      description: "Frozen current production profile"
    }
  ],
  variants: [{ id: "baseline_current", label: "Frozen current extractor", description: "Frozen current extractor" }],
  default_profile: "jp_v3_mobile_current",
  default_variant: "baseline_current"
};

export async function mockOcrProfiles(page: Page) {
  await page.route("**/api/ocr/profiles", async (route) => {
    await route.fulfill({ contentType: "application/json", body: JSON.stringify(ocrProfilePayload) });
  });
}
