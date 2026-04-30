import { expect, test, type ConsoleMessage, type Page } from "@playwright/test";

type BrowserIssue = {
  message: string;
  source: string;
};

test("ignores browser-extension context invalidation noise", () => {
  expect(isExternalExtensionNoise("Uncaught Error: Extension context invalidated.", "chrome-extension://abc/content.js")).toBe(true);
  expect(isExternalExtensionNoise("Uncaught Error: Extension context invalidated.\n    at a (content.js:11:1155)", "")).toBe(true);
  expect(isExternalExtensionNoise("Uncaught Error: Extension context invalidated.", "http://127.0.0.1:3000/_next/static/app.js")).toBe(false);
});

test("home page loads without app-owned browser runtime errors", async ({ page }) => {
  const browserIssues = recordBrowserIssues(page);

  await mockEmptyBackend(page);
  await page.goto("/");

  await expect(page.getByRole("heading", { name: "Turn workbook photos into Anki candidates you can actually trust." })).toBeVisible();
  await expect(page.getByText("No uploads yet.")).toBeVisible();
  expect(browserIssues).toEqual([]);
});

test("records app-owned console errors", async ({ page }) => {
  const browserIssues = recordBrowserIssues(page);

  await page.setContent("<script>console.error('App console error from bundled code')</script>");

  expect(browserIssues).toEqual([
    expect.objectContaining({
      message: "App console error from bundled code"
    })
  ]);
});

test("records app-owned uncaught page errors", async ({ page }) => {
  const browserIssues = recordBrowserIssues(page);

  await page.setContent("<script>setTimeout(() => { throw new Error('App page exploded'); }, 0)</script>");

  await expect.poll(() => browserIssues.map((issue) => issue.message).join("\n")).toContain("App page exploded");
});

test("does not ignore app-owned errors that merely mention extensions", async ({ page }) => {
  const browserIssues = recordBrowserIssues(page);

  await page.setContent("<script>console.error('Extension context invalidated inside app code')</script>");

  expect(browserIssues).toHaveLength(1);
});

function recordBrowserIssues(page: Page): BrowserIssue[] {
  const browserIssues: BrowserIssue[] = [];

  page.on("pageerror", (error) => {
    const details = `${error.message}\n${error.stack ?? ""}`;
    if (!isExternalExtensionNoise(details, "")) {
      browserIssues.push({ message: details, source: "pageerror" });
    }
  });

  page.on("console", (message) => {
    if (message.type() !== "error") return;
    const location = message.location();
    if (!isIgnorableConsoleError(message, location.url)) {
      browserIssues.push({ message: message.text(), source: location.url || "console" });
    }
  });

  return browserIssues;
}

function isIgnorableConsoleError(message: ConsoleMessage, sourceUrl: string): boolean {
  const text = message.text();
  return isExternalExtensionNoise(text, sourceUrl);
}

function isExternalExtensionNoise(message: string, sourceUrl: string): boolean {
  if (!message.includes("Extension context invalidated")) return false;
  return sourceUrl.startsWith("chrome-extension://") || sourceUrl.endsWith("/content.js") || message.includes("content.js:");
}

async function mockEmptyBackend(page: Page) {
  await page.route("**/api/pages", async (route) => {
    await route.fulfill({ contentType: "application/json", body: "[]" });
  });
}
