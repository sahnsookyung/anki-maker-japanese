export type PageTiming = {
  pageId: string;
  pageTitle: string;
  ms: number;
  success: boolean;
};

export type BatchTimingReport = {
  text: string;
  engineLabel: string;
  totalPages: number;
  processedPages: number;
  successfulPages: number;
  failedPages: number;
  failures: string[];
  stats: {
    total: string;
    average: string;
    min: string;
    p25: string;
    p75: string;
    max: string;
  };
};

export function durationMs(start: number, end: number): number {
  return Math.max(0, Math.round(end - start));
}

export function formatDuration(ms: number): string {
  if (ms < 1000) return `${Math.round(ms)}ms`;
  const seconds = ms / 1000;
  if (seconds < 60) return `${seconds.toFixed(seconds < 10 ? 1 : 0)}s`;
  const minutes = Math.floor(seconds / 60);
  const remainder = Math.round(seconds % 60);
  return `${minutes}m ${remainder}s`;
}

export function percentile(values: number[], percentileValue: number): number {
  if (!values.length) return 0;
  const sorted = [...values].sort((first, second) => first - second);
  const clamped = Math.min(1, Math.max(0, percentileValue));
  const index = Math.round((sorted.length - 1) * clamped);
  return sorted[index];
}

export function batchTimingSummary(
  timings: PageTiming[],
  totalPages: number,
  failures: string[],
  engineLabel: string,
  processedPages = totalPages
): string {
  return batchTimingReport(timings, totalPages, failures, engineLabel, processedPages).text;
}

export function batchTimingReport(
  timings: PageTiming[],
  totalPages: number,
  failures: string[],
  engineLabel: string,
  processedPages = totalPages
): BatchTimingReport {
  const measured = timings.map((timing) => timing.ms);
  const totalMs = measured.reduce((sum, value) => sum + value, 0);
  const averageMs = measured.length ? totalMs / measured.length : 0;
  const minMs = measured.length ? Math.min(...measured) : 0;
  const maxMs = measured.length ? Math.max(...measured) : 0;
  const successfulPages = failures.length ? Math.max(0, processedPages - failures.length) : processedPages;
  const pageWord = failures.length ? "pages" : pageLabel(processedPages);
  const prefix = `Processed ${successfulPages}/${totalPages} ${pageWord} with ${engineLabel}`;
  const stats = {
    total: formatDuration(totalMs),
    average: formatDuration(averageMs),
    min: formatDuration(minMs),
    p25: formatDuration(percentile(measured, 0.25)),
    p75: formatDuration(percentile(measured, 0.75)),
    max: formatDuration(maxMs)
  };
  const statsText = [
    `total ${stats.total}`,
    `avg ${stats.average}`,
    `min ${stats.min}`,
    `p25 ${stats.p25}`,
    `p75 ${stats.p75}`,
    `max ${stats.max}`
  ].join(", ");
  const failureText = failures.length ? ` Failed: ${failures.join(", ")}.` : ".";
  return {
    text: `${prefix} in frontend time: ${statsText}${failureText}`,
    engineLabel,
    totalPages,
    processedPages,
    successfulPages,
    failedPages: failures.length,
    failures,
    stats
  };
}

function pageLabel(count: number): string {
  return count === 1 ? "page" : "pages";
}
