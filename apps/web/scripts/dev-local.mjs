import { existsSync, rmSync } from "node:fs";
import { get } from "node:http";
import net from "node:net";
import { dirname, resolve } from "node:path";
import { spawn } from "node:child_process";
import { fileURLToPath } from "node:url";

const webDir = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const repoRoot = resolve(webDir, "../..");
const frontendPort = Number(process.env.PORT ?? 3000);
const apiBase = normalizeApiBase(process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://127.0.0.1:8000");

rmSync(resolve(webDir, ".next/dev"), { recursive: true, force: true });
await assertPortFree(frontendPort);

let backendProcess = null;
if (await backendIsHealthy(apiBase)) {
  console.log(`Backend is already running at ${apiBase}.`);
} else if (canAutoStartBackend(apiBase)) {
  backendProcess = spawn("bash", [resolve(repoRoot, "scripts/dev-backend.sh")], {
    cwd: repoRoot,
    env: process.env,
    stdio: "inherit"
  });
  await waitForBackend(apiBase);
} else {
  console.error(`Backend is not reachable at ${apiBase}. Start it separately, then run npm run dev again.`);
  process.exit(1);
}

const nextBin = resolve(webDir, "node_modules/.bin/next");
if (!existsSync(nextBin)) {
  console.error("Next.js is not installed. Run npm install in apps/web first.");
  cleanupAndExit(1);
}

const nextProcess = spawn(nextBin, ["dev", "--hostname", "127.0.0.1", "--port", String(frontendPort)], {
  cwd: webDir,
  env: process.env,
  stdio: "inherit"
});

nextProcess.on("exit", (code) => cleanupAndExit(code ?? 0));
process.on("SIGINT", () => cleanupAndExit(130));
process.on("SIGTERM", () => cleanupAndExit(143));

function normalizeApiBase(value) {
  return value.replace(/\/$/, "");
}

function canAutoStartBackend(value) {
  try {
    const url = new URL(value);
    return ["127.0.0.1", "localhost"].includes(url.hostname) && url.port === "8000";
  } catch {
    return false;
  }
}

function backendIsHealthy(value) {
  return new Promise((resolveHealth) => {
    const request = get(`${value}/api/health`, (response) => {
      response.resume();
      resolveHealth(response.statusCode === 200);
    });
    request.setTimeout(800, () => {
      request.destroy();
      resolveHealth(false);
    });
    request.on("error", () => resolveHealth(false));
  });
}

async function waitForBackend(value) {
  console.log(`Starting backend at ${value}...`);
  const deadline = Date.now() + 90_000;
  while (Date.now() < deadline) {
    if (await backendIsHealthy(value)) {
      console.log(`Backend is ready at ${value}.`);
      return;
    }
    await new Promise((resolveWait) => setTimeout(resolveWait, 1000));
  }
  console.error(`Backend did not become ready at ${value} within 90 seconds.`);
  cleanupAndExit(1);
}

function assertPortFree(port) {
  return new Promise((resolveFree, rejectFree) => {
    const server = net.createServer();
    server.once("error", (error) => {
      if (error.code === "EADDRINUSE") {
        rejectFree(new Error(`Port ${port} is already in use. Stop the existing Next dev server before running npm run dev.`));
        return;
      }
      rejectFree(error);
    });
    server.once("listening", () => {
      server.close(resolveFree);
    });
    server.listen(port, "127.0.0.1");
  }).catch((error) => {
    console.error(error.message ?? error);
    process.exit(1);
  });
}

function cleanupAndExit(code) {
  if (backendProcess && !backendProcess.killed) {
    backendProcess.kill("SIGTERM");
  }
  process.exit(code);
}
