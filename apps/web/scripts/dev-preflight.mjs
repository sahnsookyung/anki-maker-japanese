import { rmSync } from "node:fs";
import net from "node:net";

const port = Number(process.env.PORT ?? 3000);

rmSync(".next/dev", { recursive: true, force: true });

const server = net.createServer();

server.once("error", (error) => {
  if (error.code === "EADDRINUSE") {
    console.error(`Port ${port} is already in use. Stop the existing Next dev server before running npm run dev.`);
    process.exit(1);
  }
  console.error(error);
  process.exit(1);
});

server.once("listening", () => {
  server.close(() => process.exit(0));
});

server.listen(port, "127.0.0.1");
