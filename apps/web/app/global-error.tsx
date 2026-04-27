"use client";

type GlobalErrorProps = Readonly<{ error: Error & { digest?: string }; reset: () => void }>;

export default function GlobalError({ error, reset }: GlobalErrorProps) {
  return (
    <html lang="en">
      <body>
        <main className="app-shell">
          <section className="panel error-screen">
            <p className="eyebrow">Application error</p>
            <h1>The app shell could not render.</h1>
            <p className="lede">{error.message || "Restart the dev server and try again."}</p>
            <button onClick={reset}>Try again</button>
          </section>
        </main>
      </body>
    </html>
  );
}
