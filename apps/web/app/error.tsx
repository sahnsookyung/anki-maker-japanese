"use client";

export default function Error({ error, reset }: { error: Error & { digest?: string }; reset: () => void }) {
  return (
    <main className="app-shell">
      <section className="panel error-screen">
        <p className="eyebrow">Frontend error</p>
        <h1>Something went sideways while rendering the review UI.</h1>
        <p className="lede">{error.message || "The page hit an unexpected client-side error."}</p>
        <button onClick={reset}>Try again</button>
      </section>
    </main>
  );
}
