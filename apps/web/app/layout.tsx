import "./globals.css";
import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "Japanese Study Image to Anki",
  description: "Review OCR-derived Japanese Anki card candidates from study-book photos."
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
