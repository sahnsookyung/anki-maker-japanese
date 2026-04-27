import "./globals.css";
import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "Japanese Study Image to Anki",
  description: "Review OCR-derived Japanese Anki card candidates from study-book photos."
};

type RootLayoutProps = Readonly<{ children: React.ReactNode }>;

export default function RootLayout({ children }: RootLayoutProps) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
