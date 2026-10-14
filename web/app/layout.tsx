import type { Metadata } from "next";
import { Toaster } from "@/components/ui/sonner";
import "./globals.css";

export const metadata: Metadata = {
  title: "Datathon'26 · решение команды",
  description: "Демо решения DATATHON'26 (NU Data Science Club × ForteBank)",
};

export default function RootLayout({ children }: LayoutProps<"/">) {
  return (
    <html lang="ru" className="h-full antialiased">
      <body className="min-h-full bg-background font-sans text-foreground">
        <main className="mx-auto max-w-6xl px-4 py-10">{children}</main>
        <Toaster />
      </body>
    </html>
  );
}
