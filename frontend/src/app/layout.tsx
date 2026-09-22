import type { Metadata } from "next";
import { Inter } from "next/font/google";
import "./globals.css";
import { Toaster } from "react-hot-toast";

const inter = Inter({ subsets: ["latin"], variable: "--font-inter" });

export const metadata: Metadata = {
  title: "AI Research Paper Assistant | RAG-Powered Q&A",
  description: "Upload research papers and chat with them using AI. Powered by GPT-4 and ChromaDB.",
  keywords: ["AI", "research", "papers", "RAG", "GPT-4", "LLM", "PDF"],
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className={inter.variable}>
      <body className="bg-gray-950 text-gray-100 antialiased min-h-screen">
        <Toaster
          position="top-right"
          toastOptions={{
            style: { background: "#1e1b4b", color: "#e0e9ff", border: "1px solid #4f46e5" },
            duration: 4000,
          }}
        />
        {children}
      </body>
    </html>
  );
}
