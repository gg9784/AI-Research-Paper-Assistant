"use client";
import { useState, useEffect } from "react";
import { Brain, Upload, MessageSquare, Github, Sparkles } from "lucide-react";
import PDFUploader from "@/components/PDFUploader";
import ChatInterface from "@/components/ChatInterface";
import PapersSidebar from "@/components/PapersSidebar";
import { listPapers } from "@/lib/api";
import toast from "react-hot-toast";

type Tab = "upload" | "chat";

export default function Home() {
  const [papers,      setPapers]      = useState<string[]>([]);
  const [activeTab,   setActiveTab]   = useState<Tab>("upload");
  const [filterPaper, setFilterPaper] = useState<string | null>(null);
  const [loading,     setLoading]     = useState(true);

  useEffect(() => {
    listPapers()
      .then((res) => { setPapers(res.papers); setLoading(false); })
      .catch(() => { setLoading(false); });
  }, []);

  const handleUploadSuccess = (paperName: string) => {
    setPapers((prev) => prev.includes(paperName) ? prev : [...prev, paperName]);
    setActiveTab("chat");
    toast.success("Paper ready! Start asking questions 🎉");
  };

  const handleDeletePaper = (name: string) => {
    setPapers((prev) => prev.filter((p) => p !== name));
    if (filterPaper === name) setFilterPaper(null);
  };

  return (
    <div className="flex flex-col min-h-screen">
      {/* ── Header ── */}
      <header className="border-b border-white/5 backdrop-blur-xl sticky top-0 z-50">
        <div className="max-w-7xl mx-auto px-6 h-16 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-9 h-9 rounded-xl bg-gradient-to-br from-indigo-500 to-purple-600 flex items-center justify-center shadow-lg shadow-indigo-500/25">
              <Brain className="w-5 h-5 text-white" />
            </div>
            <div>
              <span className="font-bold text-white">ResearchAI</span>
              <span className="text-xs text-gray-500 ml-2 hidden sm:inline">Powered by GPT-4 + RAG</span>
            </div>
          </div>
          <div className="flex items-center gap-3">
            <div className="flex items-center gap-1.5 px-3 py-1.5 rounded-full glass-card text-xs text-emerald-400">
              <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-pulse" />
              {loading ? "Connecting..." : `${papers.length} papers indexed`}
            </div>
          </div>
        </div>
      </header>

      <div className="flex-1 max-w-7xl mx-auto w-full px-6 py-6 flex gap-6 min-h-0">
        {/* ── Sidebar ── */}
        <aside className="w-64 flex-shrink-0 flex flex-col gap-6">
          {/* Stats */}
          <div className="glass-card p-4 space-y-3">
            <div className="flex items-center gap-2">
              <Sparkles className="w-4 h-4 text-indigo-400" />
              <span className="text-xs font-semibold text-gray-400 uppercase tracking-wider">Overview</span>
            </div>
            <div className="grid grid-cols-2 gap-2">
              {[
                { label: "Papers",   value: papers.length },
                { label: "Model",    value: "GPT-4o" },
                { label: "Vector DB",value: "Chroma" },
                { label: "Embed",    value: "3-small" },
              ].map((s) => (
                <div key={s.label} className="bg-white/[0.03] rounded-lg p-2 text-center">
                  <p className="text-xs text-gray-500">{s.label}</p>
                  <p className="text-sm font-bold gradient-text">{s.value}</p>
                </div>
              ))}
            </div>
          </div>

          {/* Papers List */}
          <div className="glass-card p-4 flex-1">
            <PapersSidebar
              papers={papers}
              activePaper={filterPaper}
              onSelect={setFilterPaper}
              onDelete={handleDeletePaper}
            />
          </div>
        </aside>

        {/* ── Main Panel ── */}
        <main className="flex-1 flex flex-col glass-card overflow-hidden min-h-0">
          {/* Tabs */}
          <div className="flex border-b border-white/5 px-2 pt-2">
            {(["upload", "chat"] as Tab[]).map((tab) => (
              <button
                key={tab}
                onClick={() => setActiveTab(tab)}
                className={`flex items-center gap-2 px-4 py-2.5 text-sm font-medium rounded-t-lg transition-all ${
                  activeTab === tab
                    ? "text-indigo-300 border-b-2 border-indigo-500 -mb-px"
                    : "text-gray-500 hover:text-gray-300"
                }`}
              >
                {tab === "upload" ? <Upload className="w-4 h-4" /> : <MessageSquare className="w-4 h-4" />}
                {tab === "upload" ? "Upload Papers" : "Chat"}
                {tab === "chat" && papers.length > 0 && (
                  <span className="w-5 h-5 text-xs rounded-full bg-indigo-500/30 text-indigo-300 flex items-center justify-center">
                    {papers.length}
                  </span>
                )}
              </button>
            ))}
          </div>

          {/* Tab Content */}
          <div className="flex-1 overflow-hidden">
            {activeTab === "upload" ? (
              <div className="p-6 space-y-6 h-full overflow-y-auto">
                <div>
                  <h2 className="text-lg font-semibold text-white">Upload Research Papers</h2>
                  <p className="text-sm text-gray-500 mt-1">
                    Drop your PDF papers below. They will be automatically chunked, embedded, and indexed.
                  </p>
                </div>
                <PDFUploader onUploadSuccess={handleUploadSuccess} />

                {papers.length > 0 && (
                  <div className="glass-card p-4 border-indigo-500/20">
                    <p className="text-sm text-indigo-300 font-medium mb-1">✅ Ready to chat!</p>
                    <p className="text-xs text-gray-500">
                      {papers.length} paper{papers.length > 1 ? "s" : ""} indexed. Switch to the Chat tab to start asking questions.
                    </p>
                    <button onClick={() => setActiveTab("chat")} className="btn-gradient text-xs mt-3 px-4 py-2">
                      Go to Chat →
                    </button>
                  </div>
                )}
              </div>
            ) : (
              <div className="h-full flex flex-col">
                {filterPaper && (
                  <div className="px-4 pt-3">
                    <div className="flex items-center gap-2 px-3 py-2 rounded-xl bg-indigo-500/10 border border-indigo-500/20 text-xs text-indigo-300">
                      <span>🔍 Filtering by:</span>
                      <span className="font-medium">{filterPaper}</span>
                    </div>
                  </div>
                )}
                <div className="flex-1 min-h-0">
                  <ChatInterface papers={papers} filterPaper={filterPaper} />
                </div>
              </div>
            )}
          </div>
        </main>
      </div>
    </div>
  );
}
