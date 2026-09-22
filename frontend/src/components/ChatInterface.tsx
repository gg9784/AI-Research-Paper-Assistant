"use client";
import { useState, useRef, useEffect } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { Send, Bot, User, BookOpen, Loader2, Zap } from "lucide-react";
import { fetchStreamAnswer, SourceCitation } from "@/lib/api";
import toast from "react-hot-toast";
import clsx from "clsx";

interface Message {
  id:      string;
  role:    "user" | "assistant";
  content: string;
  sources?: SourceCitation[];
  isStreaming?: boolean;
}

interface ChatInterfaceProps {
  papers:      string[];
  filterPaper: string | null;
}

const SUGGESTED_QUESTIONS = [
  "What is the main contribution of this paper?",
  "Explain the methodology used in this research.",
  "What are the key results and findings?",
  "What datasets were used in the experiments?",
  "How does this compare to prior work?",
];

export default function ChatInterface({ papers, filterPaper }: ChatInterfaceProps) {
  const [messages,  setMessages]  = useState<Message[]>([]);
  const [input,     setInput]     = useState("");
  const [loading,   setLoading]   = useState(false);
  const bottomRef = useRef<HTMLDivElement>(null);
  const inputRef  = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const generateId = () => Math.random().toString(36).slice(2);

  const sendMessage = async (question: string) => {
    if (!question.trim() || loading) return;
    if (papers.length === 0) {
      toast.error("Please upload at least one research paper first!");
      return;
    }

    const userMsg: Message = { id: generateId(), role: "user", content: question };
    const assistantId = generateId();
    const assistantMsg: Message = {
      id: assistantId, role: "assistant", content: "", isStreaming: true,
    };

    setMessages((prev) => [...prev, userMsg, assistantMsg]);
    setInput("");
    setLoading(true);

    try {
      let fullContent = "";
      for await (const token of fetchStreamAnswer(question, filterPaper || undefined)) {
        fullContent += token;
        setMessages((prev) =>
          prev.map((m) => m.id === assistantId ? { ...m, content: fullContent } : m)
        );
      }
      setMessages((prev) =>
        prev.map((m) => m.id === assistantId ? { ...m, isStreaming: false } : m)
      );
    } catch (err) {
      toast.error("Failed to get answer. Is the backend running?");
      setMessages((prev) => prev.filter((m) => m.id !== assistantId));
    } finally {
      setLoading(false);
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); sendMessage(input); }
  };

  return (
    <div className="flex flex-col h-full">

      {/* Messages */}
      <div className="flex-1 overflow-y-auto space-y-4 p-4 min-h-0">
        {messages.length === 0 ? (
          <div className="flex flex-col items-center justify-center h-full gap-6 py-8">
            <div className="w-16 h-16 rounded-2xl bg-gradient-to-br from-indigo-500 to-purple-600 flex items-center justify-center shadow-lg shadow-indigo-500/25">
              <Bot className="w-8 h-8 text-white" />
            </div>
            <div className="text-center">
              <h3 className="text-lg font-semibold text-white">Ask me anything about your papers</h3>
              <p className="text-sm text-gray-500 mt-1">
                {papers.length > 0
                  ? `${papers.length} paper${papers.length > 1 ? "s" : ""} ready to chat`
                  : "Upload papers to get started"}
              </p>
            </div>
            {papers.length > 0 && (
              <div className="grid grid-cols-1 gap-2 w-full max-w-md">
                {SUGGESTED_QUESTIONS.map((q) => (
                  <button
                    key={q}
                    onClick={() => sendMessage(q)}
                    className="text-left text-xs text-gray-400 px-4 py-2.5 rounded-xl glass-card hover:border-indigo-500/40 hover:text-indigo-300 transition-all"
                  >
                    💡 {q}
                  </button>
                ))}
              </div>
            )}
          </div>
        ) : (
          messages.map((msg) => (
            <div key={msg.id} className={clsx("flex gap-3 message-enter", msg.role === "user" ? "flex-row-reverse" : "flex-row")}>
              {/* Avatar */}
              <div className={clsx(
                "w-8 h-8 rounded-xl flex-shrink-0 flex items-center justify-center",
                msg.role === "user"
                  ? "bg-indigo-500/20 border border-indigo-500/30"
                  : "bg-purple-500/20 border border-purple-500/30"
              )}>
                {msg.role === "user"
                  ? <User className="w-4 h-4 text-indigo-300" />
                  : <Bot  className="w-4 h-4 text-purple-300" />
                }
              </div>

              {/* Bubble */}
              <div className={clsx(
                "max-w-[80%] rounded-2xl px-4 py-3 text-sm",
                msg.role === "user"
                  ? "bg-indigo-600/80 text-white rounded-tr-sm"
                  : "glass-card text-gray-200 rounded-tl-sm"
              )}>
                {msg.role === "assistant" ? (
                  <>
                    <ReactMarkdown
                      remarkPlugins={[remarkGfm]}
                      className="prose prose-invert prose-sm max-w-none"
                    >
                      {msg.content}
                    </ReactMarkdown>
                    {msg.isStreaming && (
                      <span className="inline-flex gap-1 ml-1 mt-1">
                        <span className="typing-dot" />
                        <span className="typing-dot" />
                        <span className="typing-dot" />
                      </span>
                    )}
                  </>
                ) : (
                  <p>{msg.content}</p>
                )}
              </div>
            </div>
          ))
        )}
        <div ref={bottomRef} />
      </div>

      {/* Input Bar */}
      <div className="p-4 border-t border-white/5">
        <div className="flex gap-3 items-end glass-card p-3">
          <textarea
            ref={inputRef}
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Ask a question about your research papers..."
            disabled={loading}
            rows={1}
            className="flex-1 bg-transparent text-sm text-gray-200 placeholder-gray-600 resize-none outline-none min-h-[36px] max-h-[120px]"
            style={{ height: "auto" }}
          />
          <button
            onClick={() => sendMessage(input)}
            disabled={loading || !input.trim()}
            className="btn-gradient w-9 h-9 flex-shrink-0 flex items-center justify-center rounded-lg"
          >
            {loading
              ? <Loader2 className="w-4 h-4 animate-spin" />
              : <Send className="w-4 h-4" />
            }
          </button>
        </div>
        <p className="text-xs text-gray-600 text-center mt-2">
          Press Enter to send · Shift+Enter for new line
        </p>
      </div>
    </div>
  );
}
