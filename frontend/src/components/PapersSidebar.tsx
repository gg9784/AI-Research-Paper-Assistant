"use client";
import { BookOpen, Trash2, Filter, X } from "lucide-react";
import { deletePaper } from "@/lib/api";
import toast from "react-hot-toast";
import clsx from "clsx";

interface PapersSidebarProps {
  papers:      string[];
  activePaper: string | null;
  onSelect:    (paper: string | null) => void;
  onDelete:    (paper: string) => void;
}

export default function PapersSidebar({ papers, activePaper, onSelect, onDelete }: PapersSidebarProps) {
  const handleDelete = async (name: string, e: React.MouseEvent) => {
    e.stopPropagation();
    try {
      await deletePaper(name);
      onDelete(name);
      toast.success(`Deleted "${name}"`);
    } catch {
      toast.error("Failed to delete paper");
    }
  };

  return (
    <div className="flex flex-col gap-2">
      <div className="flex items-center justify-between px-1">
        <span className="text-xs font-semibold text-gray-500 uppercase tracking-widest">Papers ({papers.length})</span>
        {activePaper && (
          <button onClick={() => onSelect(null)} className="text-xs text-indigo-400 hover:text-indigo-300 flex items-center gap-1">
            <X className="w-3 h-3" /> Clear filter
          </button>
        )}
      </div>
      {papers.length === 0 ? (
        <p className="text-xs text-gray-600 text-center py-4">No papers uploaded yet</p>
      ) : (
        papers.map((paper) => (
          <div
            key={paper}
            onClick={() => onSelect(activePaper === paper ? null : paper)}
            className={clsx(
              "group flex items-center gap-2 px-3 py-2.5 rounded-xl cursor-pointer transition-all",
              activePaper === paper
                ? "bg-indigo-500/20 border border-indigo-500/40 text-indigo-200"
                : "glass-card hover:border-white/15 text-gray-400 hover:text-gray-200"
            )}
          >
            {activePaper === paper
              ? <Filter className="w-3.5 h-3.5 text-indigo-400 flex-shrink-0" />
              : <BookOpen className="w-3.5 h-3.5 flex-shrink-0" />
            }
            <span className="text-xs flex-1 truncate">{paper}</span>
            <button
              onClick={(e) => handleDelete(paper, e)}
              className="opacity-0 group-hover:opacity-100 text-red-400 hover:text-red-300 transition-all"
            >
              <Trash2 className="w-3 h-3" />
            </button>
          </div>
        ))
      )}
    </div>
  );
}
