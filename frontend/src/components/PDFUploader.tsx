"use client";
import { useCallback, useState } from "react";
import { useDropzone } from "react-dropzone";
import toast from "react-hot-toast";
import { uploadPaper, UploadResponse } from "@/lib/api";
import { Upload, FileText, CheckCircle, Loader2, X } from "lucide-react";
import clsx from "clsx";

interface PDFUploaderProps {
  onUploadSuccess: (paperName: string) => void;
}

interface UploadedFile {
  name: string;
  status: "uploading" | "done" | "error";
  result?: UploadResponse;
}

export default function PDFUploader({ onUploadSuccess }: PDFUploaderProps) {
  const [files, setFiles] = useState<UploadedFile[]>([]);

  const processFile = async (file: File) => {
    const entry: UploadedFile = { name: file.name, status: "uploading" };
    setFiles((prev) => [...prev, entry]);

    try {
      const result = await uploadPaper(file);
      setFiles((prev) =>
        prev.map((f) => f.name === file.name ? { ...f, status: "done", result } : f)
      );
      onUploadSuccess(result.paper_name);
      toast.success(`✅ "${result.paper_name}" indexed successfully!`);
    } catch (err: any) {
      setFiles((prev) =>
        prev.map((f) => f.name === file.name ? { ...f, status: "error" } : f)
      );
      toast.error(`❌ Failed to upload "${file.name}"`);
    }
  };

  const onDrop = useCallback((accepted: File[]) => {
    accepted.forEach(processFile);
  }, []);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: { "application/pdf": [".pdf"] },
    maxFiles: 10,
    maxSize: 50 * 1024 * 1024,
  });

  return (
    <div className="space-y-4">
      {/* Drop Zone */}
      <div
        {...getRootProps()}
        className={clsx(
          "border-2 border-dashed rounded-2xl p-8 text-center cursor-pointer transition-all duration-300",
          isDragActive
            ? "border-indigo-400 bg-indigo-500/10 scale-[1.01]"
            : "border-white/10 hover:border-indigo-500/50 hover:bg-white/[0.02]"
        )}
      >
        <input {...getInputProps()} />
        <div className="flex flex-col items-center gap-3">
          <div className={clsx(
            "w-14 h-14 rounded-xl flex items-center justify-center transition-all",
            isDragActive ? "bg-indigo-500/30" : "bg-white/5"
          )}>
            <Upload className={clsx("w-7 h-7", isDragActive ? "text-indigo-300" : "text-gray-400")} />
          </div>
          <div>
            <p className="text-sm font-medium text-gray-300">
              {isDragActive ? "Drop your PDFs here..." : "Drag & drop research papers"}
            </p>
            <p className="text-xs text-gray-500 mt-1">
              or <span className="text-indigo-400 underline">browse files</span> · PDF only · Max 50MB each
            </p>
          </div>
        </div>
      </div>

      {/* Upload List */}
      {files.length > 0 && (
        <div className="space-y-2">
          {files.map((f) => (
            <div key={f.name} className={clsx(
              "flex items-center gap-3 px-4 py-3 rounded-xl glass-card",
              f.status === "error" && "border-red-500/30"
            )}>
              <FileText className="w-4 h-4 text-indigo-400 flex-shrink-0" />
              <span className="text-sm text-gray-300 flex-1 truncate">{f.name}</span>
              {f.status === "uploading" && (
                <Loader2 className="w-4 h-4 text-indigo-400 animate-spin" />
              )}
              {f.status === "done" && (
                <CheckCircle className="w-4 h-4 text-emerald-400" />
              )}
              {f.status === "error" && (
                <X className="w-4 h-4 text-red-400" />
              )}
              {f.result && (
                <span className="text-xs text-gray-500">
                  {f.result.total_chunks} chunks
                </span>
              )}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
