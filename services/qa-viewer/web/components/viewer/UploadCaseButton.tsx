"use client";

import { useMutation, useQueryClient } from "@tanstack/react-query";
import { Loader2, Upload } from "lucide-react";
import { useRef, useState } from "react";

import { uploadCase, type CaseInfo } from "@/lib/api";
import { useQAStore } from "@/lib/store";
import { cn } from "@/lib/utils";

// Upload a reviewer's own DICOM series (.zip/.dcm) or NIfTI volume
// (.nii/.nii.gz) as an ad-hoc test case for the selected model. The server
// converts it to the cohort layout and returns the new case; we auto-select
// it so the reviewer can hit Run immediately. No new client deps — the file
// is sent as multipart and converted server-side.
export function UploadCaseButton({ modelId }: { modelId: string }) {
  const qc = useQueryClient();
  const setCase = useQAStore((s) => s.setCase);
  const reviewer = useQAStore((s) => s.reviewer);
  const inputRef = useRef<HTMLInputElement>(null);
  const [progress, setProgress] = useState<number | null>(null);
  const [err, setErr] = useState<string | null>(null);

  const mut = useMutation({
    mutationFn: (files: File[]) =>
      uploadCase({ model_id: modelId, files, reviewer, onProgress: setProgress }),
    onMutate: () => {
      setErr(null);
      setProgress(0);
    },
    onSuccess: async (c: CaseInfo) => {
      setProgress(null);
      await qc.invalidateQueries({ queryKey: ["cohort"] });
      setCase(c);
    },
    onError: (e: unknown) => {
      setProgress(null);
      setErr(e instanceof Error ? e.message : String(e));
    },
  });

  function onPick(e: React.ChangeEvent<HTMLInputElement>) {
    const files = e.target.files ? Array.from(e.target.files) : [];
    if (files.length) mut.mutate(files);
    e.target.value = ""; // allow re-picking the same file
  }

  const busy = mut.isPending;
  const pct = progress !== null ? Math.round(progress * 100) : null;
  const label = busy
    ? pct !== null && pct < 100
      ? `uploading ${pct}%`
      : "converting…"
    : "upload case";

  return (
    <div className="flex shrink-0 flex-col items-stretch">
      <input
        ref={inputRef}
        type="file"
        multiple
        accept=".nii,.nii.gz,.zip,.dcm,application/gzip,application/zip"
        className="hidden"
        onChange={onPick}
      />
      <button
        type="button"
        onClick={() => inputRef.current?.click()}
        disabled={busy}
        title="Upload a DICOM series (.zip / .dcm) or a NIfTI volume (.nii / .nii.gz) as a test case for this model"
        className={cn(
          "inline-flex items-center gap-1.5 rounded-[var(--radius-rt-sm)] border border-dashed px-3 py-1.5 text-[12px] font-medium transition-colors",
          "border-[var(--color-rt-line-2)] text-[var(--color-rt-muted)] hover:border-[var(--color-rt-accent)] hover:text-[var(--color-rt-accent)]",
          busy && "cursor-not-allowed opacity-70",
        )}
      >
        {busy ? (
          <Loader2 size={12} className="animate-spin" />
        ) : (
          <Upload size={12} />
        )}
        {label}
      </button>
      {err && (
        <span
          className="mt-0.5 max-w-[200px] truncate text-[10px] text-[var(--color-rt-pip-error)]"
          title={err}
        >
          {err}
        </span>
      )}
    </div>
  );
}
