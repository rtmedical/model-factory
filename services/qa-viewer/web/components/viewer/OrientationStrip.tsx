"use client";

import { useQAStore, type ViewerOrientation } from "@/lib/store";
import { cn } from "@/lib/utils";

const OPTIONS: { value: ViewerOrientation; label: string }[] = [
  { value: "axial", label: "Axial" },
  { value: "coronal", label: "Coronal" },
  { value: "sagittal", label: "Sagittal" },
];

export function OrientationStrip() {
  const orientation = useQAStore((s) => s.orientation);
  const setOrientation = useQAStore((s) => s.setOrientation);
  return (
    <div className="inline-flex rounded-[var(--radius-rt-sm)] border border-[var(--color-rt-line)] bg-[var(--color-rt-paper)] p-0.5 text-[11px]">
      {OPTIONS.map((opt) => (
        <button
          key={opt.value}
          type="button"
          onClick={() => setOrientation(opt.value)}
          className={cn(
            "rounded-[calc(var(--radius-rt-sm)-2px)] px-2 py-1 transition-colors",
            orientation === opt.value
              ? "bg-[color-mix(in_oklab,var(--color-rt-accent)_12%,var(--color-rt-paper))] text-[var(--color-rt-accent)]"
              : "text-[var(--color-rt-muted)] hover:bg-[var(--color-rt-mist)]",
          )}
        >
          {opt.label}
        </button>
      ))}
    </div>
  );
}
