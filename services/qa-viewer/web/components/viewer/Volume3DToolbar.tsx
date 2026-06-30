"use client";

// Top strip of Volume3DStage. Mirrors the FullscreenToolbar shape so the
// two fullscreen modes feel like siblings. Drives:
//   - per-segment visibility (one chip per non-background label, coloured
//     by the GT palette so segment colour ↔ chip colour ↔ mesh colour)
//   - reset camera
//   - exit
//
// Per the cornerstone-tools 1.86 polySeg public API, the marching-cubes /
// vtk smoothing pipeline doesn't expose a runtime smoothing knob, so the
// quality is what cornerstone's worker defaults to — already smoothed via
// VTK's WindowedSincPolyDataFilter under the hood. We expose a "regenerate"
// button as a placeholder for when the API grows a public iteration count.

import * as cornerstone from "@cornerstonejs/core";
import { Focus, X } from "lucide-react";
import { useMemo } from "react";

import { useQAStore } from "@/lib/store";
import { cn } from "@/lib/utils";

import { IconToggle } from "@/components/shell/IconToggle";

import {
  QA_RENDERING_ENGINE_ID,
  QA_VOLUME_3D_VIEWPORT_ID,
} from "./cornerstoneInit";
import { GT_PALETTE } from "./NiftiViewer";

export function Volume3DToolbar() {
  const prediction = useQAStore((s) => s.currentPrediction);
  const selectedModel = useQAStore((s) => s.selectedModel);
  const selectedCase = useQAStore((s) => s.selectedCase);
  const segmentVisibility = useQAStore((s) => s.volume3DSegmentVisibility);
  const setSegmentVisibility = useQAStore((s) => s.setVolume3DSegmentVisibility);
  const close = useQAStore((s) => s.closeVolume3D);

  const segments = useMemo(() => {
    if (!prediction?.label_map) return [] as { name: string; idx: number }[];
    return Object.entries(prediction.label_map)
      .filter(([name]) => name !== "background")
      .map(([name, idx]) => ({ name, idx: idx as number }))
      .sort((a, b) => a.idx - b.idx);
  }, [prediction]);

  const resetCamera = () => {
    const engine = cornerstone.getRenderingEngine(QA_RENDERING_ENGINE_ID);
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const vp = engine?.getViewport(QA_VOLUME_3D_VIEWPORT_ID) as any;
    try {
      vp?.resetCamera?.();
      vp?.render?.();
    } catch {
      /* viewport may have just been disposed */
    }
  };

  return (
    <div className="flex shrink-0 items-center justify-between gap-3 border-b border-[var(--color-rt-line)] bg-[var(--color-rt-paper)] px-3 py-2">
      {/* Left — identity */}
      <div className="flex min-w-0 items-center gap-2">
        <span className="rt-display text-[10px] font-semibold uppercase tracking-[0.16em] text-[var(--color-rt-muted)]">
          3D · surface mesh
        </span>
        <span className="h-4 w-px bg-[var(--color-rt-line)]" />
        {selectedModel && (
          <span className="truncate text-[11.5px] text-[var(--color-rt-ink)]">
            {selectedModel.dataset_name.replace(/^Dataset(\d+)_/, "D$1 ")}
          </span>
        )}
        {selectedCase && (
          <span className="hidden font-mono text-[10.5px] text-[var(--color-rt-muted)] md:inline">
            {selectedCase.case_id}
          </span>
        )}
      </div>

      {/* Middle — visibility chips */}
      <div className="flex max-w-[60%] flex-wrap items-center gap-1 overflow-hidden">
        {segments.length === 0 ? (
          <span className="text-[11px] italic text-[var(--color-rt-muted)]">
            no predicted segments
          </span>
        ) : (
          segments.map(({ name, idx }) => {
            const visible = segmentVisibility[idx] !== false;
            const [r, g, b] = GT_PALETTE[(idx - 1) % GT_PALETTE.length];
            return (
              <button
                key={idx}
                type="button"
                onClick={() => setSegmentVisibility(idx, !visible)}
                aria-pressed={visible}
                title={`${name} — click to ${visible ? "hide" : "show"}`}
                className={cn(
                  "inline-flex h-7 items-center gap-1.5 rounded-full border px-2 text-[11px] transition-colors",
                  visible
                    ? "border-[var(--color-rt-line)] bg-[var(--color-rt-paper)] text-[var(--color-rt-ink)]"
                    : "border-[var(--color-rt-line)] bg-[var(--color-rt-mist)] text-[var(--color-rt-muted)] line-through opacity-60",
                )}
              >
                <span
                  aria-hidden
                  className="inline-block h-2.5 w-2.5 rounded-full"
                  style={{ background: `rgb(${r}, ${g}, ${b})` }}
                />
                <span className="max-w-[14ch] truncate">{name}</span>
              </button>
            );
          })
        )}
      </div>

      {/* Right — viewport controls */}
      <div className="flex shrink-0 items-center gap-2">
        <IconToggle
          onClick={resetCamera}
          Icon={Focus}
          label="Reset camera"
        />
        <span className="h-5 w-px bg-[var(--color-rt-line)]" />
        <button
          type="button"
          onClick={() => close()}
          className="inline-flex h-8 items-center gap-1 rounded-[var(--radius-rt-sm)] border border-[var(--color-rt-line)] px-2.5 text-[11.5px] text-[var(--color-rt-muted)] transition-colors hover:bg-[var(--color-rt-mist)] hover:text-[var(--color-rt-ink)]"
          title="Exit 3D view (Esc)"
        >
          <X size={13} />
          exit
        </button>
      </div>
    </div>
  );
}
