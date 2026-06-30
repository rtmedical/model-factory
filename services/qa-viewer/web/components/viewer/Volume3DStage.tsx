"use client";

// Portal-based fullscreen overlay that renders the prediction labelmap as
// smoothed 3D surface meshes. Mirrors the FullscreenStage (MPR) structure
// so the two fullscreen modes feel like siblings: top toolbar, content
// canvas, bottom strip with case identity.
//
// The component is dynamic-imported from ViewerStage so the cornerstone
// VOLUME_3D + polySeg machinery is only loaded the first time the operator
// opens the view, keeping the cold-start bundle lean.

import { useEffect, useState } from "react";
import { createPortal } from "react-dom";

import { useQAStore } from "@/lib/store";

import { Volume3DCanvas } from "./Volume3DCanvas";
import { Volume3DToolbar } from "./Volume3DToolbar";

export function Volume3DStage() {
  const open = useQAStore((s) => s.volume3DOpen);
  const close = useQAStore((s) => s.closeVolume3D);
  const selectedCase = useQAStore((s) => s.selectedCase);
  const prediction = useQAStore((s) => s.currentPrediction);
  const selectedModel = useQAStore((s) => s.selectedModel);

  const [mounted, setMounted] = useState(false);
  useEffect(() => setMounted(true), []);

  // Best-effort browser fullscreen on open. Iframe-embedded contexts (the
  // Caddy-fronted Brev deploy) reject the request — the portal still gives
  // us a visual takeover.
  useEffect(() => {
    if (!open) return;
    let cancelled = false;
    (async () => {
      try {
        if (!document.fullscreenElement) {
          await document.documentElement.requestFullscreen?.();
        }
      } catch {
        /* permission denied or iframe — no-op */
      }
      if (cancelled) return;
    })();
    return () => {
      cancelled = true;
      try {
        if (document.fullscreenElement) {
          void document.exitFullscreen?.();
        }
      } catch {
        /* no-op */
      }
    };
  }, [open]);

  if (!open || !mounted || !selectedCase || !prediction) return null;

  return createPortal(
    <div
      className="fixed inset-0 z-[100] flex flex-col bg-[var(--color-rt-paper)] text-[var(--color-rt-ink)]"
      role="dialog"
      aria-modal="true"
      aria-label="3D surface viewer"
    >
      <Volume3DToolbar />

      <div className="relative flex-1 min-h-0 bg-[color-mix(in_oklab,#02060f_92%,var(--color-rt-mist)_8%)]">
        <Volume3DCanvas />
      </div>

      <div className="flex shrink-0 items-center justify-between gap-3 border-t border-[var(--color-rt-line)] bg-[var(--color-rt-paper)] px-3 py-1.5 text-[11px]">
        <div className="flex min-w-0 items-center gap-3 truncate">
          {selectedModel && (
            <span className="rt-display font-semibold text-[var(--color-rt-ink)]">
              {selectedModel.dataset_name.replace(/^Dataset(\d+)_/, "D$1 ")}
            </span>
          )}
          <span className="font-mono text-[10.5px] text-[var(--color-rt-muted)]">
            {selectedCase.case_id}
          </span>
          {prediction.metrics && prediction.metrics.length > 0 && (
            <span className="text-[10.5px] text-[var(--color-rt-muted)]">
              {prediction.metrics.length} segment
              {prediction.metrics.length === 1 ? "" : "s"} from{" "}
              <span className="font-mono">{prediction.prediction_id.slice(0, 8)}</span>
            </span>
          )}
        </div>
        <button
          type="button"
          onClick={() => close()}
          className="font-mono text-[10.5px] text-[var(--color-rt-muted)] underline-offset-2 hover:text-[var(--color-rt-ink)] hover:underline"
        >
          Esc to close
        </button>
      </div>
    </div>,
    document.body,
  );
}
