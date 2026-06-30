"use client";

// Immersive tri-planar (MPR) overlay. Rendered via createPortal to
// document.body so it floats above the workspace grid, with ESC binding
// + optional browser fullscreen.
//
// Behaviour:
//   - Opens when store.fullscreenMpr === true (driven by ViewerStage's
//     LayoutGrid icon or the M hotkey).
//   - Reads selectedCase + currentPrediction from the store; resolves
//     the same image / prediction-seg / GT-revision URLs as the main
//     ViewerStage so cornerstone's volume cache reuses the already-loaded
//     volumes — no extra fetches.
//   - Hosts three MprViewport instances (axial / coronal / sagittal),
//     each pinned to a dedicated rendering engine + per-axis viewport
//     ID. They all join the existing QA_TOOL_GROUP_ID, so segmentation
//     reps and tool bindings are shared with the main viewer.
//   - While gtEditMode is on, the existing floating GtEditToolbar is
//     reused (rendered inside the overlay) — it carries save / cancel /
//     brush size / undo / redo / notes, so we don't duplicate that UI.

import { Loader2 } from "lucide-react";
import { useEffect, useMemo, useState } from "react";
import { createPortal } from "react-dom";

import {
  caseGroundtruthUrl,
  caseImageUrl,
  predictionSegUrl,
} from "@/lib/api";
import { useQAStore } from "@/lib/store";

import { FullscreenToolbar } from "./FullscreenToolbar";
import { GtEditToolbar } from "./GtEditToolbar";
import { MprViewport } from "./MprViewport";

export function FullscreenStage() {
  const open = useQAStore((s) => s.fullscreenMpr);
  const close = useQAStore((s) => s.closeFullscreenMpr);
  const selectedCase = useQAStore((s) => s.selectedCase);
  const prediction = useQAStore((s) => s.currentPrediction);
  const gtEditMode = useQAStore((s) => s.gtEditMode);
  const gtActiveRevisionId = useQAStore((s) => s.gtActiveRevisionId);
  const setGtRevisions = useQAStore((s) => s.setGtRevisions);
  // Mirror the main viewer's exclusion logic: GT show toggles the
  // *prediction* overlay off, and hiding GT brings it back. Editing GT
  // always shows GT regardless of the toggle.
  const showGroundtruth = useQAStore((s) => s.showGroundtruth);

  const [mounted, setMounted] = useState(false);
  useEffect(() => setMounted(true), []);

  // Try to enter true browser fullscreen on open. Best-effort: in
  // iframe-embedded contexts the API throws, and the portal still gives
  // us the visual takeover so the operator's experience is identical.
  useEffect(() => {
    if (!open) return;
    let cancelled = false;
    const root = document.documentElement;
    (async () => {
      try {
        if (!document.fullscreenElement) {
          await root.requestFullscreen?.();
        }
      } catch {
        /* iframe / permission denied — no-op */
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

  // Resolve the same URLs the main ViewerStage uses — keeps the
  // cornerstone cache hot.
  const imageVolumeId = useMemo(() => {
    if (!selectedCase) return null;
    return `nifti:${caseImageUrl(selectedCase.case_id, 0)}`;
  }, [selectedCase]);

  const segVolumeId = useMemo(() => {
    if (!prediction || gtEditMode || showGroundtruth) return null;
    return `nifti:${predictionSegUrl(prediction.prediction_id)}`;
  }, [prediction, gtEditMode, showGroundtruth]);

  const gtVolumeId = useMemo(() => {
    if (!selectedCase?.groundtruth_path) return null;
    // Only materialize the GT volumeId when GT is actually meant to be
    // visible — same gate as the main viewer.
    if (!gtEditMode && !showGroundtruth) return null;
    const url = caseGroundtruthUrl(
      selectedCase.case_id,
      gtActiveRevisionId === null ? "active" : gtActiveRevisionId,
    );
    // Match the main viewer's edit-mode cache-buster so the editable
    // labelmap doesn't come back from a pre-edit cache hit.
    const editSuffix = gtEditMode ? (url.includes("?") ? "&edit=1" : "?edit=1") : "";
    return `nifti:${url}${editSuffix}`;
  }, [selectedCase, gtActiveRevisionId, gtEditMode, showGroundtruth]);

  if (!open || !mounted || !selectedCase) return null;

  const portalRoot = document.body;
  if (!portalRoot) return null;

  return createPortal(
    <div
      className="fixed inset-0 z-[100] flex flex-col bg-[var(--color-rt-paper)] text-[var(--color-rt-ink)]"
      role="dialog"
      aria-modal="true"
      aria-label="Tri-planar MPR viewer"
    >
      <FullscreenToolbar
        caseId={selectedCase.case_id}
        hasGroundtruth={!!selectedCase.groundtruth_path}
      />

      <div className="relative flex-1 min-h-0 bg-[color-mix(in_oklab,var(--color-rt-mist)_30%,var(--color-rt-paper))]">
        {imageVolumeId ? (
          <div className="grid h-full min-h-0 grid-cols-1 gap-2 p-2 md:grid-cols-3">
            {(["axial", "coronal", "sagittal"] as const).map((o) => (
              <div
                key={o}
                className="overflow-hidden rounded-[var(--radius-rt-sm)] border border-[var(--color-rt-line)] bg-black/2 shadow-[var(--shadow-rt-elevation-1)]"
              >
                <MprViewport
                  orientation={o}
                  imageVolumeId={imageVolumeId}
                  segmentationVolumeId={segVolumeId}
                  groundtruthVolumeId={gtVolumeId}
                />
              </div>
            ))}
          </div>
        ) : (
          <div className="flex h-full items-center justify-center text-[var(--color-rt-muted)]">
            <Loader2 className="mr-2 animate-spin" size={14} />
            preparing volume…
          </div>
        )}

        {gtEditMode && (
          <GtEditToolbar
            caseLabelMap={prediction?.label_map ?? { background: 0 }}
            caseId={selectedCase.case_id}
            basePredictionId={prediction?.prediction_id ?? null}
            onSaved={(rev) => {
              setGtRevisions(
                [
                  {
                    id: rev.id,
                    region: rev.region,
                    case_id: rev.case_id,
                    revision: rev.revision,
                    path: rev.path,
                    base_prediction_id: rev.base_prediction_id,
                    reviewer: rev.reviewer,
                    notes: rev.notes,
                    status: rev.status as "active",
                    created_at: rev.created_at,
                  },
                ],
                rev.id,
              );
            }}
          />
        )}
      </div>

      <FullscreenFooter
        prediction={prediction}
        onClose={close}
      />
    </div>,
    portalRoot,
  );
}

// Thin bottom strip with the case identity + the worst-3 dice scores so
// the operator never loses context while painting in fullscreen. Stays
// shy — single line, hairline rule, monospace numerals.
function FullscreenFooter({
  prediction,
  onClose,
}: {
  prediction: ReturnType<typeof useQAStore.getState>["currentPrediction"];
  onClose: () => void;
}) {
  const selectedCase = useQAStore((s) => s.selectedCase);
  const selectedModel = useQAStore((s) => s.selectedModel);

  const summary = useMemo(() => {
    const m = prediction?.metrics;
    if (!m || m.length === 0) return null;
    const valid = m.filter(
      (x) => x.dice !== null && !Number.isNaN(x.dice),
    ) as Array<typeof m[number] & { dice: number }>;
    if (valid.length === 0) return null;
    const mean =
      valid.reduce((acc, v) => acc + v.dice, 0) / valid.length;
    const worst = [...valid].sort((a, b) => a.dice - b.dice).slice(0, 3);
    return { mean, worst };
  }, [prediction]);

  return (
    <div className="flex shrink-0 items-center justify-between gap-3 border-t border-[var(--color-rt-line)] bg-[var(--color-rt-paper)] px-3 py-1.5 text-[11px]">
      <div className="flex min-w-0 items-center gap-3 truncate">
        {selectedModel && (
          <span className="rt-display font-semibold text-[var(--color-rt-ink)]">
            {selectedModel.dataset_name.replace(/^Dataset(\d+)_/, "D$1 ")}
          </span>
        )}
        {selectedCase && (
          <span className="font-mono text-[10.5px] text-[var(--color-rt-muted)]">
            {selectedCase.case_id}
          </span>
        )}
      </div>
      {summary && (
        <div className="flex items-baseline gap-3 text-[10.5px] text-[var(--color-rt-muted)]">
          <span>
            mean dice{" "}
            <span className="ml-1 font-mono text-[var(--color-rt-ink)]">
              {summary.mean.toFixed(3)}
            </span>
          </span>
          <span className="h-3 w-px bg-[var(--color-rt-line)]" />
          <span>
            worst:{" "}
            <span className="font-mono">
              {summary.worst
                .map((w) => `${w.label_name} ${w.dice.toFixed(2)}`)
                .join(" · ")}
            </span>
          </span>
        </div>
      )}
      <button
        type="button"
        onClick={onClose}
        className="font-mono text-[10.5px] text-[var(--color-rt-muted)] underline-offset-2 hover:text-[var(--color-rt-ink)] hover:underline"
      >
        Esc to close
      </button>
    </div>
  );
}
