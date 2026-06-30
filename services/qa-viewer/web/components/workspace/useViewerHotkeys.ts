"use client";

import { useEffect } from "react";

import { useQAStore } from "@/lib/store";

// Centralized keyboard shortcuts for the workspace.
//
//   F          toggle viewer focus mode (hides both sidebars)
//   M          open / close fullscreen MPR (tri-planar) overlay
//   V          open / close fullscreen 3D (surface mesh) overlay
//   Esc        close 3D → close MPR → cancel GT edit → exit focus
//   Ctrl-Z     undo last GT brush stroke   (forwarded via custom event)
//   Ctrl-⇧-Z   redo                        (forwarded via custom event)
//   [   ]      shrink / grow the active brush
//
// Undo/redo are dispatched as `qa-gt-undo` / `qa-gt-redo` CustomEvents on
// `window` so the NiftiViewer component (which owns the Cornerstone scalar
// buffer) can subscribe via its own effect — keeping all DOM/Cornerstone
// interactions out of the store.
export function useViewerHotkeys() {
  const toggleFocus = useQAStore((s) => s.toggleViewerFocus);
  const exitFocus = useQAStore((s) => s.exitViewerFocus);
  const toggleMpr = useQAStore((s) => s.toggleFullscreenMpr);
  const closeMpr = useQAStore((s) => s.closeFullscreenMpr);
  const toggleVol = useQAStore((s) => s.toggleVolume3D);
  const closeVol = useQAStore((s) => s.closeVolume3D);
  const cancelGt = useQAStore((s) => s.cancelGtEdit);
  const setBrushSize = useQAStore((s) => s.setGtBrushSize);

  useEffect(() => {
    function onKey(e: KeyboardEvent) {
      // Ignore when the user is typing into an input/textarea — otherwise
      // pressing "f" while writing a notes field flips the layout.
      const tgt = e.target as HTMLElement | null;
      if (
        tgt &&
        (tgt.tagName === "INPUT" ||
          tgt.tagName === "TEXTAREA" ||
          tgt.tagName === "SELECT" ||
          tgt.isContentEditable)
      ) {
        return;
      }

      const { gtEditMode, viewerFocus, gtBrushSize, fullscreenMpr, volume3DOpen, currentPrediction } =
        useQAStore.getState();

      // Esc — close 3D first, then MPR, then GT edit, then focus.
      if (e.key === "Escape") {
        if (volume3DOpen) {
          e.preventDefault();
          closeVol();
          return;
        }
        if (fullscreenMpr) {
          e.preventDefault();
          closeMpr();
          return;
        }
        if (gtEditMode) {
          e.preventDefault();
          cancelGt();
          return;
        }
        if (viewerFocus) {
          e.preventDefault();
          exitFocus();
          return;
        }
      }

      if (e.key === "f" || e.key === "F") {
        e.preventDefault();
        toggleFocus();
        return;
      }

      if (e.key === "m" || e.key === "M") {
        e.preventDefault();
        toggleMpr();
        return;
      }

      // V — toggle 3D, but only when there's actually a prediction to render.
      if (e.key === "v" || e.key === "V") {
        if (currentPrediction?.seg_url || volume3DOpen) {
          e.preventDefault();
          toggleVol();
        }
        return;
      }

      // Undo/redo. Only meaningful while editing, but we always dispatch
      // so the viewer can decide whether to consume.
      if ((e.metaKey || e.ctrlKey) && (e.key === "z" || e.key === "Z")) {
        if (gtEditMode) {
          e.preventDefault();
          window.dispatchEvent(
            new CustomEvent(e.shiftKey ? "qa-gt-redo" : "qa-gt-undo"),
          );
        }
        return;
      }

      // Brush size with the bracket keys.
      if (gtEditMode && (e.key === "[" || e.key === "]")) {
        e.preventDefault();
        const delta = e.key === "[" ? -1 : 1;
        const next = Math.max(1, Math.min(40, gtBrushSize + delta));
        setBrushSize(next);
      }
    }
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [toggleFocus, exitFocus, toggleMpr, closeMpr, toggleVol, closeVol, cancelGt, setBrushSize]);
}
