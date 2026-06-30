"use client";

import { ArrowLeft, BarChart3 } from "lucide-react";
import { useEffect } from "react";

import { InferencePanel } from "@/components/shell/InferencePanel";
import { ModelSidebar } from "@/components/shell/ModelSidebar";
import { ViewerStage } from "@/components/viewer/ViewerStage";
import { crossvalRollupHtmlUrl } from "@/lib/api";
import { useQAStore } from "@/lib/store";

import { useViewerHotkeys } from "./useViewerHotkeys";

export function Workspace() {
  const selectedModel = useQAStore((s) => s.selectedModel);
  const exitToCatalog = useQAStore((s) => s.exitToCatalog);
  const leftOpen = useQAStore((s) => s.leftSidebarOpen);
  const rightOpen = useQAStore((s) => s.rightSidebarOpen);
  const viewerFocus = useQAStore((s) => s.viewerFocus);
  const gtDirty = useQAStore((s) => s.gtDirty);

  useViewerHotkeys();

  // Sync the <html> data attributes whenever layout state changes. CSS hooks
  // off these for the grid template so the page reflows synchronously.
  useEffect(() => {
    const root = document.documentElement;
    root.setAttribute("data-layout-left", leftOpen ? "open" : "closed");
    root.setAttribute("data-layout-right", rightOpen ? "open" : "closed");
    root.setAttribute("data-layout-focus", viewerFocus ? "on" : "off");
  }, [leftOpen, rightOpen, viewerFocus]);

  // Don't let the operator navigate away with unsaved GT edits. The
  // returned string is shown by older browsers; modern browsers display a
  // generic prompt regardless.
  useEffect(() => {
    if (!gtDirty) return;
    const handler = (e: BeforeUnloadEvent) => {
      e.preventDefault();
      e.returnValue = "You have unsaved ground-truth edits.";
      return e.returnValue;
    };
    window.addEventListener("beforeunload", handler);
    return () => window.removeEventListener("beforeunload", handler);
  }, [gtDirty]);

  // Tailwind can't generate dynamic arbitrary grid templates at runtime, so
  // we resolve to a static class set from the four valid combos. Hidden
  // sidebars get `0fr` (collapse to zero) plus `overflow-hidden` via the
  // <aside> wrapper. viewerFocus overrides everything.
  let gridClasses: string;
  if (viewerFocus) {
    gridClasses = "grid-cols-[0_minmax(0,1fr)_0]";
  } else if (leftOpen && rightOpen) {
    gridClasses = "grid-cols-[280px_minmax(0,1fr)_360px]";
  } else if (!leftOpen && rightOpen) {
    gridClasses = "grid-cols-[0_minmax(0,1fr)_360px]";
  } else if (leftOpen && !rightOpen) {
    gridClasses = "grid-cols-[280px_minmax(0,1fr)_0]";
  } else {
    gridClasses = "grid-cols-[0_minmax(0,1fr)_0]";
  }

  const hideSidebars = viewerFocus;

  return (
    <div className="flex min-h-0 flex-1 flex-col gap-3">
      {!viewerFocus && (
        <div className="flex items-center justify-between gap-3 rounded-[var(--radius-rt)] border border-[var(--color-rt-line)] bg-[var(--color-rt-paper)] px-2.5 py-1.5 shadow-[var(--shadow-rt-elevation-1)]">
          <button
            type="button"
            onClick={exitToCatalog}
            className="inline-flex items-center gap-1.5 rounded-full border border-[var(--color-rt-line)] bg-[var(--color-rt-paper)] px-2.5 py-1 text-[12px] font-medium text-[var(--color-rt-muted)] transition-colors hover:border-[var(--color-rt-line-2)] hover:bg-[var(--color-rt-mist)] hover:text-[var(--color-rt-ink)]"
          >
            <ArrowLeft size={13} />
            catalog
          </button>
          {selectedModel && (
            <>
              <div className="flex min-w-0 items-center gap-2.5">
                <span className="rt-display truncate text-[13px] font-semibold text-[var(--color-rt-ink)]">
                  {selectedModel.dataset_name.replace(/^Dataset(\d+)_/, "D$1 ")}
                </span>
                <span className="hidden items-center gap-1.5 sm:flex">
                  {[
                    selectedModel.plans.replace("nnUNet", "").replace("Plans", "") || "Plans",
                    selectedModel.configuration,
                  ].map((t) => (
                    <span
                      key={t}
                      className="inline-flex items-center rounded-md border border-[var(--color-rt-line)] bg-[var(--color-rt-mist)] px-1.5 py-0.5 font-mono text-[10px] leading-none text-[var(--color-rt-ink-2)]"
                    >
                      {t}
                    </span>
                  ))}
                </span>
              </div>
              {/* Model-level cross-validation rollup — server-rendered,
                  self-contained HTML (prints to PDF). Opens in a new tab. */}
              <a
                href={crossvalRollupHtmlUrl(selectedModel.model_id)}
                target="_blank"
                rel="noopener noreferrer"
                title="Open the cross-validation rollup report for this model"
                className="inline-flex shrink-0 items-center gap-1.5 rounded-full border border-[var(--color-rt-line)] bg-[var(--color-rt-paper)] px-2.5 py-1 text-[12px] font-medium text-[var(--color-rt-muted)] transition-colors hover:border-[var(--color-rt-line-2)] hover:bg-[var(--color-rt-mist)] hover:text-[var(--color-rt-ink)]"
              >
                <BarChart3 size={13} />
                CV report
              </a>
            </>
          )}
        </div>
      )}

      <div
        className={`grid min-h-0 flex-1 gap-3 transition-[grid-template-columns] duration-200 ease-[var(--ease-rt)] ${gridClasses}`}
      >
        <aside
          aria-hidden={!leftOpen || hideSidebars}
          // `inert` (React 19) pulls the collapsed panel out of BOTH the a11y
          // tree and the keyboard tab order — aria-hidden alone leaves its
          // buttons focusable, trapping focus in a zero-width hidden region.
          inert={!leftOpen || hideSidebars}
          className="flex min-h-0 flex-col overflow-hidden"
        >
          <ModelSidebar />
        </aside>
        <ViewerStage />
        <aside
          aria-hidden={!rightOpen || hideSidebars}
          inert={!rightOpen || hideSidebars}
          className="flex min-h-0 flex-col overflow-hidden"
        >
          <InferencePanel />
        </aside>
      </div>
    </div>
  );
}
