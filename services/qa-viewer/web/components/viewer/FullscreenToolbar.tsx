"use client";

// Top toolbar for the fullscreen MPR overlay. Provides:
//   - GT revision picker (a real popover — the previous show/hide button
//     never told the operator which revision they were looking at)
//   - Quick-access contouring tools that enter GT edit mode on click
//   - Crosshair link toggle (only meaningful with CrosshairsTool registered)
//   - Window/Level reset
//   - Exit fullscreen
//
// The floating <GtEditToolbar /> still renders inside FullscreenStage
// while gtEditMode is on, providing brush size, save/cancel, undo/redo,
// and notes. The buttons here are duplicated by design — they let the
// operator start editing without first finding the small "correct GT"
// button on the right sidebar (which doesn't exist in fullscreen).

import * as cornerstone from "@cornerstonejs/core";
import * as cornerstoneTools from "@cornerstonejs/tools";
import { Crosshair, Eye, EyeOff, Focus, Pencil, X } from "lucide-react";
import { Fragment, useCallback } from "react";

import { useQAStore } from "@/lib/store";
import type { GtEditTool } from "@/lib/store";
import { cn } from "@/lib/utils";

import { IconToggle } from "@/components/shell/IconToggle";

import {
  QA_CROSSHAIRS_TOOL_NAME,
  QA_RENDERING_ENGINE_ID,
  QA_MPR_VIEWPORT_IDS,
  QA_TOOL_GROUP_ID,
} from "./cornerstoneInit";
import { GtRevisionSelect } from "./GtRevisionSelect";
import { TOOL_CATALOG, TOOL_GROUP_ORDER } from "./toolCatalog";

export function FullscreenToolbar({
  caseId,
  hasGroundtruth,
}: {
  caseId: string;
  hasGroundtruth: boolean;
}) {
  const gtEditMode = useQAStore((s) => s.gtEditMode);
  const gtActiveTool = useQAStore((s) => s.gtActiveTool);
  const setGtActiveTool = useQAStore((s) => s.setGtActiveTool);
  const enterGtEdit = useQAStore((s) => s.enterGtEdit);
  const cancelGtEdit = useQAStore((s) => s.cancelGtEdit);
  const showGroundtruth = useQAStore((s) => s.showGroundtruth);
  const toggleGroundtruth = useQAStore((s) => s.toggleGroundtruth);
  const crosshairsLinked = useQAStore((s) => s.mprCrosshairsLinked);
  const setCrosshairsLinked = useQAStore((s) => s.setMprCrosshairsLinked);
  const closeFullscreen = useQAStore((s) => s.closeFullscreenMpr);

  const pickTool = useCallback(
    (t: GtEditTool) => {
      if (!hasGroundtruth) return;
      if (!gtEditMode) {
        enterGtEdit();
      }
      setGtActiveTool(t);
    },
    [enterGtEdit, gtEditMode, hasGroundtruth, setGtActiveTool],
  );

  const toggleCrosshairs = useCallback(() => {
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const ct: any = cornerstoneTools;
    const name = QA_CROSSHAIRS_TOOL_NAME;
    if (!name) {
      // Tool not available on this runtime — user clicked anyway, surface
      // via the console only (no UI toast machinery in this codebase).
      console.warn("[FullscreenToolbar] CrosshairsTool not registered");
      return;
    }
    const group = ct.ToolGroupManager.getToolGroup(QA_TOOL_GROUP_ID);
    if (!group) return;
    const MB = ct.Enums.MouseBindings;
    try {
      if (crosshairsLinked) {
        group.setToolPassive(name);
      } else {
        group.setToolActive(name, {
          bindings: [{ mouseButton: MB.Primary }],
        });
      }
      setCrosshairsLinked(!crosshairsLinked);
    } catch (err) {
      console.warn("[FullscreenToolbar] crosshair toggle failed:", err);
    }
  }, [crosshairsLinked, setCrosshairsLinked]);

  const resetWindowLevel = useCallback(() => {
    const engine = cornerstone.getRenderingEngine(QA_RENDERING_ENGINE_ID);
    if (!engine) return;
    Object.values(QA_MPR_VIEWPORT_IDS).forEach((vpid) => {
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      const vp = engine.getViewport(vpid) as any;
      try {
        vp?.resetCamera?.(false, true, false);
        vp?.resetProperties?.();
        vp?.render?.();
      } catch {
        /* viewport may have just been disposed */
      }
    });
  }, []);

  return (
    <div className="flex shrink-0 items-center justify-between gap-3 border-b border-[var(--color-rt-line)] bg-[var(--color-rt-paper)] px-3 py-2">
      {/* Left cluster — GT identity */}
      <div className="flex min-w-0 items-center gap-2">
        <span className="rt-display text-[10px] font-semibold uppercase tracking-[0.16em] text-[var(--color-rt-muted)]">
          MPR · tri-planar
        </span>
        <span className="h-4 w-px bg-[var(--color-rt-line)]" />
        {hasGroundtruth ? (
          <>
            <GtRevisionSelect caseId={caseId} compact />
            <IconToggle
              onClick={toggleGroundtruth}
              Icon={showGroundtruth ? Eye : EyeOff}
              label={showGroundtruth ? "Hide GT overlay" : "Show GT overlay"}
              active={showGroundtruth}
            />
          </>
        ) : (
          <span className="text-[11px] italic text-[var(--color-rt-muted)]">
            no groundtruth on this case
          </span>
        )}
      </div>

      {/* Middle cluster — contouring tools, grouped by purpose */}
      <div className="flex items-center gap-1.5">
        {TOOL_GROUP_ORDER.map((group, gi) => (
          <Fragment key={group}>
            {gi > 0 && <span className="h-5 w-px bg-[var(--color-rt-line)]" />}
            <div className="flex items-center gap-1">
              {TOOL_CATALOG.filter((t) => t.group === group).map((entry) => {
                const { key, Icon, dim, label } = entry;
                const isActive = gtEditMode && gtActiveTool === key;
                return (
                  <button
                    key={key}
                    type="button"
                    disabled={!hasGroundtruth}
                    onClick={() => pickTool(key)}
                    title={label}
                    aria-label={label}
                    aria-pressed={isActive}
                    className={cn(
                      "relative inline-flex h-8 w-8 items-center justify-center rounded-[var(--radius-rt-sm)] border transition-colors",
                      !hasGroundtruth
                        ? "cursor-not-allowed border-[var(--color-rt-line)] text-[var(--color-rt-muted)] opacity-50"
                        : isActive
                          ? "border-[var(--color-rt-accent)] bg-[color-mix(in_oklab,var(--color-rt-accent)_14%,var(--color-rt-paper))] text-[var(--color-rt-accent)]"
                          : "border-[var(--color-rt-line)] text-[var(--color-rt-muted)] hover:bg-[var(--color-rt-mist)] hover:text-[var(--color-rt-ink)]",
                    )}
                  >
                    <Icon size={14} />
                    {dim && (
                      <span
                        aria-hidden
                        className={cn(
                          "pointer-events-none absolute -bottom-0.5 -right-0.5 rounded bg-[var(--color-rt-paper)] px-px font-mono leading-none",
                          "text-[8px] tracking-tight",
                          isActive ? "text-[var(--color-rt-accent)]" : "text-[var(--color-rt-muted)]",
                        )}
                      >
                        {dim}
                      </span>
                    )}
                  </button>
                );
              })}
            </div>
          </Fragment>
        ))}
        {!gtEditMode && hasGroundtruth && (
          <button
            type="button"
            onClick={() => enterGtEdit()}
            className="ml-1 inline-flex items-center gap-1 rounded-[var(--radius-rt-sm)] border border-[var(--color-rt-line)] px-2 py-1 text-[11.5px] font-medium text-[var(--color-rt-accent)] transition-colors hover:bg-[color-mix(in_oklab,var(--color-rt-accent)_10%,var(--color-rt-paper))]"
            title="Open the GT correction toolbar"
          >
            <Pencil size={12} />
            edit GT
          </button>
        )}
        {gtEditMode && (
          <button
            type="button"
            onClick={() => cancelGtEdit()}
            className="ml-1 inline-flex items-center gap-1 rounded-[var(--radius-rt-sm)] border border-[var(--color-rt-line)] px-2 py-1 text-[11.5px] text-[var(--color-rt-muted)] transition-colors hover:bg-[var(--color-rt-mist)] hover:text-[var(--color-rt-ink)]"
            title="Exit GT edit mode (asks to confirm if dirty)"
          >
            <X size={12} />
            exit edit
          </button>
        )}
      </div>

      {/* Right cluster — viewport controls */}
      <div className="flex items-center gap-2">
        <IconToggle
          onClick={resetWindowLevel}
          Icon={Focus}
          label="Reset window / level on all panes"
        />
        <IconToggle
          onClick={toggleCrosshairs}
          Icon={Crosshair}
          label={
            crosshairsLinked
              ? "Unlink crosshair (independent panes)"
              : "Link crosshair across panes"
          }
          active={crosshairsLinked}
        />
        <span className="h-5 w-px bg-[var(--color-rt-line)]" />
        <button
          type="button"
          onClick={() => closeFullscreen()}
          className="inline-flex h-8 items-center gap-1 rounded-[var(--radius-rt-sm)] border border-[var(--color-rt-line)] px-2.5 text-[11.5px] text-[var(--color-rt-muted)] transition-colors hover:bg-[var(--color-rt-mist)] hover:text-[var(--color-rt-ink)]"
          title="Exit fullscreen (Esc)"
        >
          <X size={13} />
          exit
        </button>
      </div>
    </div>
  );
}
