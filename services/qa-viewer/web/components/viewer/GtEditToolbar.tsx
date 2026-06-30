"use client";

import { Loader2, Redo2, Save, Undo2, X } from "lucide-react";
import { Fragment, useEffect, useMemo, useState } from "react";

import {
  postGtEdit,
  type GtEditSidecar,
  type GroundTruthRevision,
} from "@/lib/api";
import { useQAStore } from "@/lib/store";
import type { GtEditTool } from "@/lib/store";
import { cn } from "@/lib/utils";

// Cornerstone-style label palette (Cornerstone3D's default segmentation
// colour LUT) — used purely as the visual swatch in the label legend so
// reviewers can match the picker to what they see on the viewport. Kept
// in sync by index with cornerstone-tools' getColorLut() output. Falls
// back to a neutral grey beyond the array.
const LABEL_SWATCHES = [
  "#888888", // 0 = background
  "#ef4444", "#f97316", "#eab308", "#22c55e",
  "#06b6d4", "#3b82f6", "#8b5cf6", "#ec4899",
  "#10b981", "#f59e0b", "#a855f7", "#84cc16",
  "#14b8a6", "#6366f1", "#d946ef", "#fb7185",
];

function labelColorFor(idx: number): string {
  return LABEL_SWATCHES[idx] ?? LABEL_SWATCHES[(idx % (LABEL_SWATCHES.length - 1)) + 1];
}

import {
  TOOL_BY_KEY,
  TOOL_CATALOG,
  TOOL_GROUP_ORDER,
  type ToolEntry,
} from "./toolCatalog";
import { getNiftiViewerHandle } from "./viewerHandle";

// Floating top-centre toolbar shown while gtEditMode is true. Drives the
// active tool, brush size, segment-index, undo/redo, and the save flow.
// Save reads the current scalar buffer from the cornerstone GT volume
// (via the imperative handle in NiftiViewer) and POSTs to
// /api/cases/.../groundtruth/edits.
export function GtEditToolbar({
  caseLabelMap,
  caseId,
  basePredictionId,
  onSaved,
}: {
  caseLabelMap: Record<string, number>;
  caseId: string;
  basePredictionId: string | null;
  onSaved: (rev: GroundTruthRevision) => void;
}) {
  const tool = useQAStore((s) => s.gtActiveTool);
  const setTool = useQAStore((s) => s.setGtActiveTool);
  const brushSize = useQAStore((s) => s.gtBrushSize);
  const setBrushSize = useQAStore((s) => s.setGtBrushSize);
  const segIdx = useQAStore((s) => s.gtActiveSegmentIndex);
  const setSegIdx = useQAStore((s) => s.setGtActiveSegmentIndex);
  const undoDepth = useQAStore((s) => s.gtUndoStack.length);
  const redoDepth = useQAStore((s) => s.gtRedoStack.length);
  const dirty = useQAStore((s) => s.gtDirty);
  const saving = useQAStore((s) => s.gtSaving);
  const saveError = useQAStore((s) => s.gtSaveError);
  const cancel = useQAStore((s) => s.cancelGtEdit);
  const finish = useQAStore((s) => s.finishGtEdit);
  const beginSave = useQAStore((s) => s.beginGtSave);
  const finishSave = useQAStore((s) => s.finishGtSave);
  const reviewer = useQAStore((s) => s.reviewer);
  const activeRevisionId = useQAStore((s) => s.gtActiveRevisionId);
  const toolsUsed = useState<Set<GtEditTool>>(() => new Set([tool]))[0];

  toolsUsed.add(tool);

  const [notes, setNotes] = useState("");

  const segmentEntries = useMemo(() => {
    return Object.entries(caseLabelMap)
      .filter(([k]) => k !== "background")
      .sort((a, b) => a[1] - b[1]);
  }, [caseLabelMap]);

  // Keyboard shortcuts for fast contouring. Bound at document level so
  // they work no matter which viewport has focus. Suppress while the
  // user is typing in any text input (notes box, reviewer name, etc.) —
  // otherwise typing "b" in the verdict notes would switch to brush.
  useEffect(() => {
    function onKey(e: KeyboardEvent) {
      const tag = (e.target as HTMLElement | null)?.tagName;
      if (tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT") return;
      // Cmd/Ctrl-Z = undo, Cmd/Ctrl-Shift-Z = redo. Don't trigger on
      // page-wide undo when the user is in a different context.
      if ((e.metaKey || e.ctrlKey) && (e.key === "z" || e.key === "Z")) {
        e.preventDefault();
        if (e.shiftKey) {
          window.dispatchEvent(new CustomEvent("qa-gt-redo"));
        } else {
          window.dispatchEvent(new CustomEvent("qa-gt-undo"));
        }
        return;
      }
      if (e.metaKey || e.ctrlKey || e.altKey) return;
      switch (e.key) {
        case "b": setTool("brush2D"); break;
        case "B": setTool("brush3D"); break;
        case "e": setTool("eraser2D"); break;
        case "E": setTool("eraser3D"); break;
        case "t": setTool("thresholdBrush"); break;
        case "s": setTool("sphereScissors"); break;
        case "r": setTool("rectScissors"); break;
        case "c": setTool("circleScissors"); break;
        case "f": setTool("paintFill"); break;
        case "v": setTool("segSelect"); break;
        case "[":
          setBrushSize(Math.max(1, brushSize - 1));
          break;
        case "]":
          setBrushSize(Math.min(40, brushSize + 1));
          break;
        default:
          if (e.key >= "1" && e.key <= "9") {
            const idx = parseInt(e.key, 10);
            const ent = segmentEntries[idx - 1];
            if (ent) setSegIdx(ent[1]);
          }
      }
    }
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [brushSize, segmentEntries, setBrushSize, setSegIdx, setTool]);

  async function onSave() {
    const handle = getNiftiViewerHandle();
    if (!handle) return;
    const ex = handle.extractGt();
    if (!ex) {
      finishSave("could not read GT buffer from viewer");
      return;
    }
    const sidecar: GtEditSidecar = {
      schema_version: 1,
      dimensions: ex.dimensions,
      spacing: ex.spacing,
      origin: ex.origin,
      direction: ex.direction,
      dtype: ex.dtype,
      source_revision: activeRevisionId ?? "active",
      label_map: caseLabelMap,
      tools_used: Array.from(toolsUsed),
      stroke_count: undoDepth,
      notes: notes.trim(),
      base_prediction_id: basePredictionId,
      reviewer: reviewer.trim(),
    };
    beginSave();
    try {
      const saved = await postGtEdit({
        case_id: caseId,
        labelmap: ex.scalarData,
        sidecar,
      });
      onSaved(saved);
      finishSave(null);
      finish();
    } catch (e) {
      const err = e as Error & { status?: number };
      finishSave(
        err.status === 409
          ? "another reviewer saved a newer revision — reload and reapply your edits"
          : err.message ?? "save failed",
      );
    }
  }

  return (
    <div className="pointer-events-none absolute inset-x-0 top-2 z-30 flex justify-center">
      <div className="pointer-events-auto flex max-w-[calc(100%-1rem)] flex-wrap items-center gap-2 rounded-[var(--radius-rt)] border border-[var(--color-rt-line)] bg-[color-mix(in_oklab,var(--color-rt-paper)_92%,transparent)] px-2 py-1.5 shadow-[var(--shadow-rt-elevation-2)] backdrop-blur">
        <ToolStrip tool={tool} setTool={setTool} />
        <CurrentToolHint entry={TOOL_BY_KEY[tool]} />

        <div className="h-5 w-px bg-[var(--color-rt-line)]" />

        <label className="flex items-center gap-1.5 text-[10.5px] text-[var(--color-rt-muted)]">
          <span className="uppercase tracking-wide">brush</span>
          <input
            type="range"
            min={1}
            max={40}
            step={1}
            value={brushSize}
            onChange={(e) => setBrushSize(parseInt(e.target.value, 10))}
            className="h-1 w-24 accent-[var(--color-rt-accent)]"
            aria-label="brush size (mm)"
          />
          <span className="w-6 font-mono text-[11px] text-[var(--color-rt-ink)]">{brushSize}</span>
        </label>

        <div className="h-5 w-px bg-[var(--color-rt-line)]" />

        <label className="flex items-center gap-1.5 text-[10.5px] text-[var(--color-rt-muted)]">
          <span className="uppercase tracking-wide">label</span>
          <select
            value={segIdx}
            onChange={(e) => setSegIdx(parseInt(e.target.value, 10))}
            className="rounded border border-[var(--color-rt-line)] bg-[var(--color-rt-paper)] px-1 py-0.5 text-[11px] text-[var(--color-rt-ink)] focus:border-[var(--color-rt-accent)] focus:outline-none"
            aria-label="active segment"
          >
            {segmentEntries.length === 0 && <option value={1}>1</option>}
            {segmentEntries.map(([name, val]) => (
              <option key={val} value={val}>
                {val} · {name}
              </option>
            ))}
          </select>
        </label>

        <div className="h-5 w-px bg-[var(--color-rt-line)]" />

        <div className="flex items-center gap-1">
          <IconBtn
            onClick={() => window.dispatchEvent(new CustomEvent("qa-gt-undo"))}
            disabled={undoDepth === 0}
            label={`Undo (${undoDepth})`}
            Icon={Undo2}
          />
          <IconBtn
            onClick={() => window.dispatchEvent(new CustomEvent("qa-gt-redo"))}
            disabled={redoDepth === 0}
            label={`Redo (${redoDepth})`}
            Icon={Redo2}
          />
        </div>

        <div className="h-5 w-px bg-[var(--color-rt-line)]" />

        <input
          type="text"
          value={notes}
          onChange={(e) => setNotes(e.target.value)}
          placeholder="notes about this edit"
          className="w-40 rounded border border-[var(--color-rt-line)] bg-[var(--color-rt-paper)] px-2 py-1 text-[11px] text-[var(--color-rt-ink)] placeholder:text-[var(--color-rt-muted)] focus:border-[var(--color-rt-accent)] focus:outline-none"
        />

        <button
          type="button"
          onClick={onSave}
          disabled={!dirty || saving}
          className={cn(
            "inline-flex items-center gap-1 rounded-[var(--radius-rt-sm)] px-2.5 py-1 text-[11.5px] font-semibold transition-colors",
            dirty && !saving
              ? "bg-[var(--color-rt-accent)] text-white hover:bg-[var(--color-rt-accent-2)]"
              : "cursor-not-allowed bg-[var(--color-rt-mist)] text-[var(--color-rt-muted)]",
          )}
        >
          {saving ? <Loader2 className="animate-spin" size={12} /> : <Save size={12} />}
          save
        </button>

        <button
          type="button"
          onClick={cancel}
          className="inline-flex items-center gap-1 rounded-[var(--radius-rt-sm)] border border-[var(--color-rt-line)] px-2 py-1 text-[11px] text-[var(--color-rt-muted)] hover:bg-[var(--color-rt-mist)] hover:text-[var(--color-rt-ink)]"
        >
          <X size={12} />
          cancel
        </button>

        {saveError && (
          <div className="basis-full text-[11px] text-[var(--color-rt-pip-error)]">
            {saveError}
          </div>
        )}

        {segmentEntries.length > 1 && (
          <LabelLegend
            entries={segmentEntries}
            activeIdx={segIdx}
            onPick={setSegIdx}
          />
        )}
      </div>
    </div>
  );
}

// Per-label colour swatches with the active label highlighted. Clicking a
// row sets the active segment. Keyboard `1`-`9` selects by row index.
// Replaces the plain <select> as the primary picker once a model has 2+
// foreground classes — the <select> stays in the toolbar for accessibility
// and as a fallback.
function LabelLegend({
  entries,
  activeIdx,
  onPick,
}: {
  entries: [string, number][];
  activeIdx: number;
  onPick: (n: number) => void;
}) {
  return (
    <div className="basis-full">
      <div className="mb-0.5 text-[9.5px] uppercase tracking-[0.16em] text-[var(--color-rt-muted)]">
        labels · press 1-9 to switch
      </div>
      <div className="flex flex-wrap gap-1">
        {entries.map(([name, val], i) => {
          const active = val === activeIdx;
          return (
            <button
              key={val}
              type="button"
              onClick={() => onPick(val)}
              className={cn(
                "inline-flex items-center gap-1 rounded-[var(--radius-rt-sm)] border px-1.5 py-0.5 text-[10.5px] transition-colors",
                active
                  ? "border-[var(--color-rt-accent)] bg-[color-mix(in_oklab,var(--color-rt-accent)_10%,var(--color-rt-paper))] text-[var(--color-rt-ink)]"
                  : "border-[var(--color-rt-line)] text-[var(--color-rt-muted)] hover:bg-[var(--color-rt-mist)]",
              )}
              title={`label ${val} · ${name}`}
            >
              <span
                aria-hidden
                className="h-2 w-2 rounded-sm"
                style={{ backgroundColor: labelColorFor(val) }}
              />
              {i < 9 && (
                <span className="font-mono text-[9.5px] opacity-70">{i + 1}</span>
              )}
              <span className="truncate max-w-[16ch]">{name}</span>
            </button>
          );
        })}
      </div>
    </div>
  );
}

/** Renders the full tool catalog with thin dividers between groups so
 *  paint / erase / smart / fill / select read as distinct clusters. */
function ToolStrip({
  tool,
  setTool,
}: {
  tool: GtEditTool;
  setTool: (t: GtEditTool) => void;
}) {
  return (
    <div className="flex items-center gap-1">
      {TOOL_GROUP_ORDER.map((group, gi) => (
        <Fragment key={group}>
          {gi > 0 && <span className="h-5 w-px bg-[var(--color-rt-line)]" />}
          <div className="flex items-center gap-1">
            {TOOL_CATALOG.filter((t) => t.group === group).map((entry) => (
              <ToolButton
                key={entry.key}
                entry={entry}
                active={tool === entry.key}
                onClick={() => setTool(entry.key)}
              />
            ))}
          </div>
        </Fragment>
      ))}
    </div>
  );
}

function CurrentToolHint({ entry }: { entry: ToolEntry }) {
  if (!entry?.hint) return null;
  return (
    <span
      className="hidden max-w-[28ch] truncate text-[10.5px] text-[var(--color-rt-muted)] lg:inline"
      title={entry.hint}
    >
      {entry.hint}
    </span>
  );
}

function ToolButton({
  entry,
  active,
  onClick,
}: {
  entry: ToolEntry;
  active: boolean;
  onClick: () => void;
}) {
  const { Icon, dim, label } = entry;
  return (
    <button
      type="button"
      onClick={onClick}
      title={label}
      aria-label={label}
      aria-pressed={active}
      className={cn(
        "relative inline-flex h-7 w-7 items-center justify-center rounded-[var(--radius-rt-sm)] border transition-colors",
        active
          ? "border-[var(--color-rt-accent)] bg-[color-mix(in_oklab,var(--color-rt-accent)_12%,var(--color-rt-paper))] text-[var(--color-rt-accent)]"
          : "border-[var(--color-rt-line)] text-[var(--color-rt-muted)] hover:bg-[var(--color-rt-mist)] hover:text-[var(--color-rt-ink)]",
      )}
    >
      <Icon size={13} />
      {dim && (
        <span
          aria-hidden
          className={cn(
            "pointer-events-none absolute -bottom-0.5 -right-0.5 rounded bg-[var(--color-rt-paper)] px-px font-mono leading-none",
            "text-[7.5px] tracking-tight",
            active ? "text-[var(--color-rt-accent)]" : "text-[var(--color-rt-muted)]",
          )}
        >
          {dim}
        </span>
      )}
    </button>
  );
}

function IconBtn({
  onClick,
  disabled,
  label,
  Icon,
}: {
  onClick: () => void;
  disabled?: boolean;
  label: string;
  Icon: React.ComponentType<{ size?: number }>;
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      disabled={disabled}
      title={label}
      aria-label={label}
      className={cn(
        "inline-flex h-7 w-7 items-center justify-center rounded-[var(--radius-rt-sm)] border border-[var(--color-rt-line)] transition-colors",
        disabled
          ? "cursor-not-allowed text-[var(--color-rt-muted)] opacity-50"
          : "text-[var(--color-rt-muted)] hover:bg-[var(--color-rt-mist)] hover:text-[var(--color-rt-ink)]",
      )}
    >
      <Icon size={13} />
    </button>
  );
}
