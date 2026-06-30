"use client";

// Popover-style picker over the ground-truth revisions for the current case.
// Reads `gtRevisions` from the store (already populated by the GT-save flow
// in ViewerStage's onSaved callback) and lets the operator switch between
// the seed labelmap and any reviewer-saved revision.
//
// The previous experience was a binary show/hide button next to the
// opacity slider — invisible to anyone who didn't know revisions existed.

import { Check, ChevronDown, History, Loader2 } from "lucide-react";
import { useCallback, useEffect, useRef, useState } from "react";

import { activateGtRevision, type GroundTruthRevision } from "@/lib/api";
import { useQAStore } from "@/lib/store";
import { cn } from "@/lib/utils";

export function GtRevisionSelect({
  caseId,
  compact = false,
}: {
  caseId: string;
  /** Compact mode shrinks the trigger to fit inside the fullscreen toolbar. */
  compact?: boolean;
}) {
  const revisions = useQAStore((s) => s.gtRevisions);
  const activeId = useQAStore((s) => s.gtActiveRevisionId);
  const setGtRevisions = useQAStore((s) => s.setGtRevisions);
  const [open, setOpen] = useState(false);
  const [busy, setBusy] = useState<number | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const rootRef = useRef<HTMLDivElement | null>(null);

  // Close on outside click. We use a capture-phase listener so the click
  // that opens *another* popover anywhere on the page also closes us.
  useEffect(() => {
    if (!open) return;
    function onClick(e: MouseEvent) {
      if (!rootRef.current) return;
      if (rootRef.current.contains(e.target as Node)) return;
      setOpen(false);
    }
    function onKey(e: KeyboardEvent) {
      if (e.key === "Escape") {
        setOpen(false);
      }
    }
    document.addEventListener("mousedown", onClick, true);
    document.addEventListener("keydown", onKey);
    return () => {
      document.removeEventListener("mousedown", onClick, true);
      document.removeEventListener("keydown", onKey);
    };
  }, [open]);

  const active =
    revisions.find((r) => r.id === activeId) ??
    revisions.find((r) => r.status === "active") ??
    null;

  const onPick = useCallback(
    async (rev: GroundTruthRevision) => {
      if (rev.id === active?.id) {
        setOpen(false);
        return;
      }
      setBusy(rev.id);
      setErr(null);
      try {
        const updated = await activateGtRevision(caseId, rev.id);
        // Mark this revision as active, demote the others. The server-side
        // is authoritative, but we mirror locally for instant UI feedback.
        setGtRevisions(
          revisions.map((r) => ({
            ...r,
            status: r.id === updated.id ? "active" : "superseded",
          })),
          updated.id,
        );
        setOpen(false);
      } catch (e) {
        setErr((e as Error).message ?? "activate failed");
      } finally {
        setBusy(null);
      }
    },
    [active, caseId, revisions, setGtRevisions],
  );

  // Even when no revisions exist (seed-only), we render the trigger so
  // the toolbar layout is stable; the popover then shows a helpful empty
  // state instead of disappearing.
  return (
    <div ref={rootRef} className="relative">
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        className={cn(
          "inline-flex items-center gap-1.5 rounded-[var(--radius-rt-sm)] border border-[var(--color-rt-line)] bg-[var(--color-rt-paper)] text-[var(--color-rt-ink)] transition-colors hover:bg-[var(--color-rt-mist)]",
          compact ? "h-8 px-2 text-[11.5px]" : "h-9 px-2.5 text-[12px]",
        )}
        aria-haspopup="listbox"
        aria-expanded={open}
        title="Select ground-truth revision"
      >
        <History size={compact ? 13 : 14} className="text-[var(--color-rt-muted)]" />
        <span className="flex flex-col items-start leading-tight">
          <span className="text-[9px] uppercase tracking-[0.1em] text-[var(--color-rt-muted)]">
            GT revision
          </span>
          <span className="font-mono text-[11px] tabular-nums">
            {active
              ? `r${active.revision}${active.reviewer ? ` · ${active.reviewer}` : ""}`
              : "seed"}
          </span>
        </span>
        <ChevronDown size={12} className="text-[var(--color-rt-muted)]" />
      </button>

      {open && (
        <div
          role="listbox"
          className="absolute left-0 top-full z-40 mt-1 w-[280px] overflow-hidden rounded-[var(--radius-rt-sm)] border border-[var(--color-rt-line)] bg-[var(--color-rt-paper)] shadow-[var(--shadow-rt-elevation-2)]"
        >
          <div className="border-b border-[var(--color-rt-line)] px-3 py-2">
            <div className="rt-display text-[11px] font-semibold uppercase tracking-[0.1em] text-[var(--color-rt-muted)]">
              Ground-truth revisions
            </div>
            <div className="mt-0.5 text-[10.5px] text-[var(--color-rt-muted)]">
              {revisions.length === 0
                ? "Only the seed labelmap exists. Save an edit to create r1."
                : `${revisions.length} revision${revisions.length === 1 ? "" : "s"} · click to activate`}
            </div>
          </div>
          <ul className="max-h-[280px] overflow-y-auto py-1">
            <li>
              <RevisionRow
                title="seed"
                subtitle="the cohort-bundled labelmap"
                tag="·"
                active={!active}
                disabled={true}
                onClick={() => undefined}
              />
            </li>
            {[...revisions]
              .sort((a, b) => b.revision - a.revision)
              .map((rev) => (
                <li key={rev.id}>
                  <RevisionRow
                    title={`r${rev.revision}`}
                    subtitle={
                      rev.notes ||
                      `${rev.reviewer || "anonymous"} · ${formatTime(rev.created_at)}`
                    }
                    tag={rev.reviewer || ""}
                    active={rev.id === activeId}
                    busy={busy === rev.id}
                    onClick={() => onPick(rev)}
                  />
                </li>
              ))}
          </ul>
          {err && (
            <div className="border-t border-[var(--color-rt-line)] px-3 py-2 text-[10.5px] text-[var(--color-rt-pip-error)]">
              {err}
            </div>
          )}
        </div>
      )}
    </div>
  );
}

function RevisionRow({
  title,
  subtitle,
  tag,
  active,
  busy = false,
  disabled = false,
  onClick,
}: {
  title: string;
  subtitle: string;
  tag: string;
  active: boolean;
  busy?: boolean;
  disabled?: boolean;
  onClick: () => void;
}) {
  return (
    <button
      type="button"
      onClick={disabled ? undefined : onClick}
      disabled={disabled || busy}
      className={cn(
        "flex w-full items-center gap-2 px-3 py-1.5 text-left transition-colors",
        active
          ? "bg-[color-mix(in_oklab,var(--color-rt-accent)_10%,var(--color-rt-paper))] text-[var(--color-rt-ink)]"
          : "text-[var(--color-rt-ink)] hover:bg-[var(--color-rt-mist)]",
        disabled && "cursor-default opacity-60 hover:bg-transparent",
      )}
    >
      <span className="w-4 shrink-0">
        {busy ? (
          <Loader2 size={12} className="animate-spin text-[var(--color-rt-accent)]" />
        ) : active ? (
          <Check size={12} className="text-[var(--color-rt-accent)]" />
        ) : null}
      </span>
      <div className="min-w-0 flex-1">
        <div className="flex items-baseline gap-2">
          <span className="font-mono text-[11.5px] tabular-nums text-[var(--color-rt-ink)]">
            {title}
          </span>
          {tag && tag !== "·" && (
            <span className="truncate text-[10.5px] text-[var(--color-rt-muted)]">
              {tag}
            </span>
          )}
        </div>
        <div className="truncate text-[10.5px] text-[var(--color-rt-muted)]">
          {subtitle}
        </div>
      </div>
    </button>
  );
}

function formatTime(iso: string): string {
  try {
    const d = new Date(iso);
    if (Number.isNaN(d.getTime())) return iso;
    const today = new Date();
    const sameDay =
      d.getFullYear() === today.getFullYear() &&
      d.getMonth() === today.getMonth() &&
      d.getDate() === today.getDate();
    if (sameDay) {
      return d.toLocaleTimeString(undefined, {
        hour: "2-digit",
        minute: "2-digit",
      });
    }
    return d.toLocaleDateString(undefined, {
      month: "short",
      day: "numeric",
    });
  } catch {
    return iso;
  }
}
