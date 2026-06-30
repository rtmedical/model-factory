"use client";

import { CalendarClock, ChevronLeft, ChevronRight, Cpu, Loader2 } from "lucide-react";
import { useEffect, useMemo, useState } from "react";

import type { ModelInfo, PlannedTraining } from "@/lib/api";
import {
  formatClockDay,
  formatDuration,
  etaMs,
  foldSchedule,
  useNow,
  type FoldSchedule,
} from "@/lib/eta";
import { cn } from "@/lib/utils";

// ── geometry ───────────────────────────────────────────────────────────────
const LABEL_W = 116; // px — fold-label gutter
const HEADER_H = 38; // px — day-axis strip height
const ROW_H = 34; // px — one fold per row
const DAY_MS = 86_400_000;
const WINDOW_DAYS = 7;
const WINDOW_MS = WINDOW_DAYS * DAY_MS;
// Rows beyond this scroll inside the plot instead of growing the card — the
// planned-trainings queue can be long, and an unbounded list dominated the
// dashboard. Live folds sort first, so the visible window leads with them.
const MAX_VISIBLE_ROWS = 9;

const DOW = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"];

type Row = {
  key: string;
  datasetName: string;
  fold: number;
  schedule: FoldSchedule;
  // Live (actively training) vs planned (queued, not yet started). Planned
  // rows render as muted/dashed bars at their projected start→finish.
  planned: boolean;
  // Planned-only tooltip detail (queue notes); undefined for live rows.
  notes?: string;
};

function startOfLocalDay(ms: number): number {
  const d = new Date(ms);
  d.setHours(0, 0, 0, 0);
  return d.getTime();
}

function displayDataset(name: string): string {
  return name.replace(/^Dataset(\d+)_.*/, "D$1");
}

// A planned training carries no epoch rate — its bar spans the backend's
// projected scheduled_start → est_finish. Mirrors the FoldSchedule shape so
// FoldBar can render it through the same geometry as a live fold.
function plannedSchedule(p: PlannedTraining): FoldSchedule {
  const startMs = p.scheduled_start ? Date.parse(p.scheduled_start) : null;
  const finishMs = p.est_finish ? Date.parse(p.est_finish) : null;
  return {
    fold: p.fold,
    startMs: Number.isFinite(startMs as number) ? startMs : null,
    finishMs: Number.isFinite(finishMs as number) ? finishMs : null,
    secPerEpoch: null,
    currentEpoch: null,
    totalEpochs: null,
    schedulable: finishMs != null && Number.isFinite(finishMs),
  };
}

// Flatten actively-training folds + queued (planned) trainings into calendar
// rows, soonest projected finish first (most urgent at the top). Planned rows
// naturally sort below live ones (they finish later), with a live-before-
// planned tiebreak. Unschedulable rows (no estimate yet) sort last.
function buildRows(models: ModelInfo[], planned: PlannedTraining[]): Row[] {
  const rows: Row[] = [];
  for (const m of models) {
    for (const f of m.folds) {
      if (f.status !== "training") continue;
      rows.push({
        key: `${m.model_id}::${f.fold}`,
        datasetName: m.dataset_name,
        fold: f.fold,
        schedule: foldSchedule(f),
        planned: false,
      });
    }
  }
  for (const p of planned) {
    if (p.status !== "planned") continue;
    rows.push({
      key: `planned::${p.id}`,
      datasetName: p.dataset_name,
      fold: p.fold,
      schedule: plannedSchedule(p),
      planned: true,
      notes: p.notes,
    });
  }
  rows.sort((a, b) => {
    const af = a.schedule.finishMs ?? Infinity;
    const bf = b.schedule.finishMs ?? Infinity;
    if (af !== bf) return af - bf;
    if (a.planned !== b.planned) return a.planned ? 1 : -1; // live first on ties
    return a.key.localeCompare(b.key);
  });
  return rows;
}

export function TrainingCalendar({
  models,
  planned = [],
}: {
  models: ModelInfo[];
  planned?: PlannedTraining[];
}) {
  const rows = useMemo(() => buildRows(models, planned), [models, planned]);
  // Only arm the 1 s clock when there's live work to count down — an idle
  // catalog shouldn't re-render once a second behind an EmptyState.
  const now = useNow(1000, rows.length > 0);
  const [offsetWeeks, setOffsetWeeks] = useState(0);

  // Defer time-dependent rendering until after mount so the static-export
  // build HTML (built at a different instant) doesn't trip hydration.
  const [mounted, setMounted] = useState(false);
  useEffect(() => setMounted(true), []);

  const windowStart = startOfLocalDay(now) + offsetWeeks * WINDOW_MS;
  const windowEnd = windowStart + WINDOW_MS;
  const nowInWindow = now >= windowStart && now <= windowEnd;

  // Fraction across the visible window, clamped to [0,1].
  const xPct = (ms: number) =>
    Math.max(0, Math.min(1, (ms - windowStart) / WINDOW_MS)) * 100;

  const days = useMemo(
    () => Array.from({ length: WINDOW_DAYS }, (_, i) => windowStart + i * DAY_MS),
    [windowStart],
  );

  const liveCount = rows.filter((r) => !r.planned).length;
  const plannedCount = rows.filter((r) => r.planned).length;

  // Cap the plot height; scroll the overflow. Day axis + gridlines stay fixed.
  const visibleRows = Math.min(rows.length, MAX_VISIBLE_ROWS);
  const bodyHeight = visibleRows * ROW_H;
  const needScroll = rows.length > MAX_VISIBLE_ROWS;

  return (
    <section className="rt-console rt-grain relative p-4 sm:p-5">
      <div className="relative z-[1] flex flex-col gap-3">
        {/* ── header ─────────────────────────────────────────────────── */}
        <div className="flex items-center justify-between gap-3">
          <div className="flex items-center gap-2.5">
            <span className="inline-flex h-8 w-8 items-center justify-center rounded-[var(--radius-rt-sm)] bg-[color-mix(in_oklab,var(--color-rt-accent)_12%,var(--color-rt-paper))] text-[var(--color-rt-accent)]">
              <CalendarClock size={16} />
            </span>
            <div className="leading-tight">
              <h2 className="rt-display text-[15px] font-semibold tracking-tight text-[var(--color-rt-ink)]">
                Training schedule
              </h2>
              <p className="text-[10.5px] text-[var(--color-rt-muted)]">
                live finish estimates · derived from epoch rate
              </p>
            </div>
          </div>
          <div className="flex items-center gap-2">
            <span
              className={cn(
                "inline-flex items-center gap-1.5 rounded-full border px-2.5 py-1 text-[10.5px] font-medium tracking-wide",
                liveCount > 0
                  ? "border-[color-mix(in_oklab,var(--color-rt-accent)_30%,transparent)] bg-[color-mix(in_oklab,var(--color-rt-accent)_10%,var(--color-rt-paper))] text-[var(--color-rt-accent)]"
                  : "border-[var(--color-rt-line)] bg-[var(--color-rt-mist)] text-[var(--color-rt-muted)]",
              )}
            >
              {liveCount > 0 ? (
                <Loader2 size={11} className="animate-spin" />
              ) : (
                <Cpu size={11} />
              )}
              {liveCount > 0
                ? `${liveCount} fold${liveCount === 1 ? "" : "s"} live`
                : "GPUs idle"}
            </span>
            {plannedCount > 0 && (
              <span
                className="inline-flex items-center gap-1.5 rounded-full border border-dashed border-[var(--color-rt-line)] bg-[var(--color-rt-mist)] px-2.5 py-1 text-[10.5px] font-medium tracking-wide text-[var(--color-rt-muted)]"
                title="trainings queued in the pipeline"
              >
                <span className="h-2 w-2 rotate-45 border border-[var(--color-rt-muted)] bg-transparent" />
                {`${plannedCount} queued`}
              </span>
            )}
            <div className="inline-flex items-center rounded-full border border-[var(--color-rt-line)] bg-[var(--color-rt-paper)] p-0.5">
              <button
                type="button"
                aria-label="previous week"
                onClick={() => setOffsetWeeks((w) => w - 1)}
                className="inline-flex h-6 w-6 items-center justify-center rounded-full text-[var(--color-rt-muted)] hover:bg-[var(--color-rt-mist)] hover:text-[var(--color-rt-ink)]"
              >
                <ChevronLeft size={13} />
              </button>
              <button
                type="button"
                onClick={() => setOffsetWeeks(0)}
                className={cn(
                  "px-2 text-[10.5px] font-medium tracking-wide tabular-nums",
                  offsetWeeks === 0
                    ? "text-[var(--color-rt-accent)]"
                    : "text-[var(--color-rt-muted)] hover:text-[var(--color-rt-ink)]",
                )}
              >
                {offsetWeeks === 0
                  ? "this week"
                  : offsetWeeks < 0
                    ? `${offsetWeeks}w`
                    : `+${offsetWeeks}w`}
              </button>
              <button
                type="button"
                aria-label="next week"
                onClick={() => setOffsetWeeks((w) => w + 1)}
                className="inline-flex h-6 w-6 items-center justify-center rounded-full text-[var(--color-rt-muted)] hover:bg-[var(--color-rt-mist)] hover:text-[var(--color-rt-ink)]"
              >
                <ChevronRight size={13} />
              </button>
            </div>
          </div>
        </div>

        {/* ── plot ───────────────────────────────────────────────────── */}
        {!mounted ? (
          <div className="h-[120px] animate-pulse rounded-[var(--radius-rt-sm)] bg-[var(--color-rt-mist)]" />
        ) : rows.length === 0 ? (
          <EmptyState />
        ) : (
          <div className="relative">
            {/* day-column gridlines — fixed frame behind everything, offset
                past the label gutter so it aligns with the bar track */}
            <div
              className="pointer-events-none absolute inset-y-0 z-0 flex"
              style={{ left: LABEL_W, right: 0 }}
            >
              {days.map((d, i) => {
                const isToday = startOfLocalDay(now) === d;
                return (
                  <div
                    key={d}
                    className={cn(
                      "flex-1 border-l",
                      i === 0 ? "border-transparent" : "border-[var(--color-rt-grid)]",
                      isToday &&
                        "bg-[color-mix(in_oklab,var(--color-rt-accent)_4%,transparent)]",
                    )}
                  />
                );
              })}
              <div className="border-l border-[var(--color-rt-grid)]" />
            </div>

            {/* now line — fixed in front; stays put while the body scrolls */}
            {nowInWindow && (
              <div
                className="pointer-events-none absolute inset-y-0 z-[3]"
                style={{ left: LABEL_W, right: 0 }}
              >
                <div
                  className="absolute"
                  style={{ left: `${xPct(now)}%`, top: HEADER_H - 10, bottom: 0 }}
                >
                  <div className="absolute -left-px top-0 h-full w-px rt-nowline bg-[var(--color-rt-accent)]" />
                  <div className="absolute -left-[3px] top-0 h-1.5 w-1.5 rounded-full bg-[var(--color-rt-accent)]" />
                </div>
              </div>
            )}

            {/* day axis (fixed header) */}
            <div className="relative z-[1] flex" style={{ height: HEADER_H }}>
              <div className="shrink-0" style={{ width: LABEL_W }} />
              <div className="flex flex-1">
                {days.map((d) => {
                  const date = new Date(d);
                  const isToday = startOfLocalDay(now) === d;
                  return (
                    <div
                      key={d}
                      className="flex flex-1 flex-col items-center justify-center leading-none"
                    >
                      <span
                        className={cn(
                          "text-[9.5px] uppercase tracking-[0.12em]",
                          isToday
                            ? "text-[var(--color-rt-accent)]"
                            : "text-[var(--color-rt-muted)]",
                        )}
                      >
                        {DOW[date.getDay()]}
                      </span>
                      <span
                        className={cn(
                          "mt-0.5 font-mono text-[12px] tabular-nums",
                          isToday
                            ? "font-semibold text-[var(--color-rt-accent)]"
                            : "text-[var(--color-rt-ink)]",
                        )}
                      >
                        {date.getDate()}
                      </span>
                    </div>
                  );
                })}
              </div>
            </div>

            {/* rows — one flex row per fold (label + bar track), scrollable
                past MAX_VISIBLE_ROWS so a long pipeline doesn't grow the card */}
            <div
              className="relative z-[1]"
              style={{
                maxHeight: bodyHeight,
                overflowY: needScroll ? "auto" : "visible",
                scrollbarWidth: "thin",
                scrollbarGutter: "stable",
              }}
            >
              {rows.map((r) => (
                <div key={r.key} className="flex" style={{ height: ROW_H }}>
                  <div
                    className="flex shrink-0 items-center gap-1.5 pr-2"
                    style={{ width: LABEL_W }}
                  >
                    <span
                      className={cn(
                        "truncate rt-display text-[12px] font-semibold",
                        r.planned
                          ? "text-[var(--color-rt-muted)]"
                          : "text-[var(--color-rt-ink)]",
                      )}
                    >
                      {displayDataset(r.datasetName)}
                    </span>
                    <span className="shrink-0 font-mono text-[10px] text-[var(--color-rt-muted)]">
                      f{r.fold}
                    </span>
                  </div>
                  <div className="relative flex-1">
                    <FoldBar
                      row={r}
                      now={now}
                      windowStart={windowStart}
                      windowEnd={windowEnd}
                      xPct={xPct}
                    />
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* legend */}
        {mounted && rows.length > 0 && (
          <div
            className="flex items-center gap-4 text-[10px] text-[var(--color-rt-muted)]"
            style={{ paddingLeft: LABEL_W }}
          >
            <Legend swatch="solid">elapsed</Legend>
            <Legend swatch="stripe">remaining</Legend>
            <Legend swatch="diamond">est. finish</Legend>
            {plannedCount > 0 && <Legend swatch="planned">queued</Legend>}
          </div>
        )}
      </div>
    </section>
  );
}

function FoldBar({
  row,
  now,
  windowStart,
  windowEnd,
  xPct,
}: {
  row: Row;
  now: number;
  windowStart: number;
  windowEnd: number;
  xPct: (ms: number) => number;
}) {
  const s = row.schedule;

  // Bar extent in time. Start falls back to "now minus elapsed at current
  // rate" inside foldSchedule; if still unknown, anchor at window start.
  const startMs = s.startMs ?? windowStart;
  const finishMs = s.finishMs;

  // Unschedulable: no finish estimate yet — show a small pulsing marker at
  // "now" instead of a full bar.
  if (finishMs == null || !s.schedulable) {
    const x = now >= windowStart && now <= windowEnd ? xPct(now) : 2;
    return (
      <div className="relative flex items-center" style={{ height: ROW_H }}>
        <div
          className="absolute flex items-center gap-1.5"
          style={{ left: `${x}%` }}
        >
          <span className="h-2 w-2 rounded-full bg-[var(--color-rt-accent)] rt-pulse-dot" />
          <span className="whitespace-nowrap text-[10px] italic text-[var(--color-rt-muted)]">
            estimating ETA…
          </span>
        </div>
      </div>
    );
  }

  const planned = row.planned;
  const left = xPct(startMs);
  const right = xPct(finishMs);
  const width = Math.max(0.6, right - left);
  // Where "now" sits inside the bar → splits elapsed (solid) from remaining
  // (hatched). Clamp to the bar's own extent. Planned bars haven't started,
  // so there's no elapsed portion.
  const nowX = Math.max(left, Math.min(right, xPct(now)));
  const elapsedW = planned ? 0 : Math.max(0, nowX - left);
  // Finish lands outside the visible week: beyond the right edge (a far-out
  // run) or before the left edge (only reachable by paging to a future
  // week). Both pin to an edge and get a muted outline marker.
  const finishesBeyond = finishMs > windowEnd;
  const finishesBefore = finishMs < windowStart;
  const offWindow = finishesBeyond || finishesBefore;

  const remaining = etaMs(finishMs, now);
  const finishLabel = formatClockDay(finishMs, now);
  const startLabel = formatClockDay(startMs, now);
  // `right` is a percentage in [0,100]; flip the label to the LEFT of the
  // marker only when it's genuinely near the right edge so it never clips.
  const labelOnLeft = right > 88;

  return (
    <div className="relative flex items-center" style={{ height: ROW_H }}>
      {/* bar */}
      <div
        className={cn(
          "absolute top-1/2 h-[12px] -translate-y-1/2 overflow-hidden rounded-full border",
          planned
            ? "border-dashed border-[var(--color-rt-line)] bg-[color-mix(in_oklab,var(--color-rt-muted)_8%,var(--color-rt-paper))]"
            : "border-[color-mix(in_oklab,var(--color-rt-accent)_28%,var(--color-rt-line))] bg-[color-mix(in_oklab,var(--color-rt-accent)_8%,var(--color-rt-paper))]",
        )}
        style={{ left: `${left}%`, width: `${width}%` }}
        title={
          planned
            ? `${displayDataset(row.datasetName)} fold ${row.fold} · queued${
                row.notes ? ` · ${row.notes}` : ""
              } · projected ${startLabel} → ${finishLabel}`
            : `${displayDataset(row.datasetName)} fold ${row.fold} · epoch ${
                s.currentEpoch ?? "?"
              }/${s.totalEpochs ?? "?"} · ${
                s.secPerEpoch ? `${s.secPerEpoch.toFixed(0)}s/epoch` : "rate n/a"
              }`
        }
      >
        {planned ? (
          /* planned: static low-opacity fill — no animation, no elapsed split */
          <div className="absolute inset-0 bg-[color-mix(in_oklab,var(--color-rt-muted)_14%,var(--color-rt-paper))]" />
        ) : (
          <>
            {/* remaining (hatched, animated) fills the whole bar as the base */}
            <div className="absolute inset-0 rt-live-stripe bg-[color-mix(in_oklab,var(--color-rt-accent)_22%,var(--color-rt-paper))]" />
            {/* elapsed (solid) overlay from the left up to the now-line */}
            <div
              className="absolute inset-y-0 left-0 bg-[var(--color-rt-accent)]"
              style={{ width: `${(elapsedW / width) * 100}%` }}
            />
          </>
        )}
      </div>

      {/* est-finish diamond marker */}
      <div
        className="absolute top-1/2 z-[2] -translate-y-1/2"
        style={{ left: `${right}%` }}
      >
        <span
          className={cn(
            "block h-2.5 w-2.5 -translate-x-1/2 rotate-45 border",
            planned || offWindow
              ? "border-[var(--color-rt-muted)] bg-[var(--color-rt-paper)]"
              : "border-[var(--color-rt-accent)] bg-[var(--color-rt-accent)]",
          )}
        />
      </div>

      {/* label: live = finish + live countdown; planned = projected finish + "queued" */}
      <div
        className={cn(
          "absolute top-1/2 -translate-y-1/2 whitespace-nowrap",
          labelOnLeft ? "-translate-x-full pr-2 text-right" : "pl-2",
        )}
        style={{ left: `${right}%` }}
      >
        <span
          className={cn(
            "font-mono text-[10.5px] tabular-nums",
            planned ? "text-[var(--color-rt-muted)]" : "text-[var(--color-rt-ink)]",
          )}
        >
          {finishesBeyond
            ? `>${WINDOW_DAYS}d`
            : finishesBefore
              ? `‹ ${finishLabel}`
              : `${planned ? "~" : ""}${finishLabel}`}
        </span>
        <span className="ml-1.5 text-[10px] text-[var(--color-rt-muted)]">
          {planned ? "queued" : formatDuration(remaining)}
        </span>
      </div>
    </div>
  );
}

function Legend({
  swatch,
  children,
}: {
  swatch: "solid" | "stripe" | "diamond" | "planned";
  children: React.ReactNode;
}) {
  return (
    <span className="inline-flex items-center gap-1.5">
      {swatch === "solid" && (
        <span className="h-2 w-4 rounded-full bg-[var(--color-rt-accent)]" />
      )}
      {swatch === "stripe" && (
        <span className="rt-live-stripe h-2 w-4 rounded-full bg-[color-mix(in_oklab,var(--color-rt-accent)_22%,var(--color-rt-paper))]" />
      )}
      {swatch === "diamond" && (
        <span className="h-2 w-2 rotate-45 bg-[var(--color-rt-accent)]" />
      )}
      {swatch === "planned" && (
        <span className="h-2 w-4 rounded-full border border-dashed border-[var(--color-rt-muted)] bg-[color-mix(in_oklab,var(--color-rt-muted)_14%,var(--color-rt-paper))]" />
      )}
      {children}
    </span>
  );
}

function EmptyState() {
  return (
    <div className="flex flex-col items-center justify-center gap-2 rounded-[var(--radius-rt-sm)] border border-dashed border-[var(--color-rt-line)] py-10 text-center">
      <Cpu size={20} className="text-[var(--color-rt-muted)]" />
      <div className="rt-display text-[13px] font-semibold text-[var(--color-rt-ink)]">
        No active training
      </div>
      <p className="max-w-sm text-[11px] text-[var(--color-rt-muted)]">
        The factory GPUs are idle. When a fold starts, its projected finish
        time appears here and updates live from the epoch rate.
      </p>
    </div>
  );
}
