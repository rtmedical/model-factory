// Live training-ETA helpers.
//
// The backend (/api/models) hands each actively-training fold a snapshot
// `est_finish` (absolute UTC instant) plus the rate it was derived from.
// The catalog refetches every 30 s, but between refetches we want the
// schedule to feel alive — so we tick a 1 s clock client-side and recompute
// the countdown toward the fixed `est_finish`. As long as the training rate
// holds, the countdown is correct; each refetch re-anchors it to the true
// current epoch. Everything here is timezone-correct because the timestamps
// are UTC ISO and we only ever subtract epoch-ms (Date.parse / Date.now).

import { useEffect, useState } from "react";

import type { FoldProgress } from "./api";

// Parse a UTC ISO8601 string to epoch-ms, tolerating null/garbage.
function parseMs(iso: string | null | undefined): number | null {
  if (!iso) return null;
  const ms = Date.parse(iso);
  return Number.isFinite(ms) ? ms : null;
}

export type FoldSchedule = {
  fold: number;
  startMs: number | null;
  finishMs: number | null;
  secPerEpoch: number | null;
  currentEpoch: number | null;
  totalEpochs: number | null;
  // True when we have enough to draw a finish marker + countdown.
  schedulable: boolean;
};

export function foldSchedule(fold: FoldProgress): FoldSchedule {
  const startMs = parseMs(fold.started_at);
  const finishMs = parseMs(fold.est_finish);
  return {
    fold: fold.fold,
    startMs,
    finishMs,
    secPerEpoch: fold.sec_per_epoch ?? null,
    currentEpoch: fold.current_epoch,
    totalEpochs: fold.total_epochs,
    schedulable: finishMs != null,
  };
}

// Remaining milliseconds until the projected finish, floored at 0.
export function etaMs(finishMs: number | null, nowMs: number): number | null {
  if (finishMs == null) return null;
  return Math.max(0, finishMs - nowMs);
}

// "8h 20m" · "3d 4h" · "12m" · "<1m" · "done". Compact, two-unit max.
export function formatDuration(ms: number | null): string {
  if (ms == null) return "—";
  const s = Math.round(ms / 1000);
  if (s <= 0) return "done";
  if (s < 60) return "<1m";
  const m = Math.floor(s / 60);
  if (m < 60) return `${m}m`;
  const h = Math.floor(m / 60);
  const remM = m % 60;
  if (h < 24) return remM ? `${h}h ${remM}m` : `${h}h`;
  const d = Math.floor(h / 24);
  const remH = h % 24;
  return remH ? `${d}d ${remH}h` : `${d}d`;
}

const DOW = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"];
const MON = [
  "Jan", "Feb", "Mar", "Apr", "May", "Jun",
  "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
];

function hhmm(d: Date): string {
  return `${String(d.getHours()).padStart(2, "0")}:${String(
    d.getMinutes(),
  ).padStart(2, "0")}`;
}

// Local wall-clock label for a finish instant, relative to `nowMs`:
//   today 23:55 · tomorrow 09:10 · Fri 14:20 · May 31
function dayDiffLocal(target: Date, now: Date): number {
  const a = new Date(now.getFullYear(), now.getMonth(), now.getDate());
  const b = new Date(
    target.getFullYear(),
    target.getMonth(),
    target.getDate(),
  );
  return Math.round((b.getTime() - a.getTime()) / 86_400_000);
}

export function formatClockDay(finishMs: number | null, nowMs: number): string {
  if (finishMs == null) return "—";
  // A projected finish in the past (a stalled-but-recent run whose
  // epoch-anchored estimate has slipped behind the wall clock) reads as
  // "overdue" rather than collapsing into a misleading "today HH:MM".
  if (finishMs < nowMs) return "overdue";
  const d = new Date(finishMs);
  const now = new Date(nowMs);
  const dd = dayDiffLocal(d, now);
  if (dd <= 0) return `today ${hhmm(d)}`;
  if (dd === 1) return `tomorrow ${hhmm(d)}`;
  if (dd < 7) return `${DOW[d.getDay()]} ${hhmm(d)}`;
  return `${MON[d.getMonth()]} ${d.getDate()}`;
}

// A coarse-grained ticking clock. Returns Date.now() in ms and re-renders
// the host component every `intervalMs` while `enabled` is true. Callers
// pass `enabled=false` when nothing live is on screen (e.g. the calendar
// has zero training folds) so we don't spin a 1 Hz re-render loop on an
// idle catalog — the hook is still called unconditionally (Rules of Hooks),
// it just doesn't arm the interval.
export function useNow(intervalMs = 1000, enabled = true): number {
  const [now, setNow] = useState(() => Date.now());
  useEffect(() => {
    if (!enabled) return;
    setNow(Date.now());
    const id = setInterval(() => setNow(Date.now()), intervalMs);
    return () => clearInterval(id);
  }, [intervalMs, enabled]);
  return now;
}
