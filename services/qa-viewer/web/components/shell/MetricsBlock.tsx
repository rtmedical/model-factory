"use client";

// Editorial dice-per-label readout. Replaces the old <MetricsTable> with
// a Fraunces hero number, a colour-tiered bar per label, and the failure
// labels surfaced first (sorted worst-dice ascending).
//
// Threshold tiers (also used to colour the hero number ring):
//   >= 0.80         emerald  (--color-rt-pip-ok)
//   [0.60, 0.80)    accent   (--color-rt-accent)
//   [0.40, 0.60)    amber    (--card-amber-bg)
//   <  0.40         rose     (--color-rt-pip-error)  + leading dot

import { useMemo } from "react";

import type { LabelMetric } from "@/lib/api";
import { FAIL_THRESHOLD, formatDice, formatHd95, TIER_VAR, tierFor } from "@/lib/dice";

export function MetricsBlock({ metrics }: { metrics: LabelMetric[] }) {
  const summary = useMemo(() => {
    const valid = metrics.filter(
      (m) => m.dice !== null && !Number.isNaN(m.dice),
    ) as Array<LabelMetric & { dice: number }>;
    const n = valid.length;
    const mean = n === 0 ? null : valid.reduce((a, b) => a + b.dice, 0) / n;
    const variance =
      n === 0 || mean === null
        ? 0
        : valid.reduce((a, b) => a + (b.dice - mean) ** 2, 0) / n;
    const std = Math.sqrt(variance);
    const failed = valid.filter((m) => m.dice < FAIL_THRESHOLD).length;
    return { n, mean, std, failed };
  }, [metrics]);

  // Worst-first ordering. Failed labels at the top so the reviewer's eye
  // catches the problems before scrolling.
  const ordered = useMemo(
    () =>
      [...metrics].sort((a, b) => {
        const da = a.dice ?? -1;
        const db = b.dice ?? -1;
        return da - db;
      }),
    [metrics],
  );

  const heroTier = tierFor(summary.mean);
  const heroColor = TIER_VAR[heroTier];

  return (
    <div className="space-y-3">
      <Summary
        mean={summary.mean}
        std={summary.std}
        n={summary.n}
        failed={summary.failed}
        heroColor={heroColor}
      />
      <div className="space-y-1.5">
        {ordered.map((m) => (
          <Row key={m.label} metric={m} />
        ))}
      </div>
    </div>
  );
}

function Summary({
  mean,
  std,
  n,
  failed,
  heroColor,
}: {
  mean: number | null;
  std: number;
  n: number;
  failed: number;
  heroColor: string;
}) {
  return (
    <div className="relative rounded-[var(--radius-rt-sm)] border border-[var(--color-rt-line)] bg-[color-mix(in_oklab,var(--color-rt-mist)_55%,var(--color-rt-paper))] px-3 py-2.5 shadow-[var(--shadow-rt-elevation-1)]">
      {/* Hairline on the left edge in the tier colour — quietly anchors the
          card to the same colour language as the per-label bars. */}
      <span
        aria-hidden
        className="absolute inset-y-2 left-0 w-px"
        style={{ background: heroColor }}
      />
      <div className="flex items-baseline justify-between gap-3 pl-2">
        <div>
          <div
            className="rt-display leading-none text-[var(--color-rt-ink)]"
            style={{
              fontSize: "30px",
              fontVariationSettings: '"opsz" 30, "SOFT" 50',
              letterSpacing: "-0.02em",
            }}
          >
            {mean !== null ? mean.toFixed(3) : "—"}
          </div>
          <div className="mt-1 text-[10px] uppercase tracking-[0.08em] text-[var(--color-rt-muted)]">
            mean dice
          </div>
        </div>
        <dl className="flex flex-col items-end gap-0.5 text-right">
          <Stat label="labels" value={String(n)} />
          <Stat label="σ" value={n === 0 ? "—" : std.toFixed(3)} />
          <Stat
            label="failed"
            value={String(failed)}
            tone={failed > 0 ? "fail" : undefined}
          />
        </dl>
      </div>
    </div>
  );
}

function Stat({
  label,
  value,
  tone,
}: {
  label: string;
  value: string;
  tone?: "fail";
}) {
  return (
    <div className="flex items-baseline gap-1.5">
      <dt className="text-[10px] uppercase tracking-wide text-[var(--color-rt-muted)]">
        {label}
      </dt>
      <dd
        className="font-mono text-[11px]"
        style={{
          color:
            tone === "fail"
              ? "var(--color-rt-pip-error)"
              : "var(--color-rt-ink)",
        }}
      >
        {value}
      </dd>
    </div>
  );
}

function Row({ metric }: { metric: LabelMetric }) {
  const t = tierFor(metric.dice);
  const color = TIER_VAR[t];
  const widthPct =
    metric.dice === null || Number.isNaN(metric.dice)
      ? 0
      : Math.max(0, Math.min(1, metric.dice)) * 100;
  const isFail = t === "fail";

  return (
    <div className="group relative">
      {isFail && (
        <span
          aria-hidden
          className="absolute -left-0.5 top-1.5 h-1 w-1 rounded-full"
          style={{ background: color }}
        />
      )}
      <div className="flex items-baseline justify-between gap-3 pl-2">
        <span
          className="min-w-0 truncate text-[11.5px] text-[var(--color-rt-ink)]"
          title={metric.label_name}
        >
          {metric.label_name}
        </span>
        <span className="shrink-0 font-mono text-[11.5px] tabular-nums text-[var(--color-rt-ink)]">
          {formatDice(metric.dice)}
        </span>
      </div>
      <div
        className="mt-1 ml-2 h-[3px] overflow-hidden rounded-full"
        style={{
          background:
            "color-mix(in oklab, var(--color-rt-line) 80%, transparent)",
        }}
      >
        <div
          className="h-full rounded-full"
          style={{
            width: `${widthPct}%`,
            background: color,
            transition: "width var(--dur-rt-pip) var(--ease-rt)",
          }}
        />
      </div>
      <div className="mt-0.5 ml-2 flex items-baseline justify-between gap-3 text-[10px] text-[var(--color-rt-muted)]">
        <span>
          gt {metric.n_voxels_gt.toLocaleString()} · pred{" "}
          {metric.n_voxels_pred.toLocaleString()}
        </span>
        <span className="font-mono tabular-nums">
          hd95 {formatHd95(metric.hd95_mm)}
        </span>
      </div>
    </div>
  );
}
