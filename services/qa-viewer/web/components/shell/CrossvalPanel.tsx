"use client";

// Per-case cross-validation comparison. Renders in the InferencePanel right
// rail when runMode === "crossval". Shows the honest out-of-fold (OOF) headline,
// a per-fold comparison table (click a row → the viewer overlay swaps to that
// fold's seg), a drill-in MetricsBlock for the selected row, inter-fold
// agreement (per-label mean ± std), and export links. The OOF fold is the one
// that held this case OUT of training (from splits_final.json) — the unbiased
// result, starred.

import { AlertTriangle, ExternalLink, FileText, Loader2, Star, Table } from "lucide-react";

import {
  crossvalReportCsvUrl,
  crossvalReportHtmlUrl,
  type CrossvalEntry,
  type CrossvalRun,
} from "@/lib/api";
import { crossvalEntryKey, useQAStore } from "@/lib/store";
import { cn } from "@/lib/utils";
import { formatDice, TIER_VAR, tierFor } from "@/lib/dice";

import { MetricsBlock } from "./MetricsBlock";

export function CrossvalPanel() {
  const crossval = useQAStore((s) => s.crossval);
  const crossvalState = useQAStore((s) => s.crossvalState);
  const crossvalError = useQAStore((s) => s.crossvalError);
  const progress = useQAStore((s) => s.crossvalProgress);
  const selectedFoldKey = useQAStore((s) => s.selectedFoldKey);
  const selectCrossvalFold = useQAStore((s) => s.selectCrossvalFold);

  if (crossvalState === "running") {
    return <CrossvalProgress completed={progress.completed} total={progress.total} currentFold={progress.currentFold} />;
  }

  if (crossvalState === "error") {
    return (
      <Section title="Cross-validation">
        <div className="flex items-start gap-1.5 rounded-[var(--radius-rt-sm)] bg-[color-mix(in_oklab,var(--color-rt-pip-error)_10%,var(--color-rt-paper))] p-2 text-[11px] text-[var(--color-rt-pip-error)]">
          <AlertTriangle size={12} className="mt-0.5 shrink-0" />
          <span className="break-all">{crossvalError ?? "cross-validation failed"}</span>
        </div>
      </Section>
    );
  }

  if (crossvalState !== "ready" || !crossval) {
    return (
      <Section title="Cross-validation">
        <p className="text-[11px] text-[var(--color-rt-muted)]">
          Run cross-validation to evaluate every fold individually and flag the
          unbiased out-of-fold result.
        </p>
      </Section>
    );
  }

  const folds = crossval.entries.filter((e) => e.kind === "fold");
  const ensemble = crossval.entries.find((e) => e.kind === "ensemble") ?? null;
  const oofEntry = folds.find((e) => e.is_oof) ?? null;
  const selectedEntry =
    crossval.entries.find((e) => crossvalEntryKey(e) === selectedFoldKey) ?? null;

  return (
    <>
      <OofHeadline run={crossval} oofEntry={oofEntry} ensemble={ensemble} />

      {crossval.stale && (
        <div className="flex items-start gap-1.5 rounded-[var(--radius-rt-sm)] bg-[color-mix(in_oklab,var(--card-amber-bg)_14%,var(--color-rt-paper))] p-2 text-[11px] text-[var(--color-rt-ink)]">
          <AlertTriangle size={12} className="mt-0.5 shrink-0" />
          <span>Ground truth changed since this run — metrics may be stale. Re-run cross-validation.</span>
        </div>
      )}

      <Section title="Per-fold comparison">
        <div className="overflow-hidden rounded-[var(--radius-rt-sm)] border border-[var(--color-rt-line)]">
          <table className="w-full border-collapse text-[11px]">
            <thead>
              <tr className="bg-[var(--color-rt-mist)] text-[10px] uppercase tracking-wide text-[var(--color-rt-muted)]">
                <th className="px-2 py-1 text-left font-medium">fold</th>
                <th className="px-1 py-1 text-center font-medium" title="out-of-fold (unbiased)">oof</th>
                <th className="px-2 py-1 text-right font-medium">mean dice</th>
                <th className="px-2 py-1 text-left font-medium">worst label</th>
              </tr>
            </thead>
            <tbody>
              {folds.map((e) => (
                <FoldRow
                  key={crossvalEntryKey(e)}
                  entry={e}
                  active={selectedFoldKey === crossvalEntryKey(e)}
                  onSelect={() => selectCrossvalFold(crossvalEntryKey(e))}
                />
              ))}
              {ensemble && (
                <FoldRow
                  entry={ensemble}
                  active={selectedFoldKey === "ensemble"}
                  onSelect={() => selectCrossvalFold("ensemble")}
                  separated
                />
              )}
            </tbody>
          </table>
        </div>
        <p className="mt-1 text-[10px] text-[var(--color-rt-muted)]">
          Click a fold to load its segmentation in the viewer.
          {crossval.compute_hd95 === "none" && " HD95 omitted for speed (Dice-only)."}
        </p>
      </Section>

      {selectedEntry && selectedEntry.metrics && selectedEntry.metrics.length > 0 ? (
        <Section title={`Per-label dice — ${rowLabel(selectedEntry)}`}>
          <MetricsBlock metrics={selectedEntry.metrics} />
        </Section>
      ) : selectedEntry ? (
        <Section title={`Per-label dice — ${rowLabel(selectedEntry)}`}>
          <p className="text-[11px] text-[var(--color-rt-muted)]">
            {selectedEntry.error ?? "no per-label metrics (no ground truth?)"}
          </p>
        </Section>
      ) : null}

      {crossval.aggregate && crossval.aggregate.per_label.length > 0 && (
        <AgreementBlock run={crossval} />
      )}

      <Section title="Report">
        <div className="flex flex-wrap gap-1.5">
          <ExportLink href={crossvalReportHtmlUrl(crossval.cv_run_id)} icon={<ExternalLink size={12} />} label="open report" newTab />
          <ExportLink href={crossvalReportHtmlUrl(crossval.cv_run_id)} icon={<FileText size={12} />} label="HTML" download={`crossval_${crossval.cv_run_id}.html`} />
          <ExportLink href={crossvalReportCsvUrl(crossval.cv_run_id)} icon={<Table size={12} />} label="CSV" download={`crossval_${crossval.cv_run_id}.csv`} />
        </div>
      </Section>
    </>
  );
}

// ── headline ───────────────────────────────────────────────────────────────

function OofHeadline({
  run,
  oofEntry,
  ensemble,
}: {
  run: CrossvalRun;
  oofEntry: CrossvalEntry | null;
  ensemble: CrossvalEntry | null;
}) {
  const agg = run.aggregate;
  const headline = agg?.headline_mean_fg_dice ?? null;
  const isOof = agg?.headline_kind === "oof" && oofEntry != null;
  const color = TIER_VAR[tierFor(headline)];
  const caption = isOof
    ? `out-of-fold · fold ${run.oof_fold}`
    : run.oof_reason === "external"
      ? "no OOF · external case — every fold unbiased"
      : "cross-fold mean (no held-out fold available)";

  return (
    <Section title="Honest cross-validation score">
      <div className="relative rounded-[var(--radius-rt-sm)] border border-[var(--color-rt-line)] bg-[color-mix(in_oklab,var(--color-rt-mist)_55%,var(--color-rt-paper))] px-3 py-2.5 shadow-[var(--shadow-rt-elevation-1)]">
        <span aria-hidden className="absolute inset-y-2 left-0 w-px" style={{ background: color }} />
        <div className="flex items-baseline justify-between gap-3 pl-2">
          <div>
            <div
              className="rt-display leading-none text-[var(--color-rt-ink)]"
              style={{ fontSize: "30px", fontVariationSettings: '"opsz" 30, "SOFT" 50', letterSpacing: "-0.02em" }}
            >
              {headline !== null ? headline.toFixed(3) : "—"}
            </div>
            <div className="mt-1 flex items-center gap-1 text-[10px] uppercase tracking-[0.08em] text-[var(--color-rt-muted)]">
              {isOof && <Star size={10} className="text-[var(--color-rt-pip-ok)]" fill="currentColor" />}
              {caption}
            </div>
          </div>
          <dl className="flex flex-col items-end gap-0.5 text-right font-mono text-[10.5px] text-[var(--color-rt-muted)]">
            <div className="flex items-baseline gap-1.5">
              <dt className="uppercase tracking-wide">folds</dt>
              <dd className="text-[var(--color-rt-ink)]">{run.available_folds.length}</dd>
            </div>
            <div className="flex items-baseline gap-1.5">
              <dt className="uppercase tracking-wide">σ folds</dt>
              <dd className="text-[var(--color-rt-ink)]">{agg?.cross_fold_std != null ? agg.cross_fold_std.toFixed(3) : "—"}</dd>
            </div>
            <div className="flex items-baseline gap-1.5">
              <dt className="uppercase tracking-wide">ensemble</dt>
              <dd className="text-[var(--color-rt-ink)]">
                {ensemble?.mean_fg_dice != null ? ensemble.mean_fg_dice.toFixed(3) : "—"}
              </dd>
            </div>
          </dl>
        </div>
      </div>
    </Section>
  );
}

// ── per-fold table row ───────────────────────────────────────────────────

function FoldRow({
  entry,
  active,
  onSelect,
  separated,
}: {
  entry: CrossvalEntry;
  active: boolean;
  onSelect: () => void;
  separated?: boolean;
}) {
  const color = TIER_VAR[tierFor(entry.mean_fg_dice)];
  const worst = worstLabel(entry);
  return (
    <tr
      onClick={onSelect}
      className={cn(
        "cursor-pointer border-t border-[var(--color-rt-line)] transition-colors",
        separated && "border-t-2",
        active ? "bg-[color-mix(in_oklab,var(--color-rt-accent)_10%,var(--color-rt-paper))]" : "hover:bg-[var(--color-rt-mist)]",
      )}
      style={active ? { boxShadow: "inset 2px 0 0 var(--color-rt-accent)" } : undefined}
    >
      <td className="px-2 py-1.5 font-mono text-[var(--color-rt-ink)]">{rowLabel(entry)}</td>
      <td className="px-1 py-1.5 text-center">
        {entry.is_oof && <Star size={11} className="inline text-[var(--color-rt-pip-ok)]" fill="currentColor" />}
      </td>
      <td className="px-2 py-1.5 text-right font-mono tabular-nums" style={{ color }}>
        {formatDice(entry.mean_fg_dice)}
      </td>
      <td className="px-2 py-1.5 text-left text-[10.5px] text-[var(--color-rt-muted)]">
        {worst ? `${worst.label_name} ${formatDice(worst.dice)}` : "—"}
      </td>
    </tr>
  );
}

// ── inter-fold agreement ───────────────────────────────────────────────────

function AgreementBlock({ run }: { run: CrossvalRun }) {
  const rows = [...(run.aggregate?.per_label ?? [])].sort(
    (a, b) => (a.dice_mean ?? -1) - (b.dice_mean ?? -1),
  );
  return (
    <Section title="Inter-fold agreement">
      <p className="-mt-0.5 mb-1.5 text-[10px] text-[var(--color-rt-muted)]">
        Per-label dice mean ± σ across folds. High σ = a structure the model
        learns inconsistently depending on the training split.
      </p>
      <div className="space-y-1.5">
        {rows.map((r) => {
          const mean = r.dice_mean;
          const std = r.dice_std ?? 0;
          const color = TIER_VAR[tierFor(mean)];
          const unstable = std >= 0.1;
          const left = mean !== null ? Math.max(0, Math.min(1, (r.dice_min ?? mean))) * 100 : 0;
          const right = mean !== null ? Math.max(0, Math.min(1, (r.dice_max ?? mean))) * 100 : 0;
          const meanPct = mean !== null ? Math.max(0, Math.min(1, mean)) * 100 : 0;
          const oofPct = r.oof_dice != null ? Math.max(0, Math.min(1, r.oof_dice)) * 100 : null;
          return (
            <div key={r.label} className="group">
              <div className="flex items-baseline justify-between gap-3">
                <span className="min-w-0 truncate text-[11.5px] text-[var(--color-rt-ink)]" title={r.label_name}>
                  {r.label_name}
                </span>
                <span className="shrink-0 font-mono text-[11px] tabular-nums text-[var(--color-rt-ink)]">
                  {mean !== null ? mean.toFixed(2) : "—"}
                  <span className={cn("ml-1 text-[10px]", unstable ? "text-[var(--color-rt-pip-error)]" : "text-[var(--color-rt-muted)]")}>
                    ±{std.toFixed(2)}
                  </span>
                </span>
              </div>
              {/* min–max range track with mean tick + OOF diamond */}
              <div className="relative mt-1 h-[6px] rounded-full" style={{ background: "color-mix(in oklab, var(--color-rt-line) 80%, transparent)" }}>
                <div
                  className="absolute top-0 h-full rounded-full"
                  style={{ left: `${left}%`, width: `${Math.max(0, right - left)}%`, background: color, opacity: 0.55 }}
                />
                <div className="absolute top-[-1px] h-[8px] w-[2px]" style={{ left: `calc(${meanPct}% - 1px)`, background: color }} />
                {oofPct !== null && (
                  <div
                    className="absolute top-[1px] h-[4px] w-[4px] rotate-45 border border-white"
                    style={{ left: `calc(${oofPct}% - 2px)`, background: "var(--color-rt-pip-ok)" }}
                    title={`OOF dice ${r.oof_dice?.toFixed(2)}`}
                  />
                )}
              </div>
            </div>
          );
        })}
      </div>
    </Section>
  );
}

// ── progress ───────────────────────────────────────────────────────────────

function CrossvalProgress({
  completed,
  total,
  currentFold,
}: {
  completed: number;
  total: number;
  currentFold: number | "ensemble" | null;
}) {
  const pips = total || 1;
  const label =
    currentFold === "ensemble"
      ? "running ensemble"
      : currentFold != null
        ? `running fold ${currentFold}`
        : "starting cross-validation";
  return (
    <Section title="Cross-validation">
      <div className="rounded-[var(--radius-rt-sm)] bg-[color-mix(in_oklab,var(--color-rt-accent)_8%,var(--color-rt-paper))] p-2.5">
        <div className="flex items-center gap-1.5 text-[11px] text-[var(--color-rt-accent)]">
          <Loader2 size={12} className="animate-spin" />
          <span>{label} · {completed}/{total || "?"}</span>
        </div>
        <div className="mt-2 flex gap-1">
          {Array.from({ length: pips }).map((_, i) => (
            <div
              key={i}
              className="h-1.5 flex-1 rounded-full"
              style={{
                background:
                  i < completed
                    ? "var(--color-rt-accent)"
                    : "color-mix(in oklab, var(--color-rt-line) 80%, transparent)",
              }}
            />
          ))}
        </div>
      </div>
    </Section>
  );
}

// ── small shared bits ──────────────────────────────────────────────────────

function ExportLink({
  href,
  icon,
  label,
  download,
  newTab,
}: {
  href: string;
  icon: React.ReactNode;
  label: string;
  download?: string;
  newTab?: boolean;
}) {
  return (
    <a
      href={href}
      download={download}
      target={newTab ? "_blank" : undefined}
      rel={newTab ? "noopener noreferrer" : undefined}
      className="inline-flex items-center gap-1 rounded-[var(--radius-rt-sm)] border border-[var(--color-rt-line)] px-2 py-1 text-[11px] font-medium text-[var(--color-rt-muted)] transition-colors hover:bg-[var(--color-rt-mist)] hover:text-[var(--color-rt-ink)]"
    >
      {icon}
      {label}
    </a>
  );
}

function Section({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <section>
      <h3 className="mb-1.5 text-[10px] font-semibold uppercase tracking-[0.18em] text-[var(--color-rt-muted)]">
        {title}
      </h3>
      <div className="space-y-1">{children}</div>
    </section>
  );
}

function rowLabel(e: CrossvalEntry): string {
  return e.kind === "ensemble" ? "ensemble" : `fold ${e.fold}`;
}

function worstLabel(e: CrossvalEntry): { label_name: string; dice: number | null } | null {
  if (!e.metrics || e.metrics.length === 0) return null;
  let worst: { label_name: string; dice: number | null } | null = null;
  for (const m of e.metrics) {
    if (m.dice === null) continue;
    if (worst === null || (worst.dice !== null && m.dice < worst.dice)) {
      worst = { label_name: m.label_name, dice: m.dice };
    }
  }
  return worst;
}
