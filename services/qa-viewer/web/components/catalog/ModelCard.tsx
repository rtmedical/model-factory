"use client";

import {
  ArrowRight,
  BadgeCheck,
  BarChart3,
  Brain,
  Check,
  ChevronDown,
  ChevronUp,
  CircleHelp,
  CircleX,
  Clock3,
  Database,
  HeartPulse,
  Layers,
  Loader2,
  Palette,
  PauseCircle,
  PersonStanding,
  RotateCcw,
  Stethoscope,
  TriangleAlert,
  Wind,
  XCircle,
} from "lucide-react";
import { useEffect, useRef, useState } from "react";

import {
  crossvalRollupHtmlUrl,
  REJECT_REASON_LABEL,
  type ApprovalStatus,
  type FoldProgress,
  type ModelInfo,
  type ModelStatus,
  type RejectReason,
  type VerdictSummary,
} from "@/lib/api";
import {
  etaMs,
  foldSchedule,
  formatClockDay,
  formatDuration,
  useNow,
} from "@/lib/eta";
import { cn, formatDice, regionLabel } from "@/lib/utils";

export const REGION_ICON: Record<string, React.ComponentType<{ size?: number; className?: string }>> = {
  brain_mr: Brain,
  hn_ct: Stethoscope,
  pelvis_ct: Layers,
  abdomen_ct: HeartPulse,
  thorax_ct: Wind,
  whole_body_ct: PersonStanding,
};

// Curated palette. Must stay aligned with PALETTE in
// src/modelfactory/qa/themes.py and the --card-* CSS vars in globals.css.
export const PALETTE = [
  "slate",
  "sky",
  "indigo",
  "violet",
  "fuchsia",
  "rose",
  "amber",
  "lime",
  "emerald",
  "teal",
] as const;
export type PaletteKey = (typeof PALETTE)[number];

export type ResolvedColor = {
  key: PaletteKey;
  // The reason this card carries this swatch — surfaces in the picker so
  // the operator knows whether they're overriding a real status signal.
  origin: "user" | "approval" | "verdict" | "status";
};

// Resolve in priority order: user override → model QA approval (the derived
// approved=green / rejected=red signal) → unresolved-review → lifecycle
// status. The custom color picker still wins (per the chosen color rule);
// the approval is never hidden because it ALSO renders as a ✓/✗ badge.
export function resolveCardColor(
  model: ModelInfo,
  verdicts: VerdictSummary | null,
  override: string | null,
): ResolvedColor {
  if (override && (PALETTE as readonly string[]).includes(override)) {
    return { key: override as PaletteKey, origin: "user" };
  }
  const approval = model.approval_status ?? "pending";
  if (approval === "approved") return { key: "emerald", origin: "approval" };
  if (approval === "rejected") return { key: "rose", origin: "approval" };
  // Pending: surface an outstanding needs-review, else the lifecycle status.
  if (verdicts && verdicts.needs_review > 0) {
    return { key: "violet", origin: "verdict" };
  }
  switch (model.status) {
    case "training":
      return { key: "sky", origin: "status" };
    case "stopped":
      return { key: "violet", origin: "status" };
    case "failed":
      return { key: "rose", origin: "status" };
    case "done":
    default:
      return { key: "slate", origin: "status" };
  }
}

export function ModelCard({
  model,
  verdicts,
  override,
  onOpen,
  onColorChange,
}: {
  model: ModelInfo;
  verdicts: VerdictSummary | null;
  override: string | null;
  onOpen: (m: ModelInfo) => void;
  onColorChange: (key: string | null) => void;
}) {
  const Icon = REGION_ICON[model.region ?? ""] ?? Layers;
  const displayDataset = model.dataset_name.replace(/^Dataset(\d+)_/, "D$1 ");
  const color = resolveCardColor(model, verdicts, override);
  const [pickerOpen, setPickerOpen] = useState(false);
  const [diceOpen, setDiceOpen] = useState(false);
  const pickerRef = useRef<HTMLDivElement | null>(null);

  const hasVerdicts = !!verdicts && verdicts.total > 0;
  const hasCached = model.cohort_size > 0;
  const approval = model.approval_status ?? "pending";
  const hasApproval = approval === "approved" || approval === "rejected";
  // Top reject reasons (worst-first), shown on a rejected card so the
  // catalog reads as "what to fix" at a glance.
  const topRejectReasons =
    approval === "rejected" && verdicts?.reject_reasons
      ? Object.entries(verdicts.reject_reasons)
          .sort(([, a], [, b]) => b - a)
          .map(([k]) => REJECT_REASON_LABEL[k as RejectReason] ?? k)
      : [];
  const hasDiceExpansion =
    !!model.per_class_dice && Object.keys(model.per_class_dice).length > 0;

  useEffect(() => {
    if (!pickerOpen) return;
    const onDocClick = (e: MouseEvent) => {
      if (pickerRef.current && !pickerRef.current.contains(e.target as Node)) {
        setPickerOpen(false);
      }
    };
    document.addEventListener("mousedown", onDocClick);
    return () => document.removeEventListener("mousedown", onDocClick);
  }, [pickerOpen]);

  // Left-edge status accent strip. `done` stays neutral so the catalog
  // reads as a status board — only live / interrupted runs carry colour.
  const statusAccent: Record<ModelStatus, string | null> = {
    training: "var(--card-sky-bg)",
    stopped: "var(--card-violet-bg)",
    failed: "var(--color-rt-pip-error)",
    done: null,
  };
  const accent = statusAccent[model.status];

  return (
    <div
      className={cn(
        "rt-card group relative flex h-full flex-col items-stretch overflow-hidden p-4 text-left transition-all duration-200",
        // `--rt-card-ring` is set dynamically below; the class itself is a
        // static string so Tailwind's scanner emits it.
        "hover:-translate-y-0.5 hover:border-[var(--rt-card-ring)] hover:shadow-[var(--shadow-rt-elevation-2)]",
      )}
      style={
        {
          backgroundColor: `color-mix(in oklab, var(--card-${color.key}-bg) 7%, var(--color-rt-paper))`,
          "--rt-card-ring": `color-mix(in oklab, var(--card-${color.key}-ring) 55%, var(--color-rt-line))`,
          boxShadow:
            color.origin === "user"
              ? `inset 0 0 0 1.5px var(--card-${color.key}-ring), var(--shadow-rt-elevation-1)`
              : undefined,
        } as React.CSSProperties
      }
    >
      {accent && (
        <span
          aria-hidden
          className={cn(
            "pointer-events-none absolute left-0 top-0 h-full w-[3px]",
            model.status === "training" && "animate-pulse",
          )}
          style={{ background: accent }}
        />
      )}
      <button
        type="button"
        onClick={() => onOpen(model)}
        className="flex flex-1 flex-col items-stretch text-left focus-visible:outline-none"
      >
        <div className="flex items-start justify-between gap-2">
          <div className="flex min-w-0 items-center gap-2.5">
            <span
              className="inline-flex h-9 w-9 shrink-0 items-center justify-center rounded-[var(--radius-rt-sm)]"
              style={{
                backgroundColor: `color-mix(in oklab, var(--card-${color.key}-bg) 16%, var(--color-rt-paper))`,
                color: `var(--card-${color.key}-fg)`,
              }}
            >
              <Icon size={17} />
            </span>
            <div className="min-w-0">
              <div className="text-[9.5px] uppercase tracking-[0.16em] text-[var(--color-rt-muted)]">
                {regionLabel(model.region)}
              </div>
              <div className="rt-display truncate text-[15.5px] font-semibold leading-tight text-[var(--color-rt-ink)]">
                {displayDataset}
              </div>
            </div>
          </div>
          <ArrowRight
            size={15}
            className="mt-1.5 shrink-0 text-[var(--color-rt-muted)] transition-transform group-hover:translate-x-0.5"
          />
        </div>

        {/* config chips — replaces the old clamped 3-line paragraph */}
        <div className="mt-3 flex flex-wrap items-center gap-1.5">
          <Chip>{model.plans.replace("nnUNet", "").replace("Plans", "") || "Plans"}</Chip>
          <Chip>{model.configuration}</Chip>
          <Chip muted>{model.trainer.replace("nnUNetTrainer", "Trainer·")}</Chip>
        </div>

        {/* two headline numbers — folds + val dice */}
        <div className="mt-3 grid grid-cols-2 gap-2">
          <Stat
            label="folds"
            value={String(model.available_folds.length)}
            hint={`folds ${model.available_folds.join(", ")} · 5-fold CV`}
          />
          <Stat label="val dice" value={formatDice(model.val_mean_fg_dice)} />
        </div>

        {/* per-fold progress (+ live ETA on training folds) */}
        {model.folds.length > 0 ? (
          <FoldStack folds={model.folds} totalEpochs={model.total_epochs} />
        ) : model.current_epoch !== null && model.total_epochs !== null ? (
          <EpochRow
            current={model.current_epoch}
            total={model.total_epochs}
            status={model.status}
          />
        ) : null}

        {(hasApproval || hasVerdicts || hasCached) && (
          <div className="mt-3 flex items-center justify-between gap-2 border-t border-[var(--color-rt-line)] pt-2.5 text-[11px]">
            <div className="flex flex-wrap items-center gap-1.5">
              {(hasApproval || hasVerdicts) && <ApprovalBadge status={approval} />}
              {hasVerdicts && (
                <>
                  <VerdictPip kind="accept" count={verdicts!.accept} />
                  <VerdictPip kind="reject" count={verdicts!.reject} />
                  <VerdictPip kind="needs_review" count={verdicts!.needs_review} />
                </>
              )}
            </div>
            {hasCached && (
              <CachedBadge hits={model.cached_count} total={model.cohort_size} />
            )}
          </div>
        )}

        {topRejectReasons.length > 0 && (
          <div className="mt-2 flex items-start gap-1.5 text-[10px] leading-snug text-[var(--color-rt-pip-error)]">
            <TriangleAlert size={11} className="mt-0.5 shrink-0" />
            <span className="text-[var(--color-rt-muted)]">
              <span className="font-medium text-[var(--color-rt-pip-error)]">to fix:</span>{" "}
              {topRejectReasons.slice(0, 3).join(" · ")}
            </span>
          </div>
        )}
      </button>

      {hasDiceExpansion && (
        <div className="mt-2">
          <button
            type="button"
            onClick={(e) => {
              e.stopPropagation();
              setDiceOpen((x) => !x);
            }}
            className="inline-flex w-full items-center justify-between rounded-[var(--radius-rt-sm)] border border-[var(--color-rt-line)] bg-[color-mix(in_oklab,var(--color-rt-mist)_40%,var(--color-rt-paper))] px-2.5 py-1.5 text-[10.5px] font-medium uppercase tracking-[0.14em] text-[var(--color-rt-muted)] hover:bg-[var(--color-rt-mist)] hover:text-[var(--color-rt-ink)]"
            aria-expanded={diceOpen}
          >
            <span>per-struct dice</span>
            {diceOpen ? <ChevronUp size={12} /> : <ChevronDown size={12} />}
          </button>
          {diceOpen && <DiceExpansion perClass={model.per_class_dice!} />}
        </div>
      )}

      {/* Top-right cluster: status pill + color picker. Outside the
          card-button so clicks here don't navigate. */}
      <div className="absolute right-2 top-2 flex items-center gap-1">
        <StatusPill status={model.status} />
        {/* Cross-validation rollup — opens the server-rendered, self-contained
            report in a new tab. stopPropagation so it doesn't enter the
            workspace. Hover-revealed like the color picker. */}
        <a
          href={crossvalRollupHtmlUrl(model.model_id)}
          target="_blank"
          rel="noopener noreferrer"
          aria-label="cross-validation rollup report"
          title="cross-validation rollup report"
          onClick={(e) => e.stopPropagation()}
          className="inline-flex h-6 w-6 items-center justify-center rounded-full border border-[var(--color-rt-line)] bg-[var(--color-rt-paper)] text-[var(--color-rt-muted)] opacity-0 transition-opacity hover:text-[var(--color-rt-ink)] group-hover:opacity-100 focus-visible:opacity-100"
        >
          <BarChart3 size={11} />
        </a>
        <button
          type="button"
          aria-label="customize card color"
          title="customize card color"
          onClick={(e) => {
            e.stopPropagation();
            setPickerOpen((x) => !x);
          }}
          className="inline-flex h-6 w-6 items-center justify-center rounded-full border border-[var(--color-rt-line)] bg-[var(--color-rt-paper)] text-[var(--color-rt-muted)] opacity-0 transition-opacity hover:text-[var(--color-rt-ink)] group-hover:opacity-100 focus-visible:opacity-100"
        >
          <Palette size={11} />
        </button>
      </div>

      {pickerOpen && (
        <div
          ref={pickerRef}
          className="absolute right-2 top-9 z-20 flex flex-col gap-1 rounded-[var(--radius-rt-sm)] border border-[var(--color-rt-line)] bg-[var(--color-rt-paper)] p-2 shadow-[var(--shadow-rt-elevation-2)]"
          onClick={(e) => e.stopPropagation()}
        >
          <div className="grid grid-cols-5 gap-1">
            {PALETTE.map((key) => (
              <button
                key={key}
                type="button"
                onClick={(e) => {
                  e.stopPropagation();
                  onColorChange(key);
                  setPickerOpen(false);
                }}
                title={key}
                aria-label={`color ${key}`}
                className="relative inline-flex h-5 w-5 items-center justify-center rounded-full border border-[var(--color-rt-line)]"
                style={{ backgroundColor: `var(--card-${key}-bg)` }}
              >
                {color.key === key && (
                  <Check size={10} className="text-white drop-shadow" />
                )}
              </button>
            ))}
          </div>
          {override && (
            <button
              type="button"
              onClick={(e) => {
                e.stopPropagation();
                onColorChange(null);
                setPickerOpen(false);
              }}
              className="mt-1 inline-flex items-center justify-center gap-1 rounded border border-[var(--color-rt-line)] px-2 py-1 text-[10.5px] text-[var(--color-rt-muted)] hover:bg-[var(--color-rt-mist)] hover:text-[var(--color-rt-ink)]"
            >
              <RotateCcw size={10} />
              reset to {color.origin === "user" ? "auto" : color.origin}
            </button>
          )}
          <div className="px-1 text-[10px] leading-tight text-[var(--color-rt-muted)]">
            {color.origin === "user"
              ? "user override"
              : color.origin === "approval"
                ? "color from QA approval"
                : color.origin === "verdict"
                  ? "color from verdicts"
                  : "color from status"}
          </div>
        </div>
      )}
    </div>
  );
}

function Chip({
  children,
  muted = false,
}: {
  children: React.ReactNode;
  muted?: boolean;
}) {
  return (
    <span
      className={cn(
        "inline-flex items-center rounded-md border px-1.5 py-0.5 font-mono text-[10px] leading-none",
        muted
          ? "border-transparent bg-[color-mix(in_oklab,var(--color-rt-mist)_70%,var(--color-rt-paper))] text-[var(--color-rt-muted)]"
          : "border-[var(--color-rt-line)] bg-[var(--color-rt-paper)] text-[var(--color-rt-ink-2)]",
      )}
    >
      {children}
    </span>
  );
}

function StatusPill({ status }: { status: ModelStatus }) {
  const meta: Record<ModelStatus, { label: string; color: string; Icon: React.ComponentType<{ size?: number; className?: string }> }> = {
    training: { label: "training", color: "var(--card-sky-bg)", Icon: Loader2 },
    stopped: { label: "stopped", color: "var(--card-violet-bg)", Icon: PauseCircle },
    done: { label: "done", color: "var(--color-rt-pip-ok)", Icon: BadgeCheck },
    failed: { label: "failed", color: "var(--color-rt-pip-error)", Icon: XCircle },
  };
  // Hide the "done" pill — it's the default; showing it on every card adds
  // noise. training / stopped / failed always show.
  if (status === "done") return null;
  const { label, color, Icon } = meta[status];
  return (
    <span
      className="inline-flex items-center gap-1 rounded-full px-2 py-0.5 text-[10px] font-semibold uppercase tracking-wide"
      style={{
        color,
        backgroundColor: `color-mix(in oklab, ${color} 14%, var(--color-rt-paper))`,
      }}
    >
      <Icon size={10} className={status === "training" ? "animate-spin" : ""} />
      {label}
    </span>
  );
}

// Per-fold progress bars for an nnUNetv2 5-fold CV run. Live folds float to
// the top and carry a ticking ETA chip.
function FoldStack({
  folds,
  totalEpochs,
}: {
  folds: FoldProgress[];
  totalEpochs: number | null;
}) {
  const ordered = [...folds].sort((a, b) => {
    const aTraining = a.status === "training" ? 1 : 0;
    const bTraining = b.status === "training" ? 1 : 0;
    if (aTraining !== bTraining) return bTraining - aTraining;
    return a.fold - b.fold;
  });
  const liveCount = ordered.filter((f) => f.status === "training").length;
  return (
    <div className="mt-3 flex flex-col gap-1.5 text-[10.5px] text-[var(--color-rt-muted)]">
      <div className="flex items-baseline justify-between">
        <span className="uppercase tracking-wide">folds</span>
        <span className="text-[10px] text-[var(--color-rt-muted)]/80">
          5-fold CV{liveCount > 0 ? ` · ${liveCount} live` : ""}
        </span>
      </div>
      <ul className="flex flex-col gap-1">
        {ordered.map((f) =>
          f.status === "training" ? (
            <TrainingFoldRow key={f.fold} fold={f} totalEpochs={totalEpochs} />
          ) : (
            <FoldRow key={f.fold} fold={f} totalEpochs={totalEpochs} />
          ),
        )}
      </ul>
    </div>
  );
}

const FOLD_FILL: Record<ModelStatus, string> = {
  training: "var(--card-sky-bg)",
  stopped: "var(--card-violet-bg)",
  done: "var(--color-rt-pip-ok)",
  failed: "var(--color-rt-pip-error)",
};
const FOLD_LABEL: Record<ModelStatus, string> = {
  training: "live",
  stopped: "paused",
  done: "done",
  failed: "failed",
};

// Static (non-training) fold row.
function FoldRow({
  fold,
  totalEpochs,
}: {
  fold: FoldProgress;
  totalEpochs: number | null;
}) {
  const total = fold.total_epochs ?? totalEpochs ?? 1000;
  const epoch = fold.current_epoch;
  const pct =
    epoch !== null && total
      ? Math.max(0, Math.min(100, (epoch / Math.max(1, total)) * 100))
      : fold.has_checkpoint_best
        ? 100
        : 0;
  return (
    <li className="flex items-center gap-2">
      <span className="w-9 shrink-0 font-mono text-[10.5px] text-[var(--color-rt-muted)]">
        fold {fold.fold}
      </span>
      <span className="h-1.5 flex-1 overflow-hidden rounded-full bg-[var(--color-rt-mist)]">
        <span
          className="block h-full transition-[width] duration-500 ease-[var(--ease-rt)]"
          style={{ width: `${pct}%`, background: FOLD_FILL[fold.status] }}
        />
      </span>
      <span className="w-[72px] shrink-0 text-right font-mono text-[10.5px] tabular-nums text-[var(--color-rt-ink)]">
        {epoch ?? (fold.has_checkpoint_best ? total : "—")}
        <span className="text-[var(--color-rt-muted)]"> / {total}</span>
      </span>
      <span
        className="w-11 shrink-0 text-right text-[10px] font-medium uppercase tracking-wide"
        style={{ color: FOLD_FILL[fold.status] }}
      >
        {FOLD_LABEL[fold.status]}
      </span>
    </li>
  );
}

// Live training fold row — same bar, but the trailing column becomes a
// ticking countdown ("8h" / "12m") with the projected finish clock in the
// tooltip. Ticks coarsely (10 s) since the card ETA is minute-grained.
function TrainingFoldRow({
  fold,
  totalEpochs,
}: {
  fold: FoldProgress;
  totalEpochs: number | null;
}) {
  const now = useNow(10_000);
  const sched = foldSchedule(fold);
  const total = fold.total_epochs ?? totalEpochs ?? 1000;
  const epoch = fold.current_epoch;
  const pct =
    epoch !== null && total
      ? Math.max(0, Math.min(100, (epoch / Math.max(1, total)) * 100))
      : 0;
  const remaining = etaMs(sched.finishMs, now);
  const finishClock = formatClockDay(sched.finishMs, now);
  return (
    <li
      className="flex items-center gap-2"
      title={
        sched.finishMs
          ? `epoch ${epoch ?? "?"}/${total} · ~${
              sched.secPerEpoch?.toFixed(0) ?? "?"
            }s/epoch · finishes ${finishClock}`
          : `epoch ${epoch ?? "?"}/${total} · estimating rate…`
      }
    >
      <span className="w-9 shrink-0 font-mono text-[10.5px] text-[var(--color-rt-muted)]">
        fold {fold.fold}
      </span>
      <span className="h-1.5 flex-1 overflow-hidden rounded-full bg-[var(--color-rt-mist)]">
        <span
          className="block h-full rt-live-stripe transition-[width] duration-500 ease-[var(--ease-rt)]"
          style={{ width: `${pct}%`, background: FOLD_FILL.training }}
        />
      </span>
      <span className="w-[72px] shrink-0 text-right font-mono text-[10.5px] tabular-nums text-[var(--color-rt-ink)]">
        {epoch ?? "—"}
        <span className="text-[var(--color-rt-muted)]"> / {total}</span>
      </span>
      <span
        className="flex w-11 shrink-0 items-center justify-end gap-0.5 text-right font-mono text-[10px] font-medium tabular-nums"
        style={{ color: FOLD_FILL.training }}
      >
        <Clock3 size={9} />
        {remaining != null ? formatDuration(remaining) : "…"}
      </span>
    </li>
  );
}

function EpochRow({
  current,
  total,
  status,
}: {
  current: number;
  total: number;
  status: ModelStatus;
}) {
  const pct = Math.max(0, Math.min(100, (current / Math.max(1, total)) * 100));
  const annotation: Record<ModelStatus, string | null> = {
    training: "· live",
    stopped: "· paused",
    done: null,
    failed: null,
  };
  return (
    <div className="mt-3 text-[10px] text-[var(--color-rt-muted)]">
      <div className="flex items-baseline justify-between">
        <span className="uppercase tracking-wide">epoch</span>
        <span className="font-mono text-[11px] text-[var(--color-rt-ink)]">
          {current}
          <span className="text-[var(--color-rt-muted)]"> / {total}</span>
          {annotation[status] && (
            <span className="ml-1" style={{ color: FOLD_FILL[status] }}>
              {annotation[status]}
            </span>
          )}
        </span>
      </div>
      <div className="mt-1 h-1 overflow-hidden rounded-full bg-[var(--color-rt-mist)]">
        <div
          className="h-full transition-[width] duration-500 ease-[var(--ease-rt)]"
          style={{ width: `${pct}%`, background: FOLD_FILL[status] }}
        />
      </div>
    </div>
  );
}

function Stat({ label, value, hint }: { label: string; value: string; hint?: string }) {
  return (
    <div className="rounded-[var(--radius-rt-sm)] bg-[color-mix(in_oklab,var(--color-rt-mist)_55%,var(--color-rt-paper))] px-2.5 py-1.5">
      <div className="text-[9.5px] uppercase tracking-[0.14em] text-[var(--color-rt-muted)]">
        {label}
      </div>
      <div
        className="mt-0.5 font-mono text-[14px] tracking-normal text-[var(--color-rt-ink)] tabular-nums"
        title={hint ?? value}
      >
        {value}
      </div>
    </div>
  );
}

function VerdictPip({
  kind,
  count,
}: {
  kind: "accept" | "reject" | "needs_review";
  count: number;
}) {
  if (!count) return null;
  const styles = {
    accept: {
      icon: BadgeCheck,
      color: "text-[var(--color-rt-pip-ok)]",
      bg: "bg-[color-mix(in_oklab,var(--color-rt-pip-ok)_12%,var(--color-rt-paper))]",
    },
    reject: {
      icon: CircleX,
      color: "text-[var(--color-rt-pip-error)]",
      bg: "bg-[color-mix(in_oklab,var(--color-rt-pip-error)_10%,var(--color-rt-paper))]",
    },
    needs_review: {
      icon: CircleHelp,
      color: "text-[var(--color-rt-purple)]",
      bg: "bg-[color-mix(in_oklab,var(--color-rt-purple)_10%,var(--color-rt-paper))]",
    },
  }[kind];
  const Icon = styles.icon;
  return (
    <span
      className={cn(
        "inline-flex items-center gap-1 rounded-full px-2 py-0.5 font-mono text-[10.5px]",
        styles.bg,
        styles.color,
      )}
    >
      <Icon size={11} />
      {count}
    </span>
  );
}

// Model-level QA decision flag — a sibling to CachedBadge. Green ✓ when the
// derived approval is "approved", red ✗ when "rejected", muted "in review"
// while pending. Always rendered (even under a custom card swatch) so the QA
// signal is never hidden by the color picker.
function ApprovalBadge({ status }: { status: ApprovalStatus }) {
  const meta = {
    approved: {
      Icon: BadgeCheck,
      label: "approved",
      cls: "border-[color-mix(in_oklab,var(--color-rt-pip-ok)_35%,transparent)] bg-[color-mix(in_oklab,var(--color-rt-pip-ok)_14%,var(--color-rt-paper))] text-[var(--color-rt-pip-ok)]",
    },
    rejected: {
      Icon: CircleX,
      label: "rejected",
      cls: "border-[color-mix(in_oklab,var(--color-rt-pip-error)_35%,transparent)] bg-[color-mix(in_oklab,var(--color-rt-pip-error)_12%,var(--color-rt-paper))] text-[var(--color-rt-pip-error)]",
    },
    pending: {
      Icon: CircleHelp,
      label: "in review",
      cls: "border-[var(--color-rt-line)] bg-[var(--color-rt-mist)] text-[var(--color-rt-muted)]",
    },
  }[status];
  const { Icon, label, cls } = meta;
  return (
    <span
      className={cn(
        "inline-flex items-center gap-1 rounded-full border px-2 py-0.5 text-[10.5px] font-semibold uppercase tracking-wide",
        cls,
      )}
      title={`QA approval (derived from case verdicts): ${label}`}
    >
      <Icon size={11} />
      {label}
    </span>
  );
}

// Tag showing how many of the model's compatible QA-cohort cases are warm
// in Dragonfly.
function CachedBadge({ hits, total }: { hits: number; total: number }) {
  const isWarm = hits > 0;
  const cls = isWarm
    ? "border-[color-mix(in_oklab,var(--color-rt-accent)_30%,transparent)] bg-[color-mix(in_oklab,var(--color-rt-accent)_10%,var(--color-rt-paper))] text-[var(--color-rt-accent)]"
    : "border-[var(--color-rt-line)] bg-[var(--color-rt-mist)] text-[var(--color-rt-muted)]";
  return (
    <span
      className={cn(
        "inline-flex items-center gap-1 rounded-full border px-2 py-0.5 font-mono text-[10.5px]",
        cls,
      )}
      title={`Dragonfly cache hits across this model's compatible QA cohort cases (${hits}/${total})`}
    >
      <Database size={11} />
      {hits}/{total} cached
    </span>
  );
}

// Worst-first list of per-structure mean dice.
function DiceExpansion({ perClass }: { perClass: Record<string, number> }) {
  const ordered = Object.entries(perClass)
    .filter(([, v]) => typeof v === "number" && !Number.isNaN(v))
    .sort(([, a], [, b]) => a - b);
  const total = ordered.length;
  return (
    <div className="mt-2 rounded-[var(--radius-rt-sm)] border border-[var(--color-rt-line)] bg-[color-mix(in_oklab,var(--color-rt-mist)_25%,var(--color-rt-paper))] p-2">
      <div className="mb-1.5 flex items-center justify-between px-1 text-[10px] uppercase tracking-[0.14em] text-[var(--color-rt-muted)]">
        <span>worst → best</span>
        <span className="font-mono">{total} structures</span>
      </div>
      <ul className="rt-scroll max-h-[180px] space-y-1.5 overflow-y-auto pr-1">
        {ordered.map(([name, value]) => (
          <DiceRow key={name} label={name} value={value} />
        ))}
      </ul>
    </div>
  );
}

function DiceRow({ label, value }: { label: string; value: number }) {
  const pct = Math.max(0, Math.min(100, value * 100));
  const tone =
    value < 0.5
      ? "var(--color-rt-pip-error)"
      : value < 0.7
        ? "var(--color-rt-purple)"
        : "var(--color-rt-pip-ok)";
  const cleaned = label.replace(/_/g, " ");
  return (
    <li className="flex items-center gap-2 text-[10.5px]">
      <span
        className="min-w-0 flex-1 truncate text-[var(--color-rt-ink)]"
        title={label}
      >
        {cleaned}
      </span>
      <span
        className="h-1.5 w-16 overflow-hidden rounded-full bg-[var(--color-rt-mist)]"
        aria-hidden
      >
        <span
          className="block h-full transition-[width] duration-300"
          style={{ width: `${pct}%`, background: tone }}
        />
      </span>
      <span
        className="w-10 shrink-0 text-right font-mono text-[10.5px] tabular-nums"
        style={{ color: tone }}
      >
        {value.toFixed(2)}
      </span>
    </li>
  );
}
