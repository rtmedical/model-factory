"use client";

import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import {
  Activity,
  Database,
  Filter,
  Gauge,
  Layers,
  Loader2,
  RefreshCcw,
  Search,
  ShieldCheck,
} from "lucide-react";
import { useEffect, useMemo, useState } from "react";

import {
  deleteModelThemeApi,
  getModelThemes,
  getModels,
  getVerdictsSummary,
  listPlannedTrainings,
  setModelThemeApi,
  type ModelInfo,
  type VerdictSummary,
} from "@/lib/api";
import { useQAStore } from "@/lib/store";
import { cn, regionLabel } from "@/lib/utils";

import { TrainingCalendar } from "@/components/dashboard/TrainingCalendar";
import { ModelCard, REGION_ICON } from "./ModelCard";

// Canonical order for the region sections — brain/HN/pelvis first to
// match how the team reasons about the catalogue (clinical region
// hierarchy), then the wider-CT regions, then any unrecognized region
// the backend has surfaced.
const REGION_ORDER = [
  "brain_mr",
  "hn_ct",
  "pelvis_ct",
  "abdomen_ct",
  "thorax_ct",
  "whole_body_ct",
] as const;

export function ModelCatalog() {
  const enterWorkspace = useQAStore((s) => s.enterWorkspace);
  const themesById = useQAStore((s) => s.modelThemesById);
  const setModelThemesMap = useQAStore((s) => s.setModelThemesMap);
  const setLocalTheme = useQAStore((s) => s.setModelTheme);
  const reviewer = useQAStore((s) => s.reviewer);
  const qc = useQueryClient();
  const [query, setQuery] = useState("");
  const [region, setRegion] = useState<string>("all");
  const [approvalFilter, setApprovalFilter] = useState<
    "all" | "approved" | "rejected" | "pending"
  >("all");

  // The catalog only needs the list of trained checkpoints — that's a pure
  // filesystem scan and always succeeds. Cohort manifest (cases) is only
  // needed once the user enters the workspace, so we don't gate the grid
  // on `/api/cohort`. Refetch every 30s so a finishing training run flips
  // the card status — and re-anchors the live ETA calendar — without a
  // manual reload.
  const models = useQuery({
    queryKey: ["models"],
    queryFn: getModels,
    refetchInterval: 30_000,
  });
  const verdicts = useQuery({
    queryKey: ["verdicts-summary"],
    queryFn: getVerdictsSummary,
    refetchInterval: 60_000,
  });
  // Queued future trainings for the calendar's planned bars. Slower cadence
  // than live models — the queue only changes on a manual schedule edit.
  const planned = useQuery({
    queryKey: ["planned-trainings"],
    queryFn: listPlannedTrainings,
    refetchInterval: 60_000,
  });
  const themes = useQuery({
    queryKey: ["model-themes"],
    queryFn: getModelThemes,
  });

  // Mirror the server-side theme map into Zustand so the store remains
  // the single source of truth for the rest of the app.
  useEffect(() => {
    if (!themes.data) return;
    const next: Record<string, string> = {};
    for (const [mid, t] of Object.entries(themes.data)) {
      next[mid] = t.color_key;
    }
    setModelThemesMap(next);
  }, [themes.data, setModelThemesMap]);

  const setTheme = useMutation({
    mutationFn: async (args: { model_id: string; color_key: string | null }) => {
      if (args.color_key === null) {
        await deleteModelThemeApi(args.model_id);
        return null;
      }
      return setModelThemeApi(args.model_id, args.color_key, reviewer);
    },
    onSuccess: () => qc.invalidateQueries({ queryKey: ["model-themes"] }),
  });

  function handleColorChange(model_id: string, color_key: string | null) {
    // Optimistic update — the card recolors before the server round-trip.
    setLocalTheme(model_id, color_key);
    setTheme.mutate({ model_id, color_key });
  }

  const verdictByModel = useMemo(() => {
    const out: Record<string, VerdictSummary> = {};
    for (const v of verdicts.data ?? []) out[v.model_id] = v;
    return out;
  }, [verdicts.data]);

  const filtered = useMemo(() => {
    const all = models.data ?? [];
    const needle = query.trim().toLowerCase();
    return all.filter((m) => {
      if (region !== "all" && m.region !== region) return false;
      if (approvalFilter !== "all" && (m.approval_status ?? "pending") !== approvalFilter)
        return false;
      if (!needle) return true;
      return (
        m.dataset_name.toLowerCase().includes(needle) ||
        m.plans.toLowerCase().includes(needle) ||
        m.trainer.toLowerCase().includes(needle) ||
        (m.region ?? "").toLowerCase().includes(needle)
      );
    });
  }, [models.data, query, region, approvalFilter]);

  // Bucket the filtered list by region, sort within each, and emit the
  // canonical-order list of (region, items) for the section render.
  // - training cards float to the top so live runs never get buried.
  // - within the same status, higher val_dice ranks ahead (nulls last).
  // - ties break by dataset_name lexicographically for stable ordering.
  const grouped = useMemo(() => {
    const buckets: Record<string, ModelInfo[]> = {};
    for (const m of filtered) {
      const key = (m.region ?? "other") as string;
      (buckets[key] ??= []).push(m);
    }
    const STATUS_RANK: Record<string, number> = {
      training: 0,
      done: 1,
      stopped: 2,
      failed: 3,
    };
    // Within the same lifecycle status, surface QA state as a status board:
    // approved first, then pending, then rejected (still visible, not buried).
    const APPROVAL_RANK: Record<string, number> = {
      approved: 0,
      pending: 1,
      rejected: 2,
    };
    for (const k of Object.keys(buckets)) {
      buckets[k].sort((a, b) => {
        const sr = (STATUS_RANK[a.status] ?? 9) - (STATUS_RANK[b.status] ?? 9);
        if (sr) return sr;
        const ar =
          (APPROVAL_RANK[a.approval_status ?? "pending"] ?? 1) -
          (APPROVAL_RANK[b.approval_status ?? "pending"] ?? 1);
        if (ar) return ar;
        const aDice = a.val_mean_fg_dice ?? -1;
        const bDice = b.val_mean_fg_dice ?? -1;
        if (aDice !== bDice) return bDice - aDice;
        return a.dataset_name.localeCompare(b.dataset_name);
      });
    }
    const ordered: { region: string; items: ModelInfo[] }[] = [];
    for (const r of REGION_ORDER) {
      if (buckets[r]?.length) ordered.push({ region: r, items: buckets[r] });
    }
    const known = new Set<string>(REGION_ORDER);
    const extras = Object.keys(buckets)
      .filter((k) => !known.has(k))
      .sort();
    for (const r of extras) ordered.push({ region: r, items: buckets[r] });
    return ordered;
  }, [filtered]);

  // KPI rollup over *all* trained models (not the filtered view). The KPI
  // rail is the catalog's "stand-back" pulse; the filter only drives the
  // grid below it.
  const kpi = useMemo(() => {
    const all = models.data ?? [];
    const liveFolds = all.reduce(
      (n, m) => n + m.folds.filter((f) => f.status === "training").length,
      0,
    );
    const done = all.filter((m) => m.status === "done").length;
    const warm = all.filter((m) => m.cached_count > 0).length;
    const approved = all.filter((m) => m.approval_status === "approved").length;
    const rejected = all.filter((m) => m.approval_status === "rejected").length;
    const pending = all.length - approved - rejected;
    const diced = all
      .map((m) => m.val_mean_fg_dice)
      .filter((d): d is number => typeof d === "number" && !Number.isNaN(d));
    const avgDice = diced.length
      ? diced.reduce((a, b) => a + b, 0) / diced.length
      : null;
    return { total: all.length, liveFolds, done, warm, approved, rejected, pending, avgDice };
  }, [models.data]);

  // Filter pills track the regions actually present (canonical order first,
  // then any extras), so newly-trained regions like abdomen/thorax/
  // whole-body become filterable without a code change.
  const availableRegions = useMemo(() => {
    const present = new Set(
      (models.data ?? [])
        .map((m) => m.region)
        .filter((r): r is string => !!r),
    );
    const known = REGION_ORDER.filter((r) => present.has(r));
    const extras = [...present]
      .filter((r) => !(REGION_ORDER as readonly string[]).includes(r))
      .sort();
    return ["all", ...known, ...extras];
  }, [models.data]);

  return (
    <div className="min-h-0 flex-1 overflow-y-auto pr-1">
      <div className="flex flex-col gap-4">
        {/* ── identity + controls ──────────────────────────────────── */}
        <div className="flex flex-col gap-3 sm:flex-row sm:items-end sm:justify-between">
          <div>
            <div className="inline-flex items-center gap-1.5 text-[10.5px] font-medium uppercase tracking-[0.18em] text-[var(--color-rt-muted)]">
              <span className="inline-block h-1.5 w-1.5 rounded-full bg-[var(--color-rt-pip-ok)] rt-pulse-dot" />
              factory control
            </div>
            <h1 className="rt-display mt-1 text-[24px] font-semibold leading-tight tracking-tight text-[var(--color-rt-ink)] sm:text-[28px]">
              Model catalog
            </h1>
          </div>
          <div className="flex flex-wrap items-center gap-2">
            <div className="relative">
              <Search
                size={13}
                className="pointer-events-none absolute left-2.5 top-1/2 -translate-y-1/2 text-[var(--color-rt-muted)]"
              />
              <input
                type="text"
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                placeholder="filter by name / plans / region"
                className="w-60 rounded-full border border-[var(--color-rt-line)] bg-[var(--color-rt-paper)] py-1.5 pl-8 pr-3 text-[12px] text-[var(--color-rt-ink)] placeholder:text-[var(--color-rt-muted)] focus:border-[var(--color-rt-accent)] focus:outline-none"
              />
            </div>
            <div className="inline-flex flex-wrap rounded-full border border-[var(--color-rt-line)] bg-[var(--color-rt-paper)] p-0.5 text-[11px]">
              {availableRegions.map((r) => (
                <button
                  key={r}
                  type="button"
                  onClick={() => setRegion(r)}
                  className={cn(
                    "rounded-full px-2.5 py-1 transition-colors",
                    region === r
                      ? "bg-[color-mix(in_oklab,var(--color-rt-accent)_12%,var(--color-rt-paper))] text-[var(--color-rt-accent)]"
                      : "text-[var(--color-rt-muted)] hover:bg-[var(--color-rt-mist)]",
                  )}
                >
                  {r === "all" ? (
                    <span className="inline-flex items-center gap-1">
                      <Filter size={11} />
                      all
                    </span>
                  ) : (
                    regionLabel(r)
                  )}
                </button>
              ))}
            </div>
            <div className="inline-flex rounded-full border border-[var(--color-rt-line)] bg-[var(--color-rt-paper)] p-0.5 text-[11px]">
              {([
                ["all", "QA: all"],
                ["approved", "approved"],
                ["rejected", "rejected"],
                ["pending", "in review"],
              ] as const).map(([key, label]) => (
                <button
                  key={key}
                  type="button"
                  onClick={() => setApprovalFilter(key)}
                  className={cn(
                    "rounded-full px-2.5 py-1 transition-colors",
                    approvalFilter === key
                      ? key === "approved"
                        ? "bg-[color-mix(in_oklab,var(--color-rt-pip-ok)_14%,var(--color-rt-paper))] text-[var(--color-rt-pip-ok)]"
                        : key === "rejected"
                          ? "bg-[color-mix(in_oklab,var(--color-rt-pip-error)_12%,var(--color-rt-paper))] text-[var(--color-rt-pip-error)]"
                          : "bg-[color-mix(in_oklab,var(--color-rt-accent)_12%,var(--color-rt-paper))] text-[var(--color-rt-accent)]"
                      : "text-[var(--color-rt-muted)] hover:bg-[var(--color-rt-mist)]",
                  )}
                >
                  {label}
                </button>
              ))}
            </div>
            <button
              type="button"
              onClick={() => {
                models.refetch();
                verdicts.refetch();
              }}
              aria-label="Refresh catalog"
              className="inline-flex h-8 w-8 items-center justify-center rounded-full border border-[var(--color-rt-line)] text-[var(--color-rt-muted)] hover:bg-[var(--color-rt-mist)] hover:text-[var(--color-rt-ink)]"
            >
              <RefreshCcw size={13} className={cn(models.isFetching && "animate-spin")} />
            </button>
          </div>
        </div>

        {/* ── command center: KPI rail + live training calendar ──────── */}
        <div className="grid gap-4 lg:grid-cols-[232px_minmax(0,1fr)]">
          <KpiRail kpi={kpi} />
          <TrainingCalendar models={models.data ?? []} planned={planned.data ?? []} />
        </div>

        {/* ── catalog grid ───────────────────────────────────────────── */}
        <div>
          {models.isLoading && <SkeletonGrid />}
          {models.error instanceof Error && (
            <ErrorBanner message={models.error.message} />
          )}
          {models.data && filtered.length === 0 && <EmptyState />}
          {models.data && filtered.length > 0 && (
            <div className="flex flex-col gap-7">
              {grouped.map(({ region: r, items }) => (
                <section key={r}>
                  <RegionBanner region={r} count={items.length} />
                  <div className="grid grid-cols-1 gap-3 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4">
                    {items.map((m, i) => (
                      <div
                        key={m.model_id}
                        className="rt-rise"
                        style={{ animationDelay: `${Math.min(i, 8) * 35}ms` }}
                      >
                        <ModelCard
                          model={m}
                          verdicts={verdictByModel[m.model_id] ?? null}
                          override={themesById[m.model_id] ?? null}
                          onOpen={enterWorkspace}
                          onColorChange={(key) => handleColorChange(m.model_id, key)}
                        />
                      </div>
                    ))}
                  </div>
                </section>
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

type Kpi = {
  total: number;
  liveFolds: number;
  done: number;
  warm: number;
  approved: number;
  rejected: number;
  pending: number;
  avgDice: number | null;
};

// Instrument-cluster KPI rail. Vertical stack on wide screens (sits beside
// the calendar), 2×2 grid on narrow. Reads as a console gauge bank, not a
// row of marketing stat cards.
function KpiRail({ kpi }: { kpi: Kpi }) {
  const cells = [
    {
      label: "checkpoints",
      value: kpi.total,
      sub: kpi.done ? `${kpi.done} fully trained` : "no completed runs",
      icon: <Layers size={13} />,
      tone: "ink" as const,
    },
    {
      label: "folds live",
      value: kpi.liveFolds,
      sub: kpi.liveFolds > 0 ? "training now" : "GPUs idle",
      icon:
        kpi.liveFolds > 0 ? (
          <Loader2 size={13} className="animate-spin" />
        ) : (
          <Activity size={13} />
        ),
      tone: kpi.liveFolds > 0 ? ("accent" as const) : ("muted" as const),
    },
    {
      label: "QA approved",
      value: kpi.approved,
      sub:
        kpi.rejected > 0 || kpi.pending > 0
          ? `${kpi.rejected} rejected · ${kpi.pending} in review`
          : "none signed off yet",
      icon: <ShieldCheck size={13} />,
      tone: kpi.approved > 0 ? ("ok" as const) : ("muted" as const),
    },
    {
      label: "mean val dice",
      value: kpi.avgDice === null ? "—" : kpi.avgDice.toFixed(3),
      sub: "across scored models",
      icon: <Gauge size={13} />,
      tone: "ink" as const,
    },
    {
      label: "warm in cache",
      value: kpi.warm,
      sub: `${kpi.warm}/${kpi.total} dragonfly`,
      icon: <Database size={13} />,
      tone: kpi.warm > 0 ? ("accent" as const) : ("muted" as const),
    },
  ];
  return (
    <div className="rt-console rt-grain relative grid grid-cols-2 gap-px overflow-hidden bg-[var(--color-rt-line)] lg:grid-cols-1">
      {cells.map((c) => (
        <div
          key={c.label}
          className="relative z-[1] flex flex-col gap-1 bg-[var(--color-rt-paper)] px-3.5 py-3"
        >
          <div
            className={cn(
              "flex items-center gap-1.5 text-[10px] font-medium uppercase tracking-[0.14em]",
              c.tone === "accent"
                ? "text-[var(--color-rt-accent)]"
                : c.tone === "ok"
                  ? "text-[var(--color-rt-pip-ok)]"
                  : "text-[var(--color-rt-muted)]",
            )}
          >
            {c.icon}
            {c.label}
          </div>
          <div
            className={cn(
              "rt-display text-[26px] font-semibold leading-none tracking-tight tabular-nums",
              c.tone === "accent"
                ? "text-[var(--color-rt-accent)]"
                : c.tone === "ok"
                  ? "text-[var(--color-rt-pip-ok)]"
                  : "text-[var(--color-rt-ink)]",
            )}
          >
            {c.value}
          </div>
          <div className="text-[10px] text-[var(--color-rt-muted)]">{c.sub}</div>
        </div>
      ))}
    </div>
  );
}

// Section header for each region band. Icon + label on the left, count
// chip on the right, hairline rule running between them.
function RegionBanner({ region, count }: { region: string; count: number }) {
  const Icon = REGION_ICON[region] ?? null;
  return (
    <div className="mb-3 flex items-center gap-3">
      <div className="flex items-center gap-2 text-[var(--color-rt-ink)]">
        {Icon ? (
          <span
            className="inline-flex h-7 w-7 items-center justify-center rounded-[var(--radius-rt-sm)] bg-[color-mix(in_oklab,var(--color-rt-accent)_8%,var(--color-rt-paper))] text-[var(--color-rt-accent)]"
            aria-hidden
          >
            <Icon size={14} />
          </span>
        ) : null}
        <h2 className="rt-display text-[14px] font-semibold tracking-tight">
          {regionLabel(region)}
        </h2>
      </div>
      <span className="inline-flex items-center rounded-full border border-[var(--color-rt-line)] bg-[var(--color-rt-paper)] px-2 py-0.5 font-mono text-[10.5px] text-[var(--color-rt-muted)]">
        {count}
      </span>
      <div className="h-px flex-1 bg-[var(--color-rt-line)]" aria-hidden />
    </div>
  );
}

function SkeletonGrid() {
  return (
    <div className="grid grid-cols-1 gap-3 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4">
      {Array.from({ length: 8 }).map((_, i) => (
        <div key={i} className="rt-card h-[180px] animate-pulse bg-[var(--color-rt-mist)]" />
      ))}
    </div>
  );
}

function ErrorBanner({ message }: { message: string }) {
  return (
    <div className="rt-card border-[var(--color-rt-pip-error)] p-4 text-[12.5px] text-[var(--color-rt-pip-error)]">
      {message}
    </div>
  );
}

function EmptyState() {
  return (
    <div className="rt-card bg-rt-mesh flex flex-col items-center justify-center gap-2 py-16 text-center">
      <div className="rt-display text-[18px] font-semibold text-[var(--color-rt-ink)]">
        No trained checkpoints match
      </div>
      <p className="max-w-md text-[12px] text-[var(--color-rt-muted)]">
        Try clearing the filter, or kick off a training run from the CLI:
        <code className="ml-1 rounded bg-[var(--color-rt-mist)] px-1 py-0.5 font-mono text-[11px]">
          modelfactory train nnunet --dataset DatasetNNN ...
        </code>
      </p>
    </div>
  );
}
