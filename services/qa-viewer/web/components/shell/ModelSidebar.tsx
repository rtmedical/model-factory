"use client";

import { useQuery } from "@tanstack/react-query";
import { CheckCircle2, Layers } from "lucide-react";
import { useMemo } from "react";

import { getModels, type ModelInfo } from "@/lib/api";
import { useQAStore } from "@/lib/store";
import { cn, formatDice, regionLabel } from "@/lib/utils";

export function ModelSidebar() {
  // Always-available source — pure filesystem scan, no cohort dependency.
  const { data, isLoading, error } = useQuery({
    queryKey: ["models"],
    queryFn: getModels,
  });
  const selectedModel = useQAStore((s) => s.selectedModel);
  const selectedCase = useQAStore((s) => s.selectedCase);
  // Reuse enterWorkspace: it sets the model, resets case/prediction, and (idempotently) keeps view=workspace.
  const setModel = useQAStore((s) => s.enterWorkspace);

  const grouped = useMemo(() => groupByRegion(data ?? []), [data]);

  return (
    <aside className="rt-card flex min-h-0 flex-1 flex-col overflow-hidden">
      <div className="flex items-center justify-between border-b border-[var(--color-rt-line)] px-3 py-2">
        <div className="flex items-center gap-2">
          <span className="inline-flex h-6 w-6 items-center justify-center rounded-[var(--radius-rt-sm)] bg-[color-mix(in_oklab,var(--color-rt-accent)_12%,var(--color-rt-paper))] text-[var(--color-rt-accent)]">
            <Layers size={13} />
          </span>
          <h2 className="rt-display text-[13px] font-semibold tracking-wide text-[var(--color-rt-ink)]">
            Trained models
          </h2>
        </div>
        <span className="inline-flex items-center rounded-full border border-[var(--color-rt-line)] bg-[var(--color-rt-paper)] px-2 py-0.5 font-mono text-[10.5px] text-[var(--color-rt-muted)]">
          {data?.length ?? "—"}
        </span>
      </div>

      <div className="flex-1 overflow-y-auto">
        {isLoading && (
          <div className="p-4 text-[12px] text-[var(--color-rt-muted)]">
            Loading checkpoints…
          </div>
        )}
        {error instanceof Error && (
          <div className="p-4 text-[12px] text-[var(--color-rt-pip-error)]">
            {error.message}
          </div>
        )}
        {data &&
          // Show every region present in the model list — preserves the
          // canonical brain/HN/pelvis ordering for backwards-compatibility
          // and appends any new regions (abdomen_ct, thorax_ct, "other")
          // automatically as new datasets are trained.
          orderedRegions(grouped).map((region) => {
            const models = grouped[region] ?? [];
            if (!models.length) return null;
            return (
              <RegionGroup
                key={region}
                region={region}
                models={models}
                selectedId={selectedModel?.model_id}
                compatible={selectedCase?.compatible_models ?? null}
                onSelect={setModel}
              />
            );
          })}
      </div>
    </aside>
  );
}

function RegionGroup({
  region,
  models,
  selectedId,
  compatible,
  onSelect,
}: {
  region: string;
  models: ModelInfo[];
  selectedId?: string | null;
  compatible: string[] | null;
  onSelect: (m: ModelInfo) => void;
}) {
  return (
    <div className="border-b border-[var(--color-rt-line)] last:border-b-0">
      <div className="flex items-center gap-2 bg-[var(--color-rt-mist)] px-3 py-1.5 text-[10px] font-semibold uppercase tracking-[0.16em] text-[var(--color-rt-muted)]">
        <Layers size={11} />
        {regionLabel(region)}
        <span className="ml-auto normal-case tracking-normal text-[11px]">
          {models.length}
        </span>
      </div>
      <ul>
        {models.map((m) => {
          const isSel = m.model_id === selectedId;
          const isCompat = compatible ? compatible.includes(m.model_id) : true;
          return (
            <li key={m.model_id}>
              <button
                type="button"
                onClick={() => onSelect(m)}
                disabled={!isCompat}
                className={cn(
                  "w-full px-3 py-2 text-left transition-colors",
                  "hover:bg-[var(--color-rt-mist)]",
                  isSel &&
                    "bg-[color-mix(in_oklab,var(--color-rt-accent)_8%,var(--color-rt-paper))]",
                  !isCompat && "cursor-not-allowed opacity-40 hover:bg-transparent",
                )}
              >
                <div className="flex items-center justify-between gap-2">
                  <span className="truncate text-[12px] font-medium text-[var(--color-rt-ink)]">
                    {m.dataset_name.replace(/^Dataset(\d+)_/, "D$1 ")}
                  </span>
                  {isSel && (
                    <CheckCircle2
                      size={14}
                      className="shrink-0 text-[var(--color-rt-accent)]"
                    />
                  )}
                </div>
                <div className="mt-0.5 flex items-center gap-2 text-[10.5px] text-[var(--color-rt-muted)]">
                  <span>{m.plans.replace("nnUNet", "")}</span>
                  <span>·</span>
                  <span>
                    {m.available_folds.length} fold
                    {m.available_folds.length === 1 ? "" : "s"}
                  </span>
                  {m.val_mean_fg_dice !== null && (
                    <>
                      <span>·</span>
                      <span>dice {formatDice(m.val_mean_fg_dice)}</span>
                    </>
                  )}
                </div>
              </button>
            </li>
          );
        })}
      </ul>
    </div>
  );
}

function groupByRegion(models: ModelInfo[]): Record<string, ModelInfo[]> {
  const out: Record<string, ModelInfo[]> = {};
  for (const m of models) {
    const key = m.region ?? "other";
    (out[key] ??= []).push(m);
  }
  return out;
}

// Canonical region ordering. Anything beyond the known list is appended in
// alphabetical order so the UI is stable. Empty groups are filtered out by
// the caller.
const KNOWN_REGION_ORDER = [
  "brain_mr",
  "hn_ct",
  "pelvis_ct",
  "abdomen_ct",
  "thorax_ct",
  "whole_body_ct",
  "other",
];

function orderedRegions(grouped: Record<string, ModelInfo[]>): string[] {
  const present = new Set(Object.keys(grouped));
  const known = KNOWN_REGION_ORDER.filter((r) => present.has(r));
  const extras = [...present]
    .filter((r) => !KNOWN_REGION_ORDER.includes(r))
    .sort();
  return [...known, ...extras];
}
