"use client";

import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import {
  Activity,
  AlertTriangle,
  BadgeCheck,
  CircleHelp,
  CircleX,
  Eye,
  EyeOff,
  Layers,
  Loader2,
  Pencil,
  Play,
  Save,
  UserRound,
  Zap,
} from "lucide-react";
import { useEffect, useState } from "react";

import {
  getCohort,
  getVerdictsForModel,
  getVerdictsSummary,
  pollCrossvalUntilDone,
  pollMetrics,
  postVerdict,
  REJECT_REASONS,
  REJECT_REASON_LABEL,
  runPredictUntilSeg,
  seedGtFromPrediction,
  type ApprovalStatus,
  type PostprocessingInfo,
  type RejectReason,
  type VerdictSummary,
  type VerdictValue,
} from "@/lib/api";
import { useQAStore } from "@/lib/store";
import { cn, formatElapsed } from "@/lib/utils";

import { GtRevisionSelect } from "@/components/viewer/GtRevisionSelect";

import { CrossvalPanel } from "./CrossvalPanel";
import { MetricsBlock } from "./MetricsBlock";

export function InferencePanel() {
  const selectedModel = useQAStore((s) => s.selectedModel);
  const selectedCase = useQAStore((s) => s.selectedCase);
  const foldChoice = useQAStore((s) => s.foldChoice);
  const setFoldChoice = useQAStore((s) => s.setFoldChoice);
  const overlayOpacity = useQAStore((s) => s.overlayOpacity);
  const setOverlayOpacity = useQAStore((s) => s.setOverlayOpacity);
  const showGroundtruth = useQAStore((s) => s.showGroundtruth);
  const toggleGroundtruth = useQAStore((s) => s.toggleGroundtruth);
  const beginInference = useQAStore((s) => s.beginInference);
  const markSegReady = useQAStore((s) => s.markSegReady);
  const markMetricsReady = useQAStore((s) => s.markMetricsReady);
  const markMetricsError = useQAStore((s) => s.markMetricsError);
  const setInferenceError = useQAStore((s) => s.setInferenceError);
  const inferenceState = useQAStore((s) => s.inferenceState);
  const inferenceError = useQAStore((s) => s.inferenceError);
  const metricsState = useQAStore((s) => s.metricsState);
  const metricsError = useQAStore((s) => s.metricsError);
  const prediction = useQAStore((s) => s.currentPrediction);

  const runMode = useQAStore((s) => s.runMode);
  const setRunMode = useQAStore((s) => s.setRunMode);
  const crossval = useQAStore((s) => s.crossval);
  const crossvalState = useQAStore((s) => s.crossvalState);
  const beginCrossval = useQAStore((s) => s.beginCrossval);
  const setCrossvalProgress = useQAStore((s) => s.setCrossvalProgress);
  const markCrossvalReady = useQAStore((s) => s.markCrossvalReady);
  const markCrossvalError = useQAStore((s) => s.markCrossvalError);

  const toggleFold = useQAStore((s) => s.toggleFold);
  const availableFolds = selectedModel?.available_folds ?? [];
  const crossvalAvailable = availableFolds.length >= 2;

  const enterGtEdit = useQAStore((s) => s.enterGtEdit);
  const gtEditMode = useQAStore((s) => s.gtEditMode);

  const reviewer = useQAStore((s) => s.reviewer);
  const setInferenceQueueInfo = useQAStore((s) => s.setInferenceQueueInfo);
  const clearInferenceQueueInfo = useQAStore((s) => s.clearInferenceQueueInfo);
  const inferenceQueuePosition = useQAStore((s) => s.inferenceQueuePosition);
  const inferenceQueueEtaSec = useQAStore((s) => s.inferenceQueueEtaSec);
  const autoRunPending = useQAStore((s) => s.autoRunPending);
  const clearAutoRun = useQAStore((s) => s.clearAutoRun);

  // For the per-fold chips, decide active state from foldChoice.
  function isFoldActive(fold: number): boolean {
    if (runMode === "crossval") return false;
    if (foldChoice === "all") return true;
    if (foldChoice === "best") return fold === availableFolds[0];
    return foldChoice.includes(fold);
  }

  // Two-phase inference. Phase 1 polls for seg_ready → render the
  // Cornerstone overlay; phase 2 polls for done → metrics attached.
  // We capture the prediction_id and check ownership before mutating the
  // store on phase-2 completion — otherwise a slow phase-2 poll from a
  // previous run could overwrite the user's new prediction.
  const mut = useMutation({
    mutationFn: async (args: {
      model_id: string;
      case_id: string;
      use_folds: typeof foldChoice;
    }) => {
      const segPhase = await runPredictUntilSeg({
        ...args,
        // The queue widget shows "reviewer · gustavo" so a second user
        // knows whose job is in front of theirs.
        reviewer,
        // Every status poll updates the store's queue info — drives the
        // "position N" label inside the PhasePill while the job waits.
        onStatus: (s) => {
          setInferenceQueueInfo(
            s.position_in_queue ?? null,
            s.queue_depth ?? 0,
            // The status payload doesn't repeat eta_s; use the queue ETA
            // from the original accept response when available.
            s.accepted?.eta_s ?? useQAStore.getState().inferenceQueueEtaSec,
          );
        },
      });
      // Mirror the accepted response into the store immediately so the
      // first frame of the panel sees the right queue state.
      setInferenceQueueInfo(
        segPhase.accepted.position_in_queue ?? null,
        segPhase.accepted.queue_depth ?? 0,
        segPhase.accepted.eta_s ?? null,
      );
      markSegReady(segPhase);
      if (segPhase.metrics !== null || segPhase.metrics_error) {
        clearInferenceQueueInfo();
        return segPhase;
      }
      const myId = segPhase.prediction_id;
      try {
        const finalPhase = await pollMetrics(myId);
        const current = useQAStore.getState().currentPrediction;
        if (current && current.prediction_id === myId) {
          markMetricsReady(finalPhase.metrics, finalPhase.metrics_error);
        }
        clearInferenceQueueInfo();
        return finalPhase;
      } catch (e) {
        const current = useQAStore.getState().currentPrediction;
        if (current && current.prediction_id === myId) {
          markMetricsError(e instanceof Error ? e.message : String(e));
        }
        clearInferenceQueueInfo();
        throw e;
      }
    },
    onMutate: () => beginInference(),
    onError: (e: Error) => {
      // Only treat as inference-error if no seg was rendered yet.
      // metricsState handles the post-seg failure separately.
      const { inferenceState: cur } = useQAStore.getState();
      if (cur === "running") {
        setInferenceError(e.message);
      }
      clearInferenceQueueInfo();
    },
  });

  // Cross-validation run: one mutation that drives every fold + the ensemble
  // through POST /api/crossval and polls to completion, updating live "fold
  // N/M" progress. Distinct from `mut` (the single /api/predict path).
  const cvMut = useMutation({
    mutationFn: async (args: { model_id: string; case_id: string }) =>
      pollCrossvalUntilDone({
        ...args,
        reviewer,
        onProgress: (s) =>
          setCrossvalProgress(s.folds_done ?? 0, s.folds_total ?? 0, s.current_fold ?? null),
      }),
    onMutate: () => beginCrossval(),
    onSuccess: (run) => markCrossvalReady(run),
    onError: (e: Error) => markCrossvalError(e.message),
  });

  const isRunning =
    runMode === "crossval" ? crossvalState === "running" : inferenceState === "running";
  const canRun =
    !!selectedModel &&
    !!selectedCase &&
    !isRunning &&
    (runMode === "single" || crossvalAvailable) &&
    (selectedCase.compatible_models.includes(selectedModel.model_id));

  // Surface *why* the run button is disabled — the only non-visual cue a
  // screen-reader user gets (the disabled state is otherwise just styling).
  const runDisabledReason = !selectedModel
    ? "Select a model first"
    : !selectedCase
      ? "Select a case first"
      : isRunning
        ? "A run is already in progress"
        : runMode === "crossval" && !crossvalAvailable
          ? "Cross-validation needs at least 2 trained folds"
          : !selectedCase.compatible_models.includes(selectedModel.model_id)
            ? "Selected case is not compatible with this model"
            : "";

  function onRun() {
    if (!selectedModel || !selectedCase) return;
    if (runMode === "crossval") {
      cvMut.mutate({
        model_id: selectedModel.model_id,
        case_id: selectedCase.case_id,
      });
      return;
    }
    mut.mutate({
      model_id: selectedModel.model_id,
      case_id: selectedCase.case_id,
      use_folds: foldChoice,
    });
  }

  // Approve-and-next: when the VerdictBlock sets autoRunPending and
  // selectedCase has advanced, fire predict on the new case automatically.
  // Cleared inside the effect so a manual case-pick doesn't re-fire.
  useEffect(() => {
    if (!autoRunPending) return;
    if (runMode !== "single") return;
    if (!selectedModel || !selectedCase) return;
    if (inferenceState === "running") return;
    if (!selectedCase.compatible_models.includes(selectedModel.model_id)) {
      clearAutoRun();
      return;
    }
    clearAutoRun();
    mut.mutate({
      model_id: selectedModel.model_id,
      case_id: selectedCase.case_id,
      use_folds: foldChoice,
    });
    // mut.mutate is stable across renders; we intentionally don't depend
    // on it to avoid retriggering when the mutation object identity
    // changes.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [autoRunPending, selectedCase?.case_id, selectedModel?.model_id]);

  return (
    <aside className="rt-card flex min-h-0 flex-1 flex-col overflow-hidden">
      <div className="flex items-center justify-between border-b border-[var(--color-rt-line)] px-3 py-2">
        <div className="flex items-center gap-2">
          <span className="inline-flex h-6 w-6 items-center justify-center rounded-[var(--radius-rt-sm)] bg-[color-mix(in_oklab,var(--color-rt-accent)_12%,var(--color-rt-paper))] text-[var(--color-rt-accent)]">
            <Activity size={13} />
          </span>
          <h2 className="rt-display text-[13px] font-semibold tracking-wide text-[var(--color-rt-ink)]">
            Inference
          </h2>
        </div>
        <PhasePill
          inferenceState={inferenceState}
          metricsState={metricsState}
          metricsError={metricsError}
          fromCache={prediction?.from_cache === true}
          queuePosition={inferenceQueuePosition}
          queueEtaSec={inferenceQueueEtaSec}
        />
      </div>

      <div className="flex-1 overflow-y-auto p-3 space-y-3">
        <Section title="Model">
          <Readout
            label="dataset"
            value={selectedModel?.dataset_name ?? "—"}
          />
          <Readout
            label="config"
            value={
              selectedModel
                ? `${selectedModel.trainer} / ${selectedModel.plans} / ${selectedModel.configuration}`
                : "—"
            }
            mono
          />
          <Readout
            label="folds"
            value={
              selectedModel
                ? selectedModel.available_folds.join(", ")
                : "—"
            }
          />
          {selectedModel?.current_epoch != null && selectedModel.total_epochs != null && (
            <Readout
              label="epoch"
              value={(() => {
                const base = `${selectedModel.current_epoch} / ${selectedModel.total_epochs}`;
                switch (selectedModel.status) {
                  case "training": return `${base}  · live`;
                  case "stopped":  return `${base}  · paused`;
                  case "failed":   return `${base}  · failed`;
                  default:         return base;
                }
              })()}
            />
          )}
        </Section>

        <Section title="Case">
          <Readout label="id" value={selectedCase?.case_id ?? "—"} mono />
          <Readout
            label="source"
            value={selectedCase?.source_dataset ?? "—"}
          />
          <Readout
            label="groundtruth"
            value={selectedCase?.groundtruth_path ? "present" : "—"}
          />
        </Section>

        <Section title="Run">
          <div className="flex flex-wrap gap-1.5">
            <FoldButton
              active={runMode === "single" && foldChoice === "best"}
              onClick={() => {
                setRunMode("single");
                setFoldChoice("best");
              }}
              icon={<Zap size={12} />}
              label="best"
              hint="lowest-index fold only"
            />
            <FoldButton
              active={runMode === "crossval"}
              onClick={() => crossvalAvailable && setRunMode("crossval")}
              icon={<Layers size={12} />}
              label="cross-val"
              hint={`run the case through all ${availableFolds.length} folds individually + ensemble; stars the out-of-fold (unbiased) result`}
              disabled={!crossvalAvailable}
            />
            {availableFolds.length > 0 && (
              <span className="mx-1 self-center text-[10px] text-[var(--color-rt-muted)]">
                or pick (ensemble):
              </span>
            )}
            {availableFolds.map((f) => (
              <FoldChip
                key={f}
                fold={f}
                active={isFoldActive(f)}
                onClick={() => {
                  setRunMode("single");
                  toggleFold(f);
                }}
              />
            ))}
          </div>

          <button
            type="button"
            disabled={!canRun}
            onClick={onRun}
            title={canRun ? "Run inference on the selected case" : runDisabledReason}
            aria-label={canRun ? undefined : `Run inference (disabled — ${runDisabledReason})`}
            className={cn(
              "mt-3 flex w-full items-center justify-center gap-2 rounded-full py-2.5 text-[13px] font-semibold transition-all",
              canRun
                ? "bg-[var(--color-rt-accent)] text-white shadow-[var(--shadow-rt-glow-accent)] hover:bg-[var(--color-rt-accent-2)] active:scale-[0.99]"
                : "cursor-not-allowed bg-[var(--color-rt-mist)] text-[var(--color-rt-muted)]",
            )}
          >
            {isRunning ? (
              <>
                <Loader2 className="animate-spin" size={14} />
                {runMode === "crossval" ? "running folds…" : "running…"}
              </>
            ) : (
              <>
                <Play size={14} />
                {runMode === "crossval" ? "run cross-val" : "run inference"}
              </>
            )}
          </button>

          {metricsState === "pending" && prediction && (
            <div className="mt-2 flex items-start gap-1.5 rounded-[var(--radius-rt-sm)] bg-[color-mix(in_oklab,var(--color-rt-accent)_8%,var(--color-rt-paper))] p-2 text-[11px] text-[var(--color-rt-accent)]">
              <Loader2 size={12} className="mt-0.5 shrink-0 animate-spin" />
              <span>
                Segmentation rendered. Computing per-label Dice & HD95
                {prediction.label_map &&
                  ` for ${Object.keys(prediction.label_map).filter((k) => k !== "background").length} labels`}
                …
              </span>
            </div>
          )}

          {inferenceState === "error" && inferenceError && (
            <div className="mt-2 flex items-start gap-1.5 rounded-[var(--radius-rt-sm)] bg-[color-mix(in_oklab,var(--color-rt-pip-error)_10%,var(--color-rt-paper))] p-2 text-[11px] text-[var(--color-rt-pip-error)]">
              <AlertTriangle size={12} className="mt-0.5 shrink-0" />
              <span className="break-all">{inferenceError}</span>
            </div>
          )}
        </Section>

        {/* Cross-validation comparison — replaces the single-run Result +
            Per-label sections when runMode === "crossval". Renders during the
            run (progress), on error, and once ready. */}
        {runMode === "crossval" &&
          (crossvalState === "running" || crossvalState === "error" || crossval) && (
            <CrossvalPanel />
          )}

        {prediction && (
          <>
            {runMode === "single" && (
              <>
                <Section title="Result">
                  <Readout
                    label="inference"
                    value={formatElapsed(prediction.elapsed_s)}
                  />
                  <MetricsTimingReadout
                    startedAt={prediction.started_at}
                    metricsState={metricsState}
                    fromCache={prediction.from_cache === true}
                  />
                  <Readout
                    label="cache"
                    value={
                      prediction.from_cache
                        ? "loaded from cache"
                        : prediction.used_preprocessed_cache
                          ? "warm (preprocessed)"
                          : "cold (raw)"
                    }
                  />
                  <Readout
                    label="prediction"
                    value={prediction.prediction_id}
                    mono
                  />
                </Section>

                {prediction.postprocessing && (
                  <PostprocessingSection info={prediction.postprocessing} />
                )}
              </>
            )}

            <Section title="Overlay">
              <div>
                <label className="flex items-center justify-between text-[11px] text-[var(--color-rt-muted)]">
                  <span>opacity</span>
                  <span>{Math.round(overlayOpacity * 100)}%</span>
                </label>
                <input
                  type="range"
                  min={0}
                  max={1}
                  step={0.05}
                  value={overlayOpacity}
                  onChange={(e) => setOverlayOpacity(parseFloat(e.target.value))}
                  className="mt-1 w-full accent-[var(--color-rt-accent)]"
                />
              </div>

              {/* Uploaded case with no GT yet: seed it from the prediction so
                  the reviewer can correct the model's output and save it as a
                  training label (closes the QC → corrected-GT → retrain loop
                  for ad-hoc data). */}
              {selectedCase?.uploaded &&
                !selectedCase.groundtruth_path &&
                prediction && (
                  <SeedGtButton
                    caseId={selectedCase.case_id}
                    predictionId={prediction.prediction_id}
                  />
                )}

              {selectedCase?.groundtruth_path && (
                <div className="mt-2 flex items-stretch gap-1.5">
                  <div className="flex-1 min-w-0">
                    <GtRevisionSelect caseId={selectedCase.case_id} />
                  </div>
                  <button
                    type="button"
                    onClick={toggleGroundtruth}
                    aria-pressed={showGroundtruth}
                    aria-label={showGroundtruth ? "Hide ground-truth overlay" : "Show ground-truth overlay"}
                    title={showGroundtruth ? "Hide GT overlay" : "Show GT overlay"}
                    className={cn(
                      "inline-flex h-9 w-9 shrink-0 items-center justify-center rounded-[var(--radius-rt-sm)] border border-[var(--color-rt-line)] transition-colors",
                      showGroundtruth
                        ? "bg-[color-mix(in_oklab,var(--color-rt-pip-ok)_14%,var(--color-rt-paper))] text-[var(--color-rt-pip-ok)]"
                        : "text-[var(--color-rt-muted)] hover:bg-[var(--color-rt-mist)] hover:text-[var(--color-rt-ink)]",
                    )}
                  >
                    {showGroundtruth ? <Eye size={14} /> : <EyeOff size={14} />}
                  </button>
                </div>
              )}

              {/* GT correction: open the in-viewer toolbar so the reviewer
                  can paint over the GT labelmap. Always available — any
                  in-flight metrics computation finishes against the prior
                  GT, and re-running inference after a save produces fresh
                  metrics against the new revision. */}
              {selectedCase?.groundtruth_path && (
                <button
                  type="button"
                  disabled={gtEditMode}
                  onClick={() => enterGtEdit()}
                  className={cn(
                    "mt-2 flex w-full items-center justify-center gap-1.5 rounded-[var(--radius-rt-sm)] border border-[var(--color-rt-line)] py-1.5 text-[11px] font-medium transition-colors",
                    gtEditMode
                      ? "cursor-not-allowed bg-[var(--color-rt-mist)] text-[var(--color-rt-muted)]"
                      : "text-[var(--color-rt-accent)] hover:bg-[color-mix(in_oklab,var(--color-rt-accent)_10%,var(--color-rt-paper))]",
                  )}
                  title="open in-viewer GT correction toolbar"
                >
                  <Pencil size={12} />
                  {gtEditMode ? "editing GT…" : "correct GT"}
                </button>
              )}
            </Section>

            {runMode === "single" && (
              <Section
                title={`Per-label dice ${selectedCase?.groundtruth_path ? "(vs GT)" : ""}`}
              >
                {prediction.metrics && prediction.metrics.length > 0 ? (
                  <MetricsBlock metrics={prediction.metrics} />
                ) : metricsState === "pending" ? (
                  <div className="flex items-center gap-1.5 text-[11px] text-[var(--color-rt-muted)]">
                    <Loader2 size={11} className="animate-spin" />
                    <span>computing metrics…</span>
                  </div>
                ) : metricsState === "error" ? (
                  <div className="flex items-start gap-1.5 rounded-[var(--radius-rt-sm)] bg-[color-mix(in_oklab,var(--color-rt-purple)_10%,var(--color-rt-paper))] p-2 text-[11px] text-[var(--color-rt-purple)]">
                    <AlertTriangle size={12} className="mt-0.5 shrink-0" />
                    <span className="break-all">
                      metrics unavailable: {metricsError ?? "unknown error"}. Segmentation is still
                      valid — you can still record a verdict.
                    </span>
                  </div>
                ) : !selectedCase?.groundtruth_path ? (
                  <div className="text-[11px] text-[var(--color-rt-muted)]">
                    no groundtruth uploaded — metrics skipped
                  </div>
                ) : null}
              </Section>
            )}

            <VerdictBlock />
          </>
        )}
      </div>
    </aside>
  );
}

// Compact phase pill rendered in the InferencePanel header. Reflects the
// combined state of the two phases so the operator can read the panel at
// a glance without staring at the run button.
function PhasePill({
  inferenceState,
  metricsState,
  metricsError,
  fromCache,
  queuePosition,
  queueEtaSec,
}: {
  inferenceState: "idle" | "running" | "error";
  metricsState: "idle" | "pending" | "ready" | "error";
  metricsError: string | null;
  fromCache: boolean;
  queuePosition: number | null;
  queueEtaSec: number | null;
}) {
  let label: string;
  let Icon: typeof BadgeCheck = Activity;
  let color: string;
  let bg: string;
  let spin = false;

  if (inferenceState === "error") {
    label = "error";
    Icon = AlertTriangle;
    color = "var(--color-rt-pip-error)";
    bg = `color-mix(in oklab, ${color} 14%, var(--color-rt-paper))`;
  } else if (inferenceState === "running" && queuePosition !== null && queuePosition > 0) {
    // Queued behind another reviewer's job. The PhasePill is the only
    // place this surfaces — keeps the second user informed instead of
    // staring at a silent spinner.
    const etaText = queueEtaSec
      ? `, ~${Math.round(queueEtaSec)}s wait`
      : "";
    label = `position ${queuePosition} in queue${etaText}`;
    Icon = Loader2;
    color = "var(--color-rt-muted)";
    bg = "var(--color-rt-mist)";
    spin = true;
  } else if (inferenceState === "running") {
    label = "running inference";
    Icon = Loader2;
    color = "var(--color-rt-accent)";
    bg = `color-mix(in oklab, ${color} 12%, var(--color-rt-paper))`;
    spin = true;
  } else if (metricsState === "pending") {
    label = "seg ready · metrics";
    Icon = Loader2;
    color = "var(--color-rt-accent)";
    bg = `color-mix(in oklab, ${color} 10%, var(--color-rt-paper))`;
    spin = true;
  } else if (metricsState === "ready") {
    // Distinguish "fresh result" from "cached" so the operator
    // knows when no GPU cycles were spent.
    label = fromCache ? "loaded from cache" : "done";
    Icon = BadgeCheck;
    color = "var(--color-rt-pip-ok)";
    bg = `color-mix(in oklab, ${color} 12%, var(--color-rt-paper))`;
  } else if (metricsState === "error") {
    label = "seg only";
    Icon = AlertTriangle;
    color = "var(--color-rt-purple)";
    bg = `color-mix(in oklab, ${color} 12%, var(--color-rt-paper))`;
  } else {
    label = "idle";
    Icon = Activity;
    color = "var(--color-rt-muted)";
    bg = "var(--color-rt-mist)";
  }

  return (
    <span
      title={metricsError ?? undefined}
      className="inline-flex items-center gap-1 rounded-full px-2 py-0.5 text-[10px] font-semibold uppercase tracking-wide"
      style={{ color, backgroundColor: bg }}
    >
      <Icon size={10} className={spin ? "animate-spin" : ""} />
      {label}
    </span>
  );
}

// Live-ticking elapsed-time readout for the metrics phase. While
// metricsState === "pending" the value counts up from started_at so the
// operator sees that the backend is still working (rather than wondering
// if the polling stalled). Stops ticking once metrics resolve.
function MetricsTimingReadout({
  startedAt,
  metricsState,
  fromCache,
}: {
  startedAt: string;
  metricsState: "idle" | "pending" | "ready" | "error";
  fromCache: boolean;
}) {
  const [now, setNow] = useState<number>(() => Date.now());
  useEffect(() => {
    if (metricsState !== "pending") return;
    const id = window.setInterval(() => setNow(Date.now()), 1000);
    return () => window.clearInterval(id);
  }, [metricsState]);

  if (metricsState === "idle") return null;

  // On a cache hit, started_at is the *original* run's timestamp (often
  // hours or days ago), so a wall-clock difference would say "ran for 26
  // minutes" even though the result loaded instantly. Suppress the live
  // clock in that case — the inference timing is already shown in
  // the "inference" readout (prediction.elapsed_s) and we add a dedicated
  // "cache" indicator above.
  if (fromCache) {
    return <Readout label="metrics" value="cached" />;
  }

  const startedMs = new Date(startedAt).getTime();
  const elapsedS = Math.max(0, (now - startedMs) / 1000);

  let value: string;
  if (metricsState === "pending") {
    value = `${formatElapsed(elapsedS)} (computing)`;
  } else if (metricsState === "error") {
    value = "failed";
  } else {
    value = formatElapsed(elapsedS);
  }
  return <Readout label="metrics" value={value} />;
}

// Uploaded cases ship with no ground truth. This copies the model's
// prediction into the case as the editable GT baseline and opens the
// correction toolbar — the reviewer's fixes then save as a training label
// through the existing GT-edit pipeline.
function SeedGtButton({
  caseId,
  predictionId,
}: {
  caseId: string;
  predictionId: string;
}) {
  const updateSelectedCase = useQAStore((s) => s.updateSelectedCase);
  const enterGtEdit = useQAStore((s) => s.enterGtEdit);
  const qc = useQueryClient();
  const [err, setErr] = useState<string | null>(null);

  const mut = useMutation({
    mutationFn: () => seedGtFromPrediction(caseId, predictionId),
    onMutate: () => setErr(null),
    onSuccess: async (c) => {
      updateSelectedCase({
        groundtruth_path: c.groundtruth_path,
        uploaded: c.uploaded,
      });
      await qc.invalidateQueries({ queryKey: ["cohort"] });
      enterGtEdit();
    },
    onError: (e: unknown) => setErr(e instanceof Error ? e.message : String(e)),
  });

  return (
    <div className="mt-2">
      <button
        type="button"
        onClick={() => mut.mutate()}
        disabled={mut.isPending}
        title="Copy this prediction into the case as editable ground truth, then open the correction tools — your fixes become a training label."
        className={cn(
          "flex w-full items-center justify-center gap-1.5 rounded-[var(--radius-rt-sm)] border border-[var(--color-rt-line)] py-1.5 text-[11px] font-medium transition-colors",
          "text-[var(--color-rt-accent)] hover:bg-[color-mix(in_oklab,var(--color-rt-accent)_10%,var(--color-rt-paper))]",
          mut.isPending && "cursor-not-allowed opacity-60",
        )}
      >
        {mut.isPending ? (
          <Loader2 size={12} className="animate-spin" />
        ) : (
          <Pencil size={12} />
        )}
        {mut.isPending ? "seeding…" : "use prediction as GT to edit"}
      </button>
      {err && (
        <div
          className="mt-1 text-[10.5px] text-[var(--color-rt-pip-error)]"
          title={err}
        >
          {err}
        </div>
      )}
    </div>
  );
}

function VerdictBlock() {
  const selectedModel = useQAStore((s) => s.selectedModel);
  const selectedCase = useQAStore((s) => s.selectedCase);
  const prediction = useQAStore((s) => s.currentPrediction);
  const foldChoice = useQAStore((s) => s.foldChoice);
  const runMode = useQAStore((s) => s.runMode);
  const crossval = useQAStore((s) => s.crossval);
  const metricsState = useQAStore((s) => s.metricsState);
  const reviewer = useQAStore((s) => s.reviewer);
  const setReviewer = useQAStore((s) => s.setReviewer);
  const setCase = useQAStore((s) => s.setCase);
  const requestAutoRun = useQAStore((s) => s.requestAutoRun);

  const [verdict, setVerdict] = useState<VerdictValue | null>(null);
  const [notes, setNotes] = useState("");
  const [savedAt, setSavedAt] = useState<string | null>(null);
  // Structured reject reason — required once "reject" is chosen so every
  // rejection carries an actionable "what to fix" category.
  const [rejectReason, setRejectReason] = useState<RejectReason | "">("");
  // Approve-and-next: when true, the post-save effect navigates to
  // `next_case_id` (if present) and requests an auto-run.
  const [advanceAfterSave, setAdvanceAfterSave] = useState(true);

  const qc = useQueryClient();

  const priorQuery = useQuery({
    queryKey: ["verdicts", selectedModel?.model_id, selectedCase?.case_id],
    queryFn: () =>
      selectedModel && selectedCase
        ? getVerdictsForModel(selectedModel.model_id, selectedCase.case_id)
        : Promise.resolve([]),
    enabled: !!selectedModel && !!selectedCase,
  });

  // Model-level rollup (same query key the catalog uses, so it's cached and
  // refetches together). Drives the live "sign-off" readout: how the case
  // verdicts so far roll up to the model's approved / rejected / pending
  // decision — recomputed server-side via verdicts.approval_status_for.
  const summaryQuery = useQuery({
    queryKey: ["verdicts-summary"],
    queryFn: getVerdictsSummary,
  });
  const modelSummary =
    summaryQuery.data?.find((s) => s.model_id === selectedModel?.model_id) ?? null;

  const mut = useMutation({
    mutationFn: postVerdict,
    onSuccess: async (saved) => {
      setSavedAt(saved.created_at);
      setNotes("");
      setVerdict(null);
      setRejectReason("");
      qc.invalidateQueries({ queryKey: ["verdicts-summary"] });
      qc.invalidateQueries({
        queryKey: ["verdicts", selectedModel?.model_id, selectedCase?.case_id],
      });
      if (!advanceAfterSave) return;
      if (!saved.next_case_id) return;
      // Resolve the next case from the cohort. We use queryClient cache
      // if present to avoid a redundant fetch.
      const cohort = await qc.fetchQuery({
        queryKey: ["cohort"],
        queryFn: getCohort,
      });
      const next = cohort.cases.find((c) => c.case_id === saved.next_case_id);
      if (!next) return;
      setCase(next);
      requestAutoRun();
    },
  });

  if (!selectedModel || !selectedCase || !prediction) return null;

  // In cross-validation mode the recorded mean_dice is the honest OOF
  // headline (not the currently-previewed fold), so the verdict matches the
  // unbiased score the reviewer is signing off on.
  const singleMeanDice =
    prediction.metrics && prediction.metrics.length > 0
      ? prediction.metrics
          .map((m) => m.dice)
          .filter((d): d is number => d !== null)
          .reduce((a, b, _i, arr) => a + b / arr.length, 0)
      : null;
  const meanDice =
    runMode === "crossval" && crossval
      ? crossval.aggregate?.headline_mean_fg_dice ?? null
      : singleMeanDice;

  function onSave(verdictOverride?: VerdictValue) {
    const v = verdictOverride ?? verdict;
    if (!v) return;
    if (!selectedModel || !selectedCase || !prediction) return;
    if (metricsState === "pending") return;
    mut.mutate({
      prediction_id: prediction.prediction_id,
      model_id: selectedModel.model_id,
      case_id: selectedCase.case_id,
      verdict: v,
      notes: notes.trim(),
      reviewer: reviewer.trim(),
      // The verdict table stores fold_choice as text — serialize arrays. In
      // cross-validation mode, record the honest OOF fold instead.
      fold_choice:
        runMode === "crossval" && crossval
          ? crossval.oof_fold != null
            ? `crossval:oof=${crossval.oof_fold}`
            : "crossval:no-oof"
          : Array.isArray(foldChoice)
            ? foldChoice.join(",")
            : foldChoice,
      mean_dice: meanDice,
      // Only meaningful on a reject; the backend ignores it otherwise.
      reject_reason: v === "reject" ? rejectReason : "",
    });
  }

  // Keyboard shortcuts inside the notes textarea:
  //   Enter             → submit Accept and advance
  //   Shift-Enter       → newline (default)
  //   Cmd/Ctrl-Enter    → submit Needs-Review and advance
  // The Reject button stays click-only — destructive action.
  function onNotesKeyDown(e: React.KeyboardEvent<HTMLTextAreaElement>) {
    if (e.key !== "Enter") return;
    if (e.shiftKey) return; // default newline
    e.preventDefault();
    if (e.metaKey || e.ctrlKey) {
      onSave("needs_review");
    } else {
      onSave("accept");
    }
  }

  return (
    <Section title="QA verdict">
      {/* Live model-level sign-off: how the case verdicts so far roll up to
          the model's approved / rejected / pending decision. */}
      {modelSummary && modelSummary.total > 0 && (
        <ModelSignoff summary={modelSummary} />
      )}

      <div className="flex gap-1.5">
        <VerdictButton
          kind="accept"
          active={verdict === "accept"}
          onClick={() => {
            setVerdict("accept");
            setRejectReason("");
          }}
        />
        <VerdictButton
          kind="needs_review"
          active={verdict === "needs_review"}
          onClick={() => {
            setVerdict("needs_review");
            setRejectReason("");
          }}
        />
        <VerdictButton
          kind="reject"
          active={verdict === "reject"}
          onClick={() => setVerdict("reject")}
        />
      </div>

      {verdict === "reject" && (
        <RejectReasonPicker value={rejectReason} onChange={setRejectReason} />
      )}

      <textarea
        value={notes}
        onChange={(e) => setNotes(e.target.value)}
        onKeyDown={onNotesKeyDown}
        rows={3}
        placeholder="notes (optional) — Enter to Accept+next, ⌘/Ctrl-Enter for Needs-Review+next"
        className="mt-2 w-full resize-none rounded-[var(--radius-rt-sm)] border border-[var(--color-rt-line)] bg-[var(--color-rt-paper)] px-2 py-1.5 font-mono text-[11px] text-[var(--color-rt-ink)] placeholder:text-[var(--color-rt-muted)] focus:outline-none focus:border-[var(--color-rt-accent)]"
      />

      <div className="relative mt-2">
        <UserRound
          size={12}
          className="pointer-events-none absolute left-2 top-1/2 -translate-y-1/2 text-[var(--color-rt-muted)]"
        />
        <input
          type="text"
          value={reviewer}
          onChange={(e) => setReviewer(e.target.value)}
          placeholder="your name or email"
          className="w-full rounded-[var(--radius-rt-sm)] border border-[var(--color-rt-line)] bg-[var(--color-rt-paper)] py-1.5 pl-7 pr-2 text-[11px] text-[var(--color-rt-ink)] placeholder:text-[var(--color-rt-muted)] focus:outline-none focus:border-[var(--color-rt-accent)]"
        />
      </div>

      {/*
        Disable verdict save while metrics are still computing — otherwise
        mean_dice would be recorded as null on what is actually a valid
        prediction. If metrics computation failed (metricsState === "error"),
        we allow saving with mean_dice=null, since the seg is still valid
        and the operator should be able to record their call.
      */}
      {(() => {
        const needsReason = verdict === "reject" && !rejectReason;
        const canSave =
          !!verdict && !mut.isPending && metricsState !== "pending" && !needsReason;
        return (
          <button
            type="button"
            disabled={!canSave}
            onClick={() => onSave()}
            title={
              needsReason ? "Pick a reject reason first" : undefined
            }
            className={cn(
              "mt-2 inline-flex w-full items-center justify-center gap-1.5 rounded-[var(--radius-rt-sm)] py-1.5 text-[12px] font-semibold transition-colors",
              canSave
                ? "bg-[var(--color-rt-ink)] text-[var(--color-rt-paper)] hover:opacity-90"
                : "cursor-not-allowed bg-[var(--color-rt-mist)] text-[var(--color-rt-muted)]",
            )}
          >
            {mut.isPending ? <Loader2 className="animate-spin" size={12} /> : <Save size={12} />}
            {needsReason
              ? "pick a reason"
              : advanceAfterSave
                ? "save & next"
                : "save verdict"}
          </button>
        );
      })()}

      <label className="mt-1.5 flex items-center gap-1.5 text-[10.5px] text-[var(--color-rt-muted)]">
        <input
          type="checkbox"
          checked={advanceAfterSave}
          onChange={(e) => setAdvanceAfterSave(e.target.checked)}
          className="accent-[var(--color-rt-accent)]"
        />
        auto-advance to next compatible case after save
      </label>

      {metricsState === "pending" && (
        <div className="mt-1.5 flex items-center gap-1 text-[10.5px] text-[var(--color-rt-muted)]">
          <Loader2 size={10} className="animate-spin" />
          waiting for metrics so mean_dice can be recorded…
        </div>
      )}

      {savedAt && (
        <div className="mt-1.5 text-[10.5px] text-[var(--color-rt-pip-ok)]">
          saved at {new Date(savedAt).toLocaleTimeString()}
        </div>
      )}

      {priorQuery.data && priorQuery.data.length > 0 && (
        <div className="mt-3 border-t border-[var(--color-rt-line)] pt-2">
          <div className="mb-1 text-[10px] uppercase tracking-[0.18em] text-[var(--color-rt-muted)]">
            previous verdicts for this case
          </div>
          <ul className="space-y-1">
            {priorQuery.data.slice(0, 4).map((v) => (
              <li
                key={v.id}
                className="flex items-center justify-between rounded bg-[var(--color-rt-mist)] px-2 py-1 text-[10.5px]"
              >
                <span className="flex items-center gap-1.5">
                  <VerdictDot kind={v.verdict} />
                  <span className="truncate font-medium text-[var(--color-rt-ink)]">
                    {v.verdict.replace("_", " ")}
                  </span>
                  {v.reviewer && (
                    <span className="text-[var(--color-rt-muted)]">· {v.reviewer}</span>
                  )}
                </span>
                <span className="font-mono text-[10px] text-[var(--color-rt-muted)]">
                  {new Date(v.created_at).toLocaleDateString()}
                </span>
              </li>
            ))}
          </ul>
        </div>
      )}
    </Section>
  );
}

// Live model-level sign-off readout. Mirrors the catalog's derived approval
// (approved / rejected / pending) so the reviewer sees, while still in the
// workspace, how their per-case verdicts roll up to the model decision —
// plus the aggregated reject reasons (the "what to fix" signal).
function ModelSignoff({ summary }: { summary: VerdictSummary }) {
  const status: ApprovalStatus = summary.approval_status ?? "pending";
  const meta = {
    approved: { label: "approved", Icon: BadgeCheck, color: "var(--color-rt-pip-ok)" },
    rejected: { label: "rejected", Icon: CircleX, color: "var(--color-rt-pip-error)" },
    pending: { label: "in review", Icon: CircleHelp, color: "var(--color-rt-muted)" },
  }[status];
  const { label, Icon, color } = meta;
  const reasons = summary.reject_reasons
    ? Object.entries(summary.reject_reasons).sort(([, a], [, b]) => b - a)
    : [];
  return (
    <div
      className="mb-2 rounded-[var(--radius-rt-sm)] border p-2"
      style={{
        borderColor: `color-mix(in oklab, ${color} 30%, var(--color-rt-line))`,
        background: `color-mix(in oklab, ${color} 7%, var(--color-rt-paper))`,
      }}
    >
      <div className="flex items-center justify-between">
        <span className="text-[10px] uppercase tracking-[0.18em] text-[var(--color-rt-muted)]">
          model sign-off
        </span>
        <span
          className="inline-flex items-center gap-1 text-[11px] font-semibold uppercase tracking-wide"
          style={{ color }}
        >
          <Icon size={12} /> {label}
        </span>
      </div>
      <div className="mt-1 flex flex-wrap items-center gap-2 font-mono text-[10.5px] text-[var(--color-rt-muted)]">
        <span className="text-[var(--color-rt-pip-ok)]">✓ {summary.accept}</span>
        <span className="text-[var(--color-rt-pip-error)]">✗ {summary.reject}</span>
        <span className="text-[var(--color-rt-purple)]">? {summary.needs_review}</span>
        <span>· {summary.total} case{summary.total === 1 ? "" : "s"}</span>
      </div>
      {status === "rejected" && reasons.length > 0 && (
        <div className="mt-1.5 flex flex-wrap gap-1">
          {reasons.map(([k, n]) => (
            <span
              key={k}
              className="inline-flex items-center rounded-full border border-[color-mix(in_oklab,var(--color-rt-pip-error)_30%,transparent)] bg-[color-mix(in_oklab,var(--color-rt-pip-error)_8%,var(--color-rt-paper))] px-1.5 py-0.5 text-[10px] text-[var(--color-rt-pip-error)]"
            >
              {REJECT_REASON_LABEL[k as RejectReason] ?? k}
              {n > 1 ? ` ×${n}` : ""}
            </span>
          ))}
        </div>
      )}
    </div>
  );
}

// Structured reject-reason chips, revealed once "reject" is selected. A
// reason is required so every rejection aggregates into the model sign-off's
// "what to fix" rollup.
function RejectReasonPicker({
  value,
  onChange,
}: {
  value: RejectReason | "";
  onChange: (r: RejectReason) => void;
}) {
  return (
    <div className="mt-2">
      <div className="mb-1 text-[10px] uppercase tracking-[0.16em] text-[var(--color-rt-muted)]">
        reject reason <span className="text-[var(--color-rt-pip-error)]">*</span>
      </div>
      <div className="flex flex-wrap gap-1">
        {REJECT_REASONS.map((r) => {
          const active = value === r;
          return (
            <button
              key={r}
              type="button"
              onClick={() => onChange(r)}
              className={cn(
                "rounded-full border px-2 py-1 text-[10.5px] transition-colors",
                active
                  ? "border-[var(--color-rt-pip-error)] bg-[color-mix(in_oklab,var(--color-rt-pip-error)_12%,var(--color-rt-paper))] text-[var(--color-rt-pip-error)]"
                  : "border-[var(--color-rt-line)] text-[var(--color-rt-muted)] hover:bg-[var(--color-rt-mist)] hover:text-[var(--color-rt-ink)]",
              )}
            >
              {REJECT_REASON_LABEL[r]}
            </button>
          );
        })}
      </div>
    </div>
  );
}

function VerdictButton({
  kind,
  active,
  onClick,
}: {
  kind: VerdictValue;
  active: boolean;
  onClick: () => void;
}) {
  const meta = {
    accept: {
      label: "accept",
      icon: BadgeCheck,
      accent: "var(--color-rt-pip-ok)",
    },
    needs_review: {
      label: "review",
      icon: CircleHelp,
      accent: "var(--color-rt-purple)",
    },
    reject: {
      label: "reject",
      icon: CircleX,
      accent: "var(--color-rt-pip-error)",
    },
  }[kind];
  const Icon = meta.icon;
  return (
    <button
      type="button"
      onClick={onClick}
      className={cn(
        "inline-flex flex-1 items-center justify-center gap-1 rounded-[var(--radius-rt-sm)] border px-2 py-1.5 text-[11px] font-medium transition-colors",
      )}
      style={{
        borderColor: active ? meta.accent : "var(--color-rt-line)",
        backgroundColor: active
          ? `color-mix(in oklab, ${meta.accent} 12%, var(--color-rt-paper))`
          : "var(--color-rt-paper)",
        color: active ? meta.accent : "var(--color-rt-muted)",
      }}
    >
      <Icon size={12} />
      {meta.label}
    </button>
  );
}

function VerdictDot({ kind }: { kind: VerdictValue }) {
  const color = {
    accept: "var(--color-rt-pip-ok)",
    needs_review: "var(--color-rt-purple)",
    reject: "var(--color-rt-pip-error)",
  }[kind];
  return (
    <span
      className="inline-block h-1.5 w-1.5 rounded-full"
      style={{ backgroundColor: color }}
    />
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

function Readout({ label, value, mono = false }: { label: string; value: string; mono?: boolean }) {
  return (
    <div className="flex items-baseline justify-between gap-3">
      <span className="text-[10.5px] uppercase tracking-wide text-[var(--color-rt-muted)]">
        {label}
      </span>
      <span
        className={cn(
          "truncate text-right text-[12px] text-[var(--color-rt-ink)]",
          mono && "font-mono text-[11px]",
        )}
        title={value}
      >
        {value}
      </span>
    </div>
  );
}

function FoldButton({
  active,
  onClick,
  icon,
  label,
  hint,
  disabled,
}: {
  active: boolean;
  onClick: () => void;
  icon: React.ReactNode;
  label: string;
  hint?: string;
  disabled?: boolean;
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      disabled={disabled}
      title={hint}
      className={cn(
        "inline-flex items-center justify-center gap-1 rounded-[var(--radius-rt-sm)] border px-2 py-1 text-[11px] font-medium transition-colors",
        active
          ? "border-[var(--color-rt-accent)] bg-[color-mix(in_oklab,var(--color-rt-accent)_10%,var(--color-rt-paper))] text-[var(--color-rt-accent)]"
          : "border-[var(--color-rt-line)] text-[var(--color-rt-muted)] hover:bg-[var(--color-rt-mist)]",
        disabled && "cursor-not-allowed opacity-40 hover:bg-transparent",
      )}
    >
      {icon}
      {label}
    </button>
  );
}

function FoldChip({
  fold,
  active,
  onClick,
}: {
  fold: number;
  active: boolean;
  onClick: () => void;
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      title={`fold ${fold}`}
      className={cn(
        "inline-flex h-7 min-w-[28px] items-center justify-center rounded-full border px-1.5 font-mono text-[11px] font-medium transition-colors",
        active
          ? "border-[var(--color-rt-accent)] bg-[var(--color-rt-accent)] text-white"
          : "border-[var(--color-rt-line)] text-[var(--color-rt-muted)] hover:bg-[var(--color-rt-mist)]",
      )}
    >
      {fold}
    </button>
  );
}

function PostprocessingSection({ info }: { info: PostprocessingInfo }) {
  const [open, setOpen] = useState(false);
  function fmtSpacing(s: number[]): string {
    if (!s.length) return "—";
    return s.map((v) => v.toFixed(2)).join(" × ") + " mm";
  }
  return (
    <Section title="Postprocessing">
      <button
        type="button"
        onClick={() => setOpen((x) => !x)}
        className="-mt-0.5 flex w-full items-center justify-between rounded-[var(--radius-rt-sm)] border border-[var(--color-rt-line)] bg-[var(--color-rt-mist)] px-2 py-1 text-left text-[10.5px] uppercase tracking-wide text-[var(--color-rt-muted)] hover:text-[var(--color-rt-ink)]"
        aria-expanded={open}
      >
        <span>{open ? "hide pipeline" : "show pipeline"}</span>
        <span className="font-mono text-[10px] text-[var(--color-rt-ink)]">
          {info.test_time_augmentation ? "TTA" : "—"}
          {" · "}
          {info.gaussian_tile_blending ? "G" : "—"}
          {" · "}
          step {info.tile_step_size}
        </span>
      </button>

      {open && (
        <div className="mt-2 space-y-1">
          <Readout
            label="TTA (mirror)"
            value={info.test_time_augmentation ? "✓ enabled" : "✗ disabled"}
          />
          <Readout
            label="gaussian blend"
            value={info.gaussian_tile_blending ? "✓ enabled" : "✗ disabled"}
          />
          <Readout label="tile step" value={String(info.tile_step_size)} />
          <Readout label="network spacing" value={fmtSpacing(info.network_spacing)} />
          <Readout label="case spacing" value={fmtSpacing(info.original_spacing)} />
          <Readout
            label="resample order (seg)"
            value={String(info.resampling_order_seg)}
          />
          <Readout label="pipeline" value={info.pipeline.join(" → ")} mono />
          <Readout
            label="largest-CC per label"
            value={
              info.has_postprocessing_pkl
                ? Object.keys(info.keep_largest_component ?? {}).length > 0
                  ? "configured"
                  : "off"
                : "not configured"
            }
          />
          <Readout
            label="region merging"
            value={
              info.region_class_order && info.region_class_order.length > 0
                ? "configured"
                : "not configured"
            }
          />
        </div>
      )}
    </Section>
  );
}

