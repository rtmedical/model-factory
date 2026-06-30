// Typed fetchers for the qa-api FastAPI service.
// The browser hits /api/* and Next's rewrites proxy to QA_API_INTERNAL_URL.

export type ModelStatus = "training" | "done" | "stopped" | "failed";

// Backend-derived region codes. `abdomen_ct` and `thorax_ct` were added in
// 0.6.0 to cover datasets the original three regions (brain_mr, hn_ct,
// pelvis_ct) didn't cover (e.g. Pancreas, LUNA16). The catalog UI treats
// unknown regions gracefully — a `string` fallback lets the frontend
// keep working when the backend grows another region without a client
// release.
export type Region = "brain_mr" | "hn_ct" | "pelvis_ct" | "abdomen_ct" | "thorax_ct" | "whole_body_ct" | (string & {});

export type ModelInfo = {
  model_id: string;
  dataset_name: string;
  configuration: string;
  trainer: string;
  plans: string;
  region: Region | null;
  available_folds: number[];
  model_dir: string;
  val_mean_fg_dice: number | null;
  last_modified: string | null;
  // Derived from a filesystem heuristic on the model's fold dirs:
  // "training" = some fold has a fresh training_log but no checkpoint;
  // "failed"   = a fold has gone stale without producing checkpoint_best;
  // "done"     = every fold has checkpoint_best and no signs of liveness.
  status: ModelStatus;
  // Max `epoch` seen across all this model's folds' metrics.jsonl, and
  // the trainer-configured total. Both null if no metrics have been
  // written yet (cold fold).
  current_epoch: number | null;
  total_epochs: number | null;
  // Per-structure mean dice (averaged across cases) read from the
  // latest fold's validation/summary.json. null for in-flight training
  // and for historical folds missing the summary file.
  per_class_dice: Record<string, number> | null;
  // Dragonfly cache hits for this model across its compatible QA-cohort
  // cases. `cohort_size` is the denominator; both 0 when the cohort
  // manifest is absent or Redis is unreachable.
  cached_count: number;
  cohort_size: number;
  // Model-level QA decision DERIVED from the per-case verdict tallies
  // (backend: verdicts.approval_status_for). Drives the green/red card +
  // the ✓/✗ approval badge. Defaults to "pending" when no verdicts exist.
  approval_status?: ApprovalStatus;
  // Per-fold rollup. nnUNetv2 trains 5-fold cross-validation; each fold
  // is its own independent run, so a model can have fold 0 done while
  // folds 1+2 are mid-training. The card renders one progress row per
  // fold from this list so a live fold doesn't get hidden behind a
  // completed one.
  folds: FoldProgress[];
};

export type FoldProgress = {
  fold: number;
  status: ModelStatus;
  current_epoch: number | null;
  total_epochs: number | null;
  val_mean_fg_dice: number | null;
  has_checkpoint_best: boolean;
  // ── live training-rate + ETA — populated only for `training` folds ──
  // Median wall-clock seconds per epoch over a recent window (backend
  // derives it from the training log's "Epoch time" lines). All three
  // timestamps are UTC ISO8601, so `new Date(...)` yields a correct
  // countdown in any client timezone. `est_finish` is the absolute
  // instant the fold is projected to complete; the frontend ticks a
  // live countdown toward it between catalog refetches. All null when
  // the fold isn't actively producing epochs.
  sec_per_epoch?: number | null;
  started_at?: string | null;
  last_epoch_at?: string | null;
  eta_seconds?: number | null;
  est_finish?: string | null;
};

export type CaseInfo = {
  case_id: string; // "<region>/case_NNN"
  region: Region;
  source_dataset: string;
  source_case_stem: string;
  image_paths: string[];
  groundtruth_path: string | null;
  compatible_models: string[];
  // True for ad-hoc cases uploaded via the viewer (DICOM/NIfTI). They have
  // no GT until a reviewer seeds one from the model's prediction.
  uploaded?: boolean;
};

export type CohortResponse = {
  version: number;
  regions: string[];
  cases: CaseInfo[];
  trained_models: ModelInfo[];
};

export type LabelMetric = {
  label: number;
  label_name: string;
  dice: number | null;
  hd95_mm: number | null;
  n_voxels_gt: number;
  n_voxels_pred: number;
};

// What POST /api/predict returns immediately (202 Accepted) — or, on a
// Redis cache hit, 202 with `from_cache: true` and `status: "done"`
// (the seg + meshes already exist on disk).
export type PredictAcceptedResponse = {
  prediction_id: string;
  status: "queued" | "done";
  status_url: string;
  from_cache?: boolean;
  // Queue observability surfaced as of 0.6.0. 0 = next, N>0 = waiting
  // behind N predictions. `queue_depth` = global count at submit time.
  // `eta_s` is the rolling mean of the last few runs for this model_id.
  position_in_queue?: number | null;
  queue_depth?: number;
  eta_s?: number | null;
};

// Describes what nnUNetv2 actually did to the logits before the seg
// NIfTI was written. Populated at the seg_ready flip; null until then.
// The fields with `null` are informational only when not configured —
// e.g. our current models don't have `postprocessing.pkl` so
// `keep_largest_component` stays null, and `region_class_order` is null
// for label-not-region datasets.
export type PostprocessingInfo = {
  test_time_augmentation: boolean;
  gaussian_tile_blending: boolean;
  tile_step_size: number;
  perform_everything_on_device: boolean;
  network_spacing: number[]; // [sx, sy, sz] in mm
  original_spacing: number[]; // case-specific, [sx, sy, sz] in mm
  resampling_order_seg: number; // 0..3, nnUNet interpolation order
  pipeline: string[]; // pretty-printed ordered steps
  region_class_order: number[][] | null;
  keep_largest_component: Record<string, boolean> | null;
  has_postprocessing_pkl: boolean;
};

// What GET /api/predictions/{id}/status returns. Populated incrementally as
// the background task progresses:
//   queued → running → seg_ready → done
// `seg_ready` means seg.nii.gz is on disk and the seg overlay can render
// while metrics are still computing. `error` is reserved for inference
// failures; metrics-compute failures arrive as `done` with metrics=null
// and metrics_error set.
export type PredictionStatus = {
  prediction_id: string;
  status: "queued" | "running" | "seg_ready" | "done" | "error";
  model_id: string;
  case_id: string;
  folds: number[];
  started_at: string;
  updated_at: string;
  elapsed_s: number | null;
  used_preprocessed_cache: boolean | null;
  label_map: Record<string, number> | null;
  metrics: LabelMetric[] | null;
  metrics_error: string | null;
  seg_url: string | null;
  metrics_url: string | null;
  error_type: string | null;
  error_message: string | null;
  postprocessing: PostprocessingInfo | null;
  active_gt_revision: number | null;
  // Mesh precompute progress — runs after `done`, so it does NOT affect
  // the polling loop's termination check.
  meshes_status?: "pending" | "ready" | "failed" | null;
  meshes_elapsed_s?: number | null;
  meshes_by_label?: Record<string, string> | null;
  meshes_error?: string | null;
  // Live queue observability. `position_in_queue` is 0 when the
  // prediction is at the head of the queue (running), >0 while it
  // waits, null after the prediction terminates.
  position_in_queue?: number | null;
  queue_depth?: number | null;
};

// The shape InferencePanel consumes once the seg has rendered. metrics
// may still be null at this point — metricsState in the store tracks the
// follow-up.
export type PredictResponse = {
  prediction_id: string;
  seg_url: string;
  metrics_url: string;
  elapsed_s: number;
  used_preprocessed_cache: boolean;
  label_map: Record<string, number>;
  metrics: LabelMetric[] | null;
  metrics_error: string | null;
  started_at: string;
  postprocessing: PostprocessingInfo | null;
  active_gt_revision: number | null;
  // True iff the original POST /api/predict was answered from the Redis
  // result cache (seg + metrics were already on disk). The UI uses this
  // to suppress the "running since started_at" timing readouts that
  // would otherwise show wall-clock-since-the-original-run.
  from_cache?: boolean;
};

// Versioned ground-truth revisions. The cohort copy at
// /factory/qa-cohort/<region>/<case>/label_groundtruth.nii.gz is never
// overwritten; reviewer edits land at label_corrected_v{N}.nii.gz with
// a partial-unique-active row in the SQLite gt_corrections table.
export type GroundTruthRevision = {
  id: number;
  region: string;
  case_id: string;
  revision: number;
  path: string;
  base_prediction_id: string | null;
  reviewer: string;
  notes: string;
  status: "active" | "superseded";
  created_at: string;
};

// Returned by GET /api/cases/{region}/{case}/groundtruth/recompute-status
// while the metric recompute task is running after an activate. Tracks
// progress so the UI can show a "X of N predictions re-evaluated" pill.
export type RecomputeStatus = {
  pending: number;
  completed: number;
  total: number;
  error: string | null;
};

// One entry in the curated card-color palette. The backend stores the
// `color_key` (one of the swatch slugs); the actual --card-{name}-*
// vars live in globals.css and are theme-aware.
export type ColorTheme = {
  model_id: string;
  color_key: string;
  updated_by: string;
  updated_at: string;
};

// Same-origin by default — FastAPI serves the static export and the API
// from the same host. Override via NEXT_PUBLIC_QA_API_URL when running the
// Next.js dev server against a separately-launched uvicorn (local dev).
const API = process.env.NEXT_PUBLIC_QA_API_URL ?? "";

async function jsonFetch<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${API}${path}`, {
    ...init,
    headers: {
      "Content-Type": "application/json",
      ...(init?.headers ?? {}),
    },
  });
  if (!res.ok) {
    const body = await res.text().catch(() => "");
    throw new Error(`${res.status} ${res.statusText}: ${body}`);
  }
  return res.json() as Promise<T>;
}

export async function getCohort(): Promise<CohortResponse> {
  return jsonFetch<CohortResponse>("/api/cohort");
}

export async function getModels(): Promise<ModelInfo[]> {
  return jsonFetch<ModelInfo[]>("/api/models");
}

// ── future-trainings pipeline (scheduler) ─────────────────────────────────
//
// Queued (not-yet-started) trainings. The backend persists intent in
// qa.sqlite and projects `scheduled_start`/`est_finish` at read time from the
// live folds' rates + a configured slot count, so the home calendar can draw
// planned bars next to the live ones. `status` is "planned" until the real
// fold appears in results/ (then the backend flips it to "submitted" and it
// drops off the calendar).
export type PlannedTraining = {
  id: string;
  dataset_key: string;
  dataset_name: string;
  fold: number;
  trainer: string;
  plans: string;
  priority: number;
  status: "planned" | "submitted" | "cancelled";
  est_duration_hours: number | null;
  submitted_by: string;
  notes: string;
  created_at: string;
  // Projected at read time (null only if the backend couldn't place it).
  scheduled_start: string | null; // ISO8601 UTC
  est_finish: string | null; // ISO8601 UTC
  eta_seconds: number | null;
};

export async function listPlannedTrainings(): Promise<PlannedTraining[]> {
  return jsonFetch<PlannedTraining[]>("/api/planned-trainings");
}

export async function createPlannedTraining(args: {
  dataset_key: string;
  dataset_name: string;
  fold?: number;
  trainer?: string;
  plans?: string;
  priority?: number;
  est_duration_hours?: number | null;
  submitted_by?: string;
  notes?: string;
}): Promise<PlannedTraining> {
  return jsonFetch<PlannedTraining>("/api/planned-trainings", {
    method: "POST",
    body: JSON.stringify(args),
  });
}

export async function updatePlannedTraining(
  id: string,
  updates: { priority?: number; notes?: string; status?: string; est_duration_hours?: number },
): Promise<PlannedTraining> {
  return jsonFetch<PlannedTraining>(`/api/planned-trainings/${id}`, {
    method: "PATCH",
    body: JSON.stringify(updates),
  });
}

export async function deletePlannedTraining(id: string): Promise<void> {
  const res = await fetch(`${API}/api/planned-trainings/${id}`, { method: "DELETE" });
  if (!res.ok) {
    const body = await res.text().catch(() => "");
    throw new Error(`${res.status} ${res.statusText}: ${body}`);
  }
}

// ── donate-case (0.6.0) ──────────────────────────────────────────────────

export type DonateCaseResponse = {
  region: Region;
  dataset_name: string;
  new_cases: CaseInfo[];
  already_existed: boolean;
};

// POST /api/cohort/cases — runs `build_cohort_for_dataset` on the
// server. Used by the "Donate a case" button surfaced in ModelSidebar
// for any model whose source dataset has no compatible case yet.
// Either `model_id` or `dataset_name` may be supplied; `model_id` is
// preferred because the server can verify the dataset has a trained
// model attached.
export async function donateCase(args: {
  model_id?: string;
  dataset_name?: string;
  region?: Region;
  n_pick?: number;
}): Promise<DonateCaseResponse> {
  return jsonFetch<DonateCaseResponse>("/api/cohort/cases", {
    method: "POST",
    body: JSON.stringify(args),
  });
}

// ── upload-your-own-case ─────────────────────────────────────────────────

// Upload a DICOM series (loose .dcm or a .zip/.tar of them) or a NIfTI
// volume (.nii/.nii.gz, one file per model channel) as an ad-hoc test case
// for `model_id`. The server converts to the cohort layout (SimpleITK /
// nibabel, both already in the image) and returns the new case. Uses XHR
// rather than fetch so large DICOM .zip uploads can report progress.
export function uploadCase(args: {
  model_id: string;
  files: File[];
  reviewer?: string;
  onProgress?: (fraction: number) => void;
}): Promise<CaseInfo> {
  return new Promise<CaseInfo>((resolve, reject) => {
    const form = new FormData();
    form.append("model_id", args.model_id);
    if (args.reviewer) form.append("reviewer", args.reviewer);
    for (const f of args.files) form.append("files", f, f.name);

    const xhr = new XMLHttpRequest();
    xhr.open("POST", `${API}/api/cohort/uploads`);
    xhr.upload.onprogress = (e) => {
      if (e.lengthComputable && args.onProgress) {
        args.onProgress(e.loaded / e.total);
      }
    };
    xhr.onload = () => {
      if (xhr.status >= 200 && xhr.status < 300) {
        try {
          resolve(JSON.parse(xhr.responseText) as CaseInfo);
        } catch (err) {
          reject(new Error(`bad upload response: ${String(err)}`));
        }
      } else {
        reject(new Error(`${xhr.status} ${xhr.statusText}: ${xhr.responseText}`));
      }
    };
    xhr.onerror = () => reject(new Error("network error during upload"));
    xhr.send(form);
  });
}

// Seed an uploaded case's ground truth from a prediction so the reviewer can
// correct it in the editor and save it as a training label. Returns the
// updated case (now with groundtruth_path set).
export async function seedGtFromPrediction(
  case_id: string,
  prediction_id: string,
): Promise<CaseInfo> {
  return jsonFetch<CaseInfo>(
    `/api/cases/${case_id}/groundtruth/seed-from-prediction`,
    { method: "POST", body: JSON.stringify({ prediction_id }) },
  );
}

// ── predict queue (0.6.0) ────────────────────────────────────────────────

export type QueueEntry = {
  prediction_id: string;
  model_id: string;
  case_id: string;
  reviewer: string;
  state: "queued" | "running";
  submitted_at: string;
  started_at: string | null;
  position_in_queue: number;
  eta_s: number | null;
};

export type QueueResponse = {
  depth: number;
  in_flight: QueueEntry[];
};

export async function getQueue(): Promise<QueueResponse> {
  return jsonFetch<QueueResponse>("/api/queue");
}

// ── cross-validation (0.7.0) ─────────────────────────────────────────────
//
// A CV run executes the selected case through every available fold
// individually + the ensemble, and flags the out-of-fold (OOF) fold — the one
// that held this case OUT of training, per splits_final.json — as the unbiased
// "honest" result. These types mirror the backend cv.json manifest 1:1.

export type CrossvalComputeHd95 = "none" | "oof_and_ensemble" | "all";

// One single-fold (or ensemble) sub-run. `prediction_id` (when present) is a
// REAL prediction whose /seg + /mesh endpoints serve THIS fold's segmentation,
// so selecting a fold row swaps the viewer overlay for free.
export type CrossvalEntry = {
  kind: "fold" | "ensemble";
  fold: number | null; // fold index for "fold"; null for the ensemble row
  is_oof: boolean; // the single fold that held this case OUT of training
  state: "queued" | "running" | "seg_ready" | "done" | "error";
  prediction_id: string | null;
  mean_fg_dice: number | null;
  metrics: LabelMetric[] | null;
  elapsed_s: number | null;
  error: string | null;
};

export type CrossvalLabelAgg = {
  label: number;
  label_name: string;
  dice_mean: number | null;
  dice_std: number | null;
  dice_min: number | null;
  dice_max: number | null;
  hd95_mean_mm: number | null;
  hd95_std_mm: number | null;
  oof_dice: number | null; // this label's dice on the OOF fold
  oof_hd95_mm: number | null;
};

export type CrossvalAggregate = {
  per_label: CrossvalLabelAgg[];
  fold_mean_fg_dice: Record<string, number | null>; // fold index -> mean fg dice
  cross_fold_mean: number | null;
  cross_fold_std: number | null;
  ensemble_mean_fg_dice: number | null;
  headline_mean_fg_dice: number | null; // OOF fold's, or cross-fold-mean fallback
  headline_kind: "oof" | "cross_fold_mean";
};

// The full cv.json manifest (GET /api/crossval/{id}).
export type CrossvalRun = {
  cv_run_id: string;
  model_id: string;
  case_id: string;
  source_case_stem: string | null;
  region: string;
  dataset_name: string;
  reviewer: string;
  status: "queued" | "running" | "done" | "error";
  compute_hd95: CrossvalComputeHd95;
  oof_fold: number | null;
  oof_resolvable: boolean;
  oof_reason: string;
  available_folds: number[];
  folds_total: number;
  folds_done: number;
  current_fold: number | "ensemble" | null;
  gt_revision: number | null;
  label_map: Record<string, number>;
  started_at: string;
  updated_at: string;
  entries: CrossvalEntry[];
  aggregate: CrossvalAggregate | null;
  error: string | null;
  stale?: boolean; // GT changed since the run was computed (read-time flag)
};

export type CrossvalAccepted = {
  cv_run_id: string;
  status: "queued";
  status_url: string;
};

// Lightweight progress poll (GET /api/crossval/{id}/status). `run` carries the
// full CrossvalRun once status === "done".
export type CrossvalStatus = {
  cv_run_id: string;
  status: "queued" | "running" | "done" | "error";
  model_id: string;
  case_id: string;
  folds_total: number | null;
  folds_done: number;
  current_fold: number | "ensemble" | null;
  entries: Array<
    Pick<CrossvalEntry, "kind" | "fold" | "is_oof" | "state" | "mean_fg_dice" | "prediction_id">
  >;
  error: string | null;
  run: CrossvalRun | null;
};

export async function startCrossval(args: {
  model_id: string;
  case_id: string;
  reviewer?: string;
  compute_hd95?: CrossvalComputeHd95;
}): Promise<CrossvalAccepted> {
  return jsonFetch<CrossvalAccepted>("/api/crossval", {
    method: "POST",
    body: JSON.stringify({
      model_id: args.model_id,
      case_id: args.case_id,
      reviewer: args.reviewer ?? "",
      compute_hd95: args.compute_hd95 ?? "none",
    }),
  });
}

const CROSSVAL_POLL_MS = 2000;
const CROSSVAL_TIMEOUT_MS = 60 * 60 * 1000; // 1h — up to 6 inferences, serialized

// Kick off a CV run and poll it to completion. Mirrors runPredictUntilSeg's
// transient-error tolerance; `onProgress` fires every poll so the panel can
// show live "fold N/M" progress.
export async function pollCrossvalUntilDone(args: {
  model_id: string;
  case_id: string;
  reviewer?: string;
  compute_hd95?: CrossvalComputeHd95;
  onProgress?: (s: CrossvalStatus) => void;
}): Promise<CrossvalRun> {
  const accepted = await startCrossval(args);
  const deadline = Date.now() + CROSSVAL_TIMEOUT_MS;
  let transientErrors = 0;
  while (Date.now() < deadline) {
    await new Promise((r) => setTimeout(r, CROSSVAL_POLL_MS));
    let status: CrossvalStatus;
    try {
      status = await jsonFetch<CrossvalStatus>(accepted.status_url);
      transientErrors = 0;
    } catch (err) {
      if (isTransientError(err) && transientErrors < POLL_MAX_CONSECUTIVE_TRANSIENT_ERRORS) {
        transientErrors += 1;
        console.warn(
          `[pollCrossvalUntilDone] transient (${transientErrors}/${POLL_MAX_CONSECUTIVE_TRANSIENT_ERRORS})`,
          err,
        );
        continue;
      }
      throw err;
    }
    if (args.onProgress) {
      try { args.onProgress(status); } catch { /* noop */ }
    }
    if (status.status === "error") {
      throw new Error(status.error ?? "cross-validation failed");
    }
    if (status.status === "done" && status.run) {
      return status.run;
    }
  }
  throw new Error(`cross-validation timed out after ${CROSSVAL_TIMEOUT_MS / 1000}s`);
}

// Project one CV fold/ensemble entry into the PredictResponse shape the viewer
// + MetricsBlock already consume, so selecting a fold row swaps the overlay
// (ViewerStage / FullscreenStage / Volume3DCanvas all key off prediction_id).
export function foldResultToPrediction(
  run: CrossvalRun,
  e: CrossvalEntry,
): PredictResponse | null {
  if (!e.prediction_id) return null;
  return {
    prediction_id: e.prediction_id,
    seg_url: predictionSegUrl(e.prediction_id),
    metrics_url: `/api/predictions/${e.prediction_id}/metrics`,
    elapsed_s: e.elapsed_s ?? 0,
    used_preprocessed_cache: true,
    label_map: run.label_map,
    metrics: e.metrics,
    metrics_error: e.error,
    started_at: run.started_at,
    postprocessing: null,
    active_gt_revision: run.gt_revision,
    from_cache: false,
  };
}

// Export URLs — the backend renders self-contained HTML (print-to-PDF) + CSV.
export function crossvalReportHtmlUrl(cv_run_id: string): string {
  return `${API}/api/crossval/${cv_run_id}/report.html`;
}
export function crossvalReportCsvUrl(cv_run_id: string): string {
  return `${API}/api/crossval/${cv_run_id}/report.csv`;
}
export function crossvalRollupHtmlUrl(model_id: string): string {
  return `${API}/api/crossval/rollup.html?model_id=${encodeURIComponent(model_id)}`;
}
export function crossvalRollupCsvUrl(model_id: string): string {
  return `${API}/api/crossval/rollup.csv?model_id=${encodeURIComponent(model_id)}`;
}

// Inference is async on the server: POST kicks off a background task and
// returns a prediction_id immediately. We poll the status endpoint until
// the seg is on disk (`seg_ready`), so the Cornerstone overlay can render
// immediately, and then continue polling — on a slower cadence — for the
// metrics to attach (`done`). This avoids the old hang where 8-label
// HD95 took ~11 min and the frontend gave up just before the backend
// flipped to done.

const SEG_POLL_MS = 1500;
const SEG_TIMEOUT_MS = 10 * 60 * 1000; // 10 min hard cap for inference itself
const METRICS_POLL_MS = 3000;
const METRICS_TIMEOUT_MS = 20 * 60 * 1000; // 20 min — covers 15+ label brute-force HD95

function statusToPartialResponse(status: PredictionStatus): PredictResponse {
  if (
    !status.seg_url ||
    !status.metrics_url ||
    status.elapsed_s == null ||
    status.used_preprocessed_cache == null ||
    !status.label_map
  ) {
    throw new Error("inference reached seg_ready but status is missing required fields");
  }
  return {
    prediction_id: status.prediction_id,
    seg_url: status.seg_url,
    metrics_url: status.metrics_url,
    elapsed_s: status.elapsed_s,
    used_preprocessed_cache: status.used_preprocessed_cache,
    label_map: status.label_map,
    metrics: status.metrics,
    metrics_error: status.metrics_error,
    started_at: status.started_at,
    postprocessing: status.postprocessing,
    active_gt_revision: status.active_gt_revision,
  };
}

// Maximum number of consecutive transient errors we tolerate on a polling
// loop before giving up. A run of transient 502/503/504 or network
// hiccups within this window is logged and ignored — only when it
// persists do we surface to the user. Set so we tolerate ~30s of edge
// flapping (e.g. a Caddy/HAProxy reload during a rollout) without
// killing an in-flight inference.
const POLL_MAX_CONSECUTIVE_TRANSIENT_ERRORS = 12;

// Parse an Error thrown by jsonFetch (shape: "STATUS STATUSTEXT: body").
// Returns the HTTP status if recognizable, otherwise null. Network
// errors (no response at all) have no parseable status and surface as
// `null` here — we treat those as transient too because the most
// common cause is a proxy connection-refused during a pod restart.
function parseHttpStatus(err: unknown): number | null {
  if (!(err instanceof Error)) return null;
  const m = err.message.match(/^(\d{3})\s/);
  return m ? parseInt(m[1], 10) : null;
}

// 5xx and "no response" (network failure) are transient — the inference
// is still progressing on the backend, only the proxy hop is bouncing.
// 4xx is permanent — the request itself is wrong (404 = no such
// prediction, 400 = malformed id, etc.) so we don't keep polling.
function isTransientError(err: unknown): boolean {
  const status = parseHttpStatus(err);
  if (status === null) return true; // network error / no response
  return status >= 500 && status <= 599;
}

// Phase 1: kick off inference and poll until the seg is on disk (seg_ready
// or done; both expose seg_url + label_map + elapsed_s). Throws on
// status="error" (inference failure) or timeout.
//
// As of 0.6.0 we accept an optional `onStatus` callback so the UI can show
// live queue position while the prediction is waiting. This is what
// powers the "position 2 in queue" pill on the PhasePill.
export async function runPredictUntilSeg(args: {
  model_id: string;
  case_id: string;
  use_folds: "best" | "all" | number[];
  reviewer?: string;
  onStatus?: (s: PredictionStatus & { accepted?: PredictAcceptedResponse }) => void;
}): Promise<PredictResponse & { accepted: PredictAcceptedResponse }> {
  const accepted = await jsonFetch<PredictAcceptedResponse>("/api/predict", {
    method: "POST",
    body: JSON.stringify({
      model_id: args.model_id,
      case_id: args.case_id,
      use_folds: args.use_folds,
      reviewer: args.reviewer ?? "",
    }),
  });

  const deadline = Date.now() + SEG_TIMEOUT_MS;
  let transientErrors = 0;
  while (Date.now() < deadline) {
    await new Promise((r) => setTimeout(r, SEG_POLL_MS));
    let status: PredictionStatus;
    try {
      status = await jsonFetch<PredictionStatus>(accepted.status_url);
      transientErrors = 0;
    } catch (err) {
      if (
        isTransientError(err) &&
        transientErrors < POLL_MAX_CONSECUTIVE_TRANSIENT_ERRORS
      ) {
        transientErrors += 1;
        console.warn(
          `[runPredictUntilSeg] transient error (${transientErrors}/${POLL_MAX_CONSECUTIVE_TRANSIENT_ERRORS}), retrying:`,
          err,
        );
        continue;
      }
      throw err;
    }
    if (args.onStatus) {
      try { args.onStatus({ ...status, accepted }); } catch { /* noop */ }
    }
    if (status.status === "error") {
      throw new Error(
        `${status.error_type ?? "InferenceError"}: ${status.error_message ?? "unknown"}`,
      );
    }
    if (status.status === "seg_ready" || status.status === "done") {
      const out = statusToPartialResponse(status);
      // Forward the cache-hit signal from the POST so the panel can
      // suppress wall-clock-since-started timing readouts.
      if (accepted.from_cache) out.from_cache = true;
      return { ...out, accepted };
    }
    // queued | running — keep polling
  }
  throw new Error(`inference timed out after ${SEG_TIMEOUT_MS / 1000}s`);
}

// SSE-based status stream — opens an EventSource on
// /api/predictions/{id}/events and invokes `onStatus` on every change.
// Returns an `unsubscribe` function. Falls back to polling silently
// when EventSource is unavailable or the server doesn't support the
// endpoint (caller can detect this by absence of any status callback
// firing within ~2 s and switching to runPredictUntilSeg's poll loop).
export function openPredictionEvents(
  prediction_id: string,
  onStatus: (s: PredictionStatus) => void,
): () => void {
  if (typeof EventSource === "undefined") return () => {};
  const src = new EventSource(`${API}/api/predictions/${prediction_id}/events`);
  src.onmessage = (ev) => {
    try {
      const payload = JSON.parse(ev.data) as PredictionStatus;
      onStatus(payload);
    } catch (err) {
      console.warn("[openPredictionEvents] bad SSE frame", err);
    }
  };
  src.addEventListener("close", () => src.close());
  src.onerror = () => {
    // The browser will retry transparently on connection drop; close
    // explicitly only on terminal errors (server closed the stream).
    if (src.readyState === EventSource.CLOSED) src.close();
  };
  return () => src.close();
}

// Phase 2: poll for metrics to attach. Resolves with the final
// PredictResponse (status="done" — metrics populated, OR metrics_error
// set if computation failed). Throws on inference-error or timeout.
export async function pollMetrics(prediction_id: string): Promise<PredictResponse> {
  const deadline = Date.now() + METRICS_TIMEOUT_MS;
  const url = `/api/predictions/${prediction_id}/status`;
  let transientErrors = 0;
  while (Date.now() < deadline) {
    await new Promise((r) => setTimeout(r, METRICS_POLL_MS));
    let status: PredictionStatus;
    try {
      status = await jsonFetch<PredictionStatus>(url);
      transientErrors = 0;
    } catch (err) {
      if (
        isTransientError(err) &&
        transientErrors < POLL_MAX_CONSECUTIVE_TRANSIENT_ERRORS
      ) {
        transientErrors += 1;
        console.warn(
          `[pollMetrics] transient error (${transientErrors}/${POLL_MAX_CONSECUTIVE_TRANSIENT_ERRORS}), retrying:`,
          err,
        );
        continue;
      }
      throw err;
    }
    if (status.status === "error") {
      // Should not happen post-seg_ready, but surface it just in case.
      throw new Error(
        `${status.error_type ?? "InferenceError"}: ${status.error_message ?? "unknown"}`,
      );
    }
    if (status.status === "done") {
      return statusToPartialResponse(status);
    }
    // seg_ready — metrics still running
  }
  throw new Error(`metrics polling timed out after ${METRICS_TIMEOUT_MS / 1000}s`);
}

// Back-compat single-await wrapper. Not used by the panel any more.
export async function runPredict(args: {
  model_id: string;
  case_id: string;
  use_folds: "best" | "all" | number[];
}): Promise<PredictResponse> {
  const segPhase = await runPredictUntilSeg(args);
  return pollMetrics(segPhase.prediction_id);
}

export async function getPredictionStatus(
  prediction_id: string,
): Promise<PredictionStatus> {
  return jsonFetch<PredictionStatus>(`/api/predictions/${prediction_id}/status`);
}

export function caseImageUrl(case_id: string, channel = 0): string {
  return `${API}/api/cases/${case_id}/image?channel=${channel}`;
}

// `revision` defaults to "active" on the server; pass an integer to load a
// specific historical revision (the GT-edit toolbar uses this to defeat
// the loader cache after a save).
export function caseGroundtruthUrl(case_id: string, revision?: number | "active"): string {
  const r = revision === undefined ? "" : `?revision=${revision}`;
  return `${API}/api/cases/${case_id}/groundtruth${r}`;
}

export function predictionSegUrl(prediction_id: string): string {
  return `${API}/api/predictions/${prediction_id}/seg`;
}

// Pre-computed per-label surface mesh, served as VTK XML PolyData.
// 404 means the precompute hasn't run for this prediction (legacy) or
// failed for this label — the 3D canvas falls back to in-browser
// marching cubes in that case.
export function predictionMeshUrl(prediction_id: string, seg_idx: number): string {
  return `${API}/api/predictions/${prediction_id}/mesh/${seg_idx}`;
}

// ── GT edit revisions ────────────────────────────────────────────────────

export async function getGtRevisions(case_id: string): Promise<GroundTruthRevision[]> {
  return jsonFetch<GroundTruthRevision[]>(`/api/cases/${case_id}/groundtruth/revisions`);
}

export async function activateGtRevision(
  case_id: string,
  revision_id: number,
): Promise<GroundTruthRevision> {
  return jsonFetch<GroundTruthRevision>(
    `/api/cases/${case_id}/groundtruth/revisions/${revision_id}/activate`,
    { method: "POST" },
  );
}

export async function getRecomputeStatus(case_id: string): Promise<RecomputeStatus> {
  return jsonFetch<RecomputeStatus>(`/api/cases/${case_id}/groundtruth/recompute-status`);
}

// Sidecar that travels alongside the raw labelmap bytes. Schema-versioned
// so the server can reject incompatible posts without misinterpreting them.
export type GtEditSidecar = {
  schema_version: 1;
  dimensions: [number, number, number];
  spacing: [number, number, number];
  origin: [number, number, number];
  // Row-major 3x3 direction cosine matrix.
  direction: number[];
  dtype: "uint8" | "uint16";
  // The revision the operator started from. If the server has advanced
  // past this (someone else edited concurrently), the post returns 409.
  source_revision: number | "active";
  // Names → label codes — derived from the case's parent dataset.json;
  // the server cross-checks that every painted label exists.
  label_map: Record<string, number>;
  tools_used: string[];
  stroke_count: number;
  notes: string;
  base_prediction_id: string | null;
  reviewer: string;
};

export async function postGtEdit(args: {
  case_id: string;
  labelmap: Uint8Array;
  sidecar: GtEditSidecar;
}): Promise<GroundTruthRevision> {
  const form = new FormData();
  // Copy into a fresh ArrayBuffer so the Blob never references a
  // SharedArrayBuffer (which BlobPart rejects under cross-origin
  // isolation) and never spans more than the labelmap's bytes.
  const compactBuf = new ArrayBuffer(args.labelmap.byteLength);
  new Uint8Array(compactBuf).set(args.labelmap);
  form.append("labelmap", new Blob([compactBuf], { type: "application/octet-stream" }),
              "labelmap.bin");
  form.append("sidecar", new Blob([JSON.stringify(args.sidecar)], {
    type: "application/json",
  }), "sidecar.json");

  const res = await fetch(`${API}/api/cases/${args.case_id}/groundtruth/edits`, {
    method: "POST",
    body: form,
  });
  if (!res.ok) {
    const body = await res.text().catch(() => "");
    // 409 = optimistic-concurrency conflict (someone else has saved a
    // newer revision since we loaded). Surface that as a typed error so
    // the toolbar can prompt "reload latest GT and reapply your edits".
    const err = new Error(`${res.status} ${res.statusText}: ${body}`) as Error & {
      status?: number;
    };
    err.status = res.status;
    throw err;
  }
  return res.json() as Promise<GroundTruthRevision>;
}

// ── model card themes ────────────────────────────────────────────────────

export async function getModelThemes(): Promise<Record<string, ColorTheme>> {
  return jsonFetch<Record<string, ColorTheme>>("/api/model-themes");
}

export async function setModelThemeApi(
  model_id: string,
  color_key: string,
  updated_by?: string,
): Promise<ColorTheme> {
  return jsonFetch<ColorTheme>(`/api/model-themes/${encodeURIComponent(model_id)}`, {
    method: "POST",
    body: JSON.stringify({ color_key, updated_by: updated_by ?? "" }),
  });
}

export async function deleteModelThemeApi(model_id: string): Promise<void> {
  const res = await fetch(`${API}/api/model-themes/${encodeURIComponent(model_id)}`, {
    method: "DELETE",
  });
  if (!res.ok) {
    const body = await res.text().catch(() => "");
    throw new Error(`${res.status} ${res.statusText}: ${body}`);
  }
}

// ── verdicts ──────────────────────────────────────────────────────────────

export type VerdictValue = "accept" | "reject" | "needs_review";

// Model-level QA decision, derived server-side from the case verdicts.
export type ApprovalStatus = "approved" | "rejected" | "pending";

// Structured reject-reason taxonomy. Keep in sync with REJECT_REASONS in
// src/modelfactory/qa/verdicts.py (same contract as PALETTE ↔ themes.py).
export const REJECT_REASONS = [
  "over_segmentation",
  "misses_small_structures",
  "wrong_anatomy",
  "boundary_errors",
  "false_positives",
  "other",
] as const;
export type RejectReason = (typeof REJECT_REASONS)[number];

// Human-readable labels for the taxonomy keys (UI only).
export const REJECT_REASON_LABEL: Record<RejectReason, string> = {
  over_segmentation: "Over-segmentation",
  misses_small_structures: "Misses small structures",
  wrong_anatomy: "Wrong anatomy",
  boundary_errors: "Boundary / contour errors",
  false_positives: "False positives",
  other: "Other",
};

export type Verdict = {
  id: number;
  prediction_id: string;
  model_id: string;
  case_id: string;
  verdict: VerdictValue;
  notes: string;
  reviewer: string;
  fold_choice: string;
  mean_dice: number | null;
  created_at: string;
  // Only set on rejects — one of REJECT_REASONS or "".
  reject_reason?: string;
  // 0.6.0: server suggests the next compatible case this reviewer hasn't
  // verdicted yet. UI uses this for the "approve-and-next" auto-advance.
  next_case_id?: string | null;
};

export type VerdictSummary = {
  model_id: string;
  accept: number;
  reject: number;
  needs_review: number;
  total: number;
  last_at: string | null;
  last_verdict: VerdictValue | null;
  // Derived decision + per-reason reject breakdown (for the sign-off panel).
  approval_status?: ApprovalStatus;
  reject_reasons?: Record<string, number> | null;
};

export async function postVerdict(args: {
  prediction_id: string;
  model_id: string;
  case_id: string;
  verdict: VerdictValue;
  notes?: string;
  reviewer?: string;
  fold_choice?: string;
  mean_dice?: number | null;
  reject_reason?: string;
}): Promise<Verdict> {
  return jsonFetch<Verdict>("/api/verdicts", {
    method: "POST",
    body: JSON.stringify(args),
  });
}

export async function getVerdictsSummary(): Promise<VerdictSummary[]> {
  return jsonFetch<VerdictSummary[]>("/api/verdicts/summary");
}

export async function getVerdictsForModel(
  model_id: string,
  case_id?: string,
): Promise<Verdict[]> {
  const params = new URLSearchParams({ model_id });
  if (case_id) params.set("case_id", case_id);
  return jsonFetch<Verdict[]>(`/api/verdicts?${params.toString()}`);
}
