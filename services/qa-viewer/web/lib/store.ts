"use client";

import { create } from "zustand";

import type {
  CaseInfo,
  CrossvalEntry,
  CrossvalRun,
  GroundTruthRevision,
  LabelMetric,
  ModelInfo,
  PredictResponse,
} from "./api";
import { foldResultToPrediction } from "./api";

export type ViewerOrientation = "axial" | "sagittal" | "coronal";
// "best" → lowest-index fold; "all" → every available fold (ensemble);
// number[] → explicit fold indices the user toggled.
export type FoldChoice = "best" | "all" | number[];
export type AppView = "catalog" | "workspace";
// Run mode is orthogonal to FoldChoice: "single" uses POST /api/predict with
// `use_folds` (best / manual ensemble); "crossval" uses POST /api/crossval and
// evaluates every fold separately. Kept separate so "crossval" never leaks
// into `use_folds` or the verdict's fold_choice string.
export type RunMode = "single" | "crossval";

// Stable key for a CV entry row: "fold:0".."fold:4" or "ensemble".
export function crossvalEntryKey(e: Pick<CrossvalEntry, "kind" | "fold">): string {
  return e.kind === "ensemble" ? "ensemble" : `fold:${e.fold}`;
}

// Contouring tool selector. The brush variants all map to a single
// cornerstone BrushTool with different activeStrategy values — paint vs.
// erase × 2D-circle vs. 3D-sphere — see QA_BRUSH_STRATEGIES in
// cornerstoneInit. The non-brush entries (scissors, paint-fill, segSelect)
// map to distinct cornerstone tools.
export type GtEditTool =
  | "brush2D"
  | "brush3D"
  | "eraser2D"
  | "eraser3D"
  | "thresholdBrush"
  | "rectScissors"
  | "circleScissors"
  | "sphereScissors"
  | "paintFill"
  | "segSelect";

// One labelmap-buffer snapshot for the undo/redo stack. Cap depth at
// UNDO_DEPTH so a long editing session doesn't grow the heap unbounded —
// a 512x512x300 uint8 labelmap is ~75 MB; 25 snapshots = ~1.9 GB
// theoretical worst case, but typical cohort cases are < 256³ → ~400 MB.
export type GtSnapshot = { buffer: Uint8Array; ts: number };
export const GT_UNDO_DEPTH = 25;

type QAState = {
  // Navigation
  view: AppView;
  selectedModel: ModelInfo | null;
  selectedCase: CaseInfo | null;

  // Inference controls
  foldChoice: FoldChoice;
  currentPrediction: PredictResponse | null;
  // Phase-1 (inference) state. Flips to `idle` as soon as the seg is on
  // disk — metricsState then tracks the slower phase 2.
  inferenceState: "idle" | "running" | "error";
  inferenceError: string | null;
  // Phase-2 (metrics) state. `pending` between seg_ready and the metrics
  // landing; `ready` once metrics are attached; `error` if metrics
  // computation failed (non-fatal — the seg is still rendered).
  metricsState: "idle" | "pending" | "ready" | "error";
  metricsError: string | null;

  // ── cross-validation slice ──
  // runMode toggles the right-rail run path; "crossval" runs every fold.
  runMode: RunMode;
  crossval: CrossvalRun | null;
  crossvalState: "idle" | "running" | "ready" | "error";
  crossvalError: string | null;
  // Live "fold N/M" progress while a CV run is in flight.
  crossvalProgress: { completed: number; total: number; currentFold: number | "ensemble" | null };
  // Which fold row is active in the comparison (drives the viewer overlay).
  // "fold:0".."fold:4" | "ensemble" | null.
  selectedFoldKey: string | null;

  // Live queue observability — written while the current prediction is
  // queued/running. 0 = next to start, >0 = waiting; null when not
  // queued. `inferenceQueueDepth` is the global in-flight count.
  inferenceQueuePosition: number | null;
  inferenceQueueDepth: number;
  inferenceQueueEtaSec: number | null;

  // Approve-and-next flag. When a verdict is recorded with auto-advance
  // intent, the VerdictBlock sets this true; the InferencePanel watches
  // selectedCase + this flag and auto-fires inference on the next case.
  // Cleared once the auto-run starts so a manual case-pick does NOT
  // accidentally re-fire.
  autoRunPending: boolean;

  // Viewer controls
  orientation: ViewerOrientation;
  overlayOpacity: number;
  showGroundtruth: boolean;

  // Layout (persisted to rt-qa-layout localStorage)
  leftSidebarOpen: boolean;
  rightSidebarOpen: boolean;
  viewerFocus: boolean;
  // Fullscreen tri-planar (MPR) overlay. Not persisted — should not survive
  // a page reload since the corresponding cornerstone engine is recreated
  // on every open.
  fullscreenMpr: boolean;
  // Whether the CrosshairsTool is active on the MPR viewports. Off by
  // default so the operator can pan/zoom without the crosshair handles
  // hijacking primary-button clicks; toggled from the fullscreen toolbar.
  mprCrosshairsLinked: boolean;
  // Volume 3D (mesh) fullscreen overlay. Activated by the V hotkey or the
  // Boxes IconToggle next to the MPR button. Gated client-side on having a
  // prediction segmentation; not persisted.
  volume3DOpen: boolean;
  // Iterations passed to vtk's WindowedSincPolyDataFilter when surfaces are
  // baked. 1..50; default 20 is a clean tradeoff between smoothness and
  // detail preservation.
  volume3DSmoothing: number;
  // Per-segment visibility overrides for the 3D view. Missing entries are
  // treated as `true` so newly-discovered segments default to visible.
  volume3DSegmentVisibility: Record<number, boolean>;

  // GT correction mode
  gtEditMode: boolean;
  gtDirty: boolean;
  gtActiveSegmentIndex: number;
  gtActiveTool: GtEditTool;
  gtBrushSize: number; // mm
  gtRevisions: GroundTruthRevision[];
  gtActiveRevisionId: number | null;
  gtUndoStack: GtSnapshot[];
  gtRedoStack: GtSnapshot[];
  gtSaving: boolean;
  gtSaveError: string | null;

  // Per-model card color overrides (cached from /api/model-themes)
  modelThemesById: Record<string, string>;

  // Reviewer identity (remembered in localStorage)
  reviewer: string;

  // Actions
  enterWorkspace: (model: ModelInfo) => void;
  exitToCatalog: () => void;
  setCase: (c: CaseInfo | null) => void;
  // Merge fields into the selected case WITHOUT the reset setCase does —
  // used after seeding GT from a prediction so groundtruth_path appears
  // while the seeding prediction stays on screen.
  updateSelectedCase: (patch: Partial<CaseInfo>) => void;
  setFoldChoice: (f: FoldChoice) => void;
  toggleFold: (fold: number) => void;
  setOrientation: (o: ViewerOrientation) => void;
  setOverlayOpacity: (n: number) => void;
  toggleGroundtruth: () => void;
  beginInference: () => void;
  // Set queue position/depth/eta — called from the POST /predict response
  // and from each polled status update so the PhasePill stays live.
  setInferenceQueueInfo: (pos: number | null, depth: number, etaSec: number | null) => void;
  clearInferenceQueueInfo: () => void;
  // Approve-and-next plumbing.
  requestAutoRun: () => void;
  clearAutoRun: () => void;
  // Phase-1 finished: seg on disk, metrics still computing.
  markSegReady: (r: PredictResponse) => void;
  // Phase-2 finished: metrics attached (or error, surfaced separately).
  markMetricsReady: (metrics: LabelMetric[] | null, metricsError: string | null) => void;
  markMetricsError: (msg: string) => void;
  // Legacy alias kept for any non-panel call sites.
  setInferenceResult: (r: PredictResponse) => void;
  setInferenceError: (msg: string) => void;
  resetPrediction: () => void;
  setReviewer: (r: string) => void;

  // Cross-validation actions.
  setRunMode: (m: RunMode) => void;
  beginCrossval: () => void;
  setCrossvalProgress: (completed: number, total: number, currentFold: number | "ensemble" | null) => void;
  markCrossvalReady: (run: CrossvalRun) => void;
  markCrossvalError: (msg: string) => void;
  // Select a fold/ensemble row by key; also swaps currentPrediction so the
  // viewer overlay/MPR/3D follow the selected fold.
  selectCrossvalFold: (key: string) => void;
  resetCrossval: () => void;

  // Layout actions
  toggleLeftSidebar: () => void;
  toggleRightSidebar: () => void;
  toggleViewerFocus: () => void;
  exitViewerFocus: () => void;
  openFullscreenMpr: () => void;
  closeFullscreenMpr: () => void;
  toggleFullscreenMpr: () => void;
  setMprCrosshairsLinked: (linked: boolean) => void;
  toggleVolume3D: () => void;
  closeVolume3D: () => void;
  setVolume3DSmoothing: (n: number) => void;
  setVolume3DSegmentVisibility: (idx: number, visible: boolean) => void;

  // GT correction actions
  enterGtEdit: () => void;
  cancelGtEdit: () => void;
  finishGtEdit: () => void;
  setGtActiveTool: (t: GtEditTool) => void;
  setGtActiveSegmentIndex: (i: number) => void;
  setGtBrushSize: (mm: number) => void;
  pushGtSnapshot: (buf: Uint8Array) => void;
  popGtUndo: () => GtSnapshot | null;
  popGtRedo: () => GtSnapshot | null;
  markGtDirty: () => void;
  setGtRevisions: (rs: GroundTruthRevision[], activeId: number | null) => void;
  beginGtSave: () => void;
  finishGtSave: (err: string | null) => void;

  // Model theme actions
  setModelThemesMap: (m: Record<string, string>) => void;
  setModelTheme: (modelId: string, colorKey: string | null) => void;
};

const REVIEWER_KEY = "rt-qa-reviewer";
const LAYOUT_KEY = "rt-qa-layout";

function loadReviewer(): string {
  if (typeof window === "undefined") return "";
  try {
    return localStorage.getItem(REVIEWER_KEY) ?? "";
  } catch {
    return "";
  }
}

function saveReviewer(value: string) {
  if (typeof window === "undefined") return;
  try {
    localStorage.setItem(REVIEWER_KEY, value);
  } catch {
    /* storage unavailable */
  }
}

type LayoutPrefs = {
  leftSidebarOpen: boolean;
  rightSidebarOpen: boolean;
  viewerFocus: boolean;
};

const LAYOUT_DEFAULT: LayoutPrefs = {
  leftSidebarOpen: true,
  rightSidebarOpen: true,
  viewerFocus: false,
};

function loadLayout(): LayoutPrefs {
  if (typeof window === "undefined") return LAYOUT_DEFAULT;
  try {
    const raw = localStorage.getItem(LAYOUT_KEY);
    if (!raw) return LAYOUT_DEFAULT;
    const parsed = JSON.parse(raw) as Partial<LayoutPrefs>;
    return {
      leftSidebarOpen: parsed.leftSidebarOpen ?? LAYOUT_DEFAULT.leftSidebarOpen,
      rightSidebarOpen: parsed.rightSidebarOpen ?? LAYOUT_DEFAULT.rightSidebarOpen,
      viewerFocus: parsed.viewerFocus ?? LAYOUT_DEFAULT.viewerFocus,
    };
  } catch {
    return LAYOUT_DEFAULT;
  }
}

function saveLayout(p: LayoutPrefs) {
  if (typeof window === "undefined") return;
  try {
    localStorage.setItem(LAYOUT_KEY, JSON.stringify(p));
  } catch {
    /* storage unavailable */
  }
}

const initialLayout = loadLayout();

// When the operator has unsaved GT edits, navigations that would discard
// them get a confirm prompt. The wrapper is defined here so every nav
// action funnels through it.
function confirmIfDirty(get: () => QAState): boolean {
  if (!get().gtDirty) return true;
  if (typeof window === "undefined") return true;
  return window.confirm(
    "You have unsaved ground-truth edits. Discard them and continue?",
  );
}

function resetCrossvalSlice() {
  return {
    runMode: "single" as RunMode,
    crossval: null as CrossvalRun | null,
    crossvalState: "idle" as "idle" | "running" | "ready" | "error",
    crossvalError: null as string | null,
    crossvalProgress: {
      completed: 0,
      total: 0,
      currentFold: null as number | "ensemble" | null,
    },
    selectedFoldKey: null as string | null,
  };
}

function resetGtSlice() {
  return {
    gtEditMode: false,
    gtDirty: false,
    gtActiveSegmentIndex: 1,
    gtActiveTool: "brush2D" as GtEditTool,
    gtBrushSize: 4,
    gtUndoStack: [] as GtSnapshot[],
    gtRedoStack: [] as GtSnapshot[],
    gtSaving: false,
    gtSaveError: null as string | null,
  };
}

export const useQAStore = create<QAState>((set, get) => ({
  view: "catalog",
  selectedModel: null,
  selectedCase: null,
  foldChoice: "best",
  currentPrediction: null,
  inferenceState: "idle",
  inferenceError: null,
  metricsState: "idle",
  metricsError: null,
  ...resetCrossvalSlice(),
  inferenceQueuePosition: null,
  inferenceQueueDepth: 0,
  inferenceQueueEtaSec: null,
  autoRunPending: false,
  orientation: "axial",
  overlayOpacity: 0.55,
  showGroundtruth: false,

  leftSidebarOpen: initialLayout.leftSidebarOpen,
  rightSidebarOpen: initialLayout.rightSidebarOpen,
  viewerFocus: initialLayout.viewerFocus,
  fullscreenMpr: false,
  mprCrosshairsLinked: false,
  volume3DOpen: false,
  volume3DSmoothing: 20,
  volume3DSegmentVisibility: {},

  gtEditMode: false,
  gtDirty: false,
  gtActiveSegmentIndex: 1,
  gtActiveTool: "brush2D",
  gtBrushSize: 4,
  gtRevisions: [],
  gtActiveRevisionId: null,
  gtUndoStack: [],
  gtRedoStack: [],
  gtSaving: false,
  gtSaveError: null,

  modelThemesById: {},

  reviewer: loadReviewer(),

  enterWorkspace: (model) => {
    if (!confirmIfDirty(get)) return;
    set({
      view: "workspace",
      selectedModel: model,
      selectedCase: null,
      currentPrediction: null,
      inferenceState: "idle",
      inferenceError: null,
      metricsState: "idle",
      metricsError: null,
      gtRevisions: [],
      gtActiveRevisionId: null,
      ...resetCrossvalSlice(),
      ...resetGtSlice(),
    });
  },
  exitToCatalog: () => {
    if (!confirmIfDirty(get)) return;
    set({
      view: "catalog",
      selectedModel: null,
      selectedCase: null,
      currentPrediction: null,
      inferenceState: "idle",
      inferenceError: null,
      metricsState: "idle",
      metricsError: null,
      gtRevisions: [],
      gtActiveRevisionId: null,
      ...resetCrossvalSlice(),
      ...resetGtSlice(),
    });
  },
  setCase: (c) => {
    if (!confirmIfDirty(get)) return;
    set({
      selectedCase: c,
      currentPrediction: null,
      inferenceState: "idle",
      inferenceError: null,
      metricsState: "idle",
      metricsError: null,
      gtRevisions: [],
      gtActiveRevisionId: null,
      // setCase clears any pending auto-run; approve-and-next sets it
      // back true AFTER calling setCase so the order is right.
      autoRunPending: false,
      ...resetCrossvalSlice(),
      ...resetGtSlice(),
    });
  },
  updateSelectedCase: (patch) =>
    set((s) =>
      s.selectedCase ? { selectedCase: { ...s.selectedCase, ...patch } } : {},
    ),
  setFoldChoice: (f) => set({ foldChoice: f }),
  toggleFold: (fold) =>
    set((s) => {
      // Coming from a preset ("best"/"all") — switch to explicit-array mode
      // seeded with just this fold (deselects everything else).
      if (!Array.isArray(s.foldChoice)) return { foldChoice: [fold] };
      const has = s.foldChoice.includes(fold);
      const next = has
        ? s.foldChoice.filter((f) => f !== fold)
        : [...s.foldChoice, fold].sort((a, b) => a - b);
      // Empty array would mean "no folds" — fall back to "best" so the run
      // button isn't disabled silently.
      return { foldChoice: next.length === 0 ? "best" : next };
    }),
  setOrientation: (o) => set({ orientation: o }),
  setOverlayOpacity: (n) => set({ overlayOpacity: n }),
  toggleGroundtruth: () => set((s) => ({ showGroundtruth: !s.showGroundtruth })),
  beginInference: () =>
    set({
      inferenceState: "running",
      inferenceError: null,
      metricsState: "idle",
      metricsError: null,
      currentPrediction: null,
      // Defaults until the POST /predict response gives us real numbers
      inferenceQueuePosition: null,
      inferenceQueueDepth: 0,
      inferenceQueueEtaSec: null,
    }),
  setInferenceQueueInfo: (pos, depth, etaSec) =>
    set({
      inferenceQueuePosition: pos,
      inferenceQueueDepth: depth,
      inferenceQueueEtaSec: etaSec,
    }),
  clearInferenceQueueInfo: () =>
    set({
      inferenceQueuePosition: null,
      inferenceQueueDepth: 0,
      inferenceQueueEtaSec: null,
    }),
  requestAutoRun: () => set({ autoRunPending: true }),
  clearAutoRun: () => set({ autoRunPending: false }),
  markSegReady: (r) =>
    set({
      inferenceState: "idle",
      currentPrediction: r,
      // If the seg arrives with metrics already attached (single-fold,
      // 2-class models where the backend finished both phases between
      // our polls), skip straight to ready.
      metricsState: r.metrics ? "ready" : r.metrics_error ? "error" : "pending",
      metricsError: r.metrics_error,
    }),
  markMetricsReady: (metrics, metricsError) =>
    set((s) => ({
      currentPrediction: s.currentPrediction
        ? { ...s.currentPrediction, metrics, metrics_error: metricsError }
        : s.currentPrediction,
      metricsState: metricsError ? "error" : "ready",
      metricsError,
    })),
  markMetricsError: (msg) => set({ metricsState: "error", metricsError: msg }),
  // Legacy single-await callers: set both phases at once.
  setInferenceResult: (r) =>
    set({
      inferenceState: "idle",
      currentPrediction: r,
      metricsState: r.metrics ? "ready" : r.metrics_error ? "error" : "pending",
      metricsError: r.metrics_error,
    }),
  setInferenceError: (msg) =>
    set({
      inferenceState: "error",
      inferenceError: msg,
      metricsState: "idle",
      metricsError: null,
    }),
  resetPrediction: () =>
    set({
      currentPrediction: null,
      inferenceState: "idle",
      inferenceError: null,
      metricsState: "idle",
      metricsError: null,
    }),

  setRunMode: (m) =>
    set((s) =>
      s.runMode === m
        ? {}
        : {
            // Switching mode clears the other mode's result so a stale overlay
            // doesn't linger and the run button reflects a clean slate.
            runMode: m,
            currentPrediction: null,
            inferenceState: "idle",
            inferenceError: null,
            metricsState: "idle",
            metricsError: null,
            crossval: null,
            crossvalState: "idle",
            crossvalError: null,
            crossvalProgress: { completed: 0, total: 0, currentFold: null },
            selectedFoldKey: null,
          },
    ),
  beginCrossval: () =>
    set({
      crossvalState: "running",
      crossvalError: null,
      crossval: null,
      currentPrediction: null,
      selectedFoldKey: null,
      crossvalProgress: { completed: 0, total: 0, currentFold: null },
    }),
  setCrossvalProgress: (completed, total, currentFold) =>
    set({ crossvalProgress: { completed, total, currentFold } }),
  markCrossvalReady: (run) =>
    set(() => {
      // Auto-show the honest out-of-fold overlay (or the ensemble if the case
      // has no OOF fold), so the operator's first frame is the unbiased one.
      const oofKey = run.oof_fold != null ? `fold:${run.oof_fold}` : "ensemble";
      const pick =
        run.entries.find((e) => crossvalEntryKey(e) === oofKey && e.prediction_id) ??
        run.entries.find((e) => e.prediction_id) ??
        null;
      return {
        crossval: run,
        crossvalState: "ready",
        selectedFoldKey: pick ? crossvalEntryKey(pick) : null,
        currentPrediction: pick ? foldResultToPrediction(run, pick) : null,
      };
    }),
  markCrossvalError: (msg) => set({ crossvalState: "error", crossvalError: msg }),
  selectCrossvalFold: (key) =>
    set((s) => {
      if (!s.crossval) return {};
      const entry = s.crossval.entries.find((e) => crossvalEntryKey(e) === key);
      if (!entry) return {};
      const proj = foldResultToPrediction(s.crossval, entry);
      return {
        selectedFoldKey: key,
        currentPrediction: proj ?? s.currentPrediction,
      };
    }),
  resetCrossval: () => set(resetCrossvalSlice()),

  setReviewer: (r) => {
    saveReviewer(r);
    set({ reviewer: r });
  },

  toggleLeftSidebar: () =>
    set((s) => {
      const next = { ...s, leftSidebarOpen: !s.leftSidebarOpen };
      saveLayout({
        leftSidebarOpen: next.leftSidebarOpen,
        rightSidebarOpen: next.rightSidebarOpen,
        viewerFocus: next.viewerFocus,
      });
      return { leftSidebarOpen: next.leftSidebarOpen };
    }),
  toggleRightSidebar: () =>
    set((s) => {
      const next = { ...s, rightSidebarOpen: !s.rightSidebarOpen };
      saveLayout({
        leftSidebarOpen: next.leftSidebarOpen,
        rightSidebarOpen: next.rightSidebarOpen,
        viewerFocus: next.viewerFocus,
      });
      return { rightSidebarOpen: next.rightSidebarOpen };
    }),
  toggleViewerFocus: () =>
    set((s) => {
      const next = { ...s, viewerFocus: !s.viewerFocus };
      saveLayout({
        leftSidebarOpen: next.leftSidebarOpen,
        rightSidebarOpen: next.rightSidebarOpen,
        viewerFocus: next.viewerFocus,
      });
      return { viewerFocus: next.viewerFocus };
    }),
  exitViewerFocus: () =>
    set((s) => {
      if (!s.viewerFocus) return {};
      saveLayout({
        leftSidebarOpen: s.leftSidebarOpen,
        rightSidebarOpen: s.rightSidebarOpen,
        viewerFocus: false,
      });
      return { viewerFocus: false };
    }),
  openFullscreenMpr: () => set({ fullscreenMpr: true }),
  closeFullscreenMpr: () =>
    // Always drop crosshair-link state on close so the next open starts in
    // the default unlinked posture (matching what new operators see).
    set({ fullscreenMpr: false, mprCrosshairsLinked: false }),
  toggleFullscreenMpr: () =>
    set((s) => ({
      fullscreenMpr: !s.fullscreenMpr,
      mprCrosshairsLinked: s.fullscreenMpr ? false : s.mprCrosshairsLinked,
    })),
  setMprCrosshairsLinked: (linked) => set({ mprCrosshairsLinked: linked }),
  toggleVolume3D: () =>
    set((s) => ({
      volume3DOpen: !s.volume3DOpen,
      // Reset visibility on close so re-opening starts with every segment
      // visible — predictions can change between opens (different model).
      volume3DSegmentVisibility: s.volume3DOpen ? {} : s.volume3DSegmentVisibility,
    })),
  closeVolume3D: () => set({ volume3DOpen: false, volume3DSegmentVisibility: {} }),
  setVolume3DSmoothing: (n) =>
    set({ volume3DSmoothing: Math.max(1, Math.min(50, Math.round(n))) }),
  setVolume3DSegmentVisibility: (idx, visible) =>
    set((s) => ({
      volume3DSegmentVisibility: { ...s.volume3DSegmentVisibility, [idx]: visible },
    })),

  enterGtEdit: () =>
    set((s) => {
      if (s.gtEditMode) return {};
      // Seed active segment from the prediction's label_map when possible
      // so the first stroke writes to a real label, not background.
      const lm = s.currentPrediction?.label_map ?? null;
      const first =
        lm
          ? Object.entries(lm)
              .filter(([k]) => k !== "background")
              .map(([, v]) => v)
              .sort((a, b) => a - b)[0] ?? 1
          : 1;
      return {
        gtEditMode: true,
        gtDirty: false,
        gtActiveSegmentIndex: first,
        gtUndoStack: [],
        gtRedoStack: [],
        gtSaveError: null,
      };
    }),
  cancelGtEdit: () => {
    if (!confirmIfDirty(get)) return;
    set(resetGtSlice());
  },
  finishGtEdit: () => set(resetGtSlice()),
  setGtActiveTool: (t) => set({ gtActiveTool: t }),
  setGtActiveSegmentIndex: (i) => set({ gtActiveSegmentIndex: i }),
  setGtBrushSize: (mm) => set({ gtBrushSize: mm }),
  pushGtSnapshot: (buf) =>
    set((s) => {
      const next = [...s.gtUndoStack, { buffer: buf, ts: Date.now() }];
      // Drop the oldest snapshot once we exceed the depth cap.
      while (next.length > GT_UNDO_DEPTH) next.shift();
      return { gtUndoStack: next, gtRedoStack: [], gtDirty: true };
    }),
  popGtUndo: () => {
    const s = get();
    if (s.gtUndoStack.length === 0) return null;
    const next = [...s.gtUndoStack];
    const snap = next.pop()!;
    set({
      gtUndoStack: next,
      gtRedoStack: [...s.gtRedoStack, snap],
    });
    return snap;
  },
  popGtRedo: () => {
    const s = get();
    if (s.gtRedoStack.length === 0) return null;
    const next = [...s.gtRedoStack];
    const snap = next.pop()!;
    set({
      gtRedoStack: next,
      gtUndoStack: [...s.gtUndoStack, snap],
    });
    return snap;
  },
  markGtDirty: () => set({ gtDirty: true }),
  setGtRevisions: (rs, activeId) => set({ gtRevisions: rs, gtActiveRevisionId: activeId }),
  beginGtSave: () => set({ gtSaving: true, gtSaveError: null }),
  finishGtSave: (err) =>
    set((s) => ({
      gtSaving: false,
      gtSaveError: err,
      gtDirty: err ? s.gtDirty : false,
    })),

  setModelThemesMap: (m) => set({ modelThemesById: m }),
  setModelTheme: (modelId, colorKey) =>
    set((s) => {
      const next = { ...s.modelThemesById };
      if (colorKey) {
        next[modelId] = colorKey;
      } else {
        delete next[modelId];
      }
      return { modelThemesById: next };
    }),
}));
