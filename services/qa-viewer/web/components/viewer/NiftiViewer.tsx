"use client";

import * as cornerstone from "@cornerstonejs/core";
import * as cornerstoneTools from "@cornerstonejs/tools";
import { useEffect, useRef, useState } from "react";

import { useQAStore } from "@/lib/store";
import type { GtEditTool } from "@/lib/store";

import {
  ensureCornerstone,
  QA_BRUSH_STRATEGIES,
  QA_BRUSH_TOOL_NAME,
  QA_EDIT_TOOL_NAMES,
  QA_IDLE_PRIMARY_TOOL,
  QA_RENDERING_ENGINE_ID,
  QA_TOOL_GROUP_ID,
  QA_VIEWPORT_ID,
} from "./cornerstoneInit";
import { SliceSlider } from "./SliceSlider";
import { setNiftiViewerHandle, type GtExtract } from "./viewerHandle";

// Exported so MprViewport can re-attach the same segmentation representations
// when the fullscreen MPR viewports join the shared tool group.
export const SEGMENTATION_ID = "qa-prediction-seg";
export const GT_SEGMENTATION_ID = "qa-groundtruth-seg";
const GT_COLOR_LUT_INDEX = 1;

export type ViewerProps = {
  imageUrl: string | null;
  segmentationUrl: string | null;
  groundtruthUrl: string | null;
  orientation: "axial" | "sagittal" | "coronal";
  overlayOpacity: number;
  showGroundtruth: boolean;
};

export function NiftiViewer({
  imageUrl,
  segmentationUrl,
  groundtruthUrl,
  orientation,
  overlayOpacity,
  showGroundtruth,
}: ViewerProps) {
  const elementRef = useRef<HTMLDivElement | null>(null);
  // IRenderingEngine (the interface) is what `cornerstone.getRenderingEngine`
  // returns and what `new RenderingEngine(...)` is assignable to. Don't type
  // this as the concrete class — TypeScript flags internal fields as missing.
  const renderingEngineRef = useRef<cornerstone.Types.IRenderingEngine | null>(null);
  const currentImageVolumeId = useRef<string | null>(null);
  const currentSegVolumeId = useRef<string | null>(null);
  const currentGtVolumeId = useRef<string | null>(null);
  const [viewportReady, setViewportReady] = useState(false);

  const gtEditMode = useQAStore((s) => s.gtEditMode);
  const gtActiveTool = useQAStore((s) => s.gtActiveTool);
  const gtBrushSize = useQAStore((s) => s.gtBrushSize);
  const gtActiveSegmentIndex = useQAStore((s) => s.gtActiveSegmentIndex);
  const pushGtSnapshot = useQAStore((s) => s.pushGtSnapshot);
  const popGtUndo = useQAStore((s) => s.popGtUndo);
  const popGtRedo = useQAStore((s) => s.popGtRedo);
  const markGtDirty = useQAStore((s) => s.markGtDirty);

  // Re-render the WebGL viewport whenever the host element resizes. The
  // Workspace grid template animates between four widths when the operator
  // toggles sidebars / focus, and Cornerstone3D's rendering engine snapshots
  // the canvas size at viewport-creation time — without a resize() call the
  // viewport keeps its old dimensions and either stretches over or clips
  // the new container. ResizeObserver fires once per frame at most, and we
  // debounce one extra frame so the call lands AFTER the CSS transition
  // settles, avoiding a sequence of intermediate resizes that each cost a
  // full re-render.
  useEffect(() => {
    if (!elementRef.current) return;
    const el = elementRef.current;
    let raf = 0;
    const ro = new ResizeObserver(() => {
      cancelAnimationFrame(raf);
      raf = requestAnimationFrame(() => {
        const engine = renderingEngineRef.current;
        if (!engine) return;
        try {
          // `resize(immediate, keepCamera)` — re-fit the canvas to the host
          // element and preserve the current pan/zoom/slice.
          // eslint-disable-next-line @typescript-eslint/no-explicit-any
          (engine as any).resize?.(true, true);
          engine.getViewport(QA_VIEWPORT_ID)?.render?.();
        } catch (err) {
          console.warn("[NiftiViewer] resize failed:", err);
        }
      });
    });
    ro.observe(el);
    return () => {
      cancelAnimationFrame(raf);
      ro.disconnect();
    };
  }, []);

  // Mount: init Cornerstone, build a single Volume viewport.
  useEffect(() => {
    let cancelled = false;
    (async () => {
      await ensureCornerstone();
      if (cancelled || !elementRef.current) return;

      let engine = cornerstone.getRenderingEngine(QA_RENDERING_ENGINE_ID);
      if (!engine) {
        engine = new cornerstone.RenderingEngine(QA_RENDERING_ENGINE_ID);
      }
      renderingEngineRef.current = engine;

      // Additive `enableElement` — NOT `setViewports`. This rendering engine
      // is SHARED with the MPR + 3D stages (all four viewports live on
      // QA_RENDERING_ENGINE_ID so their cached volume textures stay in ONE
      // WebGL context — see cornerstoneInit.ts). `setViewports([...])` would
      // wipe the sibling viewports; `enableElement` only adds ours. Matches
      // MprViewport / Volume3DCanvas. Cleanup disables this viewport, so a
      // remount re-enables cleanly (no double-enable in React prod).
      engine.enableElement({
        viewportId: QA_VIEWPORT_ID,
        type: cornerstone.Enums.ViewportType.ORTHOGRAPHIC,
        element: elementRef.current,
        defaultOptions: {
          orientation: cornerstone.Enums.OrientationAxis.AXIAL,
          background: [0, 0, 0],
        },
      });

      const group = cornerstoneTools.ToolGroupManager.getToolGroup(QA_TOOL_GROUP_ID);
      group?.addViewport(QA_VIEWPORT_ID, QA_RENDERING_ENGINE_ID);

      // Register a cool-tone color LUT for the groundtruth overlay so it's
      // visually distinct from the prediction's default (warm) palette.
      // Idempotent — only runs once per page load.
      ensureGroundtruthColorLUT();

      setViewportReady(true);
    })();

    return () => {
      cancelled = true;
      setViewportReady(false);
      setNiftiViewerHandle(null);
      // Do NOT destroy the engine — it's SHARED with the MPR + 3D stages.
      // Destroying it tore down the single WebGL context out from under them;
      // cornerstone's global volume cache outlives the engine, so the next
      // render reused a cached volume whose GPU texture belonged to the dead
      // context → "bindTexture: object does not belong to this context", and
      // tool callbacks hit the now-actorless viewport → "Cannot read
      // properties of undefined (reading 'actor')". Mirror the MPR / 3D
      // cleanup: remove our viewport from the tool group and disable it;
      // leave the singleton engine alive for the page lifetime.
      try {
        const group = cornerstoneTools.ToolGroupManager.getToolGroup(QA_TOOL_GROUP_ID);
        group?.removeViewports?.(QA_RENDERING_ENGINE_ID, QA_VIEWPORT_ID);
      } catch {
        /* group already disposed — fine */
      }
      try {
        const engine = cornerstone.getRenderingEngine(QA_RENDERING_ENGINE_ID);
        engine?.disableElement?.(QA_VIEWPORT_ID);
      } catch {
        /* already disabled */
      }
      renderingEngineRef.current = null;
    };
  }, []);

  // Orientation change.
  useEffect(() => {
    const engine = renderingEngineRef.current;
    if (!engine) return;
    const vp = engine.getViewport(QA_VIEWPORT_ID) as cornerstone.Types.IVolumeViewport;
    if (!vp) return;
    const axis = orientation.toUpperCase() as keyof typeof cornerstone.Enums.OrientationAxis;
    vp.setOrientation(cornerstone.Enums.OrientationAxis[axis]);
    vp.render();
  }, [orientation]);

  // Image volume.
  useEffect(() => {
    if (!imageUrl) return;
    let cancelled = false;
    (async () => {
      await ensureCornerstone();
      const engine = renderingEngineRef.current;
      if (!engine || cancelled) return;

      const volumeId = `nifti:${imageUrl}`;
      if (currentImageVolumeId.current === volumeId) return;

      await cornerstone.volumeLoader.createAndCacheVolume(volumeId);

      const vp = engine.getViewport(QA_VIEWPORT_ID) as cornerstone.Types.IVolumeViewport;
      await vp.setVolumes([{ volumeId }]);
      vp.render();
      currentImageVolumeId.current = volumeId;

      // Reset any leftover segmentation refs when the image changes; we'll
      // re-attach below when the seg URLs change.
      currentSegVolumeId.current = null;
      currentGtVolumeId.current = null;
    })();
    return () => {
      cancelled = true;
    };
  }, [imageUrl]);

  // Prediction segmentation overlay. Hidden while editing GT so the brush
  // can't accidentally hit the prediction labelmap, AND hidden when the
  // operator has toggled GT visibility on — the two overlays share a
  // canvas and the user wants a clean "ground-truth only" view.
  useEffect(() => {
    if (!segmentationUrl || gtEditMode || showGroundtruth) {
      removeSegmentation(SEGMENTATION_ID);
      currentSegVolumeId.current = null;
      return;
    }
    let cancelled = false;
    (async () => {
      await ensureCornerstone();
      if (cancelled) return;
      const volumeId = `nifti:${segmentationUrl}`;
      if (currentSegVolumeId.current !== volumeId) {
        await cornerstone.volumeLoader.createAndCacheVolume(volumeId);
        addLabelmapSegmentation(SEGMENTATION_ID, volumeId);
        currentSegVolumeId.current = volumeId;
      }
      setSegmentationOpacity(SEGMENTATION_ID, overlayOpacity);
    })();
    return () => {
      cancelled = true;
    };
  }, [segmentationUrl, overlayOpacity, gtEditMode, showGroundtruth]);

  // Groundtruth overlay. Always shown while in edit mode (you can't edit
  // an invisible labelmap). The volumeId carries `&edit=1` while editing
  // so cache hits don't bring back the pre-edit voxels.
  useEffect(() => {
    const show = gtEditMode || (groundtruthUrl && showGroundtruth);
    if (!groundtruthUrl || !show) {
      removeSegmentation(GT_SEGMENTATION_ID);
      currentGtVolumeId.current = null;
      return;
    }
    let cancelled = false;
    (async () => {
      await ensureCornerstone();
      if (cancelled) return;
      const sep = groundtruthUrl.includes("?") ? "&" : "?";
      const editSuffix = gtEditMode ? `${sep}edit=1` : "";
      const volumeId = `nifti:${groundtruthUrl}${editSuffix}`;
      if (currentGtVolumeId.current !== volumeId) {
        await cornerstone.volumeLoader.createAndCacheVolume(volumeId);
        addLabelmapSegmentation(GT_SEGMENTATION_ID, volumeId, /* gt */ true);
        currentGtVolumeId.current = volumeId;
      }
      // Higher opacity in edit mode so the operator can see what they
      // paint; otherwise pair-with-prediction half-opacity. We apply via
      // the richer setGtSegmentationStyle so the active segment "glows"
      // (heavier outline, brighter fill) — cornerstone routes the *Active
      // vs *Inactive style variants based on setActiveSegmentIndex below.
      const target = gtEditMode
        ? Math.max(0.6, overlayOpacity)
        : Math.min(0.45, overlayOpacity * 0.7);
      setGtSegmentationStyle(GT_SEGMENTATION_ID, target, gtEditMode);

      // Activate the GT segmentation for the edit tools. We also broadcast
      // the active segment index to cornerstone *outside* edit mode so the
      // glow on the operator-selected structure shows up immediately when
      // they're just hovering / viewing.
      try {
        if (gtEditMode) {
          tseg.activeSegmentation?.setActiveSegmentationRepresentation?.(
            QA_TOOL_GROUP_ID,
            GT_SEGMENTATION_ID,
          );
        }
        tseg.segmentIndex?.setActiveSegmentIndex?.(
          GT_SEGMENTATION_ID,
          gtActiveSegmentIndex,
        );
      } catch (err) {
        console.warn("[NiftiViewer] activeSegmentation/index set failed:", err);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [groundtruthUrl, showGroundtruth, overlayOpacity, gtEditMode, gtActiveSegmentIndex]);

  // Tool bindings: when GT edit mode flips on, deactivate WindowLevel on
  // Primary and bind the active edit tool. Reverse on exit.
  useEffect(() => {
    if (!viewportReady) return;
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const ct: any = cornerstoneTools;
    const group = ct.ToolGroupManager.getToolGroup(QA_TOOL_GROUP_ID);
    if (!group) return;
    const MB = ct.Enums.MouseBindings;

    if (gtEditMode) {
      try {
        group.setToolPassive(QA_IDLE_PRIMARY_TOOL);
      } catch {
        /* WindowLevel was already passive */
      }
      const toolName = QA_EDIT_TOOL_NAMES[gtActiveTool];
      // Brush variants (paint / erase / threshold × 2D / 3D) all route to
      // the single BrushTool — distinguish them by swapping the tool's
      // activeStrategy here. The mapping lives in QA_BRUSH_STRATEGIES so
      // strategy spellings stay co-located with cornerstone-tools.
      const brushStrategy = QA_BRUSH_STRATEGIES[gtActiveTool];
      if (toolName) {
        try {
          // Always update brush size while in edit mode so switching from
          // a paint tool to an eraser carries the size over — without this
          // the eraser kept the last threshold-brush radius.
          if (brushStrategy && QA_BRUSH_TOOL_NAME) {
            ct.utilities?.segmentation?.setBrushSizeForToolGroup?.(
              QA_TOOL_GROUP_ID,
              gtBrushSize,
            );
            try {
              group.setActiveStrategy?.(QA_BRUSH_TOOL_NAME, brushStrategy);
            } catch (err) {
              console.warn(
                "[NiftiViewer] setActiveStrategy failed:",
                err,
              );
            }
          }
          group.setToolActive(toolName, {
            bindings: [{ mouseButton: MB.Primary }],
          });
        } catch (err) {
          console.warn("[NiftiViewer] setToolActive failed:", err);
        }
      }
    } else {
      // Restore the idle bindings.
      try {
        const editToolName = QA_EDIT_TOOL_NAMES[gtActiveTool];
        if (editToolName) group.setToolPassive(editToolName);
      } catch {
        /* not active */
      }
      try {
        group.setToolActive(QA_IDLE_PRIMARY_TOOL, {
          bindings: [{ mouseButton: MB.Primary }],
        });
      } catch (err) {
        console.warn("[NiftiViewer] restore WindowLevel failed:", err);
      }
    }
  }, [gtEditMode, gtActiveTool, gtBrushSize, viewportReady]);

  // Push an undo snapshot on each labelmap mutation while editing. The
  // event fires on every brush stroke / scissor commit / fill. We
  // snapshot the live buffer before the next stroke so popGtUndo can
  // restore it. The very first push happens lazily on the first stroke.
  useEffect(() => {
    if (!gtEditMode) return;
    const evtName = (cornerstoneTools as { Enums?: { Events?: { SEGMENTATION_DATA_MODIFIED?: string } } })
      .Enums?.Events?.SEGMENTATION_DATA_MODIFIED;
    if (!evtName) return;
    const onMutate = (evt: Event) => {
      const detail = (evt as CustomEvent<{ segmentationId?: string }>).detail ?? {};
      if (detail.segmentationId !== GT_SEGMENTATION_ID) return;
      const buf = snapshotGtBuffer();
      if (buf) pushGtSnapshot(buf);
      markGtDirty();
    };
    cornerstone.eventTarget.addEventListener(evtName, onMutate as EventListener);
    return () => {
      cornerstone.eventTarget.removeEventListener(evtName, onMutate as EventListener);
    };
  }, [gtEditMode, pushGtSnapshot, markGtDirty]);

  // Undo / redo via window CustomEvents (dispatched by useViewerHotkeys).
  useEffect(() => {
    if (!gtEditMode) return;
    const handleUndo = () => {
      const snap = popGtUndo();
      if (snap) applyGtSnapshot(snap.buffer);
    };
    const handleRedo = () => {
      const snap = popGtRedo();
      if (snap) applyGtSnapshot(snap.buffer);
    };
    window.addEventListener("qa-gt-undo", handleUndo);
    window.addEventListener("qa-gt-redo", handleRedo);
    return () => {
      window.removeEventListener("qa-gt-undo", handleUndo);
      window.removeEventListener("qa-gt-redo", handleRedo);
    };
  }, [gtEditMode, popGtUndo, popGtRedo]);

  // Expose the imperative handle while mounted. Re-set on every change
  // to the volumeId so a fresh extract reads from the current overlay.
  useEffect(() => {
    setNiftiViewerHandle({
      extractGt: () => extractGtVolume(currentGtVolumeId.current),
      applySnapshot: (buf) => applyGtSnapshot(buf),
      snapshotBuffer: () => snapshotGtBuffer(),
    });
    return () => {
      setNiftiViewerHandle(null);
    };
  }, [viewportReady]);

  return (
    <div className="rt-viewport-stack">
      <div
        ref={elementRef}
        className="rt-viewport-host"
        onContextMenu={(e) => e.preventDefault()}
      />
      <SliceSlider
        elementRef={elementRef}
        viewportReady={viewportReady}
        engineId={QA_RENDERING_ENGINE_ID}
        viewportId={QA_VIEWPORT_ID}
        resyncKey={`${orientation}:${imageUrl ?? ""}`}
      />
    </div>
  );
}


// Cool-tone palette for the groundtruth overlay so it doesn't visually
// collide with the prediction's default (warm) colors. The palette is
// registered once at color-LUT index 1; `addLabelmapSegmentation(..., true)`
// references it via `colorLUTOrIndex: 1`.
//
// Cornerstone-tools 1.x stores color LUTs as `Array<[r, g, b, a]>`. Entry 0
// is background (transparent); entries 1..255 are foreground label colors.
// Curated 16-color palette cycling across the full hue wheel (avoiding the
// 0-60° warm band the prediction's auto-LUT uses, so GT + prediction stay
// disambiguable when both are visible). Each segment index maps
// deterministically to a swatch via `segIdx % palette.length`, so an
// anatomy keeps the same colour as the operator moves between cases.
export const GT_PALETTE: Array<[number, number, number]> = [
  [124, 196,  64], // lime
  [ 16, 185, 129], // emerald
  [ 20, 184, 166], // teal
  [ 56, 189, 248], // sky
  [ 99, 102, 241], // indigo
  [139,  92, 246], // violet
  [217,  70, 239], // fuchsia
  [236,  72, 153], // pink
  [251, 113, 133], // rose
  [245, 158,  11], // amber-gold
  [132, 204,  22], // lime-2
  [ 34, 211, 238], // cyan
  [129, 140, 248], // periwinkle
  [192, 132, 252], // plum
  [244, 114, 182], // pink-2
  [253, 186, 116], // peach
];

let _gtLutRegistered = false;
function ensureGroundtruthColorLUT(): void {
  if (_gtLutRegistered) return;
  const lut: Array<[number, number, number, number]> = [[0, 0, 0, 0]];
  for (let i = 1; i < 256; i++) {
    const [r, g, b] = GT_PALETTE[(i - 1) % GT_PALETTE.length];
    lut.push([r, g, b, 255]);
  }
  try {
    tseg?.config?.color?.addColorLUT?.(lut, GT_COLOR_LUT_INDEX);
    _gtLutRegistered = true;
  } catch {
    /* addColorLUT shape drifts between minor versions — non-fatal. If it
       throws, addLabelmapSegmentation's catch will fall back to the default
       LUT and GT will share colors with the prediction. */
  }
}

// The `segmentation.*` namespace in @cornerstonejs/tools 1.x has had several
// breaking signature changes across patch releases (addSegmentations options
// shape, addSegmentationRepresentations vs addSegmentationRepresentations,
// config.style.setStyle, removeSegmentation). The published `.d.ts` lags the
// actual runtime by months. We already guard every call in try/catch and pin
// the major version in package.json; the runtime is the source of truth, so
// don't gate the build on the .d.ts shape. `any`-cast the namespace only
// inside these helpers — call-sites in the component body keep their types.
// eslint-disable-next-line @typescript-eslint/no-explicit-any
const tseg: any = (cornerstoneTools as any).segmentation;
// eslint-disable-next-line @typescript-eslint/no-explicit-any
const tenums: any = (cornerstoneTools as any).Enums;

function addLabelmapSegmentation(
  segId: string,
  volumeId: string,
  isGroundtruth = false,
) {
  // Always go through the hardened removeSegmentation helper so any
  // dangling per-viewport representations are torn down before we re-add.
  // Otherwise `tseg.addSegmentations` throws
  //   "Segmentation with id <segId> already exists"
  // because cornerstone-tools 1.x won't accept an add while reps are
  // still bound to the toolGroup — the inline `tseg.removeSegmentation`
  // we used to do here fails silently in that case.
  removeSegmentation(segId);
  tseg.addSegmentations([
    {
      segmentationId: segId,
      representation: {
        type: tenums.SegmentationRepresentations.Labelmap,
        data: { volumeId },
      },
    },
  ]);
  // In @cornerstonejs/tools 1.x, addSegmentationRepresentations's first arg
  // is the toolGroupId — the rep is bound to every viewport attached to
  // that group. Passing a viewportId here throws
  // `No tool group found for toolGroupId: <viewport-id>` and the overlay
  // silently never appears even though the seg volume itself loads fine.
  //
  // Only reference the GT color LUT if registration succeeded; passing
  // `colorLUTOrIndex: 1` without a LUT at that slot throws
  // `Cannot read properties of undefined (reading 'length')` deep inside
  // the renderer.
  const useGtLUT = isGroundtruth && _gtLutRegistered;
  tseg.addSegmentationRepresentations(QA_TOOL_GROUP_ID, [
    {
      segmentationId: segId,
      type: tenums.SegmentationRepresentations.Labelmap,
      options: useGtLUT ? { colorLUTOrIndex: GT_COLOR_LUT_INDEX } : undefined,
    },
  ]);
}

function removeSegmentation(segId: string) {
  // Step 1: hide first — any frame rendered between state mutation and
  // the next redraw is already invisible. Cornerstone-tools 1.x doesn't
  // always tear the rep's textures down on `removeSegmentation` (the
  // GL textures live in the rendering-engine's cache until something
  // forces a redraw), so a 0-alpha pre-pass guards against a half-cleared
  // state being visible.
  try {
    setSegmentationOpacity(segId, 0);
  } catch {
    /* rep wasn't there — fine */
  }

  // Step 2: explicitly remove this segmentation's representations from
  // the tool group. The seg state remove that follows ALSO needs the
  // reps gone first — leaving a rep bound keeps the seg "live" and any
  // subsequent `addSegmentations` for the same id throws
  //   "Segmentation with id <id> already exists"
  // even though we tried to clear it. Look up the rep UIDs for this
  // segId, then yank them via removeSegmentationsFromToolGroup
  // (top-level on `segmentation`, takes UIDs — not segIds).
  try {
    const reps =
      tseg.state?.getSegmentationRepresentations?.(QA_TOOL_GROUP_ID) ?? [];
    const matching = reps
      .filter((r: { segmentationId?: string }) => r.segmentationId === segId)
      .map(
        (r: { segmentationRepresentationUID?: string }) =>
          r.segmentationRepresentationUID,
      )
      .filter((uid: string | undefined): uid is string => typeof uid === "string");
    if (matching.length > 0 && typeof tseg.removeSegmentationsFromToolGroup === "function") {
      tseg.removeSegmentationsFromToolGroup(QA_TOOL_GROUP_ID, matching);
    }
  } catch {
    /* rep enumeration not supported on this build — fall through */
  }

  // Step 3: remove the segmentation data itself. The function lives at
  // `segmentation.state.removeSegmentation` in @cornerstonejs/tools 1.86 —
  // there is NO top-level `segmentation.removeSegmentation`. Until this
  // fix the call was silently a no-op (the try/catch was swallowing
  // "is not a function") and every subsequent addSegmentations call hit
  // "already exists".
  try {
    if (typeof tseg.state?.removeSegmentation === "function") {
      tseg.state.removeSegmentation(segId);
    } else if (typeof tseg.removeSegmentation === "function") {
      // Older 1.x builds expose it at the top level.
      tseg.removeSegmentation(segId);
    }
  } catch {
    /* nothing to remove */
  }

  // Step 4: trigger a hard redraw across every viewport in the engine.
  // engine.render() walks all viewports; triggerSegmentationRender also
  // pokes the segmentation pipeline so the now-empty rep set is reflected
  // in the next paint. Both are best-effort — the API surface varies
  // across 1.x patch releases.
  try {
    const engine = cornerstone.getRenderingEngine(QA_RENDERING_ENGINE_ID);
    engine?.render?.();
  } catch {
    /* engine not ready */
  }
  try {
    tseg.triggerSegmentationRender?.(QA_TOOL_GROUP_ID);
  } catch {
    /* not supported on this build */
  }
}

function setSegmentationOpacity(segId: string, opacity: number) {
  try {
    tseg.config.style.setStyle(
      { segmentationId: segId, type: tenums.SegmentationRepresentations.Labelmap },
      { fillAlpha: opacity, outlineWidth: 1 },
    );
  } catch {
    /* style API drift between cornerstone-tools versions — non-fatal */
  }
}

/** Style the GT labelmap so the active segment (the one currently selected
 * in the dropdown) visually "glows" — thicker outline + brighter fill —
 * while inactive segments dim. cornerstone-tools 1.86 supports this via the
 * Labelmap style's *Active / *Inactive variants; with
 * `tseg.segmentIndex.setActiveSegmentIndex(GT_SEGMENTATION_ID, idx)` (already
 * called by the GT effect) the renderer picks the right variant per voxel. */
function setGtSegmentationStyle(segId: string, baseOpacity: number, editing: boolean) {
  const active = editing
    ? {
        renderOutline: true,
        outlineWidthActive: 3,
        outlineWidthInactive: 1,
        outlineOpacity: 1.0,
        outlineOpacityInactive: 0.55,
        fillAlpha: Math.min(0.65, baseOpacity),
        fillAlphaInactive: Math.min(0.3, baseOpacity * 0.55),
      }
    : {
        renderOutline: true,
        outlineWidthActive: 2,
        outlineWidthInactive: 1,
        outlineOpacity: 0.9,
        outlineOpacityInactive: 0.55,
        fillAlpha: baseOpacity,
        fillAlphaInactive: baseOpacity * 0.6,
      };
  try {
    tseg.config.style.setStyle(
      { segmentationId: segId, type: tenums.SegmentationRepresentations.Labelmap },
      active,
    );
  } catch {
    /* style API drift across 1.x patch releases — fall back silently */
  }
}

// Read the GT segmentation's scalar buffer + geometry. Returns null when
// the GT overlay isn't loaded — the caller decides how to surface that.
function extractGtVolume(volumeId: string | null): GtExtract | null {
  if (!volumeId) return null;
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const cache: any = (cornerstone as any).cache;
  const vol = cache?.getVolume?.(volumeId);
  if (!vol) return null;

  // Cornerstone exposes the scalar buffer under several shapes across
  // 1.x patch releases — try the canonical method first, fall back to
  // direct property access.
  let scalar: ArrayBufferView | undefined;
  try {
    scalar = typeof vol.getScalarData === "function" ? vol.getScalarData() : undefined;
  } catch {
    scalar = undefined;
  }
  if (!scalar && vol.scalarData) scalar = vol.scalarData as ArrayBufferView;
  if (!scalar) return null;

  let bytes: Uint8Array;
  let dtype: "uint8" | "uint16" = "uint8";
  if (scalar instanceof Uint8Array) {
    bytes = new Uint8Array(scalar);
  } else if (scalar instanceof Uint16Array) {
    dtype = "uint16";
    bytes = new Uint8Array(scalar.buffer.slice(scalar.byteOffset, scalar.byteOffset + scalar.byteLength));
  } else {
    // Promote unexpected types to uint8 (clip to label range).
    const u8 = new Uint8Array((scalar as Uint8Array).length);
    for (let i = 0; i < u8.length; i++) {
      u8[i] = (scalar as unknown as { [k: number]: number })[i] & 0xff;
    }
    bytes = u8;
  }

  const dims = vol.dimensions ?? [0, 0, 0];
  const spacing = vol.spacing ?? [1, 1, 1];
  const origin = vol.origin ?? [0, 0, 0];
  const direction = vol.direction
    ? Array.from(vol.direction as Float64Array | number[]).slice(0, 9)
    : [1, 0, 0, 0, 1, 0, 0, 0, 1];

  return {
    scalarData: bytes,
    dimensions: [Number(dims[0]), Number(dims[1]), Number(dims[2])],
    spacing: [Number(spacing[0]), Number(spacing[1]), Number(spacing[2])],
    origin: [Number(origin[0]), Number(origin[1]), Number(origin[2])],
    direction: direction.map(Number),
    dtype,
  };
}

function snapshotGtBuffer(): Uint8Array | null {
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const cache: any = (cornerstone as any).cache;
  // Find any volume in the cache that backs the GT seg; we may not know
  // the exact volumeId here (callers don't pass it), but the GT
  // segmentation registry tracks it.
  let scalar: ArrayBufferView | undefined;
  try {
    const segs = tseg?.state?.getSegmentation?.(GT_SEGMENTATION_ID);
    const volumeId = segs?.representationData?.LABELMAP?.volumeId
      ?? segs?.representationData?.Labelmap?.volumeId
      ?? segs?.data?.volumeId;
    if (volumeId) {
      const vol = cache?.getVolume?.(volumeId);
      scalar = vol?.getScalarData?.();
    }
  } catch {
    scalar = undefined;
  }
  if (!scalar) return null;
  if (scalar instanceof Uint8Array) return new Uint8Array(scalar);
  if (scalar instanceof Uint16Array) {
    return new Uint8Array(scalar.buffer.slice(scalar.byteOffset, scalar.byteOffset + scalar.byteLength));
  }
  return null;
}

function applyGtSnapshot(buf: Uint8Array): void {
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const cache: any = (cornerstone as any).cache;
  try {
    const segs = tseg?.state?.getSegmentation?.(GT_SEGMENTATION_ID);
    const volumeId = segs?.representationData?.LABELMAP?.volumeId
      ?? segs?.representationData?.Labelmap?.volumeId
      ?? segs?.data?.volumeId;
    if (!volumeId) return;
    const vol = cache?.getVolume?.(volumeId);
    const scalar = vol?.getScalarData?.();
    if (!scalar) return;
    if (scalar instanceof Uint8Array) {
      scalar.set(buf);
    } else if (scalar instanceof Uint16Array) {
      const u16 = new Uint16Array(buf.buffer, buf.byteOffset, buf.byteLength / 2);
      scalar.set(u16);
    }
    // Mark dirty so the renderer pushes the new texture.
    tseg?.triggerSegmentationEvents?.triggerSegmentationDataModified?.(
      GT_SEGMENTATION_ID,
    );
  } catch (err) {
    console.warn("[NiftiViewer] applyGtSnapshot failed:", err);
  }
}
