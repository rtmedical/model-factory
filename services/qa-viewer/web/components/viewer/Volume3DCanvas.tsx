"use client";

// Single VOLUME_3D viewport that renders the prediction segmentation as
// surface meshes. We build the meshes ourselves using vtk.js
// (`ImageMarchingCubes` + `WindowedSincPolyDataFilter`) and add them to
// the viewport's internal vtk renderer.
//
// Why we bypass cornerstone-tools' polySeg pipeline: in 1.86 it pulls
// the labelmap volume by calling `cache.getVolume(state.LABELMAP.volumeId)`
// at compute time. That call returns `undefined` for our prediction
// volumes from inside `convertLabelmapToSurface`, even after we register
// the seg with the matching volumeId (the validator at
// `validateLabelmap.js` *passes* — so the cache had the volume — but the
// later polySeg lookup somehow doesn't see it). The crash is always
// `Cannot read properties of undefined (reading 'getScalarData')`.
// Doing marching cubes ourselves sidesteps that state coupling
// entirely; the inputs are read directly from the cornerstone volume
// object we already hold a reference to.

import * as cornerstone from "@cornerstonejs/core";
import * as cornerstoneTools from "@cornerstonejs/tools";
import vtkActor from "@kitware/vtk.js/Rendering/Core/Actor";
import vtkMapper from "@kitware/vtk.js/Rendering/Core/Mapper";
import vtkDataArray from "@kitware/vtk.js/Common/Core/DataArray";
import vtkImageData from "@kitware/vtk.js/Common/DataModel/ImageData";
import vtkImageMarchingCubes from "@kitware/vtk.js/Filters/General/ImageMarchingCubes";
import vtkWindowedSincPolyDataFilter from "@kitware/vtk.js/Filters/General/WindowedSincPolyDataFilter";
import vtkXMLPolyDataReader from "@kitware/vtk.js/IO/XML/XMLPolyDataReader";
import { Loader2 } from "lucide-react";
import { useEffect, useMemo, useRef, useState } from "react";

import { predictionMeshUrl, predictionSegUrl } from "@/lib/api";
import { useQAStore } from "@/lib/store";

import {
  ensureCornerstone,
  QA_RENDERING_ENGINE_ID,
  QA_TRACKBALL_ROTATE_TOOL_NAME,
  QA_VOLUME_3D_TOOL_GROUP_ID,
  QA_VOLUME_3D_VIEWPORT_ID,
  QA_VOLUME_ROTATE_WHEEL_TOOL_NAME,
} from "./cornerstoneInit";
import { GT_PALETTE } from "./NiftiViewer";

// Smoothing iteration count for WindowedSincPolyDataFilter — 20 is the
// vtk default that strikes a good balance between detail preservation and
// the "polished" look the user asked for.
const SMOOTH_ITERATIONS = 20;

type SegmentActor = {
  segIdx: number;
  actor: ReturnType<typeof vtkActor.newInstance>;
};

export function Volume3DCanvas() {
  const elementRef = useRef<HTMLDivElement | null>(null);
  const [viewportReady, setViewportReady] = useState(false);
  const [progress, setProgress] = useState<string | null>("preparing meshes…");
  const [error, setError] = useState<string | null>(null);
  const surfaceBuiltForRef = useRef<string | null>(null);
  const actorsRef = useRef<SegmentActor[]>([]);

  const prediction = useQAStore((s) => s.currentPrediction);
  const segmentVisibility = useQAStore((s) => s.volume3DSegmentVisibility);

  const segments = useMemo(() => {
    if (!prediction?.label_map) return [] as Array<{ name: string; idx: number }>;
    return Object.entries(prediction.label_map)
      .filter(([name]) => name !== "background")
      .map(([name, idx]) => ({ name, idx: idx as number }))
      .sort((a, b) => a.idx - b.idx);
  }, [prediction]);

  // Mount: enable VOLUME_3D viewport, build tool group, bind tools.
  useEffect(() => {
    let cancelled = false;
    (async () => {
      await ensureCornerstone();
      if (cancelled || !elementRef.current) return;

      let engine = cornerstone.getRenderingEngine(QA_RENDERING_ENGINE_ID);
      if (!engine) {
        engine = new cornerstone.RenderingEngine(QA_RENDERING_ENGINE_ID);
      }
      engine.enableElement({
        viewportId: QA_VOLUME_3D_VIEWPORT_ID,
        type: cornerstone.Enums.ViewportType.VOLUME_3D,
        element: elementRef.current,
        defaultOptions: { background: [0.02, 0.04, 0.08] },
      });

      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      const ct: any = cornerstoneTools;
      const ToolGroupManager = ct.ToolGroupManager;
      const MB = ct.Enums.MouseBindings;

      let group = ToolGroupManager.getToolGroup?.(QA_VOLUME_3D_TOOL_GROUP_ID) ?? null;
      if (!group) {
        group = ToolGroupManager.createToolGroup(QA_VOLUME_3D_TOOL_GROUP_ID);
        const trySetActive = (toolName: string | null, button: number) => {
          if (!toolName) return;
          try {
            group.addTool(toolName);
            group.setToolActive(toolName, { bindings: [{ mouseButton: button }] });
          } catch (err) {
            console.warn(`[Volume3DCanvas] couldn't activate ${toolName}:`, err);
          }
        };
        trySetActive(QA_TRACKBALL_ROTATE_TOOL_NAME, MB.Primary);
        trySetActive("Pan", MB.Auxiliary);
        trySetActive("Zoom", MB.Secondary);
        if (QA_VOLUME_ROTATE_WHEEL_TOOL_NAME) {
          try {
            group.addTool(QA_VOLUME_ROTATE_WHEEL_TOOL_NAME);
            group.setToolActive(QA_VOLUME_ROTATE_WHEEL_TOOL_NAME, {
              bindings: [{ mouseButton: MB.Wheel ?? MB.Auxiliary }],
            });
          } catch {
            /* not present in this build — fine */
          }
        }
      }
      group.addViewport(QA_VOLUME_3D_VIEWPORT_ID, QA_RENDERING_ENGINE_ID);

      setViewportReady(true);
    })();
    return () => {
      cancelled = true;
      setViewportReady(false);
      _disposeActors();
      try {
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        const ct: any = cornerstoneTools;
        const group = ct.ToolGroupManager.getToolGroup?.(QA_VOLUME_3D_TOOL_GROUP_ID);
        group?.removeViewports?.(QA_RENDERING_ENGINE_ID, QA_VOLUME_3D_VIEWPORT_ID);
      } catch {
        /* group already disposed */
      }
      try {
        const engine = cornerstone.getRenderingEngine(QA_RENDERING_ENGINE_ID);
        engine?.disableElement?.(QA_VOLUME_3D_VIEWPORT_ID);
      } catch {
        /* already disabled */
      }
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  function _disposeActors() {
    const engine = cornerstone.getRenderingEngine(QA_RENDERING_ENGINE_ID);
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const vp = engine?.getViewport(QA_VOLUME_3D_VIEWPORT_ID) as any;
    const renderer = vp?.getRenderer?.();
    for (const { actor } of actorsRef.current) {
      try {
        renderer?.removeActor?.(actor);
      } catch {
        /* renderer disposed — fine */
      }
    }
    actorsRef.current = [];
  }

  // Build the meshes whenever a new prediction lands while 3D is open.
  // The expensive bit is the marching-cubes pass per segment; on a typical
  // 256³ uint8 labelmap each segment runs in 0.5–2 s on the main thread.
  // We yield to the event loop between segments so the spinner can
  // animate and the operator can still click "exit".
  useEffect(() => {
    if (!viewportReady || !prediction?.prediction_id) return;
    if (surfaceBuiltForRef.current === prediction.prediction_id) return;

    let cancelled = false;
    (async () => {
      _disposeActors();
      setError(null);
      try {
        const volumeId = `nifti:${predictionSegUrl(prediction.prediction_id)}`;
        setProgress("loading prediction volume…");

        // Ensure the volume is cached; await both the initial load and any
        // in-flight one started by the main viewer.
        await cornerstone.volumeLoader.createAndCacheVolume(volumeId);
        if (cancelled) return;
        const volume = cornerstone.cache.getVolume(volumeId);
        if (!volume) {
          throw new Error(
            `prediction volume not in cache after createAndCacheVolume(${volumeId})`,
          );
        }
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        const v: any = volume;
        const scalarData: Uint8Array | Uint16Array | Int16Array =
          typeof v.getScalarData === "function"
            ? v.getScalarData()
            : v.scalarData;
        if (!scalarData) {
          throw new Error("volume has no scalarData");
        }
        const dimensions: [number, number, number] = v.dimensions;
        const spacing: [number, number, number] = v.spacing ?? [1, 1, 1];
        const origin: [number, number, number] = v.origin ?? [0, 0, 0];
        const direction: number[] = Array.from(v.direction ?? [1, 0, 0, 0, 1, 0, 0, 0, 1]);

        const engine = cornerstone.getRenderingEngine(QA_RENDERING_ENGINE_ID);
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        const vp = engine?.getViewport(QA_VOLUME_3D_VIEWPORT_ID) as any;
        const renderer = vp?.getRenderer?.();
        if (!renderer) {
          throw new Error("3D viewport renderer not ready");
        }

        // Build one mesh per non-background segment. Fetch all the
        // backend-precomputed .vtp files in parallel — they're tiny
        // (50 KB–2 MB) and the server can stream them at the same time.
        // For each label that 404s (precompute hasn't run for this
        // prediction, or failed for this one label), fall back to
        // in-browser marching cubes.
        setProgress(`loading ${segments.length} mesh${segments.length === 1 ? "" : "es"}…`);
        const fetches = segments.map((seg) =>
          _fetchVtpActor(prediction.prediction_id, seg.idx).catch((err) => {
            console.warn(`[Volume3DCanvas] vtp fetch failed for ${seg.name}:`, err);
            return null;
          }),
        );
        const fetched = await Promise.all(fetches);
        if (cancelled) return;

        for (let i = 0; i < segments.length; i++) {
          if (cancelled) return;
          const { name, idx } = segments[i];
          let actor = fetched[i];
          if (!actor) {
            // Fallback: in-browser marching cubes (the old hot path).
            setProgress(`building mesh ${i + 1} / ${segments.length} · ${name}`);
            await new Promise<void>((r) => requestAnimationFrame(() => r()));
            if (cancelled) return;
            actor = _buildSegmentActor(scalarData, dimensions, spacing, origin, direction, idx);
          }
          if (!actor) continue;
          // Style + color the actor (one helper so both code paths agree).
          _styleSegmentActor(actor, idx);
          // Default visibility from the store; default-visible if absent.
          actor.setVisibility(segmentVisibility[idx] !== false);
          renderer.addActor(actor);
          actorsRef.current.push({ segIdx: idx, actor });
        }

        if (cancelled) return;
        try {
          vp.resetCamera?.();
        } catch {
          /* viewport may have been disposed */
        }
        vp?.render?.();
        surfaceBuiltForRef.current = prediction.prediction_id;
      } catch (err) {
        if (cancelled) return;
        const msg = err instanceof Error ? err.message : String(err);
        console.warn("[Volume3DCanvas] mesh build failed:", err);
        setError(msg);
      } finally {
        if (!cancelled) setProgress(null);
      }
    })();
    return () => {
      cancelled = true;
    };
    // segments is derived from prediction.label_map; rebuild only on a
    // genuinely new prediction (the ref-check above gates that).
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [viewportReady, prediction?.prediction_id]);

  // Apply per-segment visibility without re-running marching cubes.
  useEffect(() => {
    if (!viewportReady || actorsRef.current.length === 0) return;
    let dirty = false;
    for (const { segIdx, actor } of actorsRef.current) {
      const want = segmentVisibility[segIdx] !== false;
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      const cur = (actor as any).getVisibility?.() ?? true;
      if (cur !== want) {
        actor.setVisibility(want);
        dirty = true;
      }
    }
    if (dirty) {
      const engine = cornerstone.getRenderingEngine(QA_RENDERING_ENGINE_ID);
      engine?.getViewport(QA_VOLUME_3D_VIEWPORT_ID)?.render?.();
    }
  }, [viewportReady, segmentVisibility]);

  return (
    <div className="relative h-full w-full">
      <div
        ref={elementRef}
        className="h-full w-full"
        style={{ touchAction: "none" }}
        onContextMenu={(e) => e.preventDefault()}
      />
      {(progress || error) && (
        <div className="pointer-events-none absolute inset-x-0 top-3 flex justify-center">
          <div
            className={
              "inline-flex items-center gap-2 rounded-full border px-3 py-1.5 text-[11px] backdrop-blur " +
              (error
                ? "border-[var(--color-rt-pip-error)] bg-[color-mix(in_oklab,var(--color-rt-paper)_88%,transparent)] text-[var(--color-rt-pip-error)]"
                : "border-[var(--color-rt-line)] bg-[color-mix(in_oklab,var(--color-rt-paper)_88%,transparent)] text-[var(--color-rt-muted)]")
            }
          >
            {!error && <Loader2 className="animate-spin" size={11} />}
            {error ? `mesh build failed: ${error}` : progress}
          </div>
        </div>
      )}
    </div>
  );
}

/** Build one smoothed surface actor for a single label value via vtk.js.
 *  Returns null if marching cubes produced no triangles (segment is empty
 *  or the threshold misses). */
function _buildSegmentActor(
  scalarData: Uint8Array | Uint16Array | Int16Array,
  dims: [number, number, number],
  spacing: [number, number, number],
  origin: [number, number, number],
  direction: number[],
  segIdx: number,
): ReturnType<typeof vtkActor.newInstance> | null {
  // Threshold to a 0/1 mask isolating this label. Marching cubes at 0.5
  // then gives a clean iso-surface around just this segment instead of
  // following whatever boundary happens to cross the labelmap value.
  const mask = new Uint8Array(scalarData.length);
  for (let i = 0; i < scalarData.length; i++) {
    mask[i] = scalarData[i] === segIdx ? 1 : 0;
  }
  // Skip empty segments to avoid an empty-polydata marching-cubes run.
  let any = false;
  for (let i = 0; i < mask.length; i++) {
    if (mask[i]) {
      any = true;
      break;
    }
  }
  if (!any) return null;

  const segImage = vtkImageData.newInstance();
  segImage.setDimensions(dims);
  segImage.setSpacing(spacing);
  segImage.setOrigin(origin);
  // setDirection's signature wants a 3x3 mat (Float32Array of length 9) —
  // the runtime is lenient about an array, but the .d.ts isn't. Coerce.
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  segImage.setDirection(direction as any);
  const dataArray = vtkDataArray.newInstance({
    numberOfComponents: 1,
    values: mask,
    name: "labelmask",
  });
  segImage.getPointData().setScalars(dataArray);

  const mc = vtkImageMarchingCubes.newInstance({
    contourValue: 0.5,
    computeNormals: true,
    mergePoints: true,
  });
  mc.setInputData(segImage);
  const polydata = mc.getOutputData();
  // Empty polydata means no surface (sparse seg) — skip the smoothing pass.
  if (!polydata.getPoints?.()?.getNumberOfPoints?.()) {
    return null;
  }

  const smoother = vtkWindowedSincPolyDataFilter.newInstance({
    numberOfIterations: SMOOTH_ITERATIONS,
    featureAngle: 60,
    passBand: 0.1,
    nonManifoldSmoothing: false,
    normalizeCoordinates: true,
    boundarySmoothing: false,
  });
  smoother.setInputData(polydata);
  const smoothed = smoother.getOutputData();

  const mapper = vtkMapper.newInstance();
  mapper.setInputData(smoothed);
  mapper.setScalarVisibility(false);

  const actor = vtkActor.newInstance();
  actor.setMapper(mapper);
  return actor;
}

/** Apply the per-segment color + lighting properties. Shared between the
 *  in-browser marching-cubes fallback and the backend .vtp fast path so
 *  both code paths render identically. */
function _styleSegmentActor(
  actor: ReturnType<typeof vtkActor.newInstance>,
  segIdx: number,
): void {
  const [r, g, b] = GT_PALETTE[(segIdx - 1) % GT_PALETTE.length];
  const prop = actor.getProperty();
  prop.setColor(r / 255, g / 255, b / 255);
  // Some specular highlights so the curvature reads on the dark backdrop.
  prop.setAmbient(0.2);
  prop.setDiffuse(0.8);
  prop.setSpecular(0.3);
  prop.setSpecularPower(20);
  prop.setOpacity(1.0);
}

/** Fetch a pre-computed .vtp from the backend and wrap it in an actor.
 *  Returns null on 404 (the caller falls back to in-browser marching
 *  cubes for that label). Other errors propagate. */
async function _fetchVtpActor(
  prediction_id: string,
  segIdx: number,
): Promise<ReturnType<typeof vtkActor.newInstance> | null> {
  const res = await fetch(predictionMeshUrl(prediction_id, segIdx));
  if (res.status === 404) return null;
  if (!res.ok) {
    throw new Error(`mesh fetch failed: ${res.status} ${res.statusText}`);
  }
  const buf = await res.arrayBuffer();

  const reader = vtkXMLPolyDataReader.newInstance();
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  (reader as any).parseAsArrayBuffer(buf);
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const polydata = (reader as any).getOutputData(0);
  if (!polydata?.getPoints?.()?.getNumberOfPoints?.()) {
    return null;
  }

  const mapper = vtkMapper.newInstance();
  mapper.setInputData(polydata);
  mapper.setScalarVisibility(false);

  const actor = vtkActor.newInstance();
  actor.setMapper(mapper);
  return actor;
}
