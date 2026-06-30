"use client";

// One ORTHOGRAPHIC viewport for a specific MPR axis (axial/coronal/sagittal).
// Mounted three times by FullscreenStage on a dedicated rendering engine
// (`QA_RENDERING_ENGINE_ID`). All three viewports join the existing
// `QA_TOOL_GROUP_ID`, so the segmentation representations already
// attached by the main viewer render here without re-adding.
//
// The image / segmentation / GT volumes are reused from the main viewer's
// cache — `cornerstone.volumeLoader.createAndCacheVolume(volumeId)` returns
// the same instance for the same volumeId regardless of which engine asks.

import * as cornerstone from "@cornerstonejs/core";
import * as cornerstoneTools from "@cornerstonejs/tools";
import { useEffect, useRef, useState } from "react";

import {
  ensureCornerstone,
  QA_RENDERING_ENGINE_ID,
  QA_MPR_VIEWPORT_IDS,
  QA_TOOL_GROUP_ID,
} from "./cornerstoneInit";
import { GT_SEGMENTATION_ID, SEGMENTATION_ID } from "./NiftiViewer";
import { SliceSlider } from "./SliceSlider";

const ORIENTATION_LABEL: Record<keyof typeof QA_MPR_VIEWPORT_IDS, string> = {
  axial: "Axial",
  coronal: "Coronal",
  sagittal: "Sagittal",
};

export function MprViewport({
  orientation,
  imageVolumeId,
  segmentationVolumeId,
  groundtruthVolumeId,
}: {
  orientation: keyof typeof QA_MPR_VIEWPORT_IDS;
  imageVolumeId: string | null;
  segmentationVolumeId: string | null;
  groundtruthVolumeId: string | null;
}) {
  const elementRef = useRef<HTMLDivElement | null>(null);
  const viewportId = QA_MPR_VIEWPORT_IDS[orientation];
  const [ready, setReady] = useState(false);

  // Re-render the WebGL viewport whenever the host element resizes —
  // matches NiftiViewer's pattern. The MPR grid reflows as the operator
  // resizes the browser window, and the cornerstone engine snapshots the
  // canvas size at creation; without resize() the viewport keeps its
  // initial dimensions and clips.
  useEffect(() => {
    if (!elementRef.current) return;
    const el = elementRef.current;
    let raf = 0;
    const ro = new ResizeObserver(() => {
      cancelAnimationFrame(raf);
      raf = requestAnimationFrame(() => {
        const engine = cornerstone.getRenderingEngine(QA_RENDERING_ENGINE_ID);
        if (!engine) return;
        try {
          // eslint-disable-next-line @typescript-eslint/no-explicit-any
          (engine as any).resize?.(true, true);
          engine.getViewport(viewportId)?.render?.();
        } catch (err) {
          console.warn(`[MprViewport ${orientation}] resize failed:`, err);
        }
      });
    });
    ro.observe(el);
    return () => {
      cancelAnimationFrame(raf);
      ro.disconnect();
    };
  }, [orientation, viewportId]);

  // Mount viewport + attach to tool group.
  useEffect(() => {
    let cancelled = false;
    (async () => {
      await ensureCornerstone();
      if (cancelled || !elementRef.current) return;

      let engine = cornerstone.getRenderingEngine(QA_RENDERING_ENGINE_ID);
      if (!engine) {
        engine = new cornerstone.RenderingEngine(QA_RENDERING_ENGINE_ID);
      }

      const axis = orientation.toUpperCase() as keyof typeof cornerstone.Enums.OrientationAxis;
      engine.enableElement({
        viewportId,
        type: cornerstone.Enums.ViewportType.ORTHOGRAPHIC,
        element: elementRef.current,
        defaultOptions: {
          orientation: cornerstone.Enums.OrientationAxis[axis],
          background: [0, 0, 0],
        },
      });

      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      const ct: any = cornerstoneTools;
      const group = ct.ToolGroupManager.getToolGroup(QA_TOOL_GROUP_ID);
      group?.addViewport(viewportId, QA_RENDERING_ENGINE_ID);

      setReady(true);
    })();
    return () => {
      cancelled = true;
      setReady(false);
      try {
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        const ct: any = cornerstoneTools;
        const group = ct.ToolGroupManager.getToolGroup(QA_TOOL_GROUP_ID);
        group?.removeViewports?.(QA_RENDERING_ENGINE_ID, viewportId);
      } catch {
        /* group already disposed — fine */
      }
      const engine = cornerstone.getRenderingEngine(QA_RENDERING_ENGINE_ID);
      try {
        engine?.disableElement(viewportId);
      } catch {
        /* already disabled */
      }
    };
  }, [orientation, viewportId]);

  // Bind image volume on change. Cornerstone caches by volumeId so the
  // main viewer's earlier load is reused — no extra network fetch.
  useEffect(() => {
    if (!ready || !imageVolumeId) return;
    let cancelled = false;
    (async () => {
      await ensureCornerstone();
      const engine = cornerstone.getRenderingEngine(QA_RENDERING_ENGINE_ID);
      if (!engine || cancelled) return;
      // Make sure the volume exists. If the main viewer already loaded it,
      // createAndCacheVolume short-circuits to the cached entry.
      await cornerstone.volumeLoader.createAndCacheVolume(imageVolumeId);
      const vp = engine.getViewport(viewportId) as cornerstone.Types.IVolumeViewport;
      if (!vp) return;
      const overlays: { volumeId: string }[] = [{ volumeId: imageVolumeId }];
      // Image volume must be set first; segmentation overlays attach via
      // the tool group's segmentation representations (already added by
      // the main viewer through addLabelmapSegmentation), so they render
      // automatically once a viewport joins the group.
      await vp.setVolumes(overlays);
      vp.render();
    })();
    return () => {
      cancelled = true;
    };
  }, [ready, imageVolumeId, viewportId]);

  // Re-attach the segmentation representations to the tool group whenever
  // a seg volume becomes available. Cornerstone-tools 1.x attaches reps
  // per-toolGroup, but viewports added to the group AFTER the rep was
  // registered don't auto-bind — calling addSegmentationRepresentations
  // again is idempotent and ensures this MPR viewport renders the same
  // overlays as the main viewer.
  //
  // Important: we gate the re-add on the *prop* (which mirrors the main
  // viewer's intent), not on the seg state. If the user toggles GT off
  // and the main viewer's GT cleanup effect hasn't fired yet, the seg
  // may still be in state — re-adding the rep here would defeat the
  // hide. Trusting the prop avoids the race.
  useEffect(() => {
    if (!ready) return;
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const ct: any = cornerstoneTools;
    const tseg = ct.segmentation;
    const tenums = ct.Enums;
    if (!tseg || !tenums) return;
    const want: Array<[string, string | null]> = [
      [SEGMENTATION_ID, segmentationVolumeId],
      [GT_SEGMENTATION_ID, groundtruthVolumeId],
    ];
    for (const [segId, volId] of want) {
      if (!volId) continue;
      const inState =
        tseg.state?.getSegmentation?.(segId) ??
        // Older 1.x shapes expose `getSegmentationData` instead; either
        // returning truthy means the seg is registered.
        tseg.state?.getSegmentationData?.(segId);
      if (!inState) continue;
      try {
        tseg.addSegmentationRepresentations(QA_TOOL_GROUP_ID, [
          {
            segmentationId: segId,
            type: tenums.SegmentationRepresentations.Labelmap,
            // GT uses the cool-tone LUT registered at index 1 by NiftiViewer.
            options: segId === GT_SEGMENTATION_ID ? { colorLUTOrIndex: 1 } : undefined,
          },
        ]);
      } catch {
        /* already bound to this group for this viewport — fine */
      }
    }
    const engine = cornerstone.getRenderingEngine(QA_RENDERING_ENGINE_ID);
    engine?.getViewport(viewportId)?.render?.();
  }, [ready, segmentationVolumeId, groundtruthVolumeId, viewportId]);

  return (
    <div className="flex h-full min-h-0 flex-col">
      <div className="flex shrink-0 items-center justify-between px-3 py-1.5">
        <span className="rt-display text-[11px] font-semibold uppercase tracking-[0.14em] text-[var(--color-rt-muted)]">
          {ORIENTATION_LABEL[orientation]}
        </span>
        <span className="font-mono text-[10px] text-[var(--color-rt-line-2)]">
          {orientation === "axial" ? "Z" : orientation === "coronal" ? "Y" : "X"}
        </span>
      </div>
      <div className="rt-viewport-stack flex-1 min-h-0">
        <div
          ref={elementRef}
          className="rt-viewport-host"
          onContextMenu={(e) => e.preventDefault()}
        />
        <SliceSlider
          elementRef={elementRef}
          viewportReady={ready}
          engineId={QA_RENDERING_ENGINE_ID}
          viewportId={viewportId}
          resyncKey={`${orientation}:${imageVolumeId ?? ""}`}
        />
      </div>
    </div>
  );
}
