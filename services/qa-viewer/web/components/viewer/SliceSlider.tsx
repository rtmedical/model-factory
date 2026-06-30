"use client";

// Reusable slice-scrubber for a single Cornerstone3D ORTHOGRAPHIC viewport.
// Subscribes to the viewport's CAMERA_MODIFIED event so mouse-wheel scrolls
// keep the slider in sync, and falls back to scroll-by-delta when
// setViewReference (1.84) is not available on the running runtime.
//
// Originally lived inline in NiftiViewer; pulled out so MprViewport can
// drive its own per-axis slider with the same logic.

import * as cornerstone from "@cornerstonejs/core";
import { useCallback, useEffect, useState } from "react";

export type SliceSliderProps = {
  /** The host <div> that owns the viewport — the CAMERA_MODIFIED event is
   *  dispatched on this element. */
  elementRef: React.RefObject<HTMLDivElement | null>;
  /** Set by the parent once the viewport has been created and the volume
   *  has been bound; used as a gate to avoid touching an undefined viewport. */
  viewportReady: boolean;
  engineId: string;
  viewportId: string;
  /** Re-subscribe whenever any of these change — they signal that the
   *  viewport state may have been swapped underneath us. */
  resyncKey?: string | null;
};

export function SliceSlider({
  elementRef,
  viewportReady,
  engineId,
  viewportId,
  resyncKey,
}: SliceSliderProps) {
  const [imageIndex, setImageIndex] = useState<number | null>(null);
  const [numberOfSlices, setNumberOfSlices] = useState<number | null>(null);

  // Keep slider state in sync with the viewport's camera state. Fires on
  // mouse-wheel scroll and on every programmatic slice change.
  useEffect(() => {
    if (!viewportReady || !elementRef.current) return;
    const el = elementRef.current;

    const sync = () => {
      const engine = cornerstone.getRenderingEngine(engineId);
      const vp = engine?.getViewport(viewportId) as
        | cornerstone.Types.IVolumeViewport
        | undefined;
      if (!vp) return;
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      const utils: any = (cornerstone as any).utilities;
      try {
        const data = utils?.getImageSliceDataForVolumeViewport?.(vp);
        if (
          data &&
          typeof data.imageIndex === "number" &&
          typeof data.numberOfSlices === "number"
        ) {
          setImageIndex(data.imageIndex);
          setNumberOfSlices(data.numberOfSlices);
        }
      } catch {
        /* utility shape drift between cornerstone versions — non-fatal */
      }
    };

    const evt = cornerstone.Enums.Events.CAMERA_MODIFIED;
    el.addEventListener(evt, sync as EventListener);
    const raf = requestAnimationFrame(sync);
    return () => {
      cancelAnimationFrame(raf);
      el.removeEventListener(evt, sync as EventListener);
    };
  }, [viewportReady, engineId, viewportId, resyncKey, elementRef]);

  const onChange = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      const idx = Number(e.target.value);
      const engine = cornerstone.getRenderingEngine(engineId);
      const vp = engine?.getViewport(viewportId) as
        | cornerstone.Types.IVolumeViewport
        | undefined;
      if (!vp) return;

      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      const vpAny: any = vp;
      const camera = vp.getCamera();
      const volumeId: string | null =
        (typeof vpAny.getVolumeId === "function" && vpAny.getVolumeId()) ||
        (typeof vpAny.getAllVolumeIds === "function" &&
          vpAny.getAllVolumeIds()?.[0]) ||
        null;

      if (
        typeof vpAny.setViewReference === "function" &&
        volumeId &&
        camera.viewPlaneNormal
      ) {
        try {
          vpAny.setViewReference({
            sliceIndex: idx,
            volumeId,
            viewPlaneNormal: camera.viewPlaneNormal,
          });
          vp.render();
          return;
        } catch (err) {
          console.warn("[SliceSlider] setViewReference failed, falling back:", err);
        }
      }

      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      const utils: any = (cornerstone as any).utilities;
      try {
        const data = utils?.getImageSliceDataForVolumeViewport?.(vp);
        const cur =
          typeof data?.imageIndex === "number" ? data.imageIndex : imageIndex;
        if (typeof cur === "number" && typeof vpAny.scroll === "function") {
          vpAny.scroll(idx - cur, false, false);
          vp.render();
        }
      } catch (err) {
        console.warn("[SliceSlider] scroll fallback failed:", err);
      }
    },
    [imageIndex, engineId, viewportId],
  );

  if (numberOfSlices === null || imageIndex === null || numberOfSlices <= 1) {
    return null;
  }

  return (
    <div className="rt-slice-slider">
      <span className="rt-slice-label">slice</span>
      <input
        type="range"
        min={0}
        max={numberOfSlices - 1}
        step={1}
        value={imageIndex}
        onChange={onChange}
        aria-label="slice index"
      />
      <span className="rt-slice-counter">
        {imageIndex + 1} / {numberOfSlices}
      </span>
    </div>
  );
}
