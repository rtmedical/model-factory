"use client";

import dynamic from "next/dynamic";
import {
  Boxes,
  LayoutGrid,
  Maximize2,
  Minimize2,
  PanelLeftClose,
  PanelLeftOpen,
  PanelRightClose,
  PanelRightOpen,
  Sparkles,
} from "lucide-react";
import { useEffect, useMemo } from "react";

import {
  caseImageUrl,
  caseGroundtruthUrl,
  getGtRevisions,
  predictionSegUrl,
} from "@/lib/api";
import { useQAStore } from "@/lib/store";

import { IconToggle } from "@/components/shell/IconToggle";

import { CaseStrip } from "./CaseStrip";
import { GtEditToolbar } from "./GtEditToolbar";
import { OrientationStrip } from "./OrientationStrip";

// Cornerstone3D pulls in WebGL + WASM that we only want loaded client-side.
const NiftiViewer = dynamic(() => import("./NiftiViewer").then((m) => m.NiftiViewer), {
  ssr: false,
  loading: () => <div className="rt-viewport-host" />,
});

// FullscreenStage transitively pulls cornerstone (via MprViewport +
// FullscreenToolbar), which references `self` at module load — the same
// reason NiftiViewer is dynamic-imported. Without `ssr: false` the
// production build's prerender fails with ReferenceError: self is not
// defined.
const FullscreenStage = dynamic(
  () => import("./FullscreenStage").then((m) => m.FullscreenStage),
  { ssr: false, loading: () => null },
);

// Volume3DStage also pulls cornerstone's VOLUME_3D + polySeg modules at
// import time — chunk it client-side too, so the heavy worker code only
// loads when the operator presses V / clicks the Boxes button.
const Volume3DStage = dynamic(
  () => import("./Volume3DStage").then((m) => m.Volume3DStage),
  { ssr: false, loading: () => null },
);

export function ViewerStage() {
  const selectedCase = useQAStore((s) => s.selectedCase);
  const prediction = useQAStore((s) => s.currentPrediction);
  const orientation = useQAStore((s) => s.orientation);
  const overlayOpacity = useQAStore((s) => s.overlayOpacity);
  const showGroundtruth = useQAStore((s) => s.showGroundtruth);
  const leftOpen = useQAStore((s) => s.leftSidebarOpen);
  const rightOpen = useQAStore((s) => s.rightSidebarOpen);
  const viewerFocus = useQAStore((s) => s.viewerFocus);
  const toggleLeft = useQAStore((s) => s.toggleLeftSidebar);
  const toggleRight = useQAStore((s) => s.toggleRightSidebar);
  const toggleFocus = useQAStore((s) => s.toggleViewerFocus);
  const toggleMpr = useQAStore((s) => s.toggleFullscreenMpr);
  const fullscreenMpr = useQAStore((s) => s.fullscreenMpr);
  const toggleVolume3D = useQAStore((s) => s.toggleVolume3D);
  const volume3DOpen = useQAStore((s) => s.volume3DOpen);
  const gtActiveRevisionId = useQAStore((s) => s.gtActiveRevisionId);
  const gtEditMode = useQAStore((s) => s.gtEditMode);
  const setGtRevisions = useQAStore((s) => s.setGtRevisions);

  const imageUrl = useMemo(
    () => (selectedCase ? caseImageUrl(selectedCase.case_id, 0) : null),
    [selectedCase],
  );

  // Hydrate the GT revision list whenever the case changes so the picker
  // in the right panel and the fullscreen toolbar is populated without
  // waiting for the first edit.
  useEffect(() => {
    if (!selectedCase) {
      setGtRevisions([], null);
      return;
    }
    let cancelled = false;
    (async () => {
      try {
        const revs = await getGtRevisions(selectedCase.case_id);
        if (cancelled) return;
        const active = revs.find((r) => r.status === "active");
        setGtRevisions(revs, active?.id ?? null);
      } catch (err) {
        if (!cancelled) console.warn("[ViewerStage] getGtRevisions failed:", err);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [selectedCase, setGtRevisions]);
  // The active revision drives the GT URL — passing a number defeats the
  // browser cache when the operator activates a corrected revision.
  const groundtruthUrl = useMemo(
    () =>
      selectedCase?.groundtruth_path
        ? caseGroundtruthUrl(
            selectedCase.case_id,
            gtActiveRevisionId === null ? "active" : gtActiveRevisionId,
          )
        : null,
    [selectedCase, gtActiveRevisionId],
  );
  const segUrl = useMemo(
    () => (prediction ? predictionSegUrl(prediction.prediction_id) : null),
    [prediction],
  );

  return (
    <section className="rt-card flex min-h-0 flex-col overflow-hidden">
      <div className="flex items-center justify-between gap-2 border-b border-[var(--color-rt-line)] px-3 py-2">
        <div className="flex min-w-0 items-center gap-2">
          <IconToggle
            onClick={toggleLeft}
            label={leftOpen ? "hide model sidebar" : "show model sidebar"}
            Icon={leftOpen ? PanelLeftClose : PanelLeftOpen}
            active={leftOpen}
          />
          <OrientationStrip />
        </div>
        <div className="flex shrink-0 items-center gap-2">
          <IconToggle
            onClick={() => segUrl && toggleVolume3D()}
            label={
              !segUrl
                ? "Run inference to enable the 3D view"
                : volume3DOpen
                  ? "exit 3D view (V)"
                  : "3D view · surfaces (V)"
            }
            Icon={Boxes}
            active={volume3DOpen}
            disabled={!segUrl}
          />
          <IconToggle
            onClick={toggleMpr}
            label={fullscreenMpr ? "exit tri-planar (M)" : "tri-planar MPR (M)"}
            Icon={LayoutGrid}
            active={fullscreenMpr}
          />
          <IconToggle
            onClick={toggleFocus}
            label={viewerFocus ? "exit focus (F)" : "focus mode (F)"}
            Icon={viewerFocus ? Minimize2 : Maximize2}
            active={viewerFocus}
          />
          <IconToggle
            onClick={toggleRight}
            label={rightOpen ? "hide inference panel" : "show inference panel"}
            Icon={rightOpen ? PanelRightClose : PanelRightOpen}
            active={rightOpen}
          />
        </div>
      </div>

      <div className="relative flex-1 min-h-0">
        {imageUrl ? (
          <>
            <NiftiViewer
              imageUrl={imageUrl}
              segmentationUrl={segUrl}
              groundtruthUrl={groundtruthUrl}
              orientation={orientation}
              overlayOpacity={overlayOpacity}
              showGroundtruth={showGroundtruth}
            />
            {gtEditMode && selectedCase && (
              <GtEditToolbar
                caseLabelMap={
                  prediction?.label_map ?? { background: 0 }
                }
                caseId={selectedCase.case_id}
                basePredictionId={prediction?.prediction_id ?? null}
                onSaved={(rev) => {
                  setGtRevisions(
                    // Prepend the new revision so the picker shows it
                    // immediately; a follow-up GET refreshes the full list.
                    [
                      {
                        id: rev.id,
                        region: rev.region,
                        case_id: rev.case_id,
                        revision: rev.revision,
                        path: rev.path,
                        base_prediction_id: rev.base_prediction_id,
                        reviewer: rev.reviewer,
                        notes: rev.notes,
                        status: rev.status as "active",
                        created_at: rev.created_at,
                      },
                    ],
                    rev.id,
                  );
                }}
              />
            )}
          </>
        ) : (
          <EmptyStage />
        )}
      </div>

      {!viewerFocus && (
        <div className="border-t border-[var(--color-rt-line)] px-2">
          <CaseStrip />
        </div>
      )}

      {/* Portals — visually independent of the section, mounted once
          and no-ops until the store opens the corresponding overlay. */}
      <FullscreenStage />
      <Volume3DStage />
    </section>
  );
}

function EmptyStage() {
  return (
    <div className="bg-rt-mesh flex h-full flex-col items-center justify-center gap-3 text-center">
      <Sparkles
        size={28}
        className="text-[color-mix(in_oklab,var(--color-rt-accent)_70%,var(--color-rt-muted))]"
      />
      <div>
        <div className="rt-display text-[18px] font-semibold tracking-tight text-[var(--color-rt-ink)]">
          Pick a model and a case
        </div>
        <div className="mt-1 max-w-md text-[12px] text-[var(--color-rt-muted)]">
          Choose a trained checkpoint on the left and a case from the strip below.
          Inference runs against the warm predictor cache on the QA MIG slice.
        </div>
      </div>
    </div>
  );
}

