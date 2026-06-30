"use client";

import { useEffect, useRef, useState } from "react";

import { AppHeader } from "@/components/shell/AppHeader";
import { ModelCatalog } from "@/components/catalog/ModelCatalog";
import { Workspace } from "@/components/workspace/Workspace";
import { useQAStore } from "@/lib/store";
import { hydrateStoreFromUrl, writeUrlState } from "@/lib/urlSync";

export default function Page() {
  const view = useQAStore((s) => s.view);
  const selectedModel = useQAStore((s) => s.selectedModel);
  const selectedCase = useQAStore((s) => s.selectedCase);
  const currentPrediction = useQAStore((s) => s.currentPrediction);

  // One-shot hydration on mount: read ?model/?case/?prediction from the
  // URL and dispatch the same store actions the click handlers use.
  // `hydrated` flips true only AFTER hydrateStoreFromUrl resolves —
  // setting it earlier would race the reverse-sync effect below,
  // which would then immediately write the empty initial store back
  // to the URL and wipe the bookmark we're trying to restore.
  const [hydrated, setHydrated] = useState(false);
  const startedRef = useRef(false);
  useEffect(() => {
    if (startedRef.current) return;
    startedRef.current = true;
    hydrateStoreFromUrl()
      .catch((err) => console.warn("[urlSync] hydrate failed:", err))
      .finally(() => setHydrated(true));
  }, []);

  // Reverse direction: whenever the selection changes, rewrite the URL
  // so a copy-paste link reflects the live workspace. replaceState
  // (inside writeUrlState) avoids back/forward history bloat.
  useEffect(() => {
    if (!hydrated) return;
    writeUrlState({
      modelId: selectedModel?.model_id ?? null,
      caseId: selectedCase?.case_id ?? null,
      predictionId: currentPrediction?.prediction_id ?? null,
    });
  }, [hydrated, selectedModel?.model_id, selectedCase?.case_id, currentPrediction?.prediction_id]);

  return (
    <div className="flex h-screen min-h-0 flex-col">
      <AppHeader />
      <main className="flex min-h-0 flex-1 flex-col px-3 pb-3 pt-3">
        {view === "catalog" ? <ModelCatalog /> : <Workspace />}
      </main>
    </div>
  );
}
