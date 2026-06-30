"use client";

// URL <-> store sync for shareable / bookmarkable workspace links.
//
// Next.js static export can't pre-render arbitrary dynamic routes (model_id
// and case_id are runtime-discovered), so we keep the SPA-style single
// app/page.tsx and use query-string parameters instead. The URL looks
// like:
//
//   /?model=Dataset090_SegRap_Aerodigestive_CT::trainer__plans__cfg
//   /?model=...&case=hn_ct/d090_case_001
//   /?model=...&case=...&prediction=abc123
//
// Two flows:
//   1. On mount, `hydrateStoreFromUrl` reads the URL, fetches the cohort,
//      and calls the same store actions the click handlers use.
//   2. On selectedModel/selectedCase/currentPrediction changes, the
//      `pushUrl` helper rewrites the URL via history.replaceState so
//      the back/forward stack isn't bloated by every click.

import type { CaseInfo, CohortResponse, ModelInfo } from "./api";
import { getCohort } from "./api";
import { useQAStore } from "./store";

const PARAM_MODEL = "model";
const PARAM_CASE = "case";
const PARAM_PREDICTION = "prediction";

export type UrlState = {
  modelId: string | null;
  caseId: string | null;
  predictionId: string | null;
};

export function readUrlState(): UrlState {
  if (typeof window === "undefined") {
    return { modelId: null, caseId: null, predictionId: null };
  }
  const params = new URLSearchParams(window.location.search);
  return {
    modelId: params.get(PARAM_MODEL),
    caseId: params.get(PARAM_CASE),
    predictionId: params.get(PARAM_PREDICTION),
  };
}

export function writeUrlState(state: UrlState): void {
  if (typeof window === "undefined") return;
  const params = new URLSearchParams(window.location.search);
  const setOrDelete = (key: string, value: string | null) => {
    if (value) params.set(key, value);
    else params.delete(key);
  };
  setOrDelete(PARAM_MODEL, state.modelId);
  setOrDelete(PARAM_CASE, state.caseId);
  setOrDelete(PARAM_PREDICTION, state.predictionId);
  const search = params.toString();
  const url = `${window.location.pathname}${search ? "?" + search : ""}${window.location.hash}`;
  // replaceState avoids growing the back/forward history on every click;
  // copy-paste links still work because the URL reflects the live state.
  window.history.replaceState(null, "", url);
}

// Pull the cohort once and find matching model + case entries by id.
export async function resolveUrlToCohort(
  state: UrlState,
): Promise<{
  cohort: CohortResponse;
  model: ModelInfo | null;
  case: CaseInfo | null;
}> {
  const cohort = await getCohort();
  const model =
    cohort.trained_models.find((m) => m.model_id === state.modelId) ?? null;
  const c =
    cohort.cases.find((cc) => cc.case_id === state.caseId) ?? null;
  return { cohort, model, case: c };
}

// One-shot hydration: parse the URL and dispatch the store actions
// equivalent to clicking through the catalog. Returns the resolved
// model + case so the caller can decide what else to load (e.g. the
// prediction status).
export async function hydrateStoreFromUrl(): Promise<{
  cohort: CohortResponse;
  model: ModelInfo | null;
  case: CaseInfo | null;
  predictionId: string | null;
}> {
  const state = readUrlState();
  if (!state.modelId) {
    return { cohort: await getCohort(), model: null, case: null, predictionId: null };
  }
  const resolved = await resolveUrlToCohort(state);
  const store = useQAStore.getState();
  if (resolved.model) {
    store.enterWorkspace(resolved.model);
  }
  if (resolved.case) {
    store.setCase(resolved.case);
  }
  return { ...resolved, predictionId: state.predictionId };
}
