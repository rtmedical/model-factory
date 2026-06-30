"use client";

// Tiny module-scoped registry so non-Cornerstone consumers (the GT edit
// toolbar) can talk to the live NiftiViewer instance without statically
// importing it — that would pull Cornerstone into the SSR graph and
// crash with `self is not defined` during `next build` static export.
//
// The viewer registers itself on mount via `setNiftiViewerHandle` and
// clears it on unmount. There is at most one viewer at a time.

export type GtExtract = {
  scalarData: Uint8Array;
  dimensions: [number, number, number];
  spacing: [number, number, number];
  origin: [number, number, number];
  direction: number[];
  dtype: "uint8" | "uint16";
};

export type NiftiViewerHandle = {
  extractGt: () => GtExtract | null;
  applySnapshot: (buf: Uint8Array) => void;
  snapshotBuffer: () => Uint8Array | null;
};

let _activeHandle: NiftiViewerHandle | null = null;

export function setNiftiViewerHandle(h: NiftiViewerHandle | null): void {
  _activeHandle = h;
}

export function getNiftiViewerHandle(): NiftiViewerHandle | null {
  return _activeHandle;
}
