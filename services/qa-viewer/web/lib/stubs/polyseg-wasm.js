// Build-time stub for @icr/polyseg-wasm.
//
// @cornerstonejs/tools eagerly imports the polySeg → wasm path during build,
// even though our viewer only uses *labelmap* segmentation (the path from a
// NIfTI volume to a Cornerstone3D labelmap representation). Bundling the
// real .wasm requires webpack experiments that don't play well with
// Emscripten-generated modules (the `Module not found: Can't resolve 'a'`
// error from the build-time scan of the wasm's import section).
//
// next.config.ts aliases the package to this file. If polygon segmentation
// is ever needed, drop the alias, install the real package, and configure
// `experiments.asyncWebAssembly` plus a proper asset rule for `.wasm`.

class ICRPolySeg {
  static async create() {
    return new ICRPolySeg();
  }
  // The real surface ships a handful of async conversion methods. Stub them
  // so callers get an explicit, traceable error rather than a silent failure
  // if they ever wander down the polySeg path.
  async convertLabelmapToSurface() {
    throw new Error(
      "[polyseg-wasm stub] polygon-segmentation is disabled in this build. " +
      "Labelmap segmentation only. See services/qa-viewer/web/lib/stubs/polyseg-wasm.js.",
    );
  }
  async convertContourToSurface() {
    throw new Error("[polyseg-wasm stub] disabled in this build");
  }
  async convertSurfaceToLabelmap() {
    throw new Error("[polyseg-wasm stub] disabled in this build");
  }
}

export default ICRPolySeg;
export { ICRPolySeg };
