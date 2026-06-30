import type { NextConfig } from "next";
import path from "node:path";

// Static export. The FastAPI backend (modelfactory.qa.api) serves /api/* AND
// the built HTML/JS/CSS at the same origin via StaticFiles(html=True), so no
// rewrites are needed and no Node runtime is required at serve time.
//
// For local dev, `npm run dev` still works against a separately-run uvicorn
// on :8080 — set NEXT_PUBLIC_QA_API_URL=http://localhost:8080 in .env.local
// to switch the typed fetchers off "" (same-origin) for the dev session.

const config: NextConfig = {
  reactStrictMode: true,
  poweredByHeader: false,
  output: "export",
  trailingSlash: false,

  // The image optimizer doesn't run in static export; declare unoptimized so
  // <Image> falls back to a plain <img> at build time without a warning.
  images: { unoptimized: true },

  serverExternalPackages: [
    "@cornerstonejs/core",
    "@cornerstonejs/tools",
    "@cornerstonejs/nifti-volume-loader",
  ],

  experimental: {
    optimizePackageImports: ["lucide-react"],
  },

  webpack: (config, { isServer }) => {
    // @cornerstonejs/tools eagerly imports the polySeg → @icr/polyseg-wasm
    // chain during build. The wasm uses Emscripten's import shape (`(import
    // "a" "memory" ...)`) which webpack 5 can't satisfy as an asyncWebAssembly
    // module. We never call polygon-segmentation (labelmap only), so alias
    // the package to a no-op stub. See lib/stubs/polyseg-wasm.js.
    config.resolve = config.resolve ?? {};
    config.resolve.alias = {
      ...(config.resolve.alias ?? {}),
      "@icr/polyseg-wasm": path.resolve(__dirname, "lib/stubs/polyseg-wasm.js"),
    };

    if (!isServer) {
      // Node-flavoured fallbacks inside transitive deps that the browser
      // doesn't actually hit at runtime — but webpack still resolves them
      // during the static analysis pass.
      config.resolve.fallback = {
        ...(config.resolve.fallback ?? {}),
        fs: false,
        path: false,
        crypto: false,
      };
    }
    return config;
  },
};

export default config;
