# model-factory services

In-cluster apps that complement the training pipeline. Today there is one:
the **QA viewer**.

## qa-viewer — Model QA on a single GPU 0

One image (`model-qa:0.1.0`), one pod, one process. FastAPI serves both the
JSON API at `/api/*` and the Next.js static export at `/` from the same
uvicorn instance.

```
                                              +--------------------+
   browser ──HTTP──>  qa-viewer  ───────────> |  /factory hostPath |
                       │  uvicorn             |  NFS root          |
                       │  FastAPI /api/*      +--------------------+
                       │  StaticFiles  /                ▲
                       │  predictor LRU cache           │
                       │  SQLite verdicts at            │
                       │    /factory/qa-cohort/qa.sqlite│
                       └──────── whole GPU 0 ───────────┘
```

GPU 0 was previously reserved for doserad; the reservation was lifted on
2026-05-15 — see CLAUDE.md non-negotiable #2.

### What you get

Workflow:
1. Open the page → **Model catalog** (grid of every trained checkpoint,
   filterable by region, search by name/plans).
2. Click a model → **QA workspace** for that model. Pick a case from the
   curated cohort, run inference (best fold or ensemble), see the
   segmentation overlay on the Cornerstone3D viewer with axial / sagittal /
   coronal MPR, toggle groundtruth overlay, read per-label dice + HD95.
3. Save a verdict: accept / needs review / reject, plus optional notes and
   a reviewer name (remembered in localStorage). Verdicts persist in SQLite
   on NFS and show up as pip counts back on the catalog card.

API endpoints (all under `/api`):
- `GET /models` — every trained checkpoint discovered under `/factory/results`
- `GET /cohort` — curated cases × compatible models
- `GET /cases/<region>/<case>/{image,groundtruth}` — NIfTI streams
- `POST /predict` — run inference; uses pre-staged `.npz` when present
- `GET /predictions/<id>/{seg,metrics}` — segmentation + per-label dice/HD95
- `POST /verdicts` — record an accept/reject/needs_review verdict
- `GET /verdicts?model_id=...&case_id=...` — list verdicts
- `GET /verdicts/summary` — per-model verdict counts (drives catalog pips)

### Local dev

```bash
# 1) cohort (one-time)
modelfactory qa cohort prepare --preprocess

# 2) backend (terminal A) — uvicorn, no static files mounted (dev server takes /)
modelfactory qa server --reload

# 3) frontend (terminal B)
cd services/qa-viewer/web
npm install
NEXT_PUBLIC_QA_API_URL=http://localhost:8080 npm run dev   # serves http://localhost:3000
```

### In-cluster deploy

```bash
# Build the unified image (multi-stage: node builds the static export,
# NGC PyTorch base runs FastAPI + nnUNetv2 inference).
make build-qa-viewer

# Apply manifests (single Deployment on GPU 0, Service, Ingress).
make deploy-qa

# Smoke
make smoke-qa     # port-forwards and curls /api/healthz
```

### Pointing a DNS name at the viewer

The qa-viewer Service is a **NodePort on `:32443`**. The host's existing
An external reverse proxy (Caddy/nginx/ingress) terminates TLS for your
public hostname and reverse-proxies it to the qa-viewer NodePort
(`<node-ip>:32443`). An edge load balancer can keep its existing backend and add an
ACL for the qa hostname.

See `infra/kustomize/qa-interface/README.md` for the full recipe (Caddyfile
block, docker-compose patch, pfSense HAProxy backend, DNS, verification).

### Repo layout

```
services/qa-viewer/
├── Dockerfile               # multi-stage: node build → NGC PyTorch runtime
├── requirements.txt         # FastAPI + nibabel — never torch (NGC base ships it)
└── web/
    ├── package.json         # Next.js 15, React 19, Tailwind v4, Cornerstone3D
    ├── next.config.ts       # output: 'export' (static export)
    ├── app/                 # globals.css (RT tokens), layout, page (catalog↔workspace)
    ├── components/
    │   ├── brand/Logo.tsx           # LogoMark + LogoLockup (RT)
    │   ├── catalog/                 # ModelCatalog + ModelCard (landing view)
    │   ├── workspace/Workspace.tsx  # three-pane QA view
    │   ├── shell/                   # AppHeader, ModelSidebar, InferencePanel
    │   └── viewer/                  # NiftiViewer, ViewerStage, CaseStrip
    ├── lib/                 # api.ts (typed fetchers), store.ts (zustand)
    └── public/brand/        # rt-medical-on-{light,dark}.png

src/modelfactory/
├── inference/
│   ├── predictor_cache.py   # LRU of warm nnUNetPredictors
│   ├── run.py               # single-case inference (preprocessed-cache fast path)
│   └── metrics.py           # dice + HD95 vs groundtruth
└── qa/
    ├── api.py               # FastAPI app (serves both /api and / from one process)
    ├── cohort.py            # build /factory/qa-cohort/
    ├── preprocess.py        # pre-stage .npz/.pkl per (model, case)
    └── verdicts.py          # SQLite verdicts store

infra/kustomize/qa-interface/
├── kustomization.yaml
├── deployment.yaml          # one container, NVIDIA_VISIBLE_DEVICES=0 (whole GPU 0)
├── service.yaml             # NodePort 32443 → containerPort 8080
└── README.md                # Caddy + docker-compose + pfSense recipe
```

### Brand parity

`services/qa-viewer/web/app/globals.css` is a verbatim copy of
the viewer's shared design system — the `--color-rt-*`
tokens, same Fraunces + Geist font setup, same `data-theme` light/dark
toggle. Logo PNGs are the literal files from medgemma-interface. If the
design system there evolves, re-sync here rather than forking.
