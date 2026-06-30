# QA viewer — reviewing models against ground truth

The QA viewer is a web app (FastAPI backend + Next.js/cornerstone.js frontend,
shipped in `services/qa-viewer/`) for **visually validating trained models**: it
runs a model on a held-out cohort and shows the predicted contours next to the
ground-truth masks, with per-class Dice (and optional HD95), so you can decide
whether a model is good enough to promote.

## What it gives you

- **Prediction vs ground truth** overlaid on the image, per structure, in 2D
  (axial/coronal/sagittal) and 3D meshes.
- **Per-class Dice / HD95** for each case, and a **model rollup**.
- **Cross-validation** view: out-of-fold predictions routed by
  `splits_final.json`, with a per-fold and per-model report.
- **Verdicts**: accept / reject / needs-review per case, with a structured
  reject-reason taxonomy, persisted to SQLite on shared storage. A model's
  approval state is derived from its case verdicts.
- A home dashboard with a training-ETA calendar.

## Deploy

```bash
make build-qa-viewer          # build the image (set registry.qaViewerTag in cluster.yaml)
make deploy-qa                # apply infra/kustomize/qa-interface
make smoke-qa                 # GET /api/healthz on the NodePort
```

The viewer runs as a single replica with one predictor cache + CUDA context. In
MIG mode it can take a whole reserved card or a slice — set its GPU in the
deployment env. It reads the same `/factory` (NFS/PVC) tree as training, so it
sees datasets, preprocessed inputs, and `nnUNet_results` checkpoints directly.

## Build a QA cohort

```bash
modelfactory qa cohort prepare --cases-per-dataset 3   # materialize a cohort
modelfactory qa cohort preprocess                      # pre-stage inputs (faster first view)
```

The cohort donates a few cases per trained dataset from its own
`imagesTr`/`labelsTr` so the ground-truth label codes line up with the model.
You can also upload ad-hoc DICOM/NIfTI cases through the UI.

## Exposing it

By default it's a `NodePort` (`network.qaNodePort`). For a hostname, set
`network.qaPublicHost` + `network.ingressEnabled: true` in `cluster.yaml` and put
**authentication** in front — the viewer has none of its own. Never hardcode the
hostname anywhere; it comes from `cluster.yaml`.

## Tips

- First inference on a large, cold (un-prestaged) volume can sit at `running` for
  several minutes — that's the single-threaded nnU-Net resample on CPU, not a
  hang. `qa cohort preprocess` / `backfill` ahead of time avoids it.
- Judge on the per-class Dice the viewer reports from `validation/summary.json`,
  not the training pseudo-dice.
