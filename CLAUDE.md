# CLAUDE.md

Guidance for Claude Code (and human contributors) working in this repository.
[`README.md`](./README.md) is the user-facing overview; this file adds the
internals and conventions. Site-specific operational notes live in a private
overlay (`overlays/private/`, git-ignored) and are not part of the public repo.

## What this repo is

A scalable, Kubernetes-based training factory for medical-image segmentation
(nnU-Net v2 + TotalSegmentator): dataset conversion, queued multi-GPU training
(Kueue + KubeRay), MLflow tracking + model registry, and a QA viewer that
compares predictions against ground truth. See `README.md` for the architecture.

## Layout

- `src/modelfactory/` — the SDK/CLI (torch-free core; trainer code is lazy-imported).
  - `infra/` — `cluster.yaml` → k8s manifests (MIG + whole-GPU), the `modelfactory infra` CLI.
  - `datasets/` — `specs.py` (framework + public examples), `sources/` adapters, `convert.py`.
  - `jobs/`, `trainers/`, `planners/`, `inference/`, `qa/`, `analysis/`.
- `infra/` — Helm values + Kustomize manifests; optional Brev/GCE `cluster-repair`.
- `services/qa-viewer/` — the QA viewer image (Next.js web + FastAPI).
- `overlays/` — drop your own private datasets/specs here (see `overlays/README.md`).

## Conventions & non-negotiables (general)

1. **Never hardcode site-specific values.** Hostnames, node names, GPU layout,
   storage paths, and image tags are parameters in `cluster.yaml` /
   `FactoryConfig`, never literals. CI fails on a "forbidden-strings" gate.
2. **Keep the SDK core torch-free.** Anything importing `torch`/`nnunetv2` must
   be lazy-imported inside the function that needs it (see `cli.py`), and lives
   behind the `[trainer]` optional dependency.
3. **Kueue routing is a LABEL** (`kueue.x-k8s.io/queue-name`), not an annotation;
   manifests use `apiVersion: kueue.x-k8s.io/v1beta2`.
4. **Don't `pip install torch`** on top of the NGC PyTorch base image — it ships
   an H100-tuned torch/cuDNN/NCCL. The trainer Dockerfile installs `nnunetv2`
   with `--no-deps`.
5. **Infra changes must keep `tests/infra/` green** — those tests prove a
   generated manifest applies cleanly (no pod churn) against a reference cluster.
6. **Respect model-weight licensing** — TotalSegmentator MR weights are
   CC-BY-NC-SA (non-commercial). See `NOTICE` and `docs/licensing.md`.
7. **Don't symlink the SDK into nnUNet's site-packages.** Trainers/planners are
   discovered via the image-installed package + variant shims.

## Where to start

- Stand up a cluster: [`docs/bootstrap.md`](./docs/bootstrap.md).
- Convert data: [`docs/conversion.md`](./docs/conversion.md).
- Train: [`docs/training.md`](./docs/training.md). QA: [`docs/qa.md`](./docs/qa.md).
- Contribute: [`CONTRIBUTING.md`](./CONTRIBUTING.md).
