# model-factory

**A scalable Kubernetes training factory for medical-image segmentation.**

model-factory turns a GPU + Kubernetes cluster into an end-to-end pipeline for
[nnU-Net v2](https://github.com/MIC-DKFZ/nnUNet) and
[TotalSegmentator](https://github.com/wasserth/TotalSegmentator)-style models:

**convert → preprocess → train → track → QA.**

It is the orchestration layer many radiotherapy / medical-imaging teams end up
rebuilding by hand — dataset registration, queued multi-GPU training, experiment
tracking, model registry, and a visual QA tool to compare predicted contours
against ground truth — packaged so you can stand it up on *your* cluster.

```
                 ┌──────────────────────────┐
   you  ──CLI──▶ │  modelfactory CLI / SDK  │  render Job manifests,
                 │  (k8s + MLflow clients)  │  apply via the k8s API
                 └────────────┬─────────────┘
                              ▼
   ┌──────────────── your Kubernetes cluster ─────────────────────┐
   │  Kueue ClusterQueue ─ priority-ordered GPU admission          │
   │  KubeRay RayCluster ─ one worker per MIG slice OR whole GPU   │
   │  nnU-Net trainers   ─ MLflow-logged, checkpointed to NFS/PVC  │
   │  MLflow + Postgres + MinIO ─ experiments, metrics, artifacts  │
   │  Prometheus + DCGM + Loki  ─ GPU + training observability     │
   │  QA viewer (FastAPI + Next.js) ─ Dice vs ground-truth, 3D     │
   └───────────────────────────────────────────────────────────────┘
```

## Why model-factory

- **MIG *and* whole-GPU.** One config switch. Pin one trainer per MIG slice
  (bypassing the device plugin) on partitioned H100/A100s, **or** request whole
  GPUs via the standard NVIDIA device plugin — whichever your cluster uses.
- **Setup is a config file, not a wiki page.** `cluster.yaml` declares your node
  labels, GPU layout, storage class, quotas, hostnames, and image tags. A
  generator renders every manifest from it — nothing site-specific is hardcoded.
- **Queued, prioritized training.** Kueue admits jobs by priority
  (`interactive-eval` > `fold-training` > `hpo-sweep`) so an eval can preempt a
  sweep; KubeRay fans 5-fold campaigns across the GPU pool.
- **Hard cases handled.** Drop-in trainers/planners for tiny/sparse structures
  (Tversky loss + foreground oversampling), sub-voxel structures (anisotropic
  high-res planner), and partial-label generalists (loss masked to annotated
  organs) — patterns that otherwise collapse to Dice 0.
- **QA you can see.** A web viewer renders each model's predictions against
  ground-truth masks with per-class Dice/HD95, cross-validation rollups, 3D
  meshes, and accept/reject verdicts.
- **Lineage + licensing aware.** Every model is tagged with its base weights and
  dataset license, so you don't accidentally ship a non-commercial fine-tune
  (see [`docs/licensing.md`](docs/licensing.md)).

## Quickstart

Prerequisites: a Kubernetes cluster with NVIDIA GPUs (GPU Operator or device
plugin installed), an RWX-capable StorageClass (e.g. NFS), `kubectl` + `helm`,
and Python ≥ 3.10. See [`docs/bootstrap.md`](docs/bootstrap.md) for the details
(including MIG, ingress, and a Brev/GCE site-repair note).

```bash
git clone https://github.com/your-org/model-factory && cd model-factory
make install-sdk                      # pip install -e ".[dev]"

cp cluster.example.yaml cluster.yaml  # edit: nodes, GPU mode, storage, hostnames
modelfactory infra validate           # check the spec
modelfactory infra render             # write manifests to .render/infra/
modelfactory infra apply --dry-run    # kubectl diff against the cluster
modelfactory infra apply              # apply (queues, RayCluster, flavor, quota)

# Deploy the services + build images (see docs/bootstrap.md)
cp infra/kustomize/secrets.example.yaml infra/kustomize/secrets.yaml  # fill creds
make deploy-mlflow deploy-kuberay deploy-monitoring
make build-images
```

Day-to-day:

```bash
modelfactory dataset register /data/Dataset100_Hippocampus --copy
modelfactory train nnunet --dataset Dataset100_Hippocampus --folds all
modelfactory runs list --dataset Dataset100_Hippocampus
modelfactory model register-ensemble --dataset Dataset100_Hippocampus --configuration 3d_fullres
make deploy-qa                         # then open the QA viewer to review Dice vs GT
```

### GPU mode in one line

`cluster.yaml` → `gpu.mode`:

| `whole` (default) | `mig` |
|---|---|
| One Ray worker per GPU; `nvidia.com/gpu: 1` via the device plugin. Simplest; what most clusters use. | One worker per MIG slice, pinned by UUID under `runtimeClassName: nvidia-legacy`. For partitioned H100/A100 fleets running many small models. `modelfactory infra mig-create` partitions the cards. |

## Documentation

| Doc | What |
|---|---|
| [`docs/bootstrap.md`](docs/bootstrap.md) | Stand up the cluster: prerequisites, `cluster.yaml` reference, MIG vs whole-GPU, ingress, post-reboot recovery |
| [`docs/conversion.md`](docs/conversion.md) | Convert your data into nnU-Net datasets: `DatasetSpec` + source adapters, adding your own |
| [`docs/training.md`](docs/training.md) | Submitting trainings & campaigns; the small-structures / high-res / partial-label trainers; MLflow |
| [`docs/qa.md`](docs/qa.md) | The QA viewer: reading Dice vs GT, cross-validation, verdicts |
| [`docs/hpo.md`](docs/hpo.md) | Hyperparameter optimization with Ray Tune |
| [`docs/licensing.md`](docs/licensing.md) | Model-weight & dataset licensing (read before shipping models) |
| [`CONTRIBUTING.md`](CONTRIBUTING.md) | Dev setup, ground rules, PR flow |

## Repository layout

```
cluster.example.yaml      the single source of truth — copy to cluster.yaml
src/modelfactory/
  infra/                  cluster.yaml -> k8s manifests (MIG + whole-GPU)
  cli.py  config.py       Click CLI + FactoryConfig (env > ~/.modelfactory.yaml)
  datasets/               conversion framework: specs.py + sources/ adapters
  jobs/                   nnU-Net Job submission + Jinja templates
  trainers/  planners/    MLflow trainer + small-structures / high-res / partial-label
  inference/  qa/          predictor cache, metrics, the QA backend (FastAPI)
  analysis/               failure mining, calibration
infra/
  kustomize/  helm/       reference manifests + Helm values per service
  cluster-repair/         OPTIONAL Brev/GCE kubelet hostname-override fix
services/qa-viewer/       the QA viewer image (Next.js web + FastAPI)
examples/smoke/           MSD-Hippocampus end-to-end smoke test
overlays/                 add your own private datasets/specs (git-ignored)
```

## Status & scope

model-factory is the orchestration + QA layer for *training new models*. It is
not an inference server for production deployment (that's a separate concern),
and it does not manage non-Kubernetes GPU workloads. It has been run in
production on an 8×H100 cluster training 180+ segmentation models across brain,
head & neck, thorax, abdomen, and pelvis.

## License

[Apache-2.0](LICENSE). See [`NOTICE`](NOTICE) for third-party components and an
important note on TotalSegmentator **MR** weights (CC-BY-NC-SA — non-commercial).
The nnU-Net weights *you* train are yours.
