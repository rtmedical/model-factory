# Bootstrapping a model-factory cluster

This guide stands up model-factory on your own Kubernetes + GPU cluster. The
whole deployment is driven by one file — `cluster.yaml` — so there are no
site-specific values baked into the manifests.

## 1. Prerequisites

- A **Kubernetes cluster** (kubeadm, k3s, or managed) with `kubectl` admin access.
- **NVIDIA GPUs** exposed to Kubernetes:
  - *Whole-GPU mode*: the [NVIDIA device plugin](https://github.com/NVIDIA/k8s-device-plugin)
    (or GPU Operator) advertising `nvidia.com/gpu`.
  - *MIG mode*: additionally a `nvidia-legacy` RuntimeClass that honours
    `NVIDIA_VISIBLE_DEVICES` (the GPU Operator installs this), and MIG-capable
    cards (H100 / A100).
- An **RWX-capable StorageClass** (NFS via the
  [nfs-subdir-external-provisioner](https://github.com/kubernetes-sigs/nfs-subdir-external-provisioner)
  is the reference; any RWX class works). model-factory stores datasets,
  preprocessed data, checkpoints, and MLflow/MinIO/Postgres state here.
- `helm` ≥ 3, and Python ≥ 3.10 on the machine you run the CLI from.

Install the CLI:

```bash
make install-sdk        # pip install -e ".[dev]"
modelfactory --help
```

## 2. Write your `cluster.yaml`

```bash
cp cluster.example.yaml cluster.yaml
$EDITOR cluster.yaml
```

`cluster.example.yaml` documents every field. The ones you almost always touch:

| Field | What |
|---|---|
| `cluster.nodeSelector` | label your GPU nodes carry (and `cluster.nodes` to have bootstrap apply it) |
| `gpu.mode` | `whole` or `mig` (see below) |
| `gpu.product` | the `nvidia.com/gpu.product` label value (H100/A100/L40S/…) |
| `storage.storageClass` / `storage.mount` | your RWX class; `pvc` (portable) or `hostPath` (single-node) |
| `kueue.{cpuQuota,memoryQuota,gpuQuota}` | how much the factory may consume |
| `network.qaPublicHost` / `network.ingressEnabled` | how the QA viewer is exposed |
| `registry.*Tag` | the trainer / QA image tags |

Validate it:

```bash
modelfactory infra validate          # prints a summary table; fails fast on errors
```

### Whole-GPU mode (most clusters)

```yaml
gpu:
  mode: whole
  whole:
    count: 4          # full GPUs to expose to the training pool
```

Each Ray worker requests `nvidia.com/gpu: 1` through the device plugin. Nothing
else to do — skip the MIG steps below.

### MIG mode (partitioned H100/A100 fleets)

```yaml
gpu:
  mode: mig
  mig:
    provisioner: nvidia-smi      # or "operator" / "preexisting"
    profile: "3g.40gb"
    layout: { 0: {mig: false}, 1: {mig: 2}, 2: {mig: 2} }   # per-GPU
    poolGpus: [1, 2]             # order fixes the slice -> worker mapping
```

Create the slices (PRIVILEGED — terminates running GPU processes on those cards):

```bash
modelfactory infra mig-create        # runs nvidia-smi mig per the layout
modelfactory infra discover          # parse nvidia-smi -L -> slice UUIDs (cached)
```

> MIG layout is **not** reboot-persistent. After a node reboot, re-run
> `infra mig-create && infra discover && infra apply`. For frequently-rebooting
> nodes, wrap those in a systemd unit (not auto-installed).

`runtimeClassName: nvidia-legacy` + UUID pinning is used because the device
plugin can advertise both MIG slices and whole un-partitioned cards on the same
node; pinning by UUID guarantees a trainer never lands on a reserved card.

## 3. Render and apply

```bash
modelfactory infra render            # cluster.yaml -> .render/infra/*.yaml
modelfactory infra apply --dry-run   # kubectl diff against the live cluster
modelfactory infra apply             # apply queues, RayCluster, flavor, quota
```

> **Applying to a cluster that already runs jobs?** Always `--dry-run` first and
> read the `kubectl diff`. A change to a Ray worker's pod template (image, env,
> resources) restarts that worker and kills its in-flight trial. If a worker was
> hand-edited out of band, capture that in `rayWorker.overrides` (keyed by group
> name, e.g. `mig-5: {memoryLimit: "300Gi"}`) so the render stays diff-clean.

## 4. Deploy the services

Create credentials, then deploy MLflow + Postgres + MinIO, KubeRay, and monitoring:

```bash
cp infra/kustomize/secrets.example.yaml infra/kustomize/secrets.yaml
$EDITOR infra/kustomize/secrets.yaml          # set admin passwords
helm repo add bitnami https://charts.bitnami.com/bitnami
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm repo add grafana https://grafana.github.io/helm-charts
helm repo add community-charts https://community-charts.github.io/helm-charts
helm repo update

make deploy-mlflow        # Postgres + MinIO + MLflow
make deploy-kuberay       # KubeRay operator + the rendered RayCluster
make deploy-monitoring    # kube-prometheus-stack + Loki
```

Build the trainer + QA images (single-node containerd, or push to a registry —
set `registry.host` in `cluster.yaml`):

```bash
make build-images         # nnunet-trainer + totalseg-trainer
make build-qa-viewer      # the QA viewer image
```

## 5. Exposing the QA viewer

By default the viewer is a `NodePort` (`network.qaNodePort`, 32443) reachable on
the node IP / via `kubectl port-forward`. For a public hostname, set:

```yaml
network:
  qaPublicHost: seg-qa.example.com
  ingressEnabled: true
  ingressClassName: nginx
```

and put authentication in front of it — it has no built-in auth. (If you instead
terminate TLS with an external proxy, point it at the NodePort; keep the
hostname in `cluster.yaml`, never hardcoded.)

## 6. Verify

```bash
make smoke-gpu            # 1-GPU pod runs nvidia-smi -L
bash examples/smoke/run_msd_hippocampus.sh   # full convert -> train -> MLflow
make smoke-qa             # QA viewer /api/healthz
```

## Appendix: optional Brev/GCE site repair

Some cloud providers rename the OS hostname on reboot, which breaks the kubelet
client-certificate identity (`CN=system:node:<name>`) and the node goes NotReady.
You almost certainly **do not need this**. If you hit it, set:

```yaml
siteRepair:
  kubeletHostnameOverride: true
  nodeName: <the name the kubelet cert expects>
```

and run `make repair` (idempotent). See
[`infra/cluster-repair/kubelet-hostname-fix.md`](../infra/cluster-repair/kubelet-hostname-fix.md).
