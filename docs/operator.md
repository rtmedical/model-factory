# Operator guide

How to bring up the factory on a fresh (or partially-broken) cluster and how
to recover from common failures.

## Prerequisites

Single H100 node, Ubuntu 24.04, Docker + containerd, kubeadm-installed
Kubernetes ≥ v1.33. NVIDIA GPU Operator with DCGM exporter. `nfs-client`
StorageClass (we use the `nfs-subdir-external-provisioner`).

This box already has all of the above. If you're standing up a new one, see
[NVIDIA GPU Operator install](https://docs.nvidia.com/datacenter/cloud-native/gpu-operator/latest/index.html)
and [kubeadm install](https://kubernetes.io/docs/setup/production-environment/tools/kubeadm/).

## First-time bootstrap

```bash
cd /data/model-factory

# 1) Cluster repair (idempotent; safe to re-run)
make repair

# 2) Create the NFS-served directory layout the PVCs will project into
make nfs-root

# 3) Set up credential secrets (copy template, fill in real values, apply)
cp infra/kustomize/secrets.example.yaml infra/kustomize/secrets.yaml
$EDITOR infra/kustomize/secrets.yaml
kubectl apply -f infra/kustomize/secrets.yaml

# 4) Add helm repos we use
helm repo add bitnami           https://charts.bitnami.com/bitnami
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm repo add grafana            https://grafana.github.io/helm-charts
helm repo add community-charts   https://community-charts.github.io/helm-charts
helm repo update

# 5) Deploy infra
make deploy-infra

# 6) Apply factory CRs (Kueue queues + PVC + alerts)
kubectl apply -f infra/kustomize/factory-resource-flavor.yaml
kubectl apply -f infra/kustomize/factory-cluster-queue.yaml
kubectl apply -f infra/kustomize/factory-local-queue.yaml
kubectl apply -f infra/kustomize/factory-priority-classes.yaml
kubectl apply -f infra/kustomize/factory-pvc.yaml
kubectl apply -f infra/kustomize/dcgm-servicemonitor.yaml
kubectl apply -f infra/kustomize/factory-alerts.yaml

# 7) Build & push trainer images
make build-images
make push-images

# 8) Install the modelfactory CLI on the operator's machine (Linux user `nvidia`)
make install-sdk

# 9) Smoke
make smoke-gpu      # 1-GPU CUDA pod, must print nvidia-smi output
make smoke          # full nnUNet smoke on MSD-Hippocampus (~3h)
```

## Recovery

### Node NotReady after a reboot
The most likely cause is hostname drift from the Brev/GCE cloud-init. See
`infra/cluster-repair/kubelet-hostname-fix.md`. Re-run `make repair`.

### Static control-plane pod CrashLoopBackOff
Usually a follow-on from a kubelet restart. Look at logs:
```bash
sudo crictl ps -a | head
sudo crictl logs <container-id>
```
Most often etcd is unhappy because the controller-manager raced it. Wait 30 s
and check again — kubelet will keep restarting the failing pod and they sort
themselves out.

### vLLM medgemma is using a GPU that we need
The medgemma containers are not in k8s — they're direct `docker run` with CDI.
As of 2026-05-12 vLLM uses only GPUs 1 and 2 (vllm-4b on GPU 1; vllm-27b tp=2
on GPUs 1+2). The factory's Kueue ClusterQueue is capped at 5 GPUs, leaving
GPU 5 as a physical buffer so vLLM can expand without preempting training.
Until the device-plugin filtering is in place (see open task #12), a training
pod could still randomly land on GPU 1 or 2 and OOM on CUDA init.

### MLflow shows stale runs after Postgres restart
The MLflow tracking server caches runs in-memory. Restart it:
```bash
kubectl -n model-factory rollout restart deploy mlflow
```

### Out of disk
The 23 TiB `/data` volume holds everything: NFS PVCs, container layers, the
nnUNet data. If you're under 15 % free, the `NFSDiskNearFull` alert fires.
Clean up old preprocessed datasets first (cheap to regenerate) and old
checkpoints with no MLflow runs pointing at them.

## Health checks

```bash
kubectl get nodes
kubectl get pods -A | grep -v Running | grep -v Completed
kubectl -n model-factory get jobs
kubectl -n monitoring get servicemonitors

# DCGM scrape working?
kubectl -n nvidia-gpu-operator port-forward svc/nvidia-dcgm-exporter 9400:9400 &
curl -s localhost:9400/metrics | grep DCGM_FI_DEV_GPU_TEMP | head

# MLflow reachable from outside the cluster?
kubectl -n model-factory port-forward svc/mlflow 5000:5000 &
curl -s localhost:5000/health
```
