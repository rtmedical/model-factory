"""Render Kubernetes manifests from a :class:`ClusterSpec`.

Pure ``dict`` builders + ``yaml.safe_dump`` — no string templating, so the output
is deterministic and unit-testable by comparing parsed dicts. Two GPU modes:

* **mig**   — one RayCluster worker group per discovered MIG slice, pinned by
              ``NVIDIA_VISIBLE_DEVICES`` under ``runtimeClassName: nvidia-legacy``.
* **whole** — a single worker group requesting ``nvidia.com/gpu: 1`` per replica
              via the standard device plugin.

IMPORTANT: container ``env``/``volumeMounts``/``volumes`` are ordered lists that
Kubernetes hashes into the pod template. To keep a re-apply diff-clean (no worker
churn), we emit them in exactly the order the live reference cluster uses.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from .discover import Slice
from .spec import ClusterSpec

_MANAGED_BY = {"app.kubernetes.io/managed-by": "modelfactory"}


# ──────────────────────────────────────────────────────────────────────────
# small builders
# ──────────────────────────────────────────────────────────────────────────


def _secret_env(name: str, secret: str, key: str) -> dict[str, Any]:
    return {"name": name, "valueFrom": {"secretKeyRef": {"name": secret, "key": key}}}


def _mlflow_env(spec: ClusterSpec) -> list[dict[str, Any]]:
    s = spec.secrets
    return [
        {"name": "MLFLOW_TRACKING_URI", "value": s.mlflow_tracking_uri},
        {"name": "MLFLOW_S3_ENDPOINT_URL", "value": s.s3_endpoint_url},
        _secret_env("AWS_ACCESS_KEY_ID", s.s3_credentials_secret, "root-user"),
        _secret_env("AWS_SECRET_ACCESS_KEY", s.s3_credentials_secret, "root-password"),
        _secret_env("MLFLOW_TRACKING_USERNAME", s.mlflow_credentials_secret, "admin-username"),
        _secret_env("MLFLOW_TRACKING_PASSWORD", s.mlflow_credentials_secret, "admin-password"),
    ]


def _nnunet_env(spec: ClusterSpec, n_proc_da: str) -> list[dict[str, Any]]:
    root = str(spec.storage.nfs_pod_root).rstrip("/")
    preproc = "/factory-ram/preprocessed" if spec.storage.ram_stage else f"{root}/preprocessed"
    return [
        {"name": "nnUNet_raw", "value": f"{root}/datasets"},
        {"name": "nnUNet_preprocessed", "value": preproc},
        {"name": "nnUNet_results", "value": f"{root}/results"},
        {"name": "nnUNet_n_proc_DA", "value": n_proc_da},
    ]


def _factory_volume(spec: ClusterSpec) -> dict[str, Any]:
    if spec.storage.mount == "pvc":
        return {"name": "factory-data", "persistentVolumeClaim": {"claimName": spec.storage.pvc_name}}
    return {
        "name": "factory-data",
        "hostPath": {"path": str(spec.storage.nfs_host_root), "type": "Directory"},
    }


def _worker_volumes(spec: ClusterSpec, eff: Any) -> list[dict[str, Any]]:
    vols = [_factory_volume(spec), {"name": "dshm", "emptyDir": {"medium": "Memory", "sizeLimit": eff.shm_size}}]
    if spec.storage.ram_stage:
        vols.append({"name": "factory-ram", "emptyDir": {"medium": "Memory", "sizeLimit": eff.ram_stage_size}})
    return vols


def _worker_volume_mounts(spec: ClusterSpec) -> list[dict[str, Any]]:
    mounts = [
        {"name": "factory-data", "mountPath": str(spec.storage.nfs_pod_root)},
        {"name": "dshm", "mountPath": "/dev/shm"},
    ]
    if spec.storage.ram_stage:
        mounts.append({"name": "factory-ram", "mountPath": "/factory-ram"})
    return mounts


def _pod_extras(spec: ClusterSpec) -> dict[str, Any]:
    """nodeSelector + optional dnsConfig shared by head and worker pod specs."""
    extras: dict[str, Any] = {"nodeSelector": dict(spec.cluster.node_selector)}
    if spec.network.dns_ndots_workaround:
        extras["dnsConfig"] = {"options": [{"name": "ndots", "value": "1"}]}
    return extras


# ──────────────────────────────────────────────────────────────────────────
# Kueue
# ──────────────────────────────────────────────────────────────────────────


def render_resource_flavor(spec: ClusterSpec) -> dict[str, Any]:
    labels = {"nvidia.com/gpu.product": spec.gpu.product, **spec.cluster.node_selector}
    return {
        "apiVersion": "kueue.x-k8s.io/v1beta2",
        "kind": "ResourceFlavor",
        "metadata": {"name": spec.kueue.flavor},
        "spec": {"nodeLabels": labels},
    }


def render_cluster_queue(spec: ClusterSpec) -> dict[str, Any]:
    return {
        "apiVersion": "kueue.x-k8s.io/v1beta2",
        "kind": "ClusterQueue",
        "metadata": {"name": spec.kueue.cluster_queue},
        "spec": {
            "namespaceSelector": {},
            "queueingStrategy": "BestEffortFIFO",
            "preemption": {"withinClusterQueue": "LowerPriority", "reclaimWithinCohort": "LowerPriority"},
            "resourceGroups": [
                {
                    "coveredResources": ["cpu", "memory", "nvidia.com/gpu"],
                    "flavors": [
                        {
                            "name": spec.kueue.flavor,
                            "resources": [
                                {"name": "cpu", "nominalQuota": spec.kueue.cpu_quota},
                                {"name": "memory", "nominalQuota": spec.kueue.memory_quota},
                                {"name": "nvidia.com/gpu", "nominalQuota": str(spec.gpu_quota)},
                            ],
                        }
                    ],
                }
            ],
        },
    }


def render_local_queue(spec: ClusterSpec) -> dict[str, Any]:
    return {
        "apiVersion": "kueue.x-k8s.io/v1beta2",
        "kind": "LocalQueue",
        "metadata": {"name": spec.kueue.local_queue, "namespace": spec.cluster.namespace},
        "spec": {"clusterQueue": spec.kueue.cluster_queue},
    }


def render_priority_classes(spec: ClusterSpec) -> list[dict[str, Any]]:
    def wpc(name: str, value: int, desc: str) -> dict[str, Any]:
        return {
            "apiVersion": "kueue.x-k8s.io/v1beta2",
            "kind": "WorkloadPriorityClass",
            "metadata": {"name": name},
            "value": value,
            "description": desc,
        }

    return [
        wpc("interactive-eval", 100, "Manual evaluation / on-demand inference. May preempt training."),
        wpc("fold-training", 50, "Default for nnUNet fold training (the bulk of factory workload)."),
        wpc("hpo-sweep", 10, "Hyperparameter sweeps / ad-hoc experiments. Preempted by anything higher."),
    ]


def render_pvc(spec: ClusterSpec) -> dict[str, Any]:
    return {
        "apiVersion": "v1",
        "kind": "PersistentVolumeClaim",
        "metadata": {"name": spec.storage.pvc_name, "namespace": spec.cluster.namespace},
        "spec": {
            "accessModes": ["ReadWriteMany"],
            "storageClassName": spec.storage.storage_class,
            "resources": {"requests": {"storage": spec.storage.pvc_size}},
        },
    }


# ──────────────────────────────────────────────────────────────────────────
# MIG ConfigMaps
# ──────────────────────────────────────────────────────────────────────────


def render_mig_uuids_configmap(spec: ClusterSpec, slices: list[Slice]) -> dict[str, Any]:
    return {
        "apiVersion": "v1",
        "kind": "ConfigMap",
        "metadata": {"name": "factory-mig-uuids", "namespace": spec.cluster.namespace, "labels": dict(_MANAGED_BY)},
        "data": {"uuids.txt": "".join(f"{s.uuid}\n" for s in slices)},
    }


def render_mig_leases_configmap(spec: ClusterSpec, slices: list[Slice]) -> dict[str, Any]:
    return {
        "apiVersion": "v1",
        "kind": "ConfigMap",
        "metadata": {"name": "factory-mig-leases", "namespace": spec.cluster.namespace, "labels": dict(_MANAGED_BY)},
        "data": {s.uuid: "available" for s in slices},
    }


def render_safe_uuids_env(slices: list[Slice]) -> str:
    header = "# Generated by `modelfactory infra render` — DO NOT EDIT BY HAND.\n# Safe-pool MIG slice UUIDs, in worker-group order.\n"
    return header + "".join(f"{s.uuid}\n" for s in slices)


# ──────────────────────────────────────────────────────────────────────────
# RayCluster
# ──────────────────────────────────────────────────────────────────────────


def _pod_labels(spec: ClusterSpec) -> dict[str, str]:
    return {**_MANAGED_BY, "ray.io/cluster": spec.ray_worker.cluster_name}


def _head_group(spec: ClusterSpec) -> dict[str, Any]:
    h = spec.ray_worker.head
    container = {
        "name": "ray-head",
        "image": spec.ray_worker.image,
        "imagePullPolicy": "IfNotPresent",
        "env": _mlflow_env(spec),
        "ports": [
            {"name": "gcs", "containerPort": 6379},
            {"name": "dashboard", "containerPort": 8265},
            {"name": "client", "containerPort": 10001},
        ],
        "resources": {
            "requests": {"cpu": h.cpu_request, "memory": h.memory_request},
            "limits": {"cpu": h.cpu_limit, "memory": h.memory_limit},
        },
        "volumeMounts": [{"name": "factory-data", "mountPath": str(spec.storage.nfs_pod_root)}],
    }
    pod_spec = {"containers": [container], "volumes": [_factory_volume(spec)], **_pod_extras(spec)}
    return {
        "rayStartParams": {"dashboard-host": "0.0.0.0", "num-cpus": h.num_cpus_ray, "num-gpus": "0"},
        "template": {"metadata": {"labels": _pod_labels(spec)}, "spec": pod_spec},
    }


def _mig_worker_group(spec: ClusterSpec, index: int, slc: Slice) -> dict[str, Any]:
    name = f"mig-{index}"
    eff = spec.ray_worker.effective(slc.gpu, name)
    disabled = name in spec.gpu.mig.disabled_worker_groups
    env = [
        {"name": "NVIDIA_VISIBLE_DEVICES", "value": slc.uuid},
        {"name": "NVIDIA_DRIVER_CAPABILITIES", "value": "compute,utility"},
        *_mlflow_env(spec),
        *_nnunet_env(spec, eff.n_proc_da),
    ]
    container = {
        "name": "ray-worker",
        "image": spec.ray_worker.image,
        "imagePullPolicy": "IfNotPresent",
        "env": env,
        "resources": {
            "requests": {"cpu": eff.cpu_request, "memory": eff.memory_request},
            "limits": {"cpu": eff.cpu_limit, "memory": eff.memory_limit},
        },
        "volumeMounts": _worker_volume_mounts(spec),
    }
    pod_spec = {
        "containers": [container],
        "runtimeClassName": spec.gpu.mig.runtime_class_name,
        "volumes": _worker_volumes(spec, eff),
        **_pod_extras(spec),
    }
    return {
        "groupName": name,
        "replicas": 0 if disabled else 1,
        "minReplicas": 0 if disabled else 1,
        "maxReplicas": 1,
        "numOfHosts": 1,
        "rayStartParams": {"num-cpus": spec.ray_worker.default.num_cpus_ray, "num-gpus": "1"},
        "template": {"metadata": {"labels": _pod_labels(spec)}, "spec": pod_spec},
    }


def _whole_worker_group(spec: ClusterSpec) -> dict[str, Any]:
    tier = spec.ray_worker.default
    count = spec.gpu.whole.count
    env = [*_mlflow_env(spec), *_nnunet_env(spec, tier.n_proc_da)]
    container = {
        "name": "ray-worker",
        "image": spec.ray_worker.image,
        "imagePullPolicy": "IfNotPresent",
        "env": env,
        "resources": {
            "requests": {"cpu": tier.cpu_request, "memory": tier.memory_request},
            "limits": {"cpu": tier.cpu_limit, "memory": tier.memory_limit, "nvidia.com/gpu": "1"},
        },
        "volumeMounts": _worker_volume_mounts(spec),
    }
    pod_spec: dict[str, Any] = {"containers": [container], "volumes": _worker_volumes(spec, tier), **_pod_extras(spec)}
    if spec.gpu.whole.runtime_class_name:
        pod_spec["runtimeClassName"] = spec.gpu.whole.runtime_class_name
    return {
        "groupName": "gpu-workers",
        "replicas": count,
        "minReplicas": count,
        "maxReplicas": count,
        "numOfHosts": 1,
        "rayStartParams": {"num-cpus": tier.num_cpus_ray, "num-gpus": "1"},
        "template": {"metadata": {"labels": _pod_labels(spec)}, "spec": pod_spec},
    }


def render_raycluster(spec: ClusterSpec, slices: list[Slice]) -> dict[str, Any]:
    if spec.gpu.mode == "mig":
        worker_groups = [_mig_worker_group(spec, i, s) for i, s in enumerate(slices)]
    else:
        worker_groups = [_whole_worker_group(spec)]
    return {
        "apiVersion": "ray.io/v1",
        "kind": "RayCluster",
        "metadata": {
            "name": spec.ray_worker.cluster_name,
            "namespace": spec.cluster.namespace,
            "labels": dict(_MANAGED_BY),
        },
        "spec": {
            "enableInTreeAutoscaling": False,
            "rayVersion": spec.ray_worker.ray_version,
            "headGroupSpec": _head_group(spec),
            "workerGroupSpecs": worker_groups,
        },
    }


# ──────────────────────────────────────────────────────────────────────────
# orchestration
# ──────────────────────────────────────────────────────────────────────────


def render_all(spec: ClusterSpec, slices: list[Slice] | None = None) -> dict[str, Any]:
    """Return ``{relative_filename: manifest|[manifests]|text}`` for the active mode."""
    slices = slices or []
    out: dict[str, Any] = {
        "factory-resource-flavor.yaml": render_resource_flavor(spec),
        "factory-cluster-queue.yaml": render_cluster_queue(spec),
        "factory-local-queue.yaml": render_local_queue(spec),
        "factory-priority-classes.yaml": render_priority_classes(spec),
        "factory-ray-cluster.yaml": render_raycluster(spec, slices),
    }
    if spec.storage.mount == "pvc":
        out["factory-pvc.yaml"] = render_pvc(spec)
    if spec.gpu.mode == "mig":
        out["factory-mig-uuids-configmap.yaml"] = render_mig_uuids_configmap(spec, slices)
        out["factory-mig-leases-configmap.yaml"] = render_mig_leases_configmap(spec, slices)
        out["safe_uuids.env"] = render_safe_uuids_env(slices)
    return out


def dump_yaml(manifest: Any) -> str:
    """Serialize a manifest (or list of manifests) to a YAML document/stream."""
    if isinstance(manifest, list):
        return yaml.safe_dump_all(manifest, sort_keys=False, default_flow_style=False)
    return yaml.safe_dump(manifest, sort_keys=False, default_flow_style=False)


def write_all(rendered: dict[str, Any], out_dir: str | Path) -> list[Path]:
    """Write the rendered artifacts to ``out_dir``. Returns the paths written."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for name, manifest in rendered.items():
        path = out_dir / name
        path.write_text(manifest if isinstance(manifest, str) else dump_yaml(manifest))
        written.append(path)
    return written
