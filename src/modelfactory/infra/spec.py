"""``ClusterSpec`` — the declarative description of one model-factory deployment.

Loaded from ``cluster.yaml`` (copy ``cluster.example.yaml``). Every field carries
a default that, taken together, describes a sensible generic deployment. A site
overrides only what differs.

The central switch is ``gpu.mode``:

* ``mig``   — GPUs are MIG-partitioned. The RayCluster gets one worker group per
              slice, each pinned to a slice UUID via ``NVIDIA_VISIBLE_DEVICES`` and
              run under ``runtimeClassName: nvidia-legacy`` (the device plugin is
              bypassed). This reproduces the reference H100 deployment.
* ``whole`` — GPUs are used whole via the standard NVIDIA device plugin. A single
              worker group requests ``nvidia.com/gpu: 1`` per replica. No UUID
              pinning, no MIG leasing. This is the path most community clusters use.

YAML keys are camelCase (k8s convention); Python fields are snake_case. Pydantic's
alias generator bridges the two, and ``populate_by_name`` lets tests construct
models with either spelling.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator
from pydantic.alias_generators import to_camel


class _Base(BaseModel):
    model_config = ConfigDict(
        alias_generator=to_camel,
        populate_by_name=True,
        extra="forbid",
    )


# ──────────────────────────────────────────────────────────────────────────
# cluster identity / scheduling
# ──────────────────────────────────────────────────────────────────────────


class ClusterMeta(_Base):
    namespace: str = "model-factory"
    # Label(s) every factory-eligible node carries. Rendered into the
    # ResourceFlavor nodeLabels and every pod's nodeSelector.
    node_selector: dict[str, str] = Field(default_factory=lambda: {"factory.io/training": "true"})
    # Node names to label during bootstrap (and, in MIG mode, to partition).
    # Empty = "select by label only; don't touch any node by name" (e.g. a
    # cloud autoscaler manages nodes).
    nodes: list[str] = Field(default_factory=list)


# ──────────────────────────────────────────────────────────────────────────
# GPU mode
# ──────────────────────────────────────────────────────────────────────────


class MigLayoutEntry(_Base):
    # `false`  → card stays whole, never advertised to the factory pool.
    # `<int>`  → number of MIG slices of `mig.profile` to create on this card.
    mig: bool | int = False

    @property
    def slices(self) -> int:
        if isinstance(self.mig, bool):
            return 0
        return int(self.mig)


class MigConfig(_Base):
    runtime_class_name: str = "nvidia-legacy"
    # How slices get created:
    #   nvidia-smi  → bootstrap runs `nvidia-smi mig` on the node(s)
    #   operator    → render the GPU-Operator mig-manager ConfigMap; user labels the node
    #   preexisting → slices already exist; just discover them
    provisioner: Literal["nvidia-smi", "operator", "preexisting"] = "preexisting"
    profile: str = "3g.40gb"
    # Per-physical-GPU layout, keyed by GPU index on the node.
    layout: dict[int, MigLayoutEntry] = Field(default_factory=dict)
    # Which cards feed the training pool, IN ORDER. Order is load-bearing: it
    # fixes the slice-UUID → worker-group-name (mig-0, mig-1, …) mapping, so a
    # re-render produces a diff-clean apply only if this matches how the live
    # cluster was built. (Reference H100 site: [3,4,5,6,7,1,2].)
    pool_gpus: list[int] = Field(default_factory=list)
    # Worker groups to render with replicas=0 (slice exists but is parked, e.g.
    # a slice that trips an NVML allocator assert and needs a GI/CI rebuild).
    disabled_worker_groups: list[str] = Field(default_factory=list)


class WholeConfig(_Base):
    # Number of full GPUs to expose to the Ray pool / Kueue quota.
    count: int = 1
    # "" → cluster default runtime (the device plugin). Advanced users can pin
    # nvidia-legacy here too, but it's not required for whole-GPU.
    runtime_class_name: str = ""


class GpuConfig(_Base):
    mode: Literal["mig", "whole"] = "whole"
    # Value of the nvidia.com/gpu.product node label (set by GPU feature
    # discovery). Used by the Kueue ResourceFlavor nodeLabels.
    product: str = "NVIDIA-H100-80GB-HBM3"
    mig: MigConfig = Field(default_factory=MigConfig)
    whole: WholeConfig = Field(default_factory=WholeConfig)

    @property
    def expected_slice_count(self) -> int:
        """Total MIG slices the layout should yield across the pool GPUs."""
        if self.mode != "mig":
            return 0
        return sum(self.mig.layout.get(g, MigLayoutEntry()).slices for g in self.mig.pool_gpus)


# ──────────────────────────────────────────────────────────────────────────
# Ray worker sizing
# ──────────────────────────────────────────────────────────────────────────


_OVERRIDE_FIELDS = (
    "cpu_request",
    "cpu_limit",
    "memory_request",
    "memory_limit",
    "n_proc_da",
    "shm_size",
    "ram_stage_size",
)


class RayResourceTier(_Base):
    # Match by GPU index; falls back to `default` for unmatched GPUs.
    gpus: list[int] = Field(default_factory=list)
    cpu_request: str | None = None
    cpu_limit: str | None = None
    memory_request: str | None = None
    memory_limit: str | None = None
    n_proc_da: str | None = None


class RayWorkerOverride(_Base):
    """Per-worker-group resource override (keyed by group name, e.g. ``mig-5``).

    Lets the operator capture ad-hoc, out-of-band live edits (e.g. a group whose
    memory limit was hand-bumped for a heavy campaign) so a re-render stays
    diff-clean instead of reconciling that group and churning its pod.
    """

    cpu_request: str | None = None
    cpu_limit: str | None = None
    memory_request: str | None = None
    memory_limit: str | None = None
    n_proc_da: str | None = None
    shm_size: str | None = None
    ram_stage_size: str | None = None


class RayWorkerDefaults(_Base):
    cpu_request: str = "18"
    cpu_limit: str = "28"
    memory_request: str = "48Gi"
    memory_limit: str = "160Gi"
    n_proc_da: str = "18"
    shm_size: str = "32Gi"
    ram_stage_size: str = "110Gi"
    # rayStartParams num-cpus advertised to the Ray scheduler.
    num_cpus_ray: str = "20"


class RayHead(_Base):
    cpu_request: str = "2"
    cpu_limit: str = "4"
    memory_request: str = "8Gi"
    memory_limit: str = "16Gi"
    num_cpus_ray: str = "2"


class RayWorkerConfig(_Base):
    image: str = "nnunet-trainer:0.3.0-ray"
    ray_version: str = "2.40.0"
    cluster_name: str = "factory-ray"
    default: RayWorkerDefaults = Field(default_factory=RayWorkerDefaults)
    tiers: list[RayResourceTier] = Field(default_factory=list)
    # Per-group overrides keyed by worker-group name (e.g. {"mig-5": {...}}).
    overrides: dict[str, RayWorkerOverride] = Field(default_factory=dict)
    head: RayHead = Field(default_factory=RayHead)

    def tier_for_gpu(self, gpu_index: int) -> RayWorkerDefaults:
        """Resolve the effective resource block for a slice on ``gpu_index``."""
        eff = self.default.model_copy(deep=True)
        for tier in self.tiers:
            if gpu_index in tier.gpus:
                for field in ("cpu_request", "cpu_limit", "memory_request", "memory_limit", "n_proc_da"):
                    val = getattr(tier, field)
                    if val is not None:
                        setattr(eff, field, val)
                break
        return eff

    def effective(self, gpu_index: int, group_name: str) -> RayWorkerDefaults:
        """Resource block for ``group_name``: default <- GPU tier <- per-group override."""
        eff = self.tier_for_gpu(gpu_index)
        ov = self.overrides.get(group_name)
        if ov is not None:
            for field in _OVERRIDE_FIELDS:
                val = getattr(ov, field)
                if val is not None:
                    setattr(eff, field, val)
        return eff


# ──────────────────────────────────────────────────────────────────────────
# Kueue / storage / network / registry / secrets / site-repair
# ──────────────────────────────────────────────────────────────────────────


class KueueConfig(_Base):
    # None → auto: MIG slice count (mig) or whole-GPU count (whole).
    gpu_quota: int | None = None
    cpu_quota: str = "196"
    memory_quota: str = "1700Gi"
    cluster_queue: str = "factory-cq"
    local_queue: str = "factory-lq"
    flavor: str = "flavor-h100"


class StorageConfig(_Base):
    storage_class: str = "nfs-client"
    # hostPath reproduces the single-node reference; pvc works on multi-node /
    # managed k8s where pods have no shared host filesystem.
    mount: Literal["hostPath", "pvc"] = "hostPath"
    nfs_host_root: str = "/data/model-factory-nfs"
    nfs_pod_root: str = "/factory"
    pvc_name: str = "factory-data-pvc"
    pvc_size: str = "10Ti"
    # Mount a tmpfs at /factory-ram so the trainer can stage preprocessed data
    # off NFS (node-local, fast epochs).
    ram_stage: bool = True


class NetworkConfig(_Base):
    # Public hostname for the QA viewer (drives ingress + the host CLI default).
    # "" → access via NodePort / port-forward only.
    qa_public_host: str = ""
    mlflow_public_host: str = ""
    qa_node_port: int = 32443
    ingress_enabled: bool = False
    ingress_class_name: str = ""
    # Emit dnsConfig.options ndots:1 on Ray pods. Site-specific workaround for
    # clusters whose resolv.conf search domain hijacks *.svc.cluster.local
    # (e.g. a public wildcard). Default off.
    dns_ndots_workaround: bool = False
    short_service_names: bool = True


class RegistryConfig(_Base):
    # "" → images are in the node's local containerd store (IfNotPresent), no
    # registry. Otherwise the registry host prefix for pulls.
    host: str = ""
    trainer_tag: str = "nnunet-trainer:0.3.0-ray"
    totalseg_tag: str = "totalseg-trainer:0.3.0-ray"
    qa_viewer_tag: str = "model-qa:0.9.2"


class SecretsConfig(_Base):
    mlflow_credentials_secret: str = "mlflow-credentials"
    s3_credentials_secret: str = "factory-minio"
    s3_bucket: str = "mlflow-artifacts"
    mlflow_tracking_uri: str = "http://mlflow:5000"
    s3_endpoint_url: str = "http://factory-minio:9000"


class SiteRepair(_Base):
    # Brev/GCE-only kubelet hostname-override fix. Almost no one needs this.
    kubelet_hostname_override: bool = False
    node_name: str = ""


# ──────────────────────────────────────────────────────────────────────────
# top-level
# ──────────────────────────────────────────────────────────────────────────


class ClusterSpec(_Base):
    api_version: str = Field(default="modelfactory.io/v1", alias="apiVersion")
    kind: str = "ClusterSpec"
    cluster: ClusterMeta = Field(default_factory=ClusterMeta)
    gpu: GpuConfig = Field(default_factory=GpuConfig)
    ray_worker: RayWorkerConfig = Field(default_factory=RayWorkerConfig)
    kueue: KueueConfig = Field(default_factory=KueueConfig)
    storage: StorageConfig = Field(default_factory=StorageConfig)
    network: NetworkConfig = Field(default_factory=NetworkConfig)
    registry: RegistryConfig = Field(default_factory=RegistryConfig)
    secrets: SecretsConfig = Field(default_factory=SecretsConfig)
    site_repair: SiteRepair = Field(default_factory=SiteRepair)

    # ── validation ──────────────────────────────────────────────────────
    @model_validator(mode="after")
    def _check(self) -> ClusterSpec:
        if self.gpu.mode == "mig":
            mig = self.gpu.mig
            unknown = [g for g in mig.pool_gpus if g not in mig.layout]
            if unknown:
                raise ValueError(f"gpu.mig.poolGpus references GPUs not in layout: {unknown}")
            not_partitioned = [g for g in mig.pool_gpus if mig.layout[g].slices == 0]
            if not_partitioned:
                raise ValueError(
                    f"gpu.mig.poolGpus includes cards with no MIG slices: {not_partitioned}"
                )
            if self.kueue.gpu_quota is not None and self.kueue.gpu_quota > self.gpu.expected_slice_count:
                raise ValueError(
                    f"kueue.gpuQuota ({self.kueue.gpu_quota}) exceeds the "
                    f"{self.gpu.expected_slice_count} MIG slices the layout yields"
                )
        if self.site_repair.kubelet_hostname_override and not self.site_repair.node_name:
            raise ValueError("siteRepair.nodeName is required when kubeletHostnameOverride is true")
        return self

    # ── derived ─────────────────────────────────────────────────────────
    @property
    def gpu_quota(self) -> int:
        """Effective Kueue nvidia.com/gpu quota."""
        if self.kueue.gpu_quota is not None:
            return self.kueue.gpu_quota
        return self.gpu.expected_slice_count if self.gpu.mode == "mig" else self.gpu.whole.count

    # ── loading ─────────────────────────────────────────────────────────
    @classmethod
    def load(cls, path: str | Path) -> ClusterSpec:
        data = yaml.safe_load(Path(path).expanduser().read_text()) or {}
        return cls.model_validate(data)
