"""Configuration loader for the modelfactory CLI/SDK.

Reads from (in order of precedence):
  1. environment variables (MFACTORY_*)
  2. ~/.modelfactory.yaml
  3. defaults baked here

Defaults assume the layout produced by `make deploy-infra`:
  - namespace `model-factory`
  - MLflow at http://mlflow.model-factory.svc.cluster.local:5000
  - MinIO at http://factory-minio.model-factory.svc.cluster.local:9000
  - in-cluster registry at registry.model-factory.svc:5000
  - NFS data root at /data/model-factory-nfs (host) → /factory (in-pod)
"""

from __future__ import annotations

import os
from pathlib import Path

import yaml
from pydantic import BaseModel, Field


class FactoryConfig(BaseModel):
    namespace: str = "model-factory"
    queue_name: str = "factory-lq"

    # QA-viewer API base URL for host-side tools (e.g. `modelfactory schedule`).
    # The viewer's qa.sqlite is written by the pod (root-owned on NFS), so the
    # host CLI mutates the queue through the API rather than the DB file. This
    # is the node's NodePort for svc/qa-viewer (port 80 → nodePort 32443).
    qa_api_url: str = "http://localhost:32443"

    # Short names only — the cluster's resolv.conf has a public wildcard
    # search domain that hijacks `*.svc.cluster.local` FQDNs to Cloudflare.
    # Short names resolve correctly via the in-pod search list (CoreDNS first).
    mlflow_tracking_uri: str = "http://mlflow:5000"
    s3_endpoint_url: str = "http://factory-minio:9000"
    s3_bucket: str = "mlflow-artifacts"

    # In-cluster registry where trainer images live (deploy-registry target
    # is pending). Until then, images are built locally and kubelet finds
    # them via containerd's image store.
    registry: str = "registry.model-factory.svc:5000"
    trainer_image: str = Field(default="nnunet-trainer:0.3.0-ray")
    totalseg_image: str = Field(default="totalseg-trainer:0.3.0-ray")

    # Filesystem layout
    nfs_host_root: Path = Path("/data/model-factory-nfs")
    nfs_pod_root: Path = Path("/factory")

    # GPU acquisition mode for direct (non-Ray) Job submission:
    #   "mig"   → pin a leased MIG slice via NVIDIA_VISIBLE_DEVICES under the
    #             nvidia-legacy runtime, no nvidia.com/gpu request (default;
    #             reproduces the reference deployment behaviour).
    #   "whole" → request nvidia.com/gpu: 1 via the standard device plugin, no
    #             UUID leasing.
    gpu_mode: str = "mig"
    runtime_class: str = "nvidia-legacy"
    # Where the curated MIG-slice UUID pool lives (mig mode). Written by
    # `modelfactory infra discover`; read at submit time to lease a slice.
    safe_uuids_path: Path = Path("/data/model-factory-nfs/safe_uuids.env")
    # nodeSelector applied to training pods (matches the Kueue ResourceFlavor).
    node_selector: dict[str, str] = Field(default_factory=lambda: {"factory.io/training": "true"})
    # Public hostname of the QA viewer, for host-side tooling. "" → use qa_api_url.
    qa_public_host: str = ""

    # Secret names (deployed alongside services; contain MLflow + MinIO creds)
    mlflow_credentials_secret: str = "mlflow-credentials"
    s3_credentials_secret: str = "factory-minio"

    # Defaults for resource requests on a single-GPU nnUNet training Job.
    # CPU limit raised above 24 so the main trainer thread + n_proc_DA=18
    # augmenter workers don't all fight for the same cores; memory limit
    # raised so the /factory-ram tmpfs (sizeLimit 110Gi) has headroom on
    # top of the trainer's ~30-40 Gi working set.
    default_cpu_request: str = "18"
    default_cpu_limit: str = "28"
    default_memory_request: str = "48Gi"
    default_memory_limit: str = "160Gi"
    default_shm_size: str = "32Gi"
    default_ram_stage_size: str = "110Gi"

    @classmethod
    def load(cls) -> FactoryConfig:
        cfg_path = Path(os.environ.get("MFACTORY_CONFIG", "~/.modelfactory.yaml")).expanduser()
        data: dict = {}
        if cfg_path.is_file():
            with cfg_path.open() as f:
                data = yaml.safe_load(f) or {}

        # Env overrides — uppercase MFACTORY_<FIELD> with underscores. Skip
        # structured fields (dict/Path collections) which can't come from a
        # bare string env var.
        for field in cls.model_fields:
            env_key = f"MFACTORY_{field.upper()}"
            if env_key in os.environ and field != "node_selector":
                data[field] = os.environ[env_key]

        return cls(**data)

    @classmethod
    def from_cluster_spec(cls, spec: object) -> FactoryConfig:
        """Derive a runtime FactoryConfig from an infra ``ClusterSpec``.

        Lets ``modelfactory infra render`` emit a matching ``~/.modelfactory.yaml``
        so the host CLI and the deployed cluster agree on namespace, endpoints,
        image, GPU mode, and the safe-UUIDs path from a single source of truth.
        """
        rc = spec.gpu.mig.runtime_class_name if spec.gpu.mode == "mig" else spec.gpu.whole.runtime_class_name
        return cls(
            namespace=spec.cluster.namespace,
            queue_name=spec.kueue.local_queue,
            mlflow_tracking_uri=spec.secrets.mlflow_tracking_uri,
            s3_endpoint_url=spec.secrets.s3_endpoint_url,
            s3_bucket=spec.secrets.s3_bucket,
            registry=spec.registry.host or "registry.model-factory.svc:5000",
            trainer_image=spec.registry.trainer_tag,
            totalseg_image=spec.registry.totalseg_tag,
            nfs_host_root=Path(spec.storage.nfs_host_root),
            nfs_pod_root=Path(spec.storage.nfs_pod_root),
            gpu_mode=spec.gpu.mode,
            runtime_class=rc or "nvidia-legacy",
            safe_uuids_path=Path(spec.storage.nfs_host_root) / "safe_uuids.env",
            node_selector=dict(spec.cluster.node_selector),
            qa_public_host=spec.network.qa_public_host,
            mlflow_credentials_secret=spec.secrets.mlflow_credentials_secret,
            s3_credentials_secret=spec.secrets.s3_credentials_secret,
        )
