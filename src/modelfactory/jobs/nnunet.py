"""Submit nnUNetv2 training Jobs to the cluster."""

from __future__ import annotations

import datetime
import hashlib
import logging
import random
import time
from dataclasses import dataclass, field
from pathlib import Path

import jinja2
import yaml
from kubernetes import client as k8s_client
from kubernetes import config as k8s_config

from modelfactory.config import FactoryConfig

logger = logging.getLogger(__name__)

# Where the curated MIG-slice UUIDs live (mirrors factory-mig-uuids ConfigMap).
# Pods on GPUs 0/1/2 are never in this list — see CLAUDE.md non-negotiable #2.
_SAFE_UUIDS_PATH = Path("/data/model-factory-nfs/safe_uuids.env")
_LEASES_CM = "factory-mig-leases"

# Wrapper run by the container before nnUNetv2_train. Rsyncs the dataset's
# preprocessed dir from NFS (/factory/preprocessed) into the pod's tmpfs
# (/factory-ram/preprocessed) so every epoch reads from RAM instead of NFS.
# Idempotent via a .stage_complete sentinel so checkpoint-resume doesn't
# re-pay the copy. Reads MFACTORY_DATASET / MFACTORY_NFS_PREPROC /
# MFACTORY_RAM_PREPROC from the pod env (set in _build_env).
_STAGE_WRAPPER_SCRIPT = """\
set -euo pipefail
DATASET="${MFACTORY_DATASET:?MFACTORY_DATASET unset}"
SRC="${MFACTORY_NFS_PREPROC}/${DATASET}"
DST="${MFACTORY_RAM_PREPROC}/${DATASET}"
if [ -d "$SRC" ] && [ ! -f "$DST/.stage_complete" ]; then
  mkdir -p "$DST"
  echo "[stage] rsyncing $SRC -> $DST"
  t0=$SECONDS
  rsync -a --delete "$SRC/" "$DST/"
  touch "$DST/.stage_complete"
  echo "[stage] done in $((SECONDS - t0))s"
fi
exec nnUNetv2_train "$@"
"""

_TEMPLATE_DIR = Path(__file__).parent / "templates"
_JINJA = jinja2.Environment(
    loader=jinja2.FileSystemLoader(_TEMPLATE_DIR),
    autoescape=False,
    trim_blocks=True,
    lstrip_blocks=True,
)


@dataclass
class TrainSpec:
    dataset: str                     # e.g. "Dataset042_KiTS23"
    configuration: str               # "3d_fullres" | "3d_lowres" | "2d" | "3d_cascade_fullres"
    fold: int                        # 0..4 (or "all")
    trainer_class: str = "nnUNetTrainerMLflow"
    num_epochs: int | None = None
    pretrained_weights: str | None = None
    continue_training: bool = False
    priority: str = "fold-training"
    parent_run_id: str | None = None
    experiment: str | None = None
    image: str | None = None         # override; defaults to config.trainer_image
    extra_env: dict[str, str] = field(default_factory=dict)


def _build_args(spec: TrainSpec) -> list[str]:
    args: list[str] = [
        spec.dataset,
        spec.configuration,
        str(spec.fold),
        "-tr",
        spec.trainer_class,
    ]
    if spec.num_epochs is not None:
        args += ["--num_epochs", str(spec.num_epochs)]
    if spec.pretrained_weights:
        args += ["-pretrained_weights", spec.pretrained_weights]
    if spec.continue_training:
        args += ["--c"]
    return args


def _build_env(
    spec: TrainSpec,
    cfg: FactoryConfig,
    experiment: str,
    leased_uuid: str | None = None,
) -> list[dict]:
    env: list[dict] = [
        {"name": "MLFLOW_TRACKING_URI", "value": cfg.mlflow_tracking_uri},
        {"name": "MLFLOW_S3_ENDPOINT_URL", "value": cfg.s3_endpoint_url},
        {"name": "AWS_ACCESS_KEY_ID",
         "valueFrom": {"secretKeyRef": {"name": cfg.s3_credentials_secret, "key": "root-user"}}},
        {"name": "AWS_SECRET_ACCESS_KEY",
         "valueFrom": {"secretKeyRef": {"name": cfg.s3_credentials_secret, "key": "root-password"}}},
        {"name": "MFACTORY_EXPERIMENT", "value": experiment},
        {"name": "nnUNet_raw", "value": "/factory/datasets"},
        {"name": "nnUNet_preprocessed", "value": "/factory-ram/preprocessed"},
        {"name": "nnUNet_results", "value": "/factory/results"},
        {"name": "nnUNet_n_proc_DA", "value": "18"},
        {"name": "MFACTORY_NFS_PREPROC", "value": "/factory/preprocessed"},
        {"name": "MFACTORY_RAM_PREPROC", "value": "/factory-ram/preprocessed"},
        {"name": "MFACTORY_DATASET", "value": spec.dataset},
    ]
    if leased_uuid:
        # The nvidia-legacy runtime reads NVIDIA_VISIBLE_DEVICES from pod-spec
        # env at container creation, so the leased MIG UUID must be baked in
        # here (init-container lease + in-container `source` doesn't work).
        env.append({"name": "NVIDIA_VISIBLE_DEVICES", "value": leased_uuid})
        env.append({"name": "NVIDIA_DRIVER_CAPABILITIES", "value": "compute,utility"})
        env.append({"name": "MFACTORY_MIG_UUID", "value": leased_uuid})
    if spec.parent_run_id:
        env.append({"name": "MFACTORY_PARENT_RUN_ID", "value": spec.parent_run_id})
    for k, v in spec.extra_env.items():
        env.append({"name": k, "value": v})
    return env


def _make_job_name(spec: TrainSpec) -> str:
    timestamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d-%H%M%S")
    return (
        f"train-{spec.dataset.lower().replace('_', '-')}-"
        f"{spec.configuration.replace('_', '-')}-f{spec.fold}-{timestamp}"
    )[:63]


def _read_safe_uuids(path: Path = _SAFE_UUIDS_PATH) -> list[str]:
    if not path.is_file():
        raise FileNotFoundError(
            f"safe-UUIDs list not found at {path}; is the NFS root mounted?"
        )
    return [
        line.strip()
        for line in path.read_text().splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


def _lease_mig_uuid(
    namespace: str,
    job_name: str,
    safe_uuids_path: Path = _SAFE_UUIDS_PATH,
    max_tries: int = 3,
) -> str:
    """Atomically claim one MIG slice from the `factory-mig-leases` ConfigMap.

    Mirrors `images/nnunet-trainer/claim-mig.sh` but runs at submit time.
    The nvidia-legacy runtime reads NVIDIA_VISIBLE_DEVICES from pod-spec env
    at container creation, so the leased UUID must be known *before* the Job
    manifest is rendered (see CLAUDE.md `gpu_runtime_class_lesson.md`).

    The lease owner is `job_name`; the reaper CronJob releases stale leases
    when the owning Job is Complete / Failed / NotFound.
    """
    candidates = _read_safe_uuids(safe_uuids_path)
    if not candidates:
        raise RuntimeError(f"no UUIDs in {safe_uuids_path}")

    # Deterministic shuffle keyed by job_name (same idea as claim-mig.sh's
    # cksum seed) so concurrent CLI invocations don't always race for the
    # same slot first.
    seed = int(hashlib.sha256(job_name.encode()).hexdigest()[:16], 16)
    rng = random.Random(seed)
    rng.shuffle(candidates)

    core_v1 = k8s_client.CoreV1Api()
    for attempt in range(max_tries):
        for uuid in candidates:
            patch = [
                {"op": "test", "path": f"/data/{uuid}", "value": "available"},
                {"op": "replace", "path": f"/data/{uuid}", "value": job_name},
            ]
            try:
                core_v1.patch_namespaced_config_map(
                    name=_LEASES_CM, namespace=namespace, body=patch,
                )
            except k8s_client.exceptions.ApiException as e:
                # 422 Unprocessable Entity = JSON-patch `test` op failed
                # (slot already taken between read and patch). Anything else
                # (403 RBAC, 404 missing ConfigMap, 5xx) bubbles up.
                if e.status != 422:
                    raise
                continue
            logger.info("leased MIG slice %s for job %s", uuid, job_name)
            return uuid
        if attempt < max_tries - 1:
            time.sleep(2 + attempt * 3)

    raise RuntimeError(
        f"no free MIG slices in {safe_uuids_path.name} after {max_tries} tries; "
        "wait for a Job to finish or run via `campaign run-trio` instead"
    )


def _release_mig_uuid(namespace: str, uuid: str) -> None:
    """Reset a leased slot back to "available". Best-effort recovery used if
    Job creation fails after the lease succeeded; the reaper CronJob sweeps
    any leftovers."""
    patch = [{"op": "replace", "path": f"/data/{uuid}", "value": "available"}]
    try:
        k8s_client.CoreV1Api().patch_namespaced_config_map(
            name=_LEASES_CM, namespace=namespace, body=patch,
        )
        logger.info("released MIG slice %s", uuid)
    except Exception as e:
        logger.warning("failed to release MIG slice %s (reaper will handle): %s", uuid, e)


def render(
    spec: TrainSpec,
    cfg: FactoryConfig,
    leased_uuid: str | None = None,
    job_name: str | None = None,
) -> str:
    """Render a Job manifest for one fold.

    `leased_uuid` is baked into the pod-spec env as NVIDIA_VISIBLE_DEVICES.
    For dry-run previews, pass a placeholder string; for real submissions,
    `submit()` calls `_lease_mig_uuid()` first and threads the result here.
    """
    experiment = spec.experiment or f"{spec.dataset}__{spec.configuration}"
    image = spec.image or cfg.trainer_image
    if job_name is None:
        job_name = _make_job_name(spec)

    # `command` invokes bash -c with the staging script; the first arg
    # ("stage-wrapper") becomes $0 inside the script so the remaining args
    # are forwarded to `exec nnUNetv2_train "$@"`.
    command = ["/bin/bash", "-c", _STAGE_WRAPPER_SCRIPT]
    wrapper_args = ["stage-wrapper", *_build_args(spec)]

    # GPU-mode-driven pod fields. These are pre-built strings (not Jinja {% if %}
    # blocks) so the template stays free of the whitespace pitfall (gotcha #5).
    # An empty line renders as whitespace-only YAML (ignored). In "mig" mode the
    # output is unchanged from the legacy template.
    runtime_class_line = f"runtimeClassName: {cfg.runtime_class}" if cfg.runtime_class else ""
    gpu_limit_line = 'nvidia.com/gpu: "1"' if cfg.gpu_mode == "whole" else ""

    return _JINJA.get_template("training-job.yaml.j2").render(
        job_name=job_name,
        namespace=cfg.namespace,
        queue_name=cfg.queue_name,
        priority_class=spec.priority,
        image=image,
        dataset=spec.dataset,
        configuration=spec.configuration,
        fold=spec.fold,
        command=command,
        args=wrapper_args,
        env_pairs=_build_env(spec, cfg, experiment, leased_uuid=leased_uuid),
        cpu_request=cfg.default_cpu_request,
        cpu_limit=cfg.default_cpu_limit,
        memory_request=cfg.default_memory_request,
        memory_limit=cfg.default_memory_limit,
        shm_size=cfg.default_shm_size,
        ram_stage_size=cfg.default_ram_stage_size,
        runtime_class_line=runtime_class_line,
        node_selector=cfg.node_selector,
        gpu_limit_line=gpu_limit_line,
    )


def submit(spec: TrainSpec, cfg: FactoryConfig | None = None, dry_run: bool = False) -> str:
    """Render + apply (or print) a Job manifest. Returns the rendered YAML
    in dry-run, otherwise the created Job name.

    Submit-time MIG lease: the nvidia-legacy runtime needs NVIDIA_VISIBLE_DEVICES
    set in the pod-spec env at container creation, so we lease a slice from
    `factory-mig-leases` *before* manifesting the Job. If the lease pool is
    saturated, `submit()` raises rather than queueing — use `campaign run-trio`
    for the Ray-queued path.
    """
    cfg = cfg or FactoryConfig.load()

    if dry_run:
        # Preview only — no lease consumed.
        return render(spec, cfg, leased_uuid="MIG-<dry-run-placeholder>")

    try:
        k8s_config.load_incluster_config()
    except k8s_config.ConfigException:
        k8s_config.load_kube_config()

    job_name = _make_job_name(spec)

    # Whole-GPU mode: the device plugin allocates a full GPU via the
    # nvidia.com/gpu request in the manifest — no MIG slice to lease.
    if cfg.gpu_mode == "whole":
        manifest = render(spec, cfg, leased_uuid=None, job_name=job_name)
        body = yaml.safe_load(manifest)
        k8s_client.BatchV1Api().create_namespaced_job(namespace=cfg.namespace, body=body)
        return body["metadata"]["name"]

    leased = _lease_mig_uuid(cfg.namespace, job_name, safe_uuids_path=cfg.safe_uuids_path)

    try:
        manifest = render(spec, cfg, leased_uuid=leased, job_name=job_name)
        body = yaml.safe_load(manifest)
        batch = k8s_client.BatchV1Api()
        batch.create_namespaced_job(namespace=cfg.namespace, body=body)
        return body["metadata"]["name"]
    except Exception:
        _release_mig_uuid(cfg.namespace, leased)
        raise


def submit_five_folds(
    dataset: str,
    configuration: str = "3d_fullres",
    trainer_class: str = "nnUNetTrainerMLflow",
    priority: str = "fold-training",
    num_epochs: int | None = None,
    cfg: FactoryConfig | None = None,
) -> list[str]:
    """Convenience: submit folds 0..4 in one call. Each fold leases a MIG
    slice at submit time; if the pool is saturated, the loop raises rather
    than queueing. For queued semantics use `campaign run-trio` (Ray-backed).
    """
    cfg = cfg or FactoryConfig.load()
    names = []
    for fold in range(5):
        spec = TrainSpec(
            dataset=dataset,
            configuration=configuration,
            fold=fold,
            trainer_class=trainer_class,
            priority=priority,
            num_epochs=num_epochs,
        )
        names.append(submit(spec, cfg))
    return names
