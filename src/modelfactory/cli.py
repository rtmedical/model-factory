"""modelfactory CLI: thin wrapper around the SDK."""

from __future__ import annotations

import datetime as _dt
import json
import sys
from pathlib import Path

import click
import jinja2
import mlflow
import yaml
from kubernetes import client as k8s_client
from kubernetes import config as k8s_config
from rich.console import Console
from rich.table import Table

from modelfactory import __version__
from modelfactory.config import FactoryConfig
from modelfactory.datasets.register import register as register_dataset
from modelfactory.datasets.register import validate as validate_dataset
from modelfactory.datasets.specs import SPECS
from modelfactory.jobs.nnunet import TrainSpec, submit, submit_five_folds
# `modelfactory.trainers.ensemble` is imported lazily inside the
# `model register-ensemble` command — it pulls in torch via the trainer
# subclass, which the host Python doesn't have (torch lives only in the
# trainer image). Top-level imports here must stay torch-free so the
# `campaign` and `train` verbs work outside the container.

console = Console()


@click.group()
@click.version_option(__version__)
def main() -> None:
    """modelfactory: train and manage segmentation models on the factory cluster."""


# ─── infra (cluster bootstrap / manifest rendering) ──────────────────────────
# Imported here (not at module top) to keep the heavy import graph local; the
# group itself only pulls in click/rich/pyyaml/pydantic — torch-free.
from modelfactory.infra.cli import infra as _infra_group  # noqa: E402

main.add_command(_infra_group)


# ─── dataset ───────────────────────────────────────────────────────────────


@main.group()
def dataset() -> None:
    """Manage nnUNetv2 datasets in the factory NFS root."""


@dataset.command("validate")
@click.argument("path", type=click.Path(exists=True, file_okay=False, path_type=Path))
def dataset_validate(path: Path) -> None:
    """Check that PATH is a well-formed nnUNetv2 dataset."""
    ds = validate_dataset(path)
    console.print(f"[green]OK[/green] {path.name}: {ds.numTraining} cases, "
                  f"{len(ds.channel_names)} channels, {len(ds.labels)} labels")


@dataset.command("register")
@click.argument("path", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option("--move/--copy", default=False, help="Move (faster, destructive) or copy")
@click.option("--overwrite", is_flag=True)
def dataset_register(path: Path, move: bool, overwrite: bool) -> None:
    """Validate and register a dataset under the factory NFS root."""
    cfg = FactoryConfig.load()
    dest = register_dataset(path, cfg.nfs_host_root, move=move, overwrite=overwrite)
    console.print(f"[green]registered[/green] {path.name} → {dest}")


@dataset.command("list")
def dataset_list() -> None:
    """List datasets registered under the factory NFS root."""
    cfg = FactoryConfig.load()
    root = cfg.nfs_host_root / "datasets"
    if not root.exists():
        console.print("[yellow](no datasets registered)[/yellow]")
        return
    t = Table("Dataset", "Cases", "Channels", "Labels", title="Registered datasets")
    for ds_dir in sorted(root.iterdir()):
        if not ds_dir.is_dir():
            continue
        try:
            ds = validate_dataset(ds_dir)
            t.add_row(ds_dir.name, str(ds.numTraining), str(len(ds.channel_names)), str(len(ds.labels)))
        except (FileNotFoundError, ValueError) as e:
            t.add_row(ds_dir.name, "[red]ERROR[/red]", str(e)[:60], "")
    console.print(t)


# ─── train ─────────────────────────────────────────────────────────────────


@main.group()
def train() -> None:
    """Submit training jobs."""


@train.command("nnunet")
@click.option("--dataset", "dataset_name", required=True, help="e.g. Dataset042_KiTS23")
@click.option("--configuration", default="3d_fullres",
              type=click.Choice(["2d", "3d_lowres", "3d_fullres", "3d_cascade_fullres"]))
@click.option("--folds", default="0,1,2,3,4",
              help="Comma-separated fold indices or 'all' (=0,1,2,3,4)")
@click.option("--trainer", "trainer_class", default="nnUNetTrainerMLflow")
@click.option("--priority", default="fold-training",
              type=click.Choice(["interactive-eval", "fold-training", "hpo-sweep"]))
@click.option("--num-epochs", type=int, default=None)
@click.option("--pretrained-weights", type=str, default=None,
              help="In-pod path under /factory/weights/...")
@click.option("--continue/--no-continue", "continue_training", default=False)
@click.option("--dry-run", is_flag=True, help="Print rendered manifest, don't submit")
def train_nnunet(
    dataset_name: str,
    configuration: str,
    folds: str,
    trainer_class: str,
    priority: str,
    num_epochs: int | None,
    pretrained_weights: str | None,
    continue_training: bool,
    dry_run: bool,
) -> None:
    """Submit one or more folds for a dataset."""
    fold_list = list(range(5)) if folds == "all" else [int(f) for f in folds.split(",")]
    cfg = FactoryConfig.load()
    for fold in fold_list:
        spec = TrainSpec(
            dataset=dataset_name,
            configuration=configuration,
            fold=fold,
            trainer_class=trainer_class,
            priority=priority,
            num_epochs=num_epochs,
            pretrained_weights=pretrained_weights,
            continue_training=continue_training,
        )
        if dry_run:
            console.print(submit(spec, cfg, dry_run=True))
            console.print("---")
        else:
            name = submit(spec, cfg)
            console.print(f"[green]submitted[/green] {name}")


# ─── runs ──────────────────────────────────────────────────────────────────


@main.group()
def runs() -> None:
    """List and inspect MLflow runs."""


@runs.command("list")
@click.option("--dataset", "dataset_name", default=None)
@click.option("--limit", type=int, default=20)
def runs_list(dataset_name: str | None, limit: int) -> None:
    cfg = FactoryConfig.load()
    mlflow.set_tracking_uri(cfg.mlflow_tracking_uri)
    client = mlflow.tracking.MlflowClient()
    exps = client.search_experiments()
    if dataset_name:
        exps = [e for e in exps if e.name.startswith(f"{dataset_name}__")]
    if not exps:
        console.print("[yellow](no experiments)[/yellow]")
        return
    t = Table("Experiment", "Run", "Fold", "Status", "val_loss", "mean_fg_dice")
    runs = client.search_runs(
        experiment_ids=[e.experiment_id for e in exps],
        max_results=limit,
        order_by=["start_time DESC"],
    )
    for r in runs:
        t.add_row(
            r.info.experiment_id,
            r.info.run_id[:8],
            r.data.tags.get("fold", ""),
            r.info.status,
            f"{r.data.metrics.get('val_loss', float('nan')):.4f}",
            f"{r.data.metrics.get('mean_fg_dice', float('nan')):.4f}",
        )
    console.print(t)


# ─── model ─────────────────────────────────────────────────────────────────


@main.group()
def model() -> None:
    """Manage MLflow Model Registry entries."""


@model.command("register-ensemble")
@click.option("--dataset", "dataset_name", required=True)
@click.option("--configuration", default="3d_fullres")
@click.option("--name", "registered_name", default=None,
              help="MLflow Registry name (default: dataset__configuration)")
def model_register_ensemble(dataset_name: str, configuration: str, registered_name: str | None) -> None:
    """Collect all 5 fold checkpoints for a dataset/config and register a pyfunc ensemble."""
    # Lazy-imported: pulls in torch via the trainer package; only this command needs it.
    from modelfactory.trainers.ensemble import register_ensemble
    cfg = FactoryConfig.load()
    mlflow.set_tracking_uri(cfg.mlflow_tracking_uri)
    uri = register_ensemble(dataset_name, configuration, registered_model_name=registered_name)
    console.print(f"[green]registered[/green] {uri}")


@model.command("promote")
@click.option("--name", "registered_name", required=True)
@click.option("--version", type=int, required=True)
@click.option("--stage", type=click.Choice(["None", "Staging", "Production", "Archived"]), required=True)
def model_promote(registered_name: str, version: int, stage: str) -> None:
    cfg = FactoryConfig.load()
    mlflow.set_tracking_uri(cfg.mlflow_tracking_uri)
    client = mlflow.tracking.MlflowClient()
    client.transition_model_version_stage(name=registered_name, version=version, stage=stage)
    console.print(f"[green]ok[/green] {registered_name} v{version} → {stage}")


# ─── campaign ──────────────────────────────────────────────────────────────


_RAY_CLUSTER_HEAD_FQDN = "factory-ray-head-svc"  # short name; see config.py for FQDN-hijack note
_RAY_CLIENT_PORT = 10001
_DRIVER_TEMPLATE = Path(__file__).resolve().parents[2] / "infra/kustomize/ray-driver-job.yaml.j2"
_HPO_DRIVER_TEMPLATE = Path(__file__).resolve().parents[2] / "infra/kustomize/hpo-driver-job.yaml.j2"


def _open_parent_runs(dataset_keys: list[str], cfg: FactoryConfig) -> dict[str, str]:
    """Start one parent MLflow run per dataset and return {key: run_id}.

    The parent runs stay open server-side; per-fold child runs nest under them
    when nnUNetTrainerMLflow reads MFACTORY_PARENT_RUN_ID. We don't `end_run`
    here because the children would lose the parent linkage if we did.
    """
    mlflow.set_tracking_uri(cfg.mlflow_tracking_uri)
    parent_ids: dict[str, str] = {}
    for key in dataset_keys:
        spec = SPECS[key]
        experiment = f"{spec.folder}__3d_fullres"
        mlflow.set_experiment(experiment)
        ts = _dt.datetime.now(_dt.UTC).strftime("%Y%m%d-%H%M%S")
        slug = key.replace("_", "-")
        with mlflow.start_run(run_name=f"campaign__{slug}__{ts}") as run:
            mlflow.set_tags({
                "campaign": "true",
                "dataset": spec.folder,
                "dataset_key": key,
                "structures": ",".join(spec.canonical_names),
            })
            parent_ids[key] = run.info.run_id
    return parent_ids


def _submit_driver_job(
    dataset_keys: list[str],
    folds: list[int],
    parent_ids: dict[str, str],
    trainer: str,
    plans: str,
    cfg: FactoryConfig,
    dry_run: bool,
    continue_training: bool = False,
    max_concurrent: int = 8,
) -> str:
    """Render and submit the Ray-driver k8s Job; returns the job name or YAML.

    `max_concurrent` caps Ray Tune's simultaneous trials. The default of 8
    preserves the historic behaviour of the `smoke`/`run-trio`/`run-brain-mr-trio`
    callers; `campaign run-wave` raises it (up to 14) to fill every MIG slice.
    """
    ts = _dt.datetime.now(_dt.UTC).strftime("%Y%m%d-%H%M%S")
    slug = "-".join(k.replace("_", "-") for k in dataset_keys)[:30]
    job_name = f"campaign-{slug}-{ts}"[:63]

    parent_arg = ",".join(f"{k}={v}" for k, v in parent_ids.items())
    env = jinja2.Environment(trim_blocks=True, lstrip_blocks=True)
    manifest = env.from_string(_DRIVER_TEMPLATE.read_text()).render(
        job_name=job_name,
        datasets=",".join(dataset_keys),
        folds=",".join(str(f) for f in folds),
        parent_run_ids=parent_arg,
        trainer=trainer,
        plans=plans,
        max_concurrent=max_concurrent,
        image=cfg.trainer_image,
        ray_address=f"ray://{_RAY_CLUSTER_HEAD_FQDN}:{_RAY_CLIENT_PORT}",
        continue_training=continue_training,
    )

    if dry_run:
        return manifest

    try:
        k8s_config.load_incluster_config()
    except k8s_config.ConfigException:
        k8s_config.load_kube_config()
    body = yaml.safe_load(manifest)
    k8s_client.BatchV1Api().create_namespaced_job(namespace=cfg.namespace, body=body)
    return job_name


@main.group()
def campaign() -> None:
    """Cross-dataset training campaigns orchestrated by Ray Tune."""


@campaign.command("smoke")
@click.option("--dataset", "dataset_key", required=True,
              help="DatasetSpec key, e.g. brain_mr_core_oar")
@click.option("--fold", type=int, default=0)
@click.option("--trainer", default="nnUNetTrainerMLflow",
              help="To smoke without a full training run, subclass nnUNetTrainerMLflow with a low num_epochs and pass that class name.")
@click.option("--plans", default="nnUNetResEncUNetLPlans")
@click.option("--continue/--no-continue", "continue_training", default=False,
              help="Pass --c to nnUNetv2_train so it resumes from checkpoint_latest.pth if present.")
@click.option("--dry-run", is_flag=True)
def campaign_smoke(dataset_key: str, fold: int, trainer: str, plans: str, continue_training: bool, dry_run: bool) -> None:
    """Run ONE (dataset, fold) trial via Ray — verifies the whole path before fanning out."""
    if dataset_key not in SPECS:
        console.print(f"[red]unknown dataset key[/red]: {dataset_key}")
        console.print(f"known: {', '.join(sorted(SPECS))}")
        sys.exit(2)
    cfg = FactoryConfig.load()
    parent_ids = {} if dry_run else _open_parent_runs([dataset_key], cfg)
    out = _submit_driver_job(
        [dataset_key], [fold], parent_ids, trainer, plans, cfg, dry_run=dry_run,
        continue_training=continue_training,
    )
    if dry_run:
        console.print(out)
    else:
        console.print(f"[green]submitted[/green] driver job {out}")
        console.print(f"  ray dashboard:  kubectl -n model-factory port-forward svc/factory-ray-head-svc 8265:8265")
        console.print(f"  mlflow ui:      kubectl -n model-factory port-forward svc/mlflow 5000:5000")


_BRAIN_MR_TRIO = ("brain_mr_core_oar", "brain_mr_lobes_bilateral", "brain_mr_deep_nuclei")


@campaign.command("run-brain-mr-trio")
@click.option("--folds", default="0,1,2,3,4")
@click.option("--trainer", default="nnUNetTrainerMLflow")
@click.option("--plans", default="nnUNetResEncUNetLPlans")
@click.option("--dry-run", is_flag=True)
def campaign_brain_mr_trio(folds: str, trainer: str, plans: str, dry_run: bool) -> None:
    """Fan datasets 045 + 047 + 048 × 5 folds out across factory-ray."""
    fold_list = [int(f) for f in folds.split(",") if f.strip()]
    cfg = FactoryConfig.load()
    parent_ids = {} if dry_run else _open_parent_runs(list(_BRAIN_MR_TRIO), cfg)
    out = _submit_driver_job(
        list(_BRAIN_MR_TRIO), fold_list, parent_ids, trainer, plans, cfg, dry_run=dry_run,
    )
    if dry_run:
        console.print(out)
    else:
        n_trials = len(_BRAIN_MR_TRIO) * len(fold_list)
        console.print(f"[green]submitted[/green] driver job {out} — {n_trials} trials")
        for key, run_id in parent_ids.items():
            console.print(f"  parent run {key}: {run_id}")
        console.print(f"  ray dashboard:  kubectl -n model-factory port-forward svc/factory-ray-head-svc 8265:8265")
        console.print(f"  mlflow ui:      kubectl -n model-factory port-forward svc/mlflow 5000:5000")


@campaign.command("run-trio")
@click.option("--datasets", required=True,
              help="Comma-separated dataset keys, exactly 3 (e.g. pelvis_male_prostate,pancreas_msd_tumor,luna16_nodules)")
@click.option("--folds", default="0,1,2,3,4")
@click.option("--trainer", default="nnUNetTrainerMLflow")
@click.option("--plans", default="nnUNetResEncUNetXLPlans")
@click.option("--dry-run", is_flag=True)
def campaign_run_trio(datasets: str, folds: str, trainer: str, plans: str, dry_run: bool) -> None:
    """Fan an arbitrary 3-dataset × N-fold campaign out across factory-ray."""
    keys = [k.strip() for k in datasets.split(",") if k.strip()]
    if len(keys) != 3:
        console.print(f"[red]--datasets must be exactly 3 keys[/red]")
        sys.exit(2)
    unknown = [k for k in keys if k not in SPECS]
    if unknown:
        console.print(f"[red]unknown keys:[/red] {unknown}")
        sys.exit(2)
    fold_list = [int(f) for f in folds.split(",") if f.strip()]
    cfg = FactoryConfig.load()
    parent_ids = {} if dry_run else _open_parent_runs(keys, cfg)
    out = _submit_driver_job(keys, fold_list, parent_ids, trainer, plans, cfg, dry_run=dry_run)
    if dry_run:
        console.print(out)
    else:
        n_trials = len(keys) * len(fold_list)
        console.print(f"[green]submitted[/green] driver job {out} — {n_trials} trials")
        for k, run_id in parent_ids.items():
            console.print(f"  parent run {k}: {run_id}")
        console.print(f"  ray dashboard:  kubectl -n model-factory port-forward svc/factory-ray-head-svc 8265:8265")
        console.print(f"  mlflow ui:      kubectl -n model-factory port-forward svc/mlflow 5000:5000")


@campaign.command("run-wave")
@click.option("--datasets", required=True,
              help="Comma-separated dataset keys, 1..14 (one trial per MIG slice).")
@click.option("--folds", default="0", show_default=True,
              help="Comma-separated folds. The wave default is fold-0 only.")
@click.option("--max-concurrent", type=int, default=14, show_default=True,
              help="Cap on simultaneous Ray Tune trials. With several waves sharing "
                   "the pool, size each so they sum to <=14; Ray's GPU=1-per-trial "
                   "accounting prevents oversubscription regardless.")
@click.option("--trainer", default="nnUNetTrainerMLflow", show_default=True,
              help="Use nnUNetTrainerSmallStructuresMLflow for sparse/thin targets, "
                   "or nnUNetTrainerPartialLabelMLflow for the partial-label generalists.")
@click.option("--plans", default="nnUNetResEncUNetLPlans", show_default=True,
              help="Pair nnUNetResEncUNetLPlans_HighRes with the small-structures trainer.")
@click.option("--continue/--no-continue", "continue_training", default=False,
              help="Pass --c to nnUNetv2_train (resume from checkpoint_latest.pth if present).")
@click.option("--dry-run", is_flag=True)
def campaign_run_wave(datasets: str, folds: str, max_concurrent: int, trainer: str,
                      plans: str, continue_training: bool, dry_run: bool) -> None:
    """Fan an N-dataset (N<=14) × M-fold wave across factory-ray in ONE driver job.

    Unlike run-trio (exactly 3), this accepts up to 14 keys and raises Ray Tune's
    max_concurrent_trials so every MIG slice fills at once. All trials in one wave
    share a single (trainer, plans) pair — span recipe classes with multiple waves
    sharing the 14-GPU pool (e.g. a default-trainer wave + a small-structures wave,
    each with --max-concurrent sized to its own trial count).
    """
    keys = [k.strip() for k in datasets.split(",") if k.strip()]
    if not 1 <= len(keys) <= 14:
        console.print("[red]--datasets must list 1..14 keys[/red]")
        sys.exit(2)
    unknown = [k for k in keys if k not in SPECS]
    if unknown:
        console.print(f"[red]unknown keys:[/red] {unknown}")
        sys.exit(2)
    fold_list = [int(f) for f in folds.split(",") if f.strip()]
    if max_concurrent > 14:
        console.print("[yellow]warn[/yellow] --max-concurrent >14 exceeds the MIG pool; "
                      "Ray's GPU=14 accounting is the real ceiling.")
    cfg = FactoryConfig.load()
    parent_ids = {} if dry_run else _open_parent_runs(keys, cfg)
    out = _submit_driver_job(
        keys, fold_list, parent_ids, trainer, plans, cfg,
        dry_run=dry_run, continue_training=continue_training,
        max_concurrent=max_concurrent,
    )
    if dry_run:
        console.print(out)
        return
    n_trials = len(keys) * len(fold_list)
    console.print(f"[green]submitted[/green] driver job {out} — {n_trials} trials, "
                  f"max_concurrent={max_concurrent}")
    for k, run_id in parent_ids.items():
        console.print(f"  parent run {k}: {run_id}")
    console.print(f"  ray dashboard:  kubectl -n model-factory port-forward svc/factory-ray-head-svc 8265:8265")
    console.print(f"  mlflow ui:      kubectl -n model-factory port-forward svc/mlflow 5000:5000")


# ─── schedule (future-trainings pipeline) ───────────────────────────────────
#
# Manages the queue of trainings we intend to submit next. The qa-viewer
# persists the queue and renders it as planned bars on the home calendar
# (projecting start/finish from live training rates). This is a plan, not an
# auto-dispatcher: submitting stays a manual `campaign` call.
#
# We talk to the qa-viewer API (NodePort), NOT the SQLite file directly: the
# DB is written by the pod (root-owned on NFS), so the pod is the single
# writer. Override the endpoint with MFACTORY_QA_API_URL / config qa_api_url.


def _qa_api(cfg: FactoryConfig, method: str, path: str, body: dict | None = None):
    """Call the qa-viewer API; exit with a clear message on HTTP/connection error."""
    import urllib.error
    import urllib.request

    url = cfg.qa_api_url.rstrip("/") + path
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(
        url, data=data, method=method, headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(req, timeout=20) as r:
            raw = r.read()
            return json.loads(raw) if raw else None
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode(errors="replace")[:300]
        console.print(f"[red]QA API {exc.code}[/red] {method} {path}: {detail}")
        sys.exit(1)
    except urllib.error.URLError as exc:
        console.print(
            f"[red]QA API unreachable[/red] at {cfg.qa_api_url} ({exc}). "
            "Set MFACTORY_QA_API_URL (or qa_api_url in ~/.modelfactory.yaml); "
            "the viewer's NodePort is svc/qa-viewer :80 → 32443."
        )
        sys.exit(1)


@main.group()
def schedule() -> None:
    """Queue future trainings; the QA home calendar renders them as planned bars."""


@schedule.command("add")
@click.option("--dataset", "dataset_key", required=True, help="DatasetSpec key, e.g. thorax_clinical_breast_l")
@click.option("--fold", type=int, default=0, show_default=True)
@click.option("--trainer", default="nnUNetTrainerMLflow", show_default=True,
              help="Use nnUNetTrainerSmallStructuresMLflow for sub-voxel/sparse datasets.")
@click.option("--plans", default="nnUNetResEncUNetLPlans", show_default=True,
              help="Use nnUNetResEncUNetLPlans_HighRes alongside the small-structures trainer.")
@click.option("--priority", type=int, default=0, show_default=True, help="Higher runs earlier.")
@click.option("--duration-hours", type=float, default=None,
              help="Override the fold wall-time prior (default 72h, or 96h for HighRes plans).")
@click.option("--notes", default="")
@click.option("--by", "submitted_by", default="cli")
def schedule_add(dataset_key: str, fold: int, trainer: str, plans: str,
                 priority: int, duration_hours: float | None, notes: str,
                 submitted_by: str) -> None:
    """Enqueue one (dataset, fold). Idempotent on (dataset, fold)."""
    if dataset_key not in SPECS:
        console.print(f"[red]unknown dataset key[/red]: {dataset_key}")
        console.print(f"known: {', '.join(sorted(SPECS))}")
        sys.exit(2)
    cfg = FactoryConfig.load()
    p = _qa_api(cfg, "POST", "/api/planned-trainings", {
        "dataset_key": dataset_key, "dataset_name": SPECS[dataset_key].folder,
        "fold": fold, "trainer": trainer, "plans": plans, "priority": priority,
        "est_duration_hours": duration_hours, "submitted_by": submitted_by, "notes": notes,
    })
    console.print(f"[green]queued[/green] {p['dataset_name']} fold {p['fold']} "
                  f"(priority {p['priority']}, id {p['id'][:8]})")


@schedule.command("list")
def schedule_list() -> None:
    """Show the planned-training queue with projected start/finish, soonest first."""
    cfg = FactoryConfig.load()
    rows = _qa_api(cfg, "GET", "/api/planned-trainings") or []
    if not rows:
        console.print("[yellow]queue empty[/yellow]")
        return
    rows.sort(key=lambda r: (r.get("est_finish") or "", -r["priority"]))
    table = Table(title=f"Future trainings — {len(rows)} queued")
    for col in ("prio", "dataset", "fold", "trainer", "plans", "~start", "~finish", "id"):
        table.add_column(col)
    for r in rows:
        table.add_row(
            str(r["priority"]), r["dataset_name"], str(r["fold"]),
            r["trainer"].replace("nnUNetTrainer", ""),
            r["plans"].replace("nnUNetResEncUNet", ""),
            (r.get("scheduled_start") or "")[:16], (r.get("est_finish") or "")[:16],
            r["id"][:8],
        )
    console.print(table)


@schedule.command("rm")
@click.argument("planned_id")
def schedule_rm(planned_id: str) -> None:
    """Remove a queued training by id (accepts the short 8-char prefix)."""
    cfg = FactoryConfig.load()
    target = planned_id
    if len(planned_id) < 32:  # short prefix → resolve against the live queue
        rows = _qa_api(cfg, "GET", "/api/planned-trainings") or []
        matches = [r["id"] for r in rows if r["id"].startswith(planned_id)]
        if len(matches) != 1:
            console.print(f"[red]{'no' if not matches else 'ambiguous'} id prefix[/red]: {planned_id}")
            sys.exit(2)
        target = matches[0]
    _qa_api(cfg, "DELETE", f"/api/planned-trainings/{target}")
    console.print("[green]removed[/green]")


@schedule.command("enqueue-remaining-folds")
@click.option("--folds", default="1,2,3,4", show_default=True,
              help="Folds to queue for every dataset whose fold 0 is already trained.")
@click.option("--priority", type=int, default=30, show_default=True)
def schedule_enqueue_remaining(folds: str, priority: int) -> None:
    """Queue folds 1-4 of every dataset that already has a trained fold.

    This is the natural 'pipeline of future trainings' after the fold-0 smokes:
    once a fold 0 lands, schedule its remaining folds for the full 5-fold CV.
    Trainer/plans are read from each model's existing results config so they
    match what fold 0 used. Posts each fold to the qa-viewer API.
    """
    from modelfactory.qa.cohort import _discover_trained_models
    cfg = FactoryConfig.load()
    fold_list = [int(f) for f in folds.split(",") if f.strip()]
    name_to_key = {SPECS[k].folder: k for k in SPECS}
    entries = _discover_trained_models(
        cfg.nfs_host_root / "results", datasets_root=cfg.nfs_host_root / "datasets"
    )
    n = 0
    for e in entries:
        have = set(e["available_folds"])
        if 0 not in have:
            continue  # only extend datasets whose fold 0 has landed
        key = name_to_key.get(e["dataset_name"], e["dataset_name"])
        for f in fold_list:
            if f in have:
                continue
            _qa_api(cfg, "POST", "/api/planned-trainings", {
                "dataset_key": key, "dataset_name": e["dataset_name"], "fold": f,
                "trainer": e["trainer"], "plans": e["plans"], "priority": priority,
                "submitted_by": "cli", "notes": "remaining-fold (5-fold CV)",
            })
            n += 1
    console.print(f"[green]queued[/green] {n} remaining fold(s) across "
                  f"{len(entries)} trained dataset(s)")


# ─── hpo ───────────────────────────────────────────────────────────────────


def _open_hpo_parent_run(dataset_key: str, cfg: FactoryConfig, num_trials: int,
                         max_epochs: int, searcher: str, scheduler: str) -> str:
    """Open one MLflow parent run for the sweep; returns its run_id.

    Per-trial child runs (created by nnUNetTrainerHPO via MFACTORY_PARENT_RUN_ID)
    nest under this one so the MLflow UI shows the whole sweep as a single row.
    """
    spec = SPECS[dataset_key]
    mlflow.set_tracking_uri(cfg.mlflow_tracking_uri)
    mlflow.set_experiment(f"{spec.folder}__3d_fullres")
    ts = _dt.datetime.now(_dt.UTC).strftime("%Y%m%d-%H%M%S")
    slug = dataset_key.replace("_", "-")
    with mlflow.start_run(run_name=f"hpo__{slug}__{ts}") as run:
        mlflow.set_tags({
            "sweep": "true",
            "dataset": spec.folder,
            "dataset_key": dataset_key,
            "hpo.num_trials": str(num_trials),
            "hpo.max_epochs": str(max_epochs),
            "hpo.searcher": searcher,
            "hpo.scheduler": scheduler,
            "structures": ",".join(spec.canonical_names),
        })
        return run.info.run_id


def _submit_hpo_driver_job(
    dataset_key: str, fold: int, parent_run_id: str,
    num_trials: int, max_epochs: int, num_iters: int, max_concurrent: int,
    trainer: str, plans: str, searcher: str, scheduler: str,
    priority: str, cfg: FactoryConfig, dry_run: bool,
) -> str:
    """Render and submit the HPO driver k8s Job; returns the job name or YAML."""
    ts = _dt.datetime.now(_dt.UTC).strftime("%Y%m%d-%H%M%S")
    slug = dataset_key.replace("_", "-")[:35]
    job_name = f"hpo-{slug}-{ts}"[:63]

    env = jinja2.Environment(trim_blocks=True, lstrip_blocks=True)
    manifest = env.from_string(_HPO_DRIVER_TEMPLATE.read_text()).render(
        job_name=job_name,
        dataset=dataset_key,
        fold=fold,
        parent_run_id=parent_run_id,
        num_trials=num_trials,
        max_epochs=max_epochs,
        num_iters=num_iters,
        max_concurrent=max_concurrent,
        trainer=trainer,
        plans=plans,
        searcher=searcher,
        scheduler=scheduler,
        priority=priority,
        image=cfg.trainer_image,
        ray_address=f"ray://{_RAY_CLUSTER_HEAD_FQDN}:{_RAY_CLIENT_PORT}",
    )

    if dry_run:
        return manifest

    try:
        k8s_config.load_incluster_config()
    except k8s_config.ConfigException:
        k8s_config.load_kube_config()
    body = yaml.safe_load(manifest)
    k8s_client.BatchV1Api().create_namespaced_job(namespace=cfg.namespace, body=body)
    return job_name


@main.group()
def hpo() -> None:
    """Hyperparameter sweeps over a single (dataset, fold) via Ray Tune + Optuna + ASHA."""


@hpo.command("run")
@click.option("--dataset", "dataset_key", required=True,
              help="DatasetSpec key, e.g. hn_optic_nerves")
@click.option("--fold", type=int, default=0,
              help="Fold to sweep on (single fold; promote winner to 5-fold full training)")
@click.option("--num-trials", type=int, default=30)
@click.option("--max-epochs", type=int, default=80,
              help="Per-trial cap; ASHA early-stops below this")
@click.option("--num-iters", type=int, default=100,
              help="num_iterations_per_epoch override (cheaper trials)")
@click.option("--max-concurrent", type=int, default=8,
              help="Cap on simultaneous trials (≤ Ray worker count)")
@click.option("--trainer", "trainer_class", default="nnUNetTrainerHPO")
@click.option("--plans", default="nnUNetResEncUNetLPlans")
@click.option("--searcher", type=click.Choice(["optuna", "random"]), default="optuna")
@click.option("--scheduler", type=click.Choice(["asha", "none"]), default="asha")
@click.option("--priority", type=click.Choice(["interactive-eval", "fold-training", "hpo-sweep"]),
              default="hpo-sweep")
@click.option("--dry-run", is_flag=True, help="Print rendered manifest, don't submit")
def hpo_run(
    dataset_key: str, fold: int, num_trials: int, max_epochs: int, num_iters: int,
    max_concurrent: int, trainer_class: str, plans: str, searcher: str, scheduler: str,
    priority: str, dry_run: bool,
) -> None:
    """Launch a Ray Tune sweep over hyperparameters for one (dataset, fold).

    The sweep varies oversample_foreground_percent, initial_lr, weight_decay,
    and enable_deep_supervision. See docs/hpo.md for the rationale and
    src/modelfactory/hpo/runner.py::param_space_for for the search space.
    """
    if dataset_key not in SPECS:
        console.print(f"[red]unknown dataset key[/red]: {dataset_key}")
        console.print(f"known: {', '.join(sorted(SPECS))}")
        sys.exit(2)

    cfg = FactoryConfig.load()
    parent_run_id = "" if dry_run else _open_hpo_parent_run(
        dataset_key, cfg, num_trials, max_epochs, searcher, scheduler,
    )
    out = _submit_hpo_driver_job(
        dataset_key, fold, parent_run_id,
        num_trials, max_epochs, num_iters, max_concurrent,
        trainer_class, plans, searcher, scheduler, priority,
        cfg, dry_run=dry_run,
    )
    if dry_run:
        console.print(out)
        return

    console.print(f"[green]submitted[/green] HPO driver job {out}")
    console.print(f"  dataset:        {dataset_key}  fold {fold}")
    console.print(f"  trials:         {num_trials}  max_epochs={max_epochs}  "
                  f"searcher={searcher} scheduler={scheduler}")
    console.print(f"  parent run:     {parent_run_id}")
    console.print(f"  ray dashboard:  kubectl -n model-factory port-forward svc/factory-ray-head-svc 8265:8265")
    console.print(f"  mlflow ui:      kubectl -n model-factory port-forward svc/mlflow 5000:5000")
    console.print(f"  follow driver:  kubectl -n model-factory logs -f job/{out}")


# ─── qa ────────────────────────────────────────────────────────────────────


@main.group()
def qa() -> None:
    """QA the trained segmentation models via the web viewer cohort."""


@qa.group("cohort")
def qa_cohort() -> None:
    """Build and preprocess the QA validation cohort."""


@qa_cohort.command("prepare")
@click.option("--cases-per-dataset", type=int, default=3,
              help="Cases per trained dataset, built for EVERY known region "
                   "(brain_mr, hn_ct, pelvis_ct, abdomen_ct, thorax_ct, "
                   "whole_body_ct). Additive: re-running with a higher number "
                   "tops each dataset up without disturbing existing case ids.")
@click.option("--region", "region_overrides", multiple=True, metavar="NAME=N",
              help="Per-region override (repeatable), e.g. --region hn_ct=5. "
                   "Regions not overridden use --cases-per-dataset; set to 0 "
                   "to skip a region.")
@click.option("--output", "output", type=click.Path(path_type=Path), default=None,
              help="Cohort root (default <nfs_host_root>/qa-cohort).")
@click.option("--preprocess/--no-preprocess", default=False,
              help="Also pre-stage nnUNetv2 preprocessed inputs for every "
                   "(model, case) pair.")
def qa_cohort_prepare(
    cases_per_dataset: int, region_overrides: tuple[str, ...],
    output: Path | None, preprocess: bool,
) -> None:
    """Materialize /factory/qa-cohort/ with N cases per trained dataset."""
    cfg = FactoryConfig.load()
    output = output or (cfg.nfs_host_root / "qa-cohort")
    datasets_root = cfg.nfs_host_root / "datasets"
    results_root = cfg.nfs_host_root / "results"

    from modelfactory.qa.cohort import KNOWN_REGIONS, build_cohort
    per_region = {r: cases_per_dataset for r in KNOWN_REGIONS}
    for ov in region_overrides:
        try:
            name, val = ov.split("=", 1)
            per_region[name.strip()] = int(val)
        except ValueError:
            console.print(f"[red]bad --region override[/red] {ov!r} (want NAME=N)")
            sys.exit(2)

    manifest = build_cohort(
        datasets_root=datasets_root,
        results_root=results_root,
        output_root=output,
        per_region=per_region,
    )
    console.print(f"[green]cohort ready[/green] {output}")
    console.print(f"  cases: {len(manifest.cases)}")
    console.print(f"  trained models discovered: {len(manifest.trained_models)}")

    if preprocess:
        from modelfactory.qa.preprocess import preprocess_cohort_for_model
        for m in manifest.trained_models:
            console.print(f"  preprocessing for {m['model_id']}...")
            preprocess_cohort_for_model(Path(m["model_dir"]), output)
        console.print("[green]preprocessing complete[/green]")


@qa_cohort.command("backfill")
@click.option("--target", type=int, default=3,
              help="Ensure every trained model's dataset has at least this "
                   "many cohort cases (additive top-up).")
@click.option("--output", "output", type=click.Path(path_type=Path), default=None)
@click.option("--preprocess/--no-preprocess", default=False)
def qa_cohort_backfill(target: int, output: Path | None, preprocess: bool) -> None:
    """Top every trained model's dataset up to --target cases (additive).

    Unlike `prepare`, this iterates discovered models directly, so datasets
    trained after the last `prepare` run get cases too. Datasets whose region
    can't be resolved are reported (they need tags.region in dataset.json).
    """
    cfg = FactoryConfig.load()
    output = output or (cfg.nfs_host_root / "qa-cohort")
    datasets_root = cfg.nfs_host_root / "datasets"
    results_root = cfg.nfs_host_root / "results"

    from modelfactory.qa.cohort import (
        DatasetNotFoundError,
        _discover_trained_models,
        build_cohort_for_dataset,
    )
    models = _discover_trained_models(results_root, datasets_root=datasets_root)
    seen: set[str] = set()
    added = 0
    skipped: list[tuple[str, str]] = []
    for m in models:
        ds = m["dataset_name"]
        if ds in seen:
            continue
        seen.add(ds)
        try:
            new = build_cohort_for_dataset(
                ds,
                datasets_root=datasets_root,
                results_root=results_root,
                output_root=output,
                n_pick=target,
                trained_models=models,
            )
        except (DatasetNotFoundError, ValueError) as exc:
            skipped.append((ds, str(exc)))
            continue
        if new:
            added += len(new)
            console.print(f"  +{len(new)} case(s) for {ds}")
    console.print(f"[green]backfill done[/green] — {added} new case(s) across "
                  f"{len(seen)} dataset(s)")
    if skipped:
        console.print("[yellow]skipped (no region / no imagesTr):[/yellow]")
        for ds, why in skipped:
            console.print(f"  {ds}: {why}")
    if preprocess and added:
        from modelfactory.qa.preprocess import preprocess_cohort_for_model
        for m in models:
            preprocess_cohort_for_model(Path(m["model_dir"]), output)
        console.print("[green]preprocessing complete[/green]")


@qa_cohort.command("prune-uploads")
@click.option("--older-than", type=float, default=None,
              help="Only prune uploads whose dir mtime is older than N days. "
                   "Omit to prune ALL uploaded cases.")
@click.option("--output", "output", type=click.Path(path_type=Path), default=None)
@click.option("--dry-run", is_flag=True)
def qa_cohort_prune_uploads(
    older_than: float | None, output: Path | None, dry_run: bool,
) -> None:
    """Remove ad-hoc uploaded cases from the cohort (dirs + manifest entries)."""
    import shutil as _shutil
    import time as _time

    cfg = FactoryConfig.load()
    output = output or (cfg.nfs_host_root / "qa-cohort")
    manifest_path = output / "manifest.json"
    if not manifest_path.is_file():
        console.print(f"[red]no cohort at {output}[/red]")
        sys.exit(2)
    data = json.loads(manifest_path.read_text())
    cutoff = (_time.time() - older_than * 86400) if older_than else None

    kept: list[dict] = []
    removed: list[str] = []
    for c in data.get("cases", []):
        is_upload = c.get("uploaded") or c.get("source_dataset") == "uploaded"
        case_dir = output / c["region"] / c["case_id"].split("/", 1)[1]
        old_enough = cutoff is None or (
            case_dir.exists() and case_dir.stat().st_mtime < cutoff
        )
        if is_upload and old_enough:
            removed.append(c["case_id"])
            if not dry_run and case_dir.exists():
                _shutil.rmtree(case_dir, ignore_errors=True)
        else:
            kept.append(c)

    if dry_run:
        console.print(f"[yellow]dry-run[/yellow] would remove {len(removed)} upload(s)")
    else:
        data["cases"] = kept
        data["regions"] = sorted({c["region"] for c in kept})
        tmp = manifest_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(data, indent=2))
        tmp.replace(manifest_path)
        console.print(f"[green]pruned[/green] {len(removed)} upload(s)")
    for cid in removed:
        console.print(f"  {cid}")


@qa_cohort.command("preprocess")
@click.option("--model", "model_id", default=None,
              help="One model id (DatasetXXX_Name::trainer__plans__cfg); "
                   "default: all discovered models.")
@click.option("--cohort", "cohort_root", type=click.Path(path_type=Path), default=None)
def qa_cohort_preprocess(model_id: str | None, cohort_root: Path | None) -> None:
    """Pre-stage nnUNetv2 preprocessed inputs (resampled + normalized .npz)."""
    cfg = FactoryConfig.load()
    cohort_root = cohort_root or (cfg.nfs_host_root / "qa-cohort")
    manifest_path = cohort_root / "manifest.json"
    if not manifest_path.is_file():
        console.print(f"[red]no cohort at {cohort_root}[/red] — run "
                      f"`modelfactory qa cohort prepare` first.")
        sys.exit(2)
    manifest = json.loads(manifest_path.read_text())

    from modelfactory.qa.preprocess import preprocess_cohort_for_model
    models = manifest["trained_models"]
    if model_id:
        models = [m for m in models if m["model_id"] == model_id]
        if not models:
            console.print(f"[red]unknown model[/red]: {model_id}")
            sys.exit(2)

    for m in models:
        console.print(f"preprocessing {m['model_id']}...")
        written = preprocess_cohort_for_model(Path(m["model_dir"]), cohort_root)
        console.print(f"  {len(written)} cases ready")


@qa.command("server")
@click.option("--host", default="0.0.0.0")
@click.option("--port", type=int, default=8080)
@click.option("--reload/--no-reload", default=False)
def qa_server(host: str, port: int, reload: bool) -> None:
    """Local dev: run the FastAPI QA backend against the live NFS mount."""
    import uvicorn
    uvicorn.run(
        "modelfactory.qa.api:app",
        host=host,
        port=port,
        reload=reload,
        workers=1,
    )


# ─── dashboard ─────────────────────────────────────────────────────────────


@main.group()
def dashboard() -> None:
    """Render the factory dashboard (docs/factory_dashboard.html)."""


@dashboard.command("render")
@click.option("--output", type=click.Path(path_type=Path),
              default=Path("docs/factory_dashboard.html"),
              help="Where to write the rendered HTML.")
@click.option("--results-root", type=click.Path(path_type=Path),
              default=Path("/data/model-factory-nfs/results"),
              help="NFS root holding <Dataset>/.../fold_0/metrics.jsonl.")
def dashboard_render(output: Path, results_root: Path) -> None:
    """Read metrics.jsonl for all tracked datasets and write the dashboard."""
    from modelfactory.dashboard import render_to_file
    out = render_to_file(output, results_root)
    console.print(f"[green]wrote[/green] {out} ({out.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
