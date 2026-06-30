"""``modelfactory infra`` — bootstrap & render the cluster from ``cluster.yaml``.

Subcommands:
    validate     check the spec and print a summary
    discover     parse ``nvidia-smi -L`` -> ordered MIG slice UUIDs (MIG mode)
    mig-create   create the MIG layout on the node(s)  [privileged]
    mig-destroy  tear MIG instances down                [privileged]
    render       write manifests to .render/infra/
    apply        kubectl diff (default) / apply the rendered manifests
    bootstrap    validate -> [mig-create] -> discover -> render -> [apply]
    get          print a single spec field (for Makefile shell-out)

Nothing mutates the cluster unless you pass ``--apply`` (or run ``mig-create``).
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import click
from rich.console import Console
from rich.table import Table

from . import discover as _discover
from . import migctl
from . import render as _render
from .spec import ClusterSpec

_console = Console()

DEFAULT_CONFIG = "cluster.yaml"
RENDER_DIR = Path(".render/infra")
SMI_CACHE = "nvidia_smi_l.txt"


def _load(config: str) -> ClusterSpec:
    path = Path(config)
    if not path.is_file():
        raise click.ClickException(
            f"{config} not found. Copy cluster.example.yaml to {config} and edit it."
        )
    return ClusterSpec.load(path)


def _slices(spec: ClusterSpec) -> list[_discover.Slice]:
    """Resolve MIG slices for rendering from the discovery cache."""
    cache = RENDER_DIR / SMI_CACHE
    if not cache.is_file():
        raise click.ClickException(
            f"no MIG discovery cache at {cache}. Run `modelfactory infra discover` first."
        )
    by_gpu = _discover.parse_nvidia_smi_l(cache.read_text())
    return _discover.ordered_slices(by_gpu, spec.gpu.mig.pool_gpus)


def _summary(spec: ClusterSpec) -> None:
    t = Table(title="model-factory cluster spec", show_header=False)
    t.add_row("namespace", spec.cluster.namespace)
    t.add_row("GPU mode", spec.gpu.mode)
    if spec.gpu.mode == "mig":
        t.add_row("MIG profile", spec.gpu.mig.profile)
        t.add_row("pool GPUs", str(spec.gpu.mig.pool_gpus))
        t.add_row("expected slices", str(spec.gpu.expected_slice_count))
        if spec.gpu.mig.disabled_worker_groups:
            t.add_row("parked groups", ", ".join(spec.gpu.mig.disabled_worker_groups))
    else:
        t.add_row("whole GPUs", str(spec.gpu.whole.count))
    t.add_row("Kueue gpu quota", str(spec.gpu_quota))
    t.add_row("storage", f"{spec.storage.mount} / {spec.storage.storage_class}")
    t.add_row("QA public host", spec.network.qa_public_host or "(NodePort/port-forward)")
    t.add_row("trainer image", spec.ray_worker.image)
    _console.print(t)


@click.group()
def infra() -> None:
    """Bootstrap and render cluster infrastructure from cluster.yaml."""


@infra.command()
@click.option("--config", default=DEFAULT_CONFIG)
def validate(config: str) -> None:
    """Validate the spec and print a summary."""
    _summary(_load(config))
    _console.print("[green]spec is valid[/green]")


@infra.command()
@click.option("--config", default=DEFAULT_CONFIG)
@click.option("--from-file", default=None, help="Read nvidia-smi -L output from a file instead of running it.")
@click.option("--ssh-host", default=None, help="Run nvidia-smi -L over SSH on this host.")
@click.option("--write-nfs/--no-write-nfs", default=True, help="Also write safe_uuids.env to storage.nfsHostRoot.")
def discover(config: str, from_file: str | None, ssh_host: str | None, write_nfs: bool) -> None:
    """Discover MIG slice UUIDs and cache them for rendering (MIG mode)."""
    spec = _load(config)
    if spec.gpu.mode != "mig":
        raise click.ClickException("discover is only meaningful in gpu.mode: mig")
    raw = Path(from_file).read_text() if from_file else _discover.run_nvidia_smi_l(ssh_host)
    by_gpu = _discover.parse_nvidia_smi_l(raw)
    slices = _discover.ordered_slices(by_gpu, spec.gpu.mig.pool_gpus)
    expected = spec.gpu.expected_slice_count
    if len(slices) != expected:
        raise click.ClickException(
            f"discovered {len(slices)} slices but the layout expects {expected}. "
            "Check gpu.mig.layout / poolGpus, or (re)create the MIG layout."
        )
    RENDER_DIR.mkdir(parents=True, exist_ok=True)
    (RENDER_DIR / SMI_CACHE).write_text(raw)
    (RENDER_DIR / "safe_uuids.env").write_text(_render.render_safe_uuids_env(slices))
    if write_nfs:
        nfs = Path(spec.storage.nfs_host_root) / "safe_uuids.env"
        try:
            nfs.write_text(_render.render_safe_uuids_env(slices))
            _console.print(f"wrote {nfs}")
        except OSError as e:  # not fatal — the .render copy still exists
            _console.print(f"[yellow]could not write {nfs}: {e}[/yellow]")
    t = Table(title=f"{len(slices)} MIG slices")
    t.add_column("group")
    t.add_column("gpu")
    t.add_column("uuid")
    for i, s in enumerate(slices):
        t.add_row(f"mig-{i}", str(s.gpu), s.uuid)
    _console.print(t)


@infra.command(name="mig-create")
@click.option("--config", default=DEFAULT_CONFIG)
@click.option("--ssh-host", default=None)
def mig_create(config: str, ssh_host: str | None) -> None:
    """Create the MIG layout on the node(s). PRIVILEGED — kills GPU processes."""
    spec = _load(config)
    if spec.gpu.mode != "mig":
        raise click.ClickException("mig-create is only valid in gpu.mode: mig")
    click.confirm(
        "Partitioning MIG will KILL any running CUDA processes on the target GPUs. Continue?",
        abort=True,
    )
    migctl.create(spec, ssh_host)


@infra.command(name="mig-destroy")
@click.option("--config", default=DEFAULT_CONFIG)
@click.option("--ssh-host", default=None)
def mig_destroy(config: str, ssh_host: str | None) -> None:
    """Destroy MIG instances on the pool GPUs. PRIVILEGED."""
    spec = _load(config)
    click.confirm("This destroys MIG instances on the pool GPUs. Continue?", abort=True)
    migctl.destroy(spec, ssh_host)


@infra.command()
@click.option("--config", default=DEFAULT_CONFIG)
@click.option("--out", default=str(RENDER_DIR), help="Output directory for rendered manifests.")
def render(config: str, out: str) -> None:
    """Render Kubernetes manifests to .render/infra/ (no cluster mutation)."""
    spec = _load(config)
    slices = _slices(spec) if spec.gpu.mode == "mig" else []
    rendered = _render.render_all(spec, slices)
    written = _render.write_all(rendered, out)
    for p in written:
        _console.print(f"  {p}")
    _console.print(f"[green]rendered {len(written)} files to {out}[/green]")


@infra.command()
@click.option("--config", default=DEFAULT_CONFIG)
@click.option("--out", default=str(RENDER_DIR))
@click.option("--dry-run/--apply", "dry_run", default=True, help="Default is a kubectl diff; pass --apply to apply.")
def apply(config: str, out: str, dry_run: bool) -> None:
    """kubectl diff (default) or apply the rendered manifests."""
    out_dir = Path(out)
    manifests = sorted(p for p in out_dir.glob("*.yaml"))
    if not manifests:
        raise click.ClickException(f"no rendered manifests in {out}. Run `infra render` first.")
    verb = ["diff"] if dry_run else ["apply"]
    rc = 0
    for m in manifests:
        _console.print(f"[bold]kubectl {verb[0]} -f {m}[/bold]")
        proc = subprocess.run(["kubectl", *verb, "-f", str(m)])
        rc = rc or (proc.returncode if not dry_run else 0)
    if dry_run:
        _console.print("[yellow]dry-run only — re-run with --apply to apply.[/yellow]")
    elif rc:
        raise click.ClickException("one or more applies failed")


@infra.command()
@click.option("--config", default=DEFAULT_CONFIG)
@click.option("--mig-create", "do_mig", is_flag=True, help="Also (re)create the MIG layout first.")
@click.option("--apply", "do_apply", is_flag=True, help="Apply the rendered manifests (default is render-only).")
@click.pass_context
def bootstrap(ctx: click.Context, config: str, do_mig: bool, do_apply: bool) -> None:
    """validate -> [label nodes] -> [mig-create] -> discover -> render -> [apply]."""
    spec = _load(config)
    _summary(spec)
    for node in spec.cluster.nodes:
        for k, v in spec.cluster.node_selector.items():
            subprocess.run(["kubectl", "label", "node", node, f"{k}={v}", "--overwrite"], check=False)
    if spec.gpu.mode == "mig":
        if do_mig:
            ctx.invoke(mig_create, config=config)
        ctx.invoke(discover, config=config)
    ctx.invoke(render, config=config)
    ctx.invoke(apply, config=config, dry_run=not do_apply)


@infra.command()
@click.argument("field")
@click.option("--config", default=DEFAULT_CONFIG)
def get(field: str, config: str) -> None:
    """Print a dotted spec field (e.g. cluster.namespace) for Makefile use."""
    obj: object = _load(config)
    for part in field.split("."):
        obj = obj[int(part)] if isinstance(obj, list) else getattr(obj, part)
    click.echo(obj)
