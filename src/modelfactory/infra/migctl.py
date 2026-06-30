"""Thin wrapper around ``scripts/mig_partition.sh`` for the MIG provisioner path.

Iterates a :class:`ClusterSpec` MIG layout and invokes the privileged shell
script per GPU. Kept minimal and explicit — partitioning is destructive and is
only ever run when the operator asks for it (``infra mig-create``), never as a
side effect of rendering.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from .spec import ClusterSpec

_SCRIPT = Path(__file__).resolve().parents[3] / "scripts" / "mig_partition.sh"


def _run(args: list[str], ssh_host: str | None) -> None:
    env = {"MIG_SSH_HOST": ssh_host} if ssh_host else None
    cmd = ["bash", str(_SCRIPT), *args]
    print("+", " ".join(cmd))
    subprocess.run(cmd, check=True, env=({**__import__("os").environ, **env} if env else None))


def create(spec: ClusterSpec, ssh_host: str | None = None) -> None:
    """Create the MIG layout described by ``spec`` (idempotent per GPU)."""
    mig = spec.gpu.mig
    for gpu, entry in sorted(mig.layout.items()):
        if entry.slices > 0:
            _run(["create", "--gpu", str(gpu), "--profile", mig.profile, "--count", str(entry.slices)], ssh_host)
        else:
            _run(["disable", "--gpu", str(gpu)], ssh_host)


def destroy(spec: ClusterSpec, ssh_host: str | None = None) -> None:
    """Destroy MIG instances on every pool GPU (for a clean re-partition)."""
    for gpu in sorted(spec.gpu.mig.pool_gpus):
        _run(["destroy", "--gpu", str(gpu)], ssh_host)
