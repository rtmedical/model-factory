"""Discover MIG slice UUIDs from ``nvidia-smi -L``.

``nvidia-smi -L`` prints one ``GPU <i>: ... (UUID: GPU-...)`` line per physical
card, with indented ``MIG <profile> Device <d>: (UUID: MIG-...)`` lines beneath
each MIG-enabled card, e.g.::

    GPU 0: NVIDIA H100 80GB HBM3 (UUID: GPU-c958...)
    GPU 1: NVIDIA H100 80GB HBM3 (UUID: GPU-d56f...)
      MIG 3g.40gb     Device  0: (UUID: MIG-14d1a71d-...)
      MIG 3g.40gb     Device  1: (UUID: MIG-649d7bfb-...)

We parse that into ``{gpu_index: [uuid, ...]}`` and then flatten in the order
given by ``spec.gpu.mig.pool_gpus`` so worker-group naming (mig-0, mig-1, …) is
deterministic and matches how the live cluster was built.
"""

from __future__ import annotations

import re
import shlex
import subprocess
from dataclasses import dataclass

_GPU_RE = re.compile(r"^GPU (\d+):")
_MIG_RE = re.compile(r"MIG\s+(\S+)\s+Device\s+\d+:\s+\(UUID:\s+(MIG-[0-9a-fA-F-]+)\)")


@dataclass(frozen=True)
class Slice:
    """One MIG slice: which physical GPU it lives on, its profile and UUID."""

    gpu: int
    profile: str
    uuid: str


def parse_nvidia_smi_l(text: str) -> dict[int, list[Slice]]:
    """Parse ``nvidia-smi -L`` output into ``{gpu_index: [Slice, ...]}``."""
    out: dict[int, list[Slice]] = {}
    cur: int | None = None
    for line in text.splitlines():
        m = _GPU_RE.match(line.strip()) if line.lstrip().startswith("GPU ") else None
        # match the GPU header at column 0 (not indented MIG lines)
        if m and not line.startswith(" "):
            cur = int(m.group(1))
            out.setdefault(cur, [])
            continue
        mm = _MIG_RE.search(line)
        if mm and cur is not None:
            out[cur].append(Slice(gpu=cur, profile=mm.group(1), uuid=mm.group(2)))
    return out


def ordered_slices(by_gpu: dict[int, list[Slice]], pool_gpus: list[int]) -> list[Slice]:
    """Flatten slices in ``pool_gpus`` order (device order within each GPU).

    This ordering fixes the slice→worker-group-index mapping, so it must match
    the order the live cluster was built with.
    """
    ordered: list[Slice] = []
    for g in pool_gpus:
        ordered.extend(by_gpu.get(g, []))
    return ordered


def run_nvidia_smi_l(ssh_host: str | None = None) -> str:
    """Run ``nvidia-smi -L`` locally (or over SSH) and return stdout."""
    cmd = ["nvidia-smi", "-L"]
    if ssh_host:
        cmd = ["ssh", ssh_host, shlex.join(cmd)]
    return subprocess.run(cmd, check=True, capture_output=True, text=True).stdout


def discover(pool_gpus: list[int], ssh_host: str | None = None) -> list[Slice]:
    """Discover and order the pool's MIG slices from a live ``nvidia-smi -L``."""
    return ordered_slices(parse_nvidia_smi_l(run_nvidia_smi_l(ssh_host)), pool_gpus)
