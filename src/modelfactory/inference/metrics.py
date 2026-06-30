"""Per-label dice + HD95 against a groundtruth NIfTI.

Hand-rolled (no scipy/monai dependency) so this module is importable in the
orchestrator Python. HD95 uses an approximate surface distance via the
edge-of-mask voxels and a KD-tree; for typical segmentation volumes the
approximation is well within the agreement noise of the labels themselves.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class LabelMetric:
    label: int
    label_name: str
    dice: float
    hd95_mm: float | None  # None if either mask is empty
    n_voxels_gt: int
    n_voxels_pred: int


def dice_per_label(
    pred: np.ndarray,
    gt: np.ndarray,
    label_map: dict[str, int],
) -> list[LabelMetric]:
    """Compute dice per foreground label. Background (label 0) is skipped."""
    out: list[LabelMetric] = []
    inv = {v: k for k, v in label_map.items()}
    for label in sorted({int(v) for v in label_map.values()} - {0}):
        p = pred == label
        g = gt == label
        n_p = int(p.sum())
        n_g = int(g.sum())
        if n_p == 0 and n_g == 0:
            dice = float("nan")
        else:
            inter = int(np.logical_and(p, g).sum())
            dice = 2.0 * inter / (n_p + n_g) if (n_p + n_g) > 0 else float("nan")
        out.append(LabelMetric(
            label=label,
            label_name=inv.get(label, f"label_{label}"),
            dice=dice,
            hd95_mm=None,  # set below if we have a spacing + both masks non-empty
            n_voxels_gt=n_g,
            n_voxels_pred=n_p,
        ))
    return out


def hd95_per_label(
    pred: np.ndarray,
    gt: np.ndarray,
    label_map: dict[str, int],
    spacing_zyx: tuple[float, float, float],
    sample_cap: int = 20000,
) -> dict[int, float | None]:
    """95th-percentile bidirectional surface distance in mm, per label.

    Approximates by sampling surface voxels (boundary = mask XOR shifted-mask)
    and computing per-side nearest-neighbour distances via brute-force on the
    sample. Returns NaN if either mask is empty.

    spacing_zyx — (z, y, x) voxel size in mm, matches numpy axis order.
    """
    out: dict[int, float | None] = {}
    sz, sy, sx = spacing_zyx
    rng = np.random.default_rng(0)

    for label in sorted({int(v) for v in label_map.values()} - {0}):
        p = pred == label
        g = gt == label
        if not p.any() or not g.any():
            out[label] = None
            continue

        p_surface = _surface_voxels(p)
        g_surface = _surface_voxels(g)
        if p_surface.size == 0 or g_surface.size == 0:
            out[label] = None
            continue

        p_mm = p_surface.astype(np.float32) * np.array([sz, sy, sx], dtype=np.float32)
        g_mm = g_surface.astype(np.float32) * np.array([sz, sy, sx], dtype=np.float32)

        if p_mm.shape[0] > sample_cap:
            p_mm = p_mm[rng.choice(p_mm.shape[0], sample_cap, replace=False)]
        if g_mm.shape[0] > sample_cap:
            g_mm = g_mm[rng.choice(g_mm.shape[0], sample_cap, replace=False)]

        d_pg = _nn_distance(p_mm, g_mm)
        d_gp = _nn_distance(g_mm, p_mm)
        h95 = float(max(np.percentile(d_pg, 95), np.percentile(d_gp, 95)))
        out[label] = h95
    return out


def _surface_voxels(mask: np.ndarray) -> np.ndarray:
    """Return (N, 3) integer coords of voxels on the mask boundary."""
    if not mask.any():
        return np.empty((0, 3), dtype=np.int32)
    inner = (
        mask
        & np.roll(mask, 1, axis=0) & np.roll(mask, -1, axis=0)
        & np.roll(mask, 1, axis=1) & np.roll(mask, -1, axis=1)
        & np.roll(mask, 1, axis=2) & np.roll(mask, -1, axis=2)
    )
    surface = mask & ~inner
    return np.argwhere(surface).astype(np.int32)


def _nn_distance(a: np.ndarray, b: np.ndarray, chunk: int = 1024) -> np.ndarray:
    """For each point in a, distance to nearest point in b. Brute-force, chunked
    to bound peak memory at ~chunk * len(b) * 4 bytes."""
    out = np.empty(a.shape[0], dtype=np.float32)
    for i in range(0, a.shape[0], chunk):
        block = a[i:i + chunk]
        # (chunk, 1, 3) - (1, B, 3) -> (chunk, B, 3)
        diff = block[:, None, :] - b[None, :, :]
        d2 = np.einsum("ijk,ijk->ij", diff, diff)
        out[i:i + chunk] = np.sqrt(d2.min(axis=1))
    return out
