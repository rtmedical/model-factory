"""Calibration helpers: ECE + reliability diagram from voxel-subsampled probabilities.

nnUNetv2 doesn't surface softmax probabilities by default — pass
`--save_probabilities` to prediction (or call `nnUNetPredictor.predict_*` with
save_probabilities=True). This module assumes you have a directory of {.npz, .npy}
probability volumes and matching ground-truth labels.

Voxel-subsampling is essential for 3D: a 256³ volume has ~17M voxels per channel.
We sample ~1M voxels per case, stratified by ground-truth label to keep rare
classes representable.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

try:
    from torchmetrics.classification import MulticlassCalibrationError
except ImportError:
    MulticlassCalibrationError = None  # type: ignore


def stratified_voxel_sample(
    probs: np.ndarray,        # (C, D, H, W)
    labels: np.ndarray,       # (D, H, W) int
    per_class: int = 50_000,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (probs_NxC, labels_N) subsampled at per_class per class."""
    rng = rng or np.random.default_rng(0)
    n_classes = probs.shape[0]
    chosen_probs, chosen_labels = [], []
    flat_probs = probs.reshape(n_classes, -1).T  # (V, C)
    flat_labels = labels.ravel()
    for c in range(n_classes):
        idx = np.where(flat_labels == c)[0]
        if len(idx) == 0:
            continue
        sel = rng.choice(idx, size=min(per_class, len(idx)), replace=False)
        chosen_probs.append(flat_probs[sel])
        chosen_labels.append(flat_labels[sel])
    return np.concatenate(chosen_probs), np.concatenate(chosen_labels)


def expected_calibration_error(
    probs: np.ndarray,        # (N, C)
    labels: np.ndarray,       # (N,)
    n_bins: int = 15,
) -> float:
    """Sklearn-free ECE for multiclass.  Uses top-1 confidence."""
    confidences = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    accuracies = (predictions == labels).astype(float)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(labels)
    for lo, hi in zip(bins[:-1], bins[1:], strict=True):
        mask = (confidences > lo) & (confidences <= hi) if hi < 1.0 else \
               (confidences > lo) & (confidences <= hi + 1e-9)
        if not mask.any():
            continue
        bin_acc = accuracies[mask].mean()
        bin_conf = confidences[mask].mean()
        ece += (mask.sum() / n) * abs(bin_acc - bin_conf)
    return float(ece)
