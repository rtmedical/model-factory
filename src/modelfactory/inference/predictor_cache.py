"""LRU cache of warm nnUNetPredictor instances.

The QA web API keeps up to N predictors resident in VRAM. First click on a
model pays the disk-load cost (~3-5 s for a ResEncL plan on a 3g.40gb MIG
slice); subsequent clicks reuse the cached predictor and skip everything but
the forward pass.

Eviction is strict LRU. The cache is process-local — the FastAPI app runs
single-worker because the underlying CUDA context cannot be shared across
processes anyway.
"""

from __future__ import annotations

import logging
import threading
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ModelKey:
    """Identifies one nnUNet model + which folds to use."""

    model_dir: Path  # …/Dataset###_X/<trainer>__<plans>__<config>
    folds: tuple[int, ...]
    checkpoint_name: str = "checkpoint_best.pth"

    def as_id(self) -> str:
        return (
            f"{self.model_dir.parent.name}/{self.model_dir.name}"
            f"::folds={','.join(str(f) for f in self.folds)}"
            f"::ckpt={self.checkpoint_name}"
        )


class PredictorCache:
    """Process-wide LRU cache. Thread-safe (one global lock — predictor.predict
    is the bottleneck, not the lookup)."""

    def __init__(self, max_size: int = 3) -> None:
        self.max_size = max_size
        self._lock = threading.Lock()
        # ordered: most-recently-used at the END
        self._cache: OrderedDict[ModelKey, object] = OrderedDict()

    def get(self, key: ModelKey):
        """Return a cached `nnUNetPredictor` or build, cache, and return one."""
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                return self._cache[key]
            predictor = self._build(key)
            self._cache[key] = predictor
            while len(self._cache) > self.max_size:
                evicted_key, evicted = self._cache.popitem(last=False)
                logger.info("evicting predictor %s", evicted_key.as_id())
                # Best-effort VRAM release. Predictor holds .network on GPU.
                try:
                    network = getattr(evicted, "network", None)
                    if network is not None:
                        network.cpu()
                    del evicted
                    import torch  # local: torch lives in trainer image
                    torch.cuda.empty_cache()
                except Exception as exc:  # noqa: BLE001
                    logger.warning("eviction cleanup failed for %s: %s",
                                   evicted_key.as_id(), exc)
            return predictor

    def loaded(self) -> list[str]:
        with self._lock:
            return [k.as_id() for k in self._cache]

    def _build(self, key: ModelKey):
        # Deferred imports — keep this module importable in the orchestrator
        # Python (no torch).
        from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

        from modelfactory.inference.run import PREDICTOR_FLAGS

        logger.info("loading predictor for %s", key.as_id())
        # Best-accuracy defaults for QA review (not the production hot path):
        #   - tile_step_size=0.25 — ~2-3x more sliding-window tiles than the
        #     0.5 default; visibly smoother boundaries on small structures
        #     (e.g. D056 brainstem, optic chiasm in HN datasets). Trade
        #     ~30-60s/case on whole-H100 GPU 0 for sharper masks.
        #   - use_mirroring=True — TTA across nnUNet's allowed mirror axes
        #     (+1-3% dice on most CT/MR cohorts).
        #   - use_gaussian=True — Gaussian-weighted tile blending.
        #   - perform_everything_on_device=True — logits stay on GPU (GPU 0
        #     is whole H100, no MIG slice; VRAM is not the bottleneck).
        # The flags live in inference.run.PREDICTOR_FLAGS so the QA API can
        # surface them in PredictionStatus.postprocessing without drift.
        predictor = nnUNetPredictor(
            tile_step_size=PREDICTOR_FLAGS["tile_step_size"],
            use_gaussian=PREDICTOR_FLAGS["use_gaussian"],
            use_mirroring=PREDICTOR_FLAGS["use_mirroring"],
            perform_everything_on_device=PREDICTOR_FLAGS["perform_everything_on_device"],
            verbose=False,
            verbose_preprocessing=False,
            allow_tqdm=False,
        )
        predictor.initialize_from_trained_model_folder(
            str(key.model_dir),
            use_folds=key.folds,
            checkpoint_name=key.checkpoint_name,
        )
        _log_predictor_diagnostics(key, predictor)
        return predictor


def _log_predictor_diagnostics(key: ModelKey, predictor) -> None:
    """One INFO line summarising the predictor's plans/dataset config.

    Makes "MR works, CT doesn't" debugging readable from pod logs without
    digging into plans.json by hand. Kept defensive — anything we can't
    read we print as `?` rather than tanking the predictor build.
    """
    try:
        cm = predictor.configuration_manager
        ds_json = predictor.dataset_json or {}
        channel_names = ds_json.get("channel_names") or ds_json.get("modality") or {}
        n_channels = len(channel_names) if channel_names else "?"
        norm = getattr(cm, "normalization_schemes", None) or "?"
        spacing = getattr(cm, "spacing", None) or "?"
        patch = getattr(cm, "patch_size", None) or "?"
        # dataset_fingerprint.json is loaded next to plans.json by
        # initialize_from_trained_model_folder; presence on disk is the
        # cheapest tell for "CTNormalization will have its percentiles".
        fp_inside = (key.model_dir / "dataset_fingerprint.json").is_file()
        fp_legacy = (key.model_dir.parent / "dataset_fingerprint.json").is_file()
        logger.info(
            "predictor ready %s | channels=%s | norm=%s | spacing=%s | "
            "patch=%s | fingerprint=%s",
            key.as_id(),
            n_channels,
            norm,
            spacing,
            patch,
            "inside" if fp_inside else ("legacy" if fp_legacy else "MISSING"),
        )
    except Exception as exc:  # noqa: BLE001 — diagnostics must never break inference
        logger.warning("could not log predictor diagnostics for %s: %s",
                       key.as_id(), exc)
