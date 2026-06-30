"""nnUNetv2 trainer subclass that pipes metrics + artifacts to MLflow.

Discovery: nnUNetv2's `recursive_find_python_class` walks
`<site-packages>/nnunetv2/training/nnUNetTrainer/` — it does NOT honour
Python entry-point groups. We install a thin shim file at image build time
(see images/nnunet-trainer/Dockerfile and services/qa-viewer/Dockerfile)
that imports this class into the variants/factory/ package so the walker
picks it up.

mlflow is imported lazily inside the `on_*` methods so this module is
importable in environments without mlflow (e.g. the qa-viewer container
which only needs `build_network_architecture` for inference).

Reads from env:
    MLFLOW_TRACKING_URI         e.g. http://mlflow.model-factory.svc:5000  OR  file:///factory/mlflow
    MLFLOW_S3_ENDPOINT_URL      MinIO endpoint (only if http tracking URI)
    AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY
    MFACTORY_EXPERIMENT         experiment name (defaults to dataset__configuration)
    MFACTORY_RUN_NAME           optional override for the run name
    MFACTORY_PARENT_RUN_ID      if set, this fold's run is nested under it (for 5-fold ensembles)
    MFACTORY_METRICS_JSONL      if "1"/"true", also append a JSONL line per epoch
                                to <output_folder>/metrics.jsonl — this is the
                                primary surface Claude agents read when
                                babysitting runs (tail-able, jq-able, no
                                MLflow client required).

The trainer never owns mlflow.start_run() at the parent level — that's the job
of the orchestrator (CLI command `modelfactory train submit`) so all 5 folds
nest under one parent.
"""

from __future__ import annotations

import datetime as _dt
import json
import math
import os
import time
from pathlib import Path

import torch

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer


class nnUNetTrainerMLflow(nnUNetTrainer):
    """nnUNetv2 trainer that logs to MLflow on every epoch and at end-of-training."""

    def on_train_start(self):
        super().on_train_start()

        import mlflow
        from mlflow.tracking import MlflowClient

        tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")
        if not tracking_uri:
            self.print_to_log_file(
                "[MLflow] MLFLOW_TRACKING_URI not set — falling back to local file://mlruns"
            )
        else:
            mlflow.set_tracking_uri(tracking_uri)

        experiment = os.environ.get(
            "MFACTORY_EXPERIMENT", f"{self.plans_manager.dataset_name}__{self.configuration_name}"
        )
        mlflow.set_experiment(experiment)

        parent_run_id = os.environ.get("MFACTORY_PARENT_RUN_ID") or None
        run_name = os.environ.get(
            "MFACTORY_RUN_NAME",
            f"{self.plans_manager.dataset_name}__{self.configuration_name}__fold{self.fold}",
        )

        self._mlflow_run = mlflow.start_run(
            run_name=run_name,
            nested=parent_run_id is not None,
            parent_run_id=parent_run_id,
            tags={
                "dataset": self.plans_manager.dataset_name,
                "configuration": self.configuration_name,
                "fold": str(self.fold),
                "trainer": type(self).__name__,
                "factory.version": os.environ.get("MFACTORY_VERSION", "0.1.0"),
            },
        )

        # Parent-run mirror: `mlflow.log_*` is pinned to the child run we just
        # started, but we also want per-fold metrics + best-dice tags visible
        # at the campaign (parent) level so the MLflow run list shows useful
        # columns without expanding nested children. Use the explicit
        # MlflowClient to address the parent by ID.
        self._parent_run_id = parent_run_id
        self._client = MlflowClient() if parent_run_id else None
        try:
            self._parent_mirror_every = max(
                1, int(os.environ.get("MFACTORY_PARENT_MIRROR_EVERY", "5"))
            )
        except ValueError:
            self._parent_mirror_every = 5

        # Params: the plans dict is large; flatten the architecture-relevant bits
        arch = self.plans_manager.plans.get("configurations", {}).get(
            self.configuration_name, {}
        )
        flat = {
            "patch_size": arch.get("patch_size"),
            "batch_size": arch.get("batch_size"),
            "spacing": arch.get("spacing"),
            "preprocessor_name": arch.get("preprocessor_name"),
            "num_epochs": self.num_epochs,
            "initial_lr": self.initial_lr,
            "weight_decay": self.weight_decay,
            "fold": self.fold,
        }
        mlflow.log_params({k: json.dumps(v) if not isinstance(v, (int, float, str)) else v
                           for k, v in flat.items() if v is not None})

        self.print_to_log_file(f"[MLflow] run_id={self._mlflow_run.info.run_id}")

        # Agent-readable JSONL sidecar (opt-in)
        self._jsonl_enabled = os.environ.get("MFACTORY_METRICS_JSONL", "").lower() in {"1", "true", "yes"}
        self._jsonl_path = Path(self.output_folder) / "metrics.jsonl"
        if self._jsonl_enabled:
            self._jsonl_path.parent.mkdir(parents=True, exist_ok=True)
            self._jsonl_append({
                "event": "run_start",
                "run_id": self._mlflow_run.info.run_id,
                "dataset": self.plans_manager.dataset_name,
                "configuration": self.configuration_name,
                "fold": self.fold,
                "num_epochs": self.num_epochs,
            })

    def _jsonl_append(self, record: dict) -> None:
        """Append one JSON record to the metrics sidecar. Never raises."""
        if not getattr(self, "_jsonl_enabled", False):
            return
        record = {"ts": _dt.datetime.utcnow().isoformat(timespec="seconds") + "Z", **record}
        try:
            with self._jsonl_path.open("a") as f:
                f.write(json.dumps(record) + "\n")
        except OSError as e:
            # Don't let metrics I/O kill the trainer
            self.print_to_log_file(f"[JSONL] append failed: {e}")

    def _per_class_dice(self) -> dict[str, float] | None:
        """Map nnUNet's per-class dice array to label names, if available."""
        log = self.logger.my_fantastic_logging
        try:
            per_class = log["dice_per_class_or_region"][-1]
        except (KeyError, IndexError):
            return None
        return self._row_to_per_class(per_class)

    def _row_to_per_class(self, per_class) -> dict[str, float] | None:
        """Map one per-class dice row (one epoch's slice) to {name: value}."""
        # foreground_labels is the source of truth for ordering inside nnUNet
        fg_labels = list(getattr(self.label_manager, "foreground_labels", []))
        labels_dict = (self.dataset_json or {}).get("labels", {})
        # A label value may be an int (softmax) OR a list/tuple (region-based /
        # the partial-label generalists' region-format dataset.json, e.g.
        # "Organ": [1]). Take the first id for region values so int() doesn't
        # choke on a list. Without this, on_validation_epoch_end raises
        # TypeError mid-training and the whole trial dies.
        def _label_id(v):
            return int(v[0]) if isinstance(v, (list, tuple)) else int(v)
        int_to_name = {_label_id(v): k for k, v in labels_dict.items() if k != "background"}
        out: dict[str, float] = {}
        for i, label_val in enumerate(fg_labels):
            if i >= len(per_class):
                break
            name = int_to_name.get(int(label_val), f"class_{label_val}")
            try:
                out[name] = float(per_class[i])
            except (TypeError, ValueError):
                continue
        return out or None

    def _mirror_to_parent(self, metrics: dict[str, float]) -> None:
        """Best-effort: copy per-fold metrics to the parent run via MlflowClient.

        Throttled by MFACTORY_PARENT_MIRROR_EVERY (default 5 validation epochs).
        Errors are swallowed — parent mirroring must never break training.
        """
        if self._client is None or self._parent_run_id is None:
            return
        if (self.current_epoch % self._parent_mirror_every) != 0:
            return
        for key, value in metrics.items():
            try:
                self._client.log_metric(
                    self._parent_run_id, key, float(value), step=self.current_epoch
                )
            except Exception as e:
                self.print_to_log_file(f"[MLflow] parent mirror {key} failed: {e}")
                # one failure is enough; don't spam logs for every metric
                return

    def on_epoch_start(self):
        super().on_epoch_start()
        # Wall-clock anchor for epoch_time_s, measured inside on_train_epoch_end
        # (validation hasn't run yet then, but training dominates and this avoids
        # the on_epoch_end ordering issue that previously produced negative
        # epoch_time_s in metrics.jsonl).
        self._epoch_t0 = time.monotonic()

    def on_train_epoch_end(self, train_outputs):
        super().on_train_epoch_end(train_outputs)
        import mlflow
        log = self.logger.my_fantastic_logging
        try:
            train_loss = float(log["train_losses"][-1])
            lr = float(log["lrs"][-1])
            gpu_mem_mb = torch.cuda.max_memory_allocated() / 2**20
            t0 = getattr(self, "_epoch_t0", None)
            epoch_time_s = (time.monotonic() - t0) if t0 is not None else 0.0
            mlflow.log_metrics(
                {
                    "train_loss": train_loss,
                    "lr": lr,
                    "gpu_mem_mb": gpu_mem_mb,
                    "epoch_time_s": epoch_time_s,
                },
                step=self.current_epoch,
            )
            self._jsonl_append({
                "phase": "train",
                "epoch": self.current_epoch,
                "train_loss": train_loss,
                "lr": lr,
                "gpu_mem_mb": gpu_mem_mb,
                "epoch_time_s": epoch_time_s,
            })
        except (KeyError, IndexError, ValueError) as e:
            self.print_to_log_file(f"[MLflow] train metric log failed: {e}")

    def on_validation_epoch_end(self, val_outputs):
        super().on_validation_epoch_end(val_outputs)
        import mlflow
        log = self.logger.my_fantastic_logging
        try:
            val_loss = float(log["val_losses"][-1])
            mean_fg_dice = float(log["mean_fg_dice"][-1])
            ema_fg_dice = float(log["ema_fg_dice"][-1])
            mlflow.log_metrics(
                {"val_loss": val_loss, "mean_fg_dice": mean_fg_dice, "ema_fg_dice": ema_fg_dice},
                step=self.current_epoch,
            )
            per_class = self._per_class_dice()
            record: dict = {
                "phase": "val",
                "epoch": self.current_epoch,
                "val_loss": val_loss,
                "mean_fg_dice": mean_fg_dice,
                "ema_fg_dice": ema_fg_dice,
            }
            parent_metrics: dict[str, float] = {
                f"fold{self.fold}/val_loss": val_loss,
                f"fold{self.fold}/mean_fg_dice": mean_fg_dice,
            }
            if per_class:
                record["per_class_dice"] = per_class
                # Also push per-class to MLflow so the file-store / UI shows them
                mlflow.log_metrics(
                    {f"dice/{k}": v for k, v in per_class.items()}, step=self.current_epoch
                )
                for k, v in per_class.items():
                    parent_metrics[f"fold{self.fold}/dice/{k}"] = v
            self._mirror_to_parent(parent_metrics)
            self._jsonl_append(record)

            if math.isnan(val_loss):
                # NaN sentinel — Loki/Promtail will pick this line up; matches the
                # factory.jobs alert rule that scans for "NNUNET_NAN_LOSS".
                self.print_to_log_file(
                    f"NNUNET_NAN_LOSS dataset={self.plans_manager.dataset_name} "
                    f"configuration={self.configuration_name} fold={self.fold} "
                    f"epoch={self.current_epoch}"
                )
                mlflow.set_tag("nan_loss", "true")
                self._jsonl_append({"event": "nan_loss", "epoch": self.current_epoch})
        except (KeyError, IndexError, ValueError) as e:
            self.print_to_log_file(f"[MLflow] val metric log failed: {e}")

    def on_train_end(self):
        super().on_train_end()
        import mlflow
        self._jsonl_append({"event": "run_end", "epoch": self.current_epoch})
        try:
            self._write_parent_summary_tags()
        except Exception as e:
            self.print_to_log_file(f"[MLflow] parent summary tags failed: {e}")
        try:
            output = Path(self.output_folder)
            # progress.png, training_log_*.txt, debug.json, validation/, checkpoint_final.pth
            # Excludes checkpoint_latest.pth (interim, large, not load-bearing)
            for path in output.iterdir():
                if path.name == "checkpoint_latest.pth":
                    continue
                if path.is_file():
                    mlflow.log_artifact(str(path), artifact_path=f"fold_{self.fold}")
                elif path.is_dir() and path.name in {"validation"}:
                    mlflow.log_artifacts(str(path), artifact_path=f"fold_{self.fold}/{path.name}")

            # Also log the dataset/plans/fingerprint at experiment level (cheap, deduplicated)
            preprocessed = Path(os.environ["nnUNet_preprocessed"]) / self.plans_manager.dataset_name
            for fname in ("dataset.json", "dataset_fingerprint.json", "splits_final.json"):
                src = preprocessed / fname
                if src.exists():
                    mlflow.log_artifact(str(src), artifact_path="dataset_meta")
        finally:
            mlflow.end_run()

    def _write_parent_summary_tags(self) -> None:
        """At end-of-training, set `best/dice/<struct>/fold{N}` tags on the
        parent campaign run so the run list shows per-structure scores as
        sortable columns without expanding the nested children.
        """
        if self._client is None or self._parent_run_id is None:
            return
        log = self.logger.my_fantastic_logging
        per_class_history = log.get("dice_per_class_or_region") or []
        mean_fg_history = log.get("mean_fg_dice") or []
        if not per_class_history:
            return

        # Argmax per class across the full epoch history.
        best_per_class: dict[str, float] = {}
        for row in per_class_history:
            row_map = self._row_to_per_class(row)
            if not row_map:
                continue
            for name, v in row_map.items():
                prev = best_per_class.get(name)
                if prev is None or v > prev:
                    best_per_class[name] = v

        final_row = self._row_to_per_class(per_class_history[-1]) or {}
        try:
            best_mean = float(max(mean_fg_history)) if mean_fg_history else None
            final_mean = float(mean_fg_history[-1]) if mean_fg_history else None
        except (TypeError, ValueError):
            best_mean = final_mean = None

        fold = self.fold
        try:
            for name, v in best_per_class.items():
                self._client.set_tag(
                    self._parent_run_id, f"best/dice/{name}/fold{fold}", f"{v:.4f}"
                )
            for name, v in final_row.items():
                self._client.set_tag(
                    self._parent_run_id, f"final/dice/{name}/fold{fold}", f"{v:.4f}"
                )
            if best_mean is not None:
                self._client.set_tag(
                    self._parent_run_id, f"best/mean_fg_dice/fold{fold}", f"{best_mean:.4f}"
                )
            if final_mean is not None:
                self._client.set_tag(
                    self._parent_run_id, f"final/mean_fg_dice/fold{fold}", f"{final_mean:.4f}"
                )
        except Exception as e:
            self.print_to_log_file(f"[MLflow] set_tag on parent failed: {e}")


# ──────────────────────────────────────────────────────────────────────────
# Small-structures trainer — for datasets where dice collapses to 0 because
# the foreground is too small or too sparse for the default loss + sampler.
# Used by: D036 LUNA16 nodules, D054 PDDCA optic nerves, D084 SegRap lenses,
# and similar tiny-target datasets.
#
# Two changes vs the parent:
#   1. oversample_foreground_percent: 0.33 -> 0.95
#      Forces 95% of patches to be centred on a foreground voxel. The
#      default 33% means a 0.0003% foreground dataset (e.g. optic nerves)
#      effectively never sees foreground in training patches.
#   2. Loss: symmetric DC + CE  ->  asymmetric Tversky (FN-weighted) + CE
#      Tversky(alpha, beta) = TP / (TP + alpha*FP + beta*FN)
#      With alpha=0.3 < beta=0.7, false negatives cost 2.3x more than false
#      positives -> gradients push the network to *find* foreground rather
#      than playing it safe with background everywhere.
# ──────────────────────────────────────────────────────────────────────────


class MemoryEfficientSoftTverskyLoss(torch.nn.Module):
    """Tversky-formulated soft Dice with asymmetric FN/FP weighting."""

    def __init__(
        self,
        apply_nonlin=None,
        batch_dice: bool = False,
        do_bg: bool = True,
        smooth: float = 1.0,
        ddp: bool = True,
        alpha: float = 0.3,
        beta: float = 0.7,
    ):
        super().__init__()
        self.do_bg = do_bg
        self.batch_dice = batch_dice
        self.apply_nonlin = apply_nonlin
        self.smooth = smooth
        self.ddp = ddp
        self.alpha = alpha
        self.beta = beta

    def forward(self, x, y, loss_mask=None):
        if self.apply_nonlin is not None:
            x = self.apply_nonlin(x)
        axes = tuple(range(2, x.ndim))

        with torch.no_grad():
            if x.ndim != y.ndim:
                y = y.view((y.shape[0], 1, *y.shape[1:]))
            if x.shape == y.shape:
                y_onehot = y.to(torch.float32)
            else:
                y_onehot = torch.zeros(x.shape, device=x.device, dtype=torch.float32)
                y_onehot.scatter_(1, y.long(), 1)
            if not self.do_bg:
                y_onehot = y_onehot[:, 1:]
            if loss_mask is None:
                sum_gt = y_onehot.sum(axes, dtype=torch.float32)
            else:
                sum_gt = (y_onehot * loss_mask).sum(axes, dtype=torch.float32)

        if not self.do_bg:
            x = x[:, 1:]

        if loss_mask is None:
            intersect = (x * y_onehot).sum(axes, dtype=torch.float32)
            sum_pred = x.sum(axes, dtype=torch.float32)
        else:
            intersect = (x * y_onehot * loss_mask).sum(axes, dtype=torch.float32)
            sum_pred = (x * loss_mask).sum(axes, dtype=torch.float32)

        if self.batch_dice:
            if self.ddp:
                from nnunetv2.utilities.ddp_allgather import AllGatherGrad
                intersect = AllGatherGrad.apply(intersect).sum(0, dtype=torch.float32)
                sum_pred = AllGatherGrad.apply(sum_pred).sum(0, dtype=torch.float32)
                sum_gt = AllGatherGrad.apply(sum_gt).sum(0, dtype=torch.float32)
            intersect = intersect.sum(0, dtype=torch.float32)
            sum_pred = sum_pred.sum(0, dtype=torch.float32)
            sum_gt = sum_gt.sum(0, dtype=torch.float32)

        fp = sum_pred - intersect
        fn = sum_gt - intersect
        tversky = (intersect + self.smooth) / (
            intersect + self.alpha * fp + self.beta * fn + self.smooth
        ).clamp_min(1e-8)
        return -tversky.mean()


class nnUNetTrainerSmallStructuresMLflow(nnUNetTrainerMLflow):
    """nnUNetTrainerMLflow tuned for tiny / sparse foreground structures.

    Drop-in replacement: same MLflow + JSONL metric piping, but with the
    foreground-sampling and loss tweaks that prevent the "all-background"
    collapse mode we hit on D036 LUNA16 nodules and D054 PDDCA optic nerves.
    """

    OVERSAMPLE_FG = 0.95
    TVERSKY_ALPHA = 0.3
    TVERSKY_BETA = 0.7

    def __init__(self, plans, configuration, fold, dataset_json, device=torch.device("cuda")):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.oversample_foreground_percent = self.OVERSAMPLE_FG
        self.print_to_log_file(
            f"[small-structures] oversample_foreground_percent={self.oversample_foreground_percent}"
        )
        self.print_to_log_file(
            f"[small-structures] loss: Tversky(alpha={self.TVERSKY_ALPHA}, "
            f"beta={self.TVERSKY_BETA}) + CE"
        )

    def _build_loss(self):
        import numpy as np
        from nnunetv2.training.loss.compound_losses import DC_and_CE_loss
        from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper

        if self.label_manager.has_regions:
            return super()._build_loss()

        loss = DC_and_CE_loss(
            {
                "batch_dice": self.configuration_manager.batch_dice,
                "smooth": 1e-5,
                "do_bg": False,
                "ddp": self.is_ddp,
                "alpha": self.TVERSKY_ALPHA,
                "beta": self.TVERSKY_BETA,
            },
            {},
            weight_ce=1,
            weight_dice=2,
            ignore_label=self.label_manager.ignore_label,
            dice_class=MemoryEfficientSoftTverskyLoss,
        )

        if self._do_i_compile():
            loss.dc = torch.compile(loss.dc)

        if self.enable_deep_supervision:
            deep_supervision_scales = self._get_deep_supervision_scales()
            weights = np.array([1 / (2 ** i) for i in range(len(deep_supervision_scales))])
            if self.is_ddp and not self._do_i_compile():
                weights[-1] = 1e-6
            else:
                weights[-1] = 0
            weights = weights / weights.sum()
            loss = DeepSupervisionWrapper(loss, weights)

        return loss


class nnUNetTrainerPartialLabelMLflow(nnUNetTrainerMLflow):
    """Partial-label single generalist over a NON-co-occurring organ union.

    Trains ONE region-based (sigmoid-per-organ) model on a cohort where each
    case annotates only SOME of the declared organs (e.g. a pelvis cohort where
    male and female OARs never appear on the same scan). The converter
    (convert.py, spec.partial_label=True) writes a region-based dataset.json and
    embeds a per-case annotation map under dataset_json["partial_label_annotations"].

    The core mechanism: for every training/validation sample, the loss is masked
    to that case's annotated organ channels only. An organ NOT contoured on a
    case is painted background in the label file (its true location is unknown),
    which would otherwise teach the model "no organ-C here" — a false negative.
    We prevent that by substituting the network output with the (detached) target
    on un-annotated channels, so those channels contribute ~0 loss and ZERO
    gradient for that sample. Organs are thus learned only from the cases that
    actually annotated them, while the model still predicts all organs at once.

    ── VALIDATE IN CONTAINER BEFORE THE FULL WAVE ──────────────────────────────
    This trainer cannot be exercised on the host (no torch / nnunetv2). Run the
    60-case prototype in docs/runbooks/partial_label_generalist.md and confirm:
      1. dataset_json["partial_label_annotations"] is present (it is, post-
         preprocess — nnUNet copies dataset.json to the preprocessed dir).
      2. batch["keys"] is populated in train_step/validation_step for the pinned
         nnUNetv2 (the dataloader contract). If absent, masking degrades to FULL
         supervision with a loud warning — correctness depends on this, so the
         prototype MUST confirm keys flow through.
      3. region-based output/target are [B, R, *spatial] (a list across scales
         under deep supervision); _apply_partial_mask handles both.
      4. A case missing organ C produces ZERO gradient on channel C (assert in
         the runbook) and C's val Dice is computed only over cases that have it.
    """

    def __init__(self, plans, configuration, fold, dataset_json, device=torch.device("cuda")):
        super().__init__(plans, configuration, fold, dataset_json, device)
        pla = (dataset_json or {}).get("partial_label_annotations")
        if not pla:
            raise RuntimeError(
                "nnUNetTrainerPartialLabelMLflow requires dataset.json to carry "
                "'partial_label_annotations' (produced by convert.py with "
                "spec.partial_label=True). None found — was this dataset converted "
                "in partial-label mode?"
            )
        # nnUNet trains this as SOFTMAX over C = n_organs + 1 channels (channel 0
        # = background, channel k = organ with label id k). (Single-label "regions"
        # in dataset.json are NOT treated as regions by nnUNet — has_regions needs
        # len>1 — so the network is a softmax head, not per-organ sigmoid.) So the
        # annotated set is stored as SOFTMAX CHANNEL INDICES == label ids (1..N).
        self._label_ids: dict[str, int] = pla["label_ids"]          # organ -> label id (1..N)
        self._n_organs: int = len(pla["structures"])
        # case_id -> set of annotated softmax channel indices (label ids, 1..N)
        self._annotated: dict[str, set[int]] = {
            cid: {self._label_ids[name] for name in names}
            for cid, names in pla["per_case"].items()
        }
        self._coverage: dict[str, int] = pla.get("coverage", {})
        self._warned_no_keys = False

    def on_train_start(self):
        super().on_train_start()
        self.print_to_log_file(
            f"[partial-label] softmax generalist over {self._n_organs} organs (+bg); "
            f"per-case MARGINAL masking active. coverage(cases/organ)={self._coverage}"
        )

    # ── per-case channel masking ────────────────────────────────────────────

    def _allowed_mask(self, keys, n_channels: int) -> "torch.Tensor | None":
        """Bool tensor [B, C] (C = n_organs+1): True for the softmax channels this
        case is ALLOWED to predict — background (0) plus its annotated organ labels.
        Un-annotated organ channels are False and get marginalised out of the loss.

        Returns None if `keys` is unusable (older nnUNet not passing keys) so the
        caller can fall back to full supervision with a loud warning.
        """
        # `keys` may be a list OR a numpy array of case ids — avoid `if not keys:`
        # (ambiguous truth value on arrays).
        if keys is None or len(keys) == 0:
            return None
        mask = torch.zeros((len(keys), n_channels), dtype=torch.bool)
        mask[:, 0] = True  # background channel is always supervised
        for b, k in enumerate(keys):
            ann = self._annotated.get(str(k))
            if ann is None:
                mask[b, :] = True  # unknown case — supervise all, to be safe
            else:
                for ch in ann:           # ch = softmax channel index == label id
                    if 0 <= ch < n_channels:
                        mask[b, ch] = True
        return mask

    @staticmethod
    def _marginal_mask(output, allowed: "torch.Tensor"):
        """Marginal masking for SOFTMAX partial labels: set the logits of
        un-annotated organ channels to -1e9 so softmax gives them ~0 probability.
        The model is then neither rewarded nor penalised for predicting an organ
        the case never annotated (its CE mass redistributes over allowed channels;
        its Dice channel is empty-vs-empty ≈ no loss, no gradient). Handles the
        deep-supervision list and the single-tensor case. `allowed` is [B, C] bool."""
        NEG = -1e4   # fp16-safe (autocast); softmax(-1e4) ≈ 0 without overflow to -inf
        def _one(o):
            a = allowed.to(o.device)
            view = (o.shape[0], o.shape[1]) + (1,) * (o.ndim - 2)
            return o.masked_fill(~a.view(view), NEG)
        if isinstance(output, (list, tuple)):
            return [_one(o) for o in output]
        return _one(output)

    def _prep_batch(self, batch):
        """Move data/target to device and build the per-case allowed-channel mask
        from batch['keys']. Returns (data, target, allowed_or_None)."""
        data = batch["data"]
        target = batch["target"]
        keys = batch.get("keys")
        data = data.to(self.device, non_blocking=True)
        if isinstance(target, (list, tuple)):
            target = [t.to(self.device, non_blocking=True) for t in target]
        else:
            target = target.to(self.device, non_blocking=True)
        return data, target, keys

    def _warn_no_keys_once(self):
        if not self._warned_no_keys:
            self.print_to_log_file(
                "[partial-label] WARNING: batch has no 'keys' — marginal masking "
                "DISABLED, training as if every organ were annotated on every case. "
                "This silently corrupts the partial-label objective. Fix the "
                "dataloader to pass keys before trusting results."
            )
            self._warned_no_keys = True

    def train_step(self, batch: dict) -> dict:
        # Mirror nnUNetTrainer.train_step but marginalise un-annotated organ
        # channels (softmax) before the loss.
        from torch import autocast, nn
        from nnunetv2.utilities.helpers import dummy_context

        data, target, keys = self._prep_batch(batch)
        self.optimizer.zero_grad(set_to_none=True)
        with autocast(self.device.type, enabled=True) if self.device.type == "cuda" else dummy_context():
            output = self.network(data)
            o0 = output[0] if isinstance(output, (list, tuple)) else output
            allowed = self._allowed_mask(keys, o0.shape[1])
            if allowed is None:
                self._warn_no_keys_once()
            else:
                output = self._marginal_mask(output, allowed)
            l = self.loss(output, target)
        if self.grad_scaler is not None:
            self.grad_scaler.scale(l).backward()
            self.grad_scaler.unscale_(self.optimizer)
            nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            l.backward()
            nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.optimizer.step()
        return {"loss": l.detach().cpu().numpy()}

    def validation_step(self, batch: dict) -> dict:
        from torch import autocast
        from nnunetv2.utilities.helpers import dummy_context

        data, target, keys = self._prep_batch(batch)
        with autocast(self.device.type, enabled=True) if self.device.type == "cuda" else dummy_context():
            output = self.network(data)
            del data
            o0 = output[0] if isinstance(output, (list, tuple)) else output
            allowed = self._allowed_mask(keys, o0.shape[1])
            if allowed is None:
                self._warn_no_keys_once()
            else:
                output = self._marginal_mask(output, allowed)
            l = self.loss(output, target)
        return self._softmax_val_metrics(output, target, l)

    def _softmax_val_metrics(self, output, target, l):
        """Hard-Dice tp/fp/fn per FOREGROUND class via softmax argmax, matching
        nnUNetTrainer.validation_step's return contract (arrays length n_organs).
        Un-annotated organs are already marginalised (-1e9) so argmax never picks
        them on cases that didn't annotate them → each organ's Dice is computed
        only over the cases that have it."""
        import numpy as np

        out = output[0] if isinstance(output, (list, tuple)) else output
        tgt = target[0] if isinstance(target, (list, tuple)) else target
        C = out.shape[1]
        pred = out.argmax(1)                       # [B, *spatial]
        if tgt.shape[1:] != pred.shape[1:]:        # target is [B,1,*spatial]
            tgt = tgt[:, 0]
        tgt = tgt.long()
        tp = np.zeros(C - 1); fp = np.zeros(C - 1); fn = np.zeros(C - 1)
        for c in range(1, C):                      # foreground classes only
            p = (pred == c); g = (tgt == c)
            tp[c - 1] = float((p & g).sum().item())
            fp[c - 1] = float((p & ~g).sum().item())
            fn[c - 1] = float((~p & g).sum().item())
        return {
            "loss": l.detach().cpu().numpy(),
            "tp_hard": tp, "fp_hard": fp, "fn_hard": fn,
        }


class nnUNetTrainerPartialLabelBalancedMLflow(nnUNetTrainerPartialLabelMLflow):
    """Partial-label generalist tuned to recover THIN classes (e.g. SacralPlex,
    NVB) that the plain generalist under-segments (0.64 vs the 0.84 a dedicated
    specialist gets), WITHOUT the Tversky over-segmentation that collapses dense
    organs on final validation (see memory tversky_collapses_dense_structures).

    The only change vs the parent: raise oversample_foreground_percent from the
    0.33 default to 0.60 so more training patches are centred on a foreground
    voxel. nnUNet picks the centre class uniformly at random among the present
    foreground classes, so a higher fg-oversample gives the rare/thin classes
    proportionally more exposure per epoch. Loss stays DC+CE (NOT Tversky) and
    the per-case marginal masking is inherited unchanged — so dense organs are
    NOT pushed toward over-segmentation. Pair with XL plans (capacity helps the
    many-class generalist; verified D151-XL > D151-ResEncL).
    """

    OVERSAMPLE_FG = 0.60

    def __init__(self, plans, configuration, fold, dataset_json, device=torch.device("cuda")):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.oversample_foreground_percent = self.OVERSAMPLE_FG

    def on_train_start(self):
        super().on_train_start()
        self.print_to_log_file(
            f"[partial-label-balanced] oversample_foreground_percent="
            f"{self.oversample_foreground_percent} (thin-class recovery; DC+CE, no Tversky)"
        )
