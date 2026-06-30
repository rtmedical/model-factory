# Training models

## Submitting a training

```bash
# One or more folds of one dataset (direct Job submission, queued by Kueue)
modelfactory train nnunet --dataset Dataset100_Hippocampus --configuration 3d_fullres --folds all

# Options: --trainer, --priority {interactive-eval|fold-training|hpo-sweep},
#          --num-epochs, --pretrained-weights, --continue, --dry-run
```

Each fold becomes a Kubernetes Job labeled for Kueue (`queue-name`,
`priority-class`). In **MIG mode** the submitter leases a slice from the
`factory-mig-leases` ConfigMap and pins it via `NVIDIA_VISIBLE_DEVICES`; in
**whole-GPU mode** the Job requests `nvidia.com/gpu: 1`. Use `--dry-run` to print
the manifest without submitting.

Multi-dataset / multi-fold **campaigns** fan out across the Ray pool:

```bash
modelfactory campaign run-trio --datasets a,b,c --folds 0,1,2,3,4
```

## Tracking

Every run logs to MLflow (`nnUNetTrainerMLflow`): per-epoch `train_loss`,
`val_loss`, `mean_fg_dice`, `lr`, GPU memory, and epoch time; final artifacts and
the dataset/splits/fingerprint JSON. Browse with `modelfactory runs list` or the
MLflow UI (`kubectl -n model-factory port-forward svc/mlflow 5000:5000`).

> **Judge models on `validation/summary.json` per-class Dice, not the per-epoch
> "pseudo" Dice.** Pseudo-dice is patch-based + EMA and can look great while the
> final, whole-volume Dice is poor.

## Trainer & planner variants

Pick by the structures in the dataset:

| Situation | Trainer | Planner |
|---|---|---|
| Normal OARs (≥ ~0.5% foreground, ≥ 5 voxels thick) | `nnUNetTrainerMLflow` (default) | default ResEnc plans |
| Pixel-sparse but resolved (e.g. lung nodules) | `nnUNetTrainerSmallStructuresMLflow` (95% foreground oversampling + Tversky loss, α=0.3/β=0.7) | default |
| Sub-voxel / thin (optic nerves, lenses, fine vessels) | `nnUNetTrainerSmallStructuresMLflow` | `nnUNetPlannerResEncL_HighRes` (anisotropic in-plane 0.7 mm, preserves axial) |
| One generalist over organs that never co-occur (e.g. male+female pelvic OARs) | `nnUNetTrainerPartialLabelMLflow` (loss masked to annotated channels) | default |

```bash
modelfactory train nnunet --dataset Dataset054_OpticNerves --fold 0 \
  --trainer nnUNetTrainerSmallStructuresMLflow --plans nnUNetResEncUNetLPlans_HighRes
```

**Caution — Tversky/HighRes only when the dataset is 100% thin/sparse.** On
datasets that also contain a dense organ, the FN-weighted Tversky loss
over-segments and can wreck the dense class on *final* validation even while
pseudo-dice looks fine. For anything with a dense structure, use the default
trainer.

## Ensembling & registry

```bash
modelfactory model register-ensemble --dataset Dataset100_Hippocampus --configuration 3d_fullres
modelfactory model promote --name Dataset100_Hippocampus__3d_fullres --version 1 --stage Staging
```

See [`hpo.md`](hpo.md) for Ray Tune hyperparameter sweeps and
[`qa.md`](qa.md) for visual review before promotion.
