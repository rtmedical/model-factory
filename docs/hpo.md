# Ray Tune HPO for nnUNetv2

Status: **proposal / not yet implemented**. This doc captures what is and
isn't tunable in nnUNetv2, how the existing Ray cluster + driver scaffolding
fits, and a concrete plan for a first useful sweep.

Author: Gustavo + Claude session, 2026-05-14.

## TL;DR

- The `factory-ray` cluster already runs Tune trials (`ray_driver.py`)
  but the param space is just `grid_search(dataset_key × fold)` — there is
  no actual hyperparameter optimization yet.
- nnUNetv2's CLI does **not** accept hyperparameter flags. Every knob is
  a class attribute on `nnUNetTrainer`; the only way to change one is to
  subclass the trainer.
- Adding HPO is two small files: one trainer subclass (`nnUNetTrainerHPO`)
  that reads overrides from env vars, and one CLI verb (`modelfactory hpo
  run …`) that wraps `ray_driver.py` with a real Optuna + ASHA setup.
- Trial cost is the binding constraint: at ~3 min/epoch on XL plans,
  ASHA-style short trials (≤80 epochs) with aggressive pruning is the only
  way to do meaningful search on this cluster.

## Current state of the scaffolding

`src/modelfactory/jobs/ray_driver.py`:
- L195–224: `tune.Tuner` is already built, configured with
  `tune.with_resources(_run_one_fold, resources={"cpu": 16, "gpu": 1})`,
  `tune.TuneConfig(metric="mean_fg_dice", mode="max")`, and
  `train.RunConfig(storage_path="/factory/results/.tune")`.
- L195–205: `param_space` is `grid_search` over `(dataset_key, fold)` only.
- L57–102: `_run_one_fold` shells out to `nnUNetv2_train` via subprocess.
- L91–93 comment is the load-bearing fact: **`nnUNetv2_train` does not
  accept `--num_epochs` (or `--lr`, etc.); to change them you must
  subclass `nnUNetTrainer` and set the class attribute before
  `super().__init__()`**.

`src/modelfactory/trainers/mlflow_trainer.py`:
- Logs the hyperparams it sees (L83–97) to MLflow params, but does not
  accept overrides from env vars or config.

Kueue: a `hpo-sweep` priority class is already registered
(`infra/kustomize/factory-priority-classes.yaml`, priority 10 — lowest).
Sweep workloads land in `factory-lq` like everything else but never
preempt `fold-training` (priority 50) or `interactive-eval` (100).

Ray Tune building blocks available in the trainer image (verified on
`factory-ray-head`, Ray 2.40):
- Schedulers: `ASHAScheduler`, `PopulationBasedTraining`,
  `MedianStoppingRule`.
- Searchers: `OptunaSearch` (TPE), `HyperOptSearch`, `HEBOSearch`,
  `BayesOptSearch`, plain RandomSearch / GridSearch.

## What's tunable in nnUNetv2

From the base `nnUNetTrainer.__init__` (path:
`<site-packages>/nnunetv2/training/nnUNetTrainer/nnUNetTrainer.py`).

### Free (just set the class attr before `super().__init__()`)

| Knob | Default | Line | Notes |
|---|---|---|---|
| `initial_lr` | 1e-2 | 152 | Most impactful. Sweep on log scale. |
| `weight_decay` | 3e-5 | 153 | Log scale. |
| `oversample_foreground_percent` | 0.33 | 154 | **Critical for sparse-target tasks** (LUNA16 nodules, optic chiasm). |
| `probabilistic_oversampling` | False | 155 | Toggle. |
| `num_iterations_per_epoch` | 250 | 156 | Lower → cheaper trials. |
| `num_val_iterations_per_epoch` | 50 | 157 | Same. |
| `num_epochs` | 1000 | 158 | Short for sweep, long for winner. |
| `enable_deep_supervision` | True | 160 | Can help or hurt; try off. |

### Hard-coded, need a `_build_*` override to vary

- **Optimizer** (`_build_optimizer`-like, L552–553): SGD,
  `momentum=0.99, nesterov=True`. AdamW is doable but requires
  subclassing the optimizer-build step.
- **Loss** (`_build_loss`, L435–469): `DC_and_CE_loss` (or
  `DC_and_BCE_loss` for region-based). Dice/CE weight ratio is
  hard-coded. Tunable only by overriding `_build_loss`.
- **Augmentation** (`get_training_transforms`, L738–829): rotation
  prob/range, scaling, gamma, mirror, noise, blur, brightness,
  contrast — all hard-coded literals. To vary, copy the whole method
  into a subclass and parameterize. Worth it only if you suspect
  augmentation is hurting (e.g. mirror-prob on lateralized H&N
  structures).

### Architecture (plans-level) knobs — sweep via `-p plans_name`

`patch_size`, `batch_size`, `features_per_stage`, `n_blocks_per_stage`
live in `plans.json`, not the trainer. To vary the architecture, run
`nnUNetv2_plan_and_preprocess` once per planner (`nnUNetPlannerResEncL`
[24 GB], `nnUNetPlannerResEncXL` [40 GB], or default
`ExperimentPlanner`), then sweep across the resulting plans names via
the existing `-p` flag. Already supported.

## Trial cost on this cluster

10 MIG 3g.40gb slices × ~3 min/epoch on XL plans (240–280 s/epoch
verified, pancreas + DeepNuclei runs):

| Trial length | Wall time | 10-slice throughput |
|---|---|---|
| 50 epochs (ASHA early-stop rung) | ~2.5 h | ~95 trials/day |
| 80 epochs (typical sweep budget) | ~4 h | ~60 trials/day |
| 200 epochs (mid-quality bake) | ~10 h | ~24 trials/day |
| 1000 epochs (full default) | ~50 h | ~5 trials/day |

The shape of any sweep we run is set by this table: ASHA-style short
trials are the only way to explore a non-trivial space. Full 1000-epoch
training is for the *winner* of a sweep, never for trial bodies.

## Recommended searcher / scheduler combo

- **Search**: `OptunaSearch` (TPE) — mixed continuous/discrete, no GP
  scaling pain, mature, deterministic with a seed.
- **Scheduler**: `ASHAScheduler(metric="mean_fg_dice", mode="max",
  max_t=80, grace_period=10, reduction_factor=3)` — runs everything
  for 10 epochs, kills the bottom 2/3, runs the rest for 30, kills
  bottom 2/3, etc. ~95% of trials die early.
- **PBT** is tempting (mutates LR / WD live) but needs the trainer to
  support checkpoint + warm-load of hyperparams mid-run, which
  nnUNetv2 doesn't do cleanly. Defer.

## Plumbing options (ranked)

1. **(Recommended)** Subclass `nnUNetTrainerMLflow` → `nnUNetTrainerHPO`
   that reads overrides from env vars:
   `MFACTORY_LR`, `MFACTORY_WD`, `MFACTORY_OS_FG`, `MFACTORY_NUM_EPOCHS`,
   `MFACTORY_NUM_ITERS`, `MFACTORY_NUM_VAL_ITERS`, `MFACTORY_DEEP_SUP`.
   `ray_driver._run_one_fold` already controls env per-trial — just
   inject from the trial `config`. Additive, no monkey-patching of the
   base class. Loss / aug knobs deferred to a v2.
2. Single JSON env var (`MFACTORY_TRAINER_OVERRIDES='{...}'`). Slightly
   more boilerplate to parse but easier to extend. Functionally the
   same as (1); pick whichever feels cleaner.
3. Generate a fresh trainer subclass `.py` per trial via templating —
   avoid. Fragile, hard to test.

## Proposed first sweep — sparse-target rescue

The HN PDDCA Optic run (50 epochs, dice 0 across chiasm + both optic
nerves; killed 2026-05-14) and the LUNA16 nodules run (5 epochs, dice
0; killed same session) are both classic **sparse-target failure
modes**. The default `oversample_foreground_percent = 0.33` is too low
when foreground voxels are <0.1% of the volume — most batches have
*no* foreground at all and the network learns "predict background".

First sweep (target: D011 HN PDDCA Optic):

```python
search_space = {
    "oversample_foreground_percent": tune.choice([0.33, 0.66, 0.90, 1.00]),
    "initial_lr":                    tune.loguniform(1e-4, 3e-2),
    "weight_decay":                  tune.loguniform(1e-6, 1e-3),
    "enable_deep_supervision":       tune.choice([True, False]),
    "num_epochs":                    80,
    "num_iterations_per_epoch":      100,   # cheaper epochs
}
scheduler = ASHAScheduler(metric="mean_fg_dice", mode="max",
                           max_t=80, grace_period=10, reduction_factor=3)
searcher = OptunaSearch(metric="mean_fg_dice", mode="max", seed=42)
num_trials = 30
```

Expected wall-clock: 30 trials × ~80 epochs × ~100 iter × ~0.7 s/iter
÷ 10 slices ≈ **~5–6 hours**.

Output: a `best_config` JSON. Promote the winner to a full 500-epoch
training, register as the production model for D011.

## Proposed rollout

1. Implement `src/modelfactory/trainers/hpo_trainer.py` —
   `nnUNetTrainerHPO(nnUNetTrainerMLflow)` that reads the
   `MFACTORY_*` env vars in `__init__` *before* `super().__init__()`.
   Register it in `pyproject.toml` (`nnunetv2.trainers` entry-point
   group — already wired) and add the
   `nnunetv2/training/nnUNetTrainer/variants/factory/hpo_trainer.py`
   shim in the Dockerfile next to `mlflow_trainer.py` (see Dockerfile
   L129–133 for the existing shim pattern).
2. Implement `src/modelfactory/hpo/runner.py` with a
   `param_space_for(dataset_key)` lookup + a thin Tune.run wrapper
   that calls into `_run_one_fold` from `ray_driver.py` (reuse, don't
   fork). Expose via:
   `modelfactory hpo run --dataset <key> --searcher optuna
   --scheduler asha --num-trials 30 --max-epochs 80
   --priority hpo-sweep`.
3. Reuse the MLflow parent-mirror pattern from
   `nnUNetTrainerMLflow._mirror_to_parent`: every trial's
   `mean_fg_dice` lands as `trial{N}/mean_fg_dice` on a single
   campaign parent, so the run-list view shows the sweep at a glance.
4. Run sweep on D011 Optic. Lock the winner. Train 500 epochs full.
5. Repeat for LUNA16, then for other under-performing datasets as
   needed.

## Open questions (defer)

- Should HPO sweeps live in their own MLflow experiment
  (`Dataset###__hpo`) or share with the main one? Current bias: share,
  flag with `tags.sweep=true`.
- Loss-weight HPO (`DC_and_CE_loss` Dice/CE ratio). Worth a v2 — needs
  `_build_loss` override.
- Augmentation HPO. Same — v2, needs `get_training_transforms` copy.
- PBT once nnUNetv2 grows hyperparam-aware checkpoint loading.
- KubeRay autoscaler for HPO bursts (CLAUDE.md "deferred to v2" — still
  applies; current 10-slice manual capacity is fine for sweep sizes
  ≤30).

## References

- `src/modelfactory/jobs/ray_driver.py` (existing Tune scaffolding)
- `src/modelfactory/trainers/mlflow_trainer.py` (parent-run mirror
  pattern to reuse)
- `infra/kustomize/factory-priority-classes.yaml` (`hpo-sweep` exists)
- nnUNetv2 source: `nnunetv2.training.nnUNetTrainer.nnUNetTrainer`
- Ray 2.40 docs: `ASHAScheduler`, `OptunaSearch`
