"""Ray Tune HPO sweep driver for nnUNetv2 + nnUNetTrainerHPO.

Runs inside a k8s Job pod (rendered from
`infra/kustomize/hpo-driver-job.yaml.j2`) that:

  1. Connects to the running `factory-ray` RayCluster via the Ray Client port.
  2. Builds a `tune.Tuner` with OptunaSearch + ASHAScheduler (defaults) over
     `param_space_for(dataset_key)` × the fixed (dataset_key, fold) cell.
  3. Each trial subprocesses `nnUNetv2_train -tr nnUNetTrainerHPO`, with the
     trial's hyperparameter cell injected as `MFACTORY_*` env vars that
     `nnUNetTrainerHPO.__init__` consumes (see
     `src/modelfactory/trainers/hpo_trainer.py`).
  4. The trial tails `<output_folder>/metrics.jsonl` and calls
     `train.report(mean_fg_dice=…)` per epoch so ASHA can early-stop.
  5. After the sweep finishes the driver prints a single-line JSON to stdout
     with `best_config` + `best_metric` + `n_trials` / `n_errors`. Promotion
     to a full 500-epoch training is a separate step (run the winner config
     via `modelfactory campaign smoke` / `modelfactory train nnunet`).

Invocation (typical):

    python -m modelfactory.hpo.runner \\
        --ray-address ray://factory-ray-head-svc:10001 \\
        --dataset hn_optic_nerves \\
        --fold 0 \\
        --parent-run-id 1a2b3c... \\
        --num-trials 30 --max-epochs 80
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import ray
from ray import train, tune
from ray.tune.schedulers import ASHAScheduler

try:
    from ray.tune.search.optuna import OptunaSearch
except ImportError:
    OptunaSearch = None  # type: ignore[assignment]

from modelfactory.datasets.specs import SPECS


# ─── search-space lookup ────────────────────────────────────────────────────


def param_space_for(dataset_key: str) -> dict:
    """Return the Ray Tune search space for `dataset_key`.

    Default = the sparse-target rescue space from docs/hpo.md L151-158
    (oversample_foreground_percent + lr + weight_decay + deep-supervision).
    Per-dataset overrides slot in here once we learn what each cohort needs.
    """
    return {
        "oversample_foreground_percent": tune.choice([0.33, 0.66, 0.90, 1.00]),
        "initial_lr":                    tune.loguniform(1e-4, 3e-2),
        "weight_decay":                  tune.loguniform(1e-6, 1e-3),
        "enable_deep_supervision":       tune.choice([True, False]),
    }


# ─── trial body ─────────────────────────────────────────────────────────────


def _sidecar_path(dataset_key: str, fold: int, trainer: str, plans: str) -> Path:
    """Locate the metrics.jsonl that nnUNetTrainerMLflow writes per-epoch.

    Mirrors the path construction at ray_driver.py:107-113.
    """
    spec = SPECS[dataset_key]
    return (
        Path(os.environ.get("nnUNet_results", "/factory/results"))
        / spec.folder
        / f"{trainer}__{plans}__3d_fullres"
        / f"fold_{fold}"
        / "metrics.jsonl"
    )


def _run_one_hpo_trial(config: dict) -> dict:
    """Executed on a Ray worker. Subprocesses nnUNetv2_train and reports
    mean_fg_dice per epoch via train.report so ASHA can early-stop.

    The hyperparameter cell from `config` is injected as MFACTORY_* env vars;
    `nnUNetTrainerHPO.__init__` reads them (see hpo_trainer.py).
    """
    spec = SPECS[config["dataset_key"]]
    fold = int(config["fold"])
    trainer = config.get("trainer", "nnUNetTrainerHPO")
    plans = config.get("plans", "nnUNetResEncUNetLPlans")

    env = os.environ.copy()
    # MIG inductor crash workaround — same as ray_driver.py:78
    env["nnUNet_compile"] = "False"
    env["MFACTORY_EXPERIMENT"] = f"{spec.folder}__3d_fullres"
    env["MFACTORY_RUN_NAME"] = (
        f"{spec.folder}__3d_fullres__fold{fold}__trial_{train.get_context().get_trial_id()}"
    )
    env["MFACTORY_METRICS_JSONL"] = "1"
    if config.get("parent_run_id"):
        env["MFACTORY_PARENT_RUN_ID"] = config["parent_run_id"]

    # Hyperparameter cell → MFACTORY_* env (consumed by nnUNetTrainerHPO)
    env["MFACTORY_LR"] = repr(float(config["initial_lr"]))
    env["MFACTORY_WD"] = repr(float(config["weight_decay"]))
    env["MFACTORY_OS_FG"] = repr(float(config["oversample_foreground_percent"]))
    env["MFACTORY_DEEP_SUP"] = "1" if config["enable_deep_supervision"] else "0"
    if "num_epochs" in config:
        env["MFACTORY_NUM_EPOCHS"] = str(int(config["num_epochs"]))
    if "num_iterations_per_epoch" in config:
        env["MFACTORY_NUM_ITERS"] = str(int(config["num_iterations_per_epoch"]))
    if "num_val_iterations_per_epoch" in config:
        env["MFACTORY_NUM_VAL_ITERS"] = str(int(config["num_val_iterations_per_epoch"]))

    cmd = [
        "nnUNetv2_train",
        str(spec.dataset_id), "3d_fullres", str(fold),
        "-p", plans, "-tr", trainer,
    ]
    print(
        f"[hpo-trial] dataset={spec.folder} fold={fold} "
        f"trial={train.get_context().get_trial_id()}",
        flush=True,
    )
    print(
        f"[hpo-trial]   LR={env['MFACTORY_LR']} WD={env['MFACTORY_WD']} "
        f"OS_FG={env['MFACTORY_OS_FG']} DEEP_SUP={env['MFACTORY_DEEP_SUP']}",
        flush=True,
    )
    print(
        f"[hpo-trial]   NVIDIA_VISIBLE_DEVICES={env.get('NVIDIA_VISIBLE_DEVICES','<unset>')}",
        flush=True,
    )

    sidecar = _sidecar_path(config["dataset_key"], fold, trainer, plans)
    proc = subprocess.Popen(cmd, env=env)

    final_dice = float("nan")
    final_val_loss = float("nan")
    seen = 0
    poll_interval_s = 2.0
    try:
        while True:
            rc = proc.poll()
            if sidecar.is_file():
                try:
                    lines = sidecar.read_text().splitlines()
                except OSError:
                    lines = []
                new_lines = lines[seen:]
                seen = len(lines)
                for line in new_lines:
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if "mean_fg_dice" not in rec:
                        continue
                    final_dice = float(rec["mean_fg_dice"])
                    if "val_loss" in rec:
                        final_val_loss = float(rec["val_loss"])
                    # Per-epoch report so ASHA can early-stop trial bodies
                    # that aren't improving. Without this call, the scheduler
                    # only sees the final value and never prunes.
                    train.report(
                        {"mean_fg_dice": final_dice, "val_loss": final_val_loss}
                    )
            if rc is not None:
                break
            time.sleep(poll_interval_s)
    finally:
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                proc.kill()

    if proc.returncode != 0:
        raise RuntimeError(f"nnUNetv2_train exited {proc.returncode}")
    return {"mean_fg_dice": final_dice, "val_loss": final_val_loss}


# ─── driver entry-point ─────────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="HPO sweep driver for nnUNetv2.")
    p.add_argument("--ray-address", required=True,
                   help="Ray Client URI, e.g. ray://factory-ray-head-svc:10001")
    p.add_argument("--dataset", required=True,
                   help="DatasetSpec key (must exist in SPECS)")
    p.add_argument("--fold", type=int, default=0,
                   help="Fold index to sweep on (single fold; full 5-fold "
                        "training is reserved for the winning config)")
    p.add_argument("--parent-run-id", default="",
                   help="Optional MLflow parent run id; per-trial child runs nest under it")
    p.add_argument("--num-trials", type=int, default=30)
    p.add_argument("--max-epochs", type=int, default=80,
                   help="Cap per trial; ASHA early-stops below this")
    p.add_argument("--num-iters", type=int, default=100,
                   help="num_iterations_per_epoch override (cheaper trials)")
    p.add_argument("--max-concurrent", type=int, default=8,
                   help="Cap on simultaneous trials (≤ Ray worker count)")
    p.add_argument("--trainer", default="nnUNetTrainerHPO")
    p.add_argument("--plans", default="nnUNetResEncUNetLPlans")
    p.add_argument("--searcher", choices=["optuna", "random"], default="optuna")
    p.add_argument("--scheduler", choices=["asha", "none"], default="asha")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args(argv)

    if args.dataset not in SPECS:
        print(f"unknown dataset key: {args.dataset}", file=sys.stderr)
        return 2

    if args.searcher == "optuna" and OptunaSearch is None:
        print(
            "OptunaSearch unavailable (ray[tune,optuna] not installed); "
            "fallback to --searcher random",
            file=sys.stderr,
        )
        return 2

    print(f"[hpo-driver] connecting to Ray at {args.ray_address}", flush=True)
    ray.init(
        address=args.ray_address,
        runtime_env={
            "env_vars": {
                "PYTHONUNBUFFERED": "1",
                "nnUNet_compile": "False",
            }
        },
    )

    space = param_space_for(args.dataset)
    # Fixed cell — the sweep varies hyperparameters within one (dataset, fold).
    space["dataset_key"] = args.dataset
    space["fold"] = args.fold
    space["trainer"] = args.trainer
    space["plans"] = args.plans
    space["parent_run_id"] = args.parent_run_id
    space["num_epochs"] = args.max_epochs
    space["num_iterations_per_epoch"] = args.num_iters

    searcher = None
    if args.searcher == "optuna":
        searcher = OptunaSearch(metric="mean_fg_dice", mode="max", seed=args.seed)

    scheduler = None
    if args.scheduler == "asha":
        scheduler = ASHAScheduler(
            metric="mean_fg_dice", mode="max",
            max_t=args.max_epochs,
            grace_period=max(1, args.max_epochs // 8),
            reduction_factor=3,
        )

    print(
        f"[hpo-driver] sweep: dataset={args.dataset} fold={args.fold} "
        f"trials={args.num_trials} max_epochs={args.max_epochs} "
        f"searcher={args.searcher} scheduler={args.scheduler}",
        flush=True,
    )

    tuner = tune.Tuner(
        tune.with_resources(_run_one_hpo_trial, resources={"cpu": 16, "gpu": 1}),
        param_space=space,
        tune_config=tune.TuneConfig(
            num_samples=args.num_trials,
            max_concurrent_trials=args.max_concurrent,
            metric="mean_fg_dice",
            mode="max",
            search_alg=searcher,
            scheduler=scheduler,
        ),
        run_config=train.RunConfig(
            name=f"hpo-{args.dataset.replace('_', '-')}",
            storage_path="/factory/results/.tune",
            log_to_file=True,
        ),
    )

    results = tuner.fit()

    print("[hpo-driver] sweep complete", flush=True)
    n_err = sum(1 for r in results if r.error is not None)
    try:
        best = results.get_best_result(metric="mean_fg_dice", mode="max")
        # Strip Tune scaffolding from the published config — keep only the
        # hyperparameters that actually drove the trial.
        keep = {"oversample_foreground_percent", "initial_lr", "weight_decay",
                "enable_deep_supervision", "num_epochs",
                "num_iterations_per_epoch", "num_val_iterations_per_epoch"}
        best_config = {k: v for k, v in (best.config or {}).items() if k in keep}
        summary = {
            "dataset": args.dataset,
            "fold": args.fold,
            "best_config": best_config,
            "best_metric": float(best.metrics.get("mean_fg_dice", float("nan"))),
            "n_trials": len(results),
            "n_errors": n_err,
        }
    except Exception as e:  # all trials errored, etc.
        summary = {
            "dataset": args.dataset,
            "fold": args.fold,
            "best_config": None,
            "best_metric": None,
            "n_trials": len(results),
            "n_errors": n_err,
            "error": str(e),
        }
    print(json.dumps(summary), flush=True)
    return 1 if n_err else 0


if __name__ == "__main__":
    sys.exit(main())
