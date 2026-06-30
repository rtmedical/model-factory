"""Ray Tune driver for cross-dataset, multi-fold nnUNetv2 training campaigns.

The driver runs inside a small k8s Job pod (see
`infra/kustomize/ray-driver-job.yaml.j2`) that:

  1. Connects to the running `factory-ray` RayCluster via the Ray Client port.
  2. Builds a `tune.Tuner` whose search space is the cartesian product
     `(dataset_key, fold)` — no HPO, just orchestration. The grid lets Ray
     fan trials out across the 8-worker pool one MIG slice at a time.
  3. Each trial shells out to `nnUNetv2_train` on the worker that holds it.
     Because each worker pod already exported `NVIDIA_VISIBLE_DEVICES=<MIG-UUID>`
     at start-up (init container `claim-mig.sh`), the subprocess inherits it
     and CUDA is pinned to a single 40 GB slice.
  4. Per-fold metrics flow into MLflow via the existing `nnUNetTrainerMLflow`
     subclass (env-driven; we just pass `MFACTORY_PARENT_RUN_ID` + the
     experiment name in). Parent runs are created by the CLI before the
     driver is launched and threaded in via `--parent-run-ids`.

Invocation (typical):

    python -m modelfactory.jobs.ray_driver \\
        --ray-address ray://factory-ray-head-svc.model-factory.svc.cluster.local:10001 \\
        --datasets brain_mr_core_oar,brain_mr_lobes_bilateral,brain_mr_deep_nuclei \\
        --folds 0,1,2,3,4 \\
        --parent-run-ids brain_mr_core_oar=abc123,brain_mr_lobes_bilateral=def456,brain_mr_deep_nuclei=ghi789 \\
        --trainer nnUNetTrainerMLflow \\
        --plans nnUNetResEncUNetLPlans
"""

from __future__ import annotations

import argparse
import errno
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import ray
from ray import tune, train  # train.RunConfig replaces tune.RunConfig in Ray 2.10+

from modelfactory.datasets.specs import SPECS


@dataclass(frozen=True)
class TrialConfig:
    """One trial = train one fold of one dataset on whichever MIG slice the
    Ray worker carrying it currently owns."""
    dataset_key: str
    fold: int
    parent_run_id: str | None
    trainer: str
    plans: str


def _is_enospc(exc: BaseException) -> bool:
    """True if `exc` is (or wraps) an ENOSPC 'no space left on device'.

    `shutil.copytree` raises `shutil.Error` wrapping a list of per-file
    `(src, dst, why)` tuples where `why` is the stringified OSError, so the
    errno isn't directly accessible — match on the message as well.
    """
    if isinstance(exc, OSError) and exc.errno == errno.ENOSPC:
        return True
    text = str(exc)
    return f"Errno {errno.ENOSPC}" in text or "No space left on device" in text


def _stage_preprocessed(
    spec,
    src_root: str = "/factory/preprocessed",
    dst_root: str = "/factory-ram/preprocessed",
) -> str:
    """Copy this fold's preprocessed dataset from NFS into the pod's
    `/factory-ram` tmpfs and return the staged root. Idempotent — the
    first task for a dataset on a given worker pod does the rsync
    (~30-60 s for the 65-92 GiB brain-MR sets); subsequent tasks for
    the same dataset reuse the cached copy.

    Only the read-hot `preprocessed/` dir is moved to RAM. `nnUNet_raw`
    and `nnUNet_results` stay on NFS so checkpoints + MLflow artifacts
    are durable. Falls back to `src_root` if `/factory-ram` is unmounted
    (e.g. running against an old worker pod without the tmpfs volume).
    """
    dst_parent = Path(dst_root)
    if not dst_parent.parent.is_dir():
        print(f"[stage] /factory-ram unavailable, falling back to {src_root}", flush=True)
        return src_root
    src = Path(src_root) / spec.folder
    dst = dst_parent / spec.folder
    sentinel = dst / ".stage_complete"

    # Size guard: /factory-ram is RAM-backed (counts against the pod's 160Gi memory
    # limit, alongside the 32Gi dshm + the trainer's RSS). Staging a large dataset
    # (e.g. the ~1000-case partial-label generalists at ~97 GiB) OOM-kills the
    # augmenter's background workers a few epochs in. Train those off NFS instead —
    # slower per epoch but stable (same as the >tmpfs ENOSPC fallback below). 50 GiB
    # leaves headroom for dshm + training within the 160Gi pod limit.
    MAX_STAGE_BYTES = 50 * 1024**3
    try:
        total = sum(p.stat().st_size for p in src.rglob("*") if p.is_file())
    except OSError:
        total = 0
    if total > MAX_STAGE_BYTES:
        print(f"[stage] {spec.folder} is {total/1024**3:.0f} GiB > 50 GiB tmpfs budget "
              f"— training off NFS {src_root} to avoid OOM", flush=True)
        return src_root

    # Evict stale sibling datasets — the worker pod survives across trials, so
    # the previous trial's dataset may still occupy tmpfs. With sizeLimit 110 Gi
    # and brain-MR datasets at 65-92 Gi, leaving two of them around overflows.
    if dst_parent.is_dir():
        for child in dst_parent.iterdir():
            if child.is_dir() and child.name != spec.folder:
                print(f"[stage] evicting stale {child}", flush=True)
                shutil.rmtree(child, ignore_errors=True)

    if sentinel.is_file():
        return dst_root
    print(f"[stage] copying {src} → {dst}", flush=True)
    t0 = time.monotonic()
    # shutil.copytree avoids depending on the rsync binary, which isn't in
    # the NGC pytorch base image. Single-threaded but NFS→tmpfs is sequential
    # read + memcpy, ~250 MB/s; the 11 GiB SegRap glands set takes ~45 s.
    if dst.is_dir():
        shutil.rmtree(dst)
    try:
        shutil.copytree(src, dst)
    except (shutil.Error, OSError) as exc:
        # Most commonly ENOSPC: the dataset is larger than the /factory-ram
        # tmpfs (e.g. a HighRes set bigger than sizeLimit — D132 CoronaryLAD is
        # 112 GiB vs a 110 GiB tmpfs). Don't lose the trial: wipe the partial
        # copy and train directly off NFS (slower per epoch but correct). Same
        # graceful-degradation intent as the unmounted-tmpfs case above.
        if _is_enospc(exc):
            print(
                f"[stage] /factory-ram full for {spec.folder} "
                f"({exc.__class__.__name__}: no space) — falling back to NFS {src_root}",
                flush=True,
            )
            shutil.rmtree(dst, ignore_errors=True)
            return src_root
        raise
    sentinel.touch()
    print(f"[stage] done in {time.monotonic() - t0:.1f}s", flush=True)
    return dst_root


def _run_one_fold(config: dict) -> dict:
    """Executed on a Ray worker. Subprocesses `nnUNetv2_train` and reports the
    last `mean_fg_dice` from the trainer's JSONL sidecar back to Tune."""
    spec = SPECS[config["dataset_key"]]
    fold = int(config["fold"])
    experiment = f"{spec.folder}__3d_fullres"

    env = os.environ.copy()
    # Inherits NVIDIA_VISIBLE_DEVICES from the worker pod's pod-spec env.
    preproc_root = _stage_preprocessed(spec)
    env["nnUNet_preprocessed"] = preproc_root
    env.update({
        "MFACTORY_EXPERIMENT": experiment,
        "MFACTORY_RUN_NAME": f"{spec.folder}__3d_fullres__fold{fold}",
        "MFACTORY_METRICS_JSONL": "1",
        # Disable torch.compile inside nnUNetv2. NGC PyTorch 2.7 + driver
        # 580.x's caching allocator hits an NVML assert during the FIRST
        # inductor-compiled `convolution_backward` on a MIG slice
        # (CUDACachingAllocator.cpp:1016, `NVML_SUCCESS == r`). The race is
        # non-deterministic — 045/048 got past it, 047 didn't on the first
        # attempt. Skipping torch.compile bypasses the inductor codegen
        # path entirely; runtime cost is roughly +20-30% per epoch but
        # makes the trial deterministic on MIG.
        "nnUNet_compile": "False",
    })
    # Partial-label generalists are the largest cohorts (~1000 cases) and train
    # off NFS. nnUNet's default DA-worker count (12-18) × per-worker prefetch +
    # NFS read page-cache OOM-kills the augmenter on the 160Gi pod around ep5-8.
    # Cap the augmenter workers to shrink that host-RAM footprint (slower aug,
    # but stable). Only for partial_label specs — the others are fine at default.
    if getattr(spec, "partial_label", False):
        env["nnUNet_n_proc_DA"] = "6"
    if config.get("parent_run_id"):
        env["MFACTORY_PARENT_RUN_ID"] = config["parent_run_id"]

    cmd = [
        "nnUNetv2_train",
        str(spec.dataset_id),
        "3d_fullres",
        str(fold),
        "-p", config["plans"],
        "-tr", config["trainer"],
    ]
    if config.get("continue_training"):
        # nnUNetv2_train's --c flag resumes from checkpoint_latest.pth if it
        # exists; otherwise it prints "Cannot continue training... Starting a
        # new training" and falls through to fresh training. Safe to set on
        # every trial in a mixed campaign (some have checkpoints, some don't).
        cmd.append("--c")
    # NB: nnUNetv2_train does NOT accept --num_epochs on the CLI; the cap is
    # a class attribute (nnUNetTrainer.num_epochs). To run a short smoke,
    # subclass nnUNetTrainerMLflow and override num_epochs in __init__.

    print(f"[trial] {spec.folder} fold={fold} → {' '.join(cmd)}", flush=True)
    print(f"[trial] NVIDIA_VISIBLE_DEVICES={env.get('NVIDIA_VISIBLE_DEVICES','<unset>')}", flush=True)

    proc = subprocess.run(cmd, env=env, check=False)
    if proc.returncode != 0:
        # Tune will mark the trial as ERROR; the failure surfaces in the dashboard
        # and the parent MLflow run still holds the partial children.
        raise RuntimeError(f"nnUNetv2_train exited {proc.returncode}")

    # Pull the final mean_fg_dice out of the JSONL sidecar (more reliable than
    # parsing stdout; the sidecar is what the trainer itself writes).
    results_root = Path(os.environ.get("nnUNet_results", "/factory/results"))
    sidecar = (
        results_root
        / spec.folder
        / f"{config['trainer']}__{config['plans']}__3d_fullres"
        / f"fold_{fold}"
        / "metrics.jsonl"
    )
    final_dice = float("nan")
    final_val_loss = float("nan")
    if sidecar.is_file():
        for line in sidecar.read_text().splitlines():
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "mean_fg_dice" in rec:
                final_dice = float(rec["mean_fg_dice"])
            if "val_loss" in rec:
                final_val_loss = float(rec["val_loss"])
    else:
        print(f"[trial] WARN sidecar missing at {sidecar}", flush=True)

    return {"mean_fg_dice": final_dice, "val_loss": final_val_loss}


def _parse_parent_run_ids(arg: str) -> dict[str, str]:
    """`key1=run_id1,key2=run_id2` → dict."""
    if not arg:
        return {}
    out: dict[str, str] = {}
    for chunk in arg.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "=" not in chunk:
            raise ValueError(f"parent-run-ids entry must be key=run_id, got {chunk!r}")
        k, v = chunk.split("=", 1)
        out[k.strip()] = v.strip()
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ray-address", required=True,
                   help="Ray Client URI, e.g. ray://factory-ray-head-svc.model-factory.svc.cluster.local:10001")
    p.add_argument("--datasets", required=True,
                   help="Comma-separated DatasetSpec keys (e.g. brain_mr_core_oar,brain_mr_lobes_bilateral)")
    p.add_argument("--folds", default="0,1,2,3,4",
                   help="Comma-separated folds")
    p.add_argument("--parent-run-ids", default="",
                   help="Comma-separated dataset_key=MLflowRunID pairs")
    p.add_argument("--trainer", default="nnUNetTrainerMLflow")
    p.add_argument("--plans", default="nnUNetResEncUNetLPlans")
    p.add_argument("--max-concurrent", type=int, default=8,
                   help="Cap on simultaneous trials (≤ worker count)")
    p.add_argument("--continue", dest="continue_training", action="store_true",
                   help="Pass --c to nnUNetv2_train (resume from checkpoint_latest.pth if present; "
                        "else fall back to fresh training).")
    args = p.parse_args(argv)

    dataset_keys = [k.strip() for k in args.datasets.split(",") if k.strip()]
    folds = [int(f.strip()) for f in args.folds.split(",") if f.strip()]
    parent_run_ids = _parse_parent_run_ids(args.parent_run_ids)

    unknown = [k for k in dataset_keys if k not in SPECS]
    if unknown:
        print(f"unknown dataset keys: {unknown}", file=sys.stderr)
        return 2

    print(f"[driver] connecting to Ray at {args.ray_address}", flush=True)
    ray.init(
        address=args.ray_address,
        runtime_env={
            "env_vars": {
                "PYTHONUNBUFFERED": "1",
                # Propagated to every Ray task this driver spawns (including
                # `_run_one_fold`). Need this here, NOT just inside the
                # task body, because the workers run their image's baked-in
                # ray_driver.py (only the driver Job mounts host source).
                # nnUNet_compile=False bypasses torch.compile inside
                # nnUNetv2 → skips the inductor → conv_backward → NVML
                # caching-allocator path that crashed 047 on first backward.
                "nnUNet_compile": "False",
            }
        },
    )

    n_trials = len(dataset_keys) * len(folds)
    print(f"[driver] {n_trials} trials = {len(dataset_keys)} datasets × {len(folds)} folds", flush=True)

    param_space = {
        "dataset_key": tune.grid_search(dataset_keys),
        "fold": tune.grid_search(folds),
        # tune.grid_search nesting requires parent_run_id to be resolved per trial
        # via a sample_from; cleaner to look it up in-trial from a closure.
        "trainer": args.trainer,
        "plans": args.plans,
        "continue_training": args.continue_training,
        # Resolved at trial start by looking up dataset_key.
        "parent_run_id": tune.sample_from(
            lambda spec: parent_run_ids.get(spec.config["dataset_key"], "")
        ),
    }

    # Each trial consumes 1 Ray GPU + 20 CPUs; matches the worker pod's
    # rayStartParams.num-gpus=1 / num-cpus=20. max_concurrent_trials gives an
    # extra ceiling in case the cluster autoscales beyond what we want.
    tuner = tune.Tuner(
        tune.with_resources(_run_one_fold, resources={"cpu": 20, "gpu": 1}),
        param_space=param_space,
        tune_config=tune.TuneConfig(
            num_samples=1,
            max_concurrent_trials=args.max_concurrent,
            metric="mean_fg_dice",
            mode="max",
        ),
        run_config=train.RunConfig(
            name=f"campaign-{','.join(dataset_keys)}",
            storage_path="/factory/results/.tune",
            log_to_file=True,
        ),
    )

    results = tuner.fit()

    print("[driver] campaign complete")
    df = results.get_dataframe()
    print(df[["config/dataset_key", "config/fold", "mean_fg_dice", "val_loss"]].to_string(index=False))

    n_err = sum(1 for r in results if r.error is not None)
    return 1 if n_err else 0


if __name__ == "__main__":
    sys.exit(main())
