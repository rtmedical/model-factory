# Ray worker cold-start failures — runbook

**Severity**: HIGH. Has cost us multiple hours of GPU time silently.
**Frequency**: Every newly-recreated or long-idle Ray worker pod.
**Symptom**: Campaigns submitted via `modelfactory campaign smoke` /
`run-trio` show `Status: Running` in `kubectl get jobs`, but the trial
inside never produces a single epoch — silently consuming a MIG slot
"forever" because the Tune driver hangs on a pandas KeyError in its
failed-trial summary.

## TL;DR — never submit a campaign without running this first

```bash
./scripts/precheck_workers.sh --fix
```

If it prints `All workers pass — safe to submit.`, you can submit.
Otherwise it patches the failing workers in place; rerun until it passes.

## Failure mode #1 — Missing `nnUNetTrainerSmallStructuresMLflow`

**Error from worker stdout**:
```
RuntimeError: Could not find requested nnunet trainer nnUNetTrainerSmallStructuresMLflow
in nnunetv2.training.nnUNetTrainer
(/usr/local/lib/python3.12/dist-packages/nnunetv2/training/nnUNetTrainer).
If it is located somewhere else, please move it there.
```

**Root cause**: the trainer image (`nnunet-trainer:0.3.0-ray`) was built
*before* the SmallStructures variant was added. The Dockerfile on disk now
writes a shim at `…/nnUNetTrainer/variants/factory/small_structures_trainer.py`,
but the image's baked-in `RUN` only writes the `mlflow_trainer.py` and
`hpo_trainer.py` shims (verify with `sudo crictl inspecti <image-id>` →
search for `small_structures_trainer.py` in the image config; it's
absent).

Why some workers work anyway: long-lived workers (5+ days old) were
manually patched in place at some point. That patch is a mutable container
layer and is lost when KubeRay recreates the pod.

**Why nnUNet can't find it via the class discovery**: nnUNet's class
finder walks `nnunetv2/training/nnUNetTrainer/` for `.py` files and
imports each one to look up classes by name. If
`small_structures_trainer.py` doesn't exist, the class is invisible —
entry-points and packaging metadata don't help (see memory:
`qa_viewer_trainer_discovery_lesson`).

**Why the import inside the shim ALSO fails on a recreated worker**: the
recreated worker's `/opt/modelfactory/src/modelfactory/trainers/mlflow_trainer.py`
is the **image-baked** (May 14, 350-line) snapshot, not the live host
source (May 18+, 509 lines). There is no hostPath mount of
`/data/model-factory/src` into the Ray worker pod (only the
campaign-driver pod has that mount). Long-lived workers got the 509-line
file written into their writable layer manually; recreated workers come
up with the 350-line image-baked version.

**Fix**: `scripts/patch_worker.sh <worker-pod-name>` — copies the host
`mlflow_trainer.py` over the stale one and writes the missing shim. Idempotent.

## Failure mode #2 — `No CUDA GPUs are available` on an idle OR just-freed worker

**Error from worker stdout**:
```
File "/usr/local/bin/nnUNetv2_train", line 8, in <module>
  ...
RuntimeError: No CUDA GPUs are available

# OR — fires inside torch.cuda init when nnUNet tries to .to(device):
File "/usr/local/lib/python3.12/dist-packages/torch/cuda/__init__.py", line 372, in _lazy_init
    torch._C._cuda_init()
RuntimeError: No CUDA GPUs are available
```

**Root cause**: the CUDA context on the Ray worker is poisoned. Two paths
get you there:

1. **Long-idle worker**: NVML descriptor cache goes stale after the worker
   has been sitting without a trial for hours. See memory:
   `nvml_allocator_assert_cold_workers`.
2. **Just-cancelled trial** (observed 2026-05-20, second time):
   when you delete a campaign Job to free a slot for another, Ray Tune
   cancels the trial via SIGTERM. The trial process exits without fully
   tearing down its CUDA context — the worker process inherits a half-dead
   state. The next trial assigned to that worker fails immediately at
   `torch._C._cuda_init()`. **Even a "warm" worker that just had a trial
   running is not safe** to receive a new trial without a recycle.

**Detection**: hard. The exclusive-MIG model means `nvidia-smi` via
`kubectl exec` always reports `Failed to initialize NVML` whenever the
slice is in use AND when it's stale. We can't distinguish the two without
actually running a CUDA process. So we rely on the trial-failure signal
itself.

**Operational rule**: any time you `kubectl delete job campaign-*` to free a
slot, **also recycle the worker pod that was running it** before letting
Ray assign a new trial to that slot:

```bash
# Find which worker the cancelled campaign was on
ip=$(kubectl -n model-factory logs -l job-name=<deleted-job> --tail=200 | \
  grep -oE 'ip=192\.168\.[0-9]+\.[0-9]+' | tail -1 | cut -d= -f2)
worker=$(kubectl -n model-factory get pods -o wide | awk -v ip="$ip" '$6==ip{print $1}')
kubectl -n model-factory delete pod "$worker"
# Wait ~30s for recreation, then patch
./scripts/patch_worker.sh "$(kubectl -n model-factory get pods -l ray.io/group=mig-N -o name | head -1 | cut -d/ -f2)"
./scripts/precheck_workers.sh
```

**Detection**: hard. The exclusive-MIG model means `nvidia-smi` via
`kubectl exec` always reports `Failed to initialize NVML` whenever the
slice is in use AND when it's stale. We can't distinguish the two without
actually running a CUDA process. So we rely on the trial-failure signal
itself.

**Fix**: recycle the worker pod. KubeRay's RayCluster controller recreates
it with the same MIG UUID environment but a fresh process state.

```bash
kubectl -n model-factory delete pod factory-ray-mig-N-worker-XXXX
# Wait ~30s for the recreated pod to be Running
# Then patch the recreated pod for failure mode #1
./scripts/patch_worker.sh factory-ray-mig-N-worker-YYYY
```

## Failure mode #3 — Tune driver hangs in `Running` after the trial dies

**Symptom**: trial errored within minutes, but the Job's `STATUS` column
stays `Running` for hours; `kubectl get jobs ... -o json | jq .status.active`
keeps reporting `1`.

**Root cause**: a pandas KeyError in Ray Tune's failed-trial summary:
```
KeyError: "None of [Index(['config/dataset_key', 'config/fold',
'mean_fg_dice', 'val_loss'], dtype='object')] are in the [columns]"
```
The Tune main loop catches the trial exception, then tries to build a
summary DataFrame from the trial's reported metrics — but a failed trial
reported no metrics, so the columns don't exist. Tune logs the
traceback and gets stuck somewhere in the post-mortem.

This means **the Job's `Running` status is not trustworthy as a "training
is happening" signal**. To check if a trial is actually training, look at
the metrics.jsonl files:

```bash
find /data/model-factory-nfs/results -name 'metrics.jsonl' \
  -newer /tmp/.5min-ago 2>/dev/null
# Where /tmp/.5min-ago = a marker touched 5 min ago:
#   touch -d '5 minutes ago' /tmp/.5min-ago
```

A driver Job in `Running` whose corresponding `metrics.jsonl` hasn't been
written in 10+ min is almost certainly hung.

## The combined failure that cost us 6+ hours

Sequence of events on 2026-05-20:

1. 06:47 — Submitted 3 melanoma smokes (D100, D102, D106)
2. Ray Tune assigned all 3 to the same idle worker `mig-0-worker-8vzlp`
3. All 3 trials crashed inside their containers within seconds with
   `No CUDA GPUs are available` (failure mode #2)
4. Tune controller hit failure mode #3 → all 3 driver Jobs stayed
   `Running` for 6.5 h before being noticed
5. ~13:24 — Diagnosed via `kubectl logs <driver-pod>` grep for `RuntimeError`
6. Fixed by recycling cold workers + patching all 10

Net loss: 1 MIG slot × 6.5 h = 6.5 GPU-hours, plus the slot that should
have been running productive work.

## How to prevent this — pre-flight checklist

Before EVERY campaign submission:

```bash
# 1. Sanity-check all workers
./scripts/precheck_workers.sh --fix

# 2. Make sure your spec is in SPECS and the preprocessed data exists
PYTHONPATH=src python3 -c "
from modelfactory.datasets.specs import SPECS
spec = SPECS['<your-key>']
print(f'D{spec.dataset_id:03d}', spec.folder)
"
ls /data/model-factory-nfs/preprocessed/Dataset<NNN>_*/nnUNetPlans_3d_fullres/ \
  | head -3

# 3. Use --dry-run + kubectl apply (host-side MLflow URI is unreachable;
#    see memory: host_cli_mlflow_unreachable)
modelfactory campaign smoke ... --dry-run > smoke.yaml
kubectl apply -f smoke.yaml

# 4. WITHIN 5 minutes of submission, check that the trial actually started:
job=campaign-<slug>-<timestamp>
kubectl -n model-factory logs -l job-name=$job --tail=200 | grep -E \
  'RuntimeError|Could not find|stage|Trial.*started|epoch'
```

If you see `RuntimeError` in step 4: delete the failed Job, recycle the
worker that hosted the failed trial (`grep -oE 'ip=192\.168\.[0-9]+\.[0-9]+'`
in the logs to find which one), patch it, resubmit.

## Permanent fixes (preferred, not yet done)

1. **Rebuild `nnunet-trainer:0.3.0-ray`** from the current Dockerfile so
   the `small_structures_trainer.py` shim AND the current `mlflow_trainer.py`
   are baked in. Eliminates failure mode #1.
   ```bash
   make build-trainer
   ```

2. **Add hostPath mount to the RayCluster worker pod template** so workers
   see the live host `/data/model-factory/src` like the campaign-driver
   does. Eliminates the divergence between long-lived and recreated
   workers. Edit `infra/kustomize/factory-raycluster.yaml` (or equivalent)
   and add:
   ```yaml
   volumes:
   - name: factory-src
     hostPath: { path: /data/model-factory/src, type: Directory }
   ...
   volumeMounts:
   - name: factory-src
     mountPath: /opt/modelfactory/src
     readOnly: true
   ```
   Then `kubectl rollout restart` the RayCluster. Existing trials will
   need to checkpoint or restart; coordinate with whoever owns the
   running campaigns.

3. **Fix Ray Tune's failed-trial summary** to not hang on KeyError.
   Upstream bug; may already be fixed in a newer Ray version.

Until those three land, `precheck_workers.sh` is the operational guardrail.

## Related memory notes

- `recreated_worker_image_drift.md` — the SDK-source divergence between
  recreated and long-lived workers
- `nvml_allocator_assert_cold_workers.md` — the original NVML cold-start
  finding
- `host_cli_mlflow_unreachable.md` — why we use dry-run + kubectl apply
- `qa_viewer_trainer_discovery_lesson.md` — why nnUNet only finds trainers
  via `variants/<name>/<file>.py` shims
