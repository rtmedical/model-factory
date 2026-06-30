# User quickstart

Five-minute tour of the daily flow. Assumes the operator has finished bootstrap.

## Install the CLI

```bash
cd /data/model-factory
pip install -e .
modelfactory --help
```

## Register a dataset

A dataset must already be in nnUNetv2 layout:

```
/somewhere/Dataset042_KiTS23/
  imagesTr/case_0001_0000.nii.gz   # CT, channel 0
            case_0001_0001.nii.gz   # (optional) channel 1
            ...
  labelsTr/case_0001.nii.gz
  dataset.json                       # nnUNet schema
```

If you have raw DICOM, convert to NIfTI first (`dicom2nifti` is in the trainer
image). For Medical Segmentation Decathlon datasets,
`nnUNetv2_convert_MSD_dataset` (also bundled) does the layout for you.

```bash
modelfactory dataset validate /somewhere/Dataset042_KiTS23
modelfactory dataset register /somewhere/Dataset042_KiTS23 --copy
```

Listed at:
```bash
modelfactory dataset list
```

## Preprocess

Run `nnUNetv2_plan_and_preprocess` once per dataset, on CPU (no GPU quota
needed). The factory exposes this as a one-off Kueue Job — see
`examples/smoke/run_msd_hippocampus.sh` for the full invocation.

## Submit training

```bash
# All 5 folds, default trainer, default priority. Kueue admits all 5 at once
# given the current 5-GPU quota (one fold per GPU). Bump folds in parallel with
# --folds 0,1,2,3,4; if quota shrinks, Kueue serializes appropriately.
modelfactory train nnunet --dataset Dataset042_KiTS23 --folds all

# A single fold with a custom epoch count and high priority:
modelfactory train nnunet --dataset Dataset042_KiTS23 \
  --folds 0 --num-epochs 200 --priority interactive-eval

# Dry run — print the rendered Job manifest without submitting:
modelfactory train nnunet --dataset Dataset042_KiTS23 --folds 0 --dry-run
```

The CLI prints each Job name. Watch progress with:
```bash
kubectl -n model-factory get jobs -L factory.io/dataset,factory.io/fold
kubectl -n model-factory logs -f job/<name>
```

## Watch MLflow

```bash
modelfactory runs list --dataset Dataset042_KiTS23
```
or open the UI:
```bash
kubectl -n model-factory port-forward svc/mlflow 5000:5000
# browse to http://localhost:5000
```
Each fold is a separate run. Folds for the same dataset/configuration share
an experiment named `{dataset}__{configuration}`. Optional: nest all 5 folds
under one parent run by passing `--parent-run` (TODO once we wire the
orchestrator-side `mlflow.start_run(parent=...)`).

## Register a 5-fold ensemble

When all 5 folds finish, wrap them in a single pyfunc Model Registry entry:
```bash
modelfactory model register-ensemble --dataset Dataset042_KiTS23 --configuration 3d_fullres
```
This downloads each fold's `checkpoint_final.pth` from the MLflow artifact
store, builds an `EnsemblePredictor` that loads all 5 and averages softmax
probabilities, and registers it as `Dataset042_KiTS23__3d_fullres` version N.

Promote to Staging or Production:
```bash
modelfactory model promote --name Dataset042_KiTS23__3d_fullres --version 1 --stage Staging
```

## Inspect failure cases

After a fold finishes, the trainer logs `validation/summary.json` as an MLflow
artifact. The `failure_mining` module produces a worst-decile parquet:

```python
from modelfactory.analysis.failure_mining import write_failure_parquet
from pathlib import Path
write_failure_parquet(
    Path("/data/model-factory-nfs/results/Dataset042_KiTS23/nnUNetTrainerMLflow__nnUNetPlans__3d_fullres/fold_0"),
    Path("/tmp/fold_0_failures.parquet"),
)
```
Plug this into Voxel51 FiftyOne for visual review (deferred Phase 7b).

## Grafana

```bash
kubectl -n monitoring port-forward svc/kps-grafana 3001:80
# browse to http://localhost:3001  (default user admin / value from grafana.adminPassword)
```
Pre-loaded dashboards: GPU (DCGM), node, k8s, Loki logs, MLflow run query.

## Common gotchas

- **NaN loss**: trainer logs `NNUNET_NAN_LOSS dataset=... fold=...` and the
  `NaNLossEmitted` alert fires. Usually means fp16 autocast overflowed —
  switch to bf16 or shrink learning rate.
- **CUDA OOM**: H100 has 80GB but a default 3d_fullres patch can need 60+GB;
  shrink patch_size in `nnUNetPlans.json` and re-preprocess.
- **Augmenter deadlock**: increase `--shm-size` in the Job template (default
  32 GiB; for very large patches bump to 48 GiB).
- **Slow MLflow uploads**: 5-fold ensemble registration uploads ~1.5 GiB of
  checkpoints. Use `kubectl exec` into the MLflow pod and check `du -sh`
  on `/tmp` while it runs if it stalls.
