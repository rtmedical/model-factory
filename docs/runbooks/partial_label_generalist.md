# Runbook: Generalist Wave (partial-label generalists + gold specialists)

Companion to the plan `~/.claude/plans/your-task-is-prepare-concurrent-wombat.md`.
This wave (a) builds **one partial-label single generalist per region** (pelvis
D151, thorax D152) over the full ~1000-case clinical union, (b) **quarantines
Colon_Sigmoid** out of the pelvis generalist into a gold specialist (D153) and
rebuilds the female model from gold (D154), and (c) fixes the weak TS-gap
tubulars (RCA D155, Bronchus D156). All training is **fold-0 only**; folds 1-4
are deferred. Targets fill all 14 MIG slices.

## What changed in the repo (this wave)

| File | Change |
|---|---|
| `cli.py` | new `campaign run-wave` (1-14 datasets, `--max-concurrent`); `max_concurrent` param on `_submit_driver_job` |
| `infra/kustomize/ray-driver-job.yaml.j2` | threads `--max-concurrent` (was hard-capped at 8) |
| `datasets/aliases.py` | `UteroCervix`/`Uterus` added to `PELVIS_CLINICAL_ALIASES` |
| `datasets/specs.py` | new specs D151,D152 (partial-label generalists), D153,D154,D155,D156,D158,D159,D160 (gold specialists + cluster A/Bs); `DatasetSpec.partial_label` field |
| `datasets/convert.py` | partial-label mode (OR-filter, skip-missing, region-based `dataset.json`, per-case annotation sidecar embedded in `dataset.json`) |
| `datasets/sources/clinical_rtstruct.py` | `partial_label` OR-filter in `discover()` |
| `trainers/mlflow_trainer.py` | `nnUNetTrainerPartialLabelMLflow` (per-case channel masking) |
| shims: `runtime_shim_install.sh`, `patch_worker.sh`, `precheck_workers.sh`, `images/nnunet-trainer/Dockerfile`, `services/qa-viewer/Dockerfile` | register `partial_label_trainer.py` |

**Verified on host:** converter partial-label logic (region `dataset.json`,
coverage sidecar, OR-filter, non-partial regression), all spec registration,
`run-wave` dry-run (14-key + guardrails), `run-trio` regression (still caps 8).
**NOT yet validated (needs the container):** `nnUNetTrainerPartialLabelMLflow`
(no torch/nnunetv2 on host) — see the prototype gate below.

## Phase A — rebuild the trainer image

The partial-label trainer + its discovery shim must be in the image, and the
`_stage_preprocessed` ENOSPC→NFS fallback must be baked in (a ~1000-case
generalist's preprocessed dir may exceed the 110 GiB `/factory-ram` tmpfs).

```bash
cd /data/model-factory
make build-images && make push-images        # bakes partial_label_trainer.py shim
```

## Phase B — convert the new datasets (CPU Jobs, hpo-sweep priority)

All nine new datasets convert from the clinical trees via the `clinical_rtstruct`
source. **Mount note:** the clinical specs root at `/in/pelvis` and `/in/thorax`,
so the convert Job must hostMount `/data/<cohort> → /in/<cohort>` and
(the stock `convert-job.yaml.j2` mounts
`/data/source → /in` — adjust the volume for clinical specs, or run a
clinical convert-job variant). Per-spec source + cohort:

| Dataset | spec key | source | ~cohort |
|---|---|---|---|
| D151 | `pelvis_generalist_partial` | clinical_rtstruct (partial) | ~1000 (OR-union) |
| D152 | `thorax_generalist_partial` | clinical_rtstruct (partial) | ~1000 (OR-union) |
| D153 | `pelvis_colon_sigmoid_gold` | clinical_rtstruct | ~700 |
| D154 | `pelvis_female_gyn_gold` | clinical_rtstruct | ~116 (UteroCervix-gated) |
| D155 | `thorax_rca_gold` | clinical_rtstruct | ~210 |
| D156 | `thorax_bronchus_anchored` | clinical_rtstruct | ~88 |
| D158 | `pelvis_clinical_core_v2` | clinical_rtstruct | ~700 |
| D159 | `thorax_breast_l_gen` | clinical_rtstruct | ~298 |
| D160 | `thorax_coronary_highres` | clinical_rtstruct | ~148 |

Render + apply per dataset (example for D151):
```bash
export COLUMNS=10000
PYTHONPATH=src python3 - <<'PY'
import jinja2, pathlib
from modelfactory.datasets.specs import SPECS
key="pelvis_generalist_partial"; s=SPECS[key]
t=pathlib.Path("infra/kustomize/convert-job.yaml.j2").read_text()
# NB: edit the rendered YAML's hostPath to your source-data root
#     (and /in -> keep, since the spec root is /in/pelvis) for clinical specs.
print(jinja2.Environment().from_string(t).render(
    spec_key=key, spec_slug=key.replace("_","-"), source_type="clinical_rtstruct",
    workers=16, memory_request="48Gi", memory_limit="160Gi"))
PY
# kubectl apply -f convert-d151.yaml ; kubectl wait --for=condition=complete job/convert-... --timeout=12h
```

After convert, **inspect the partial-label coverage** (thin channels are the risk):
```bash
python3 -c "import json;d=json.load(open('/data/model-factory-nfs/datasets/Dataset151_Pelvis_Generalist_Partial/partial_label_annotations.json'));print(d['coverage'])"
```

## Phase C — preprocess

```bash
# render infra/kustomize/preprocess-job.yaml.j2 per dataset, then kubectl apply.
# Generalists D151/D152 + ResEncL gold D154/D158/D159: planner nnUNetPlannerResEncL.
# HighRes specialists D153/D155/D156/D160: planner nnUNetPlannerResEncL_HighRes, np=4.
# Also preprocess D151/D152 at nnUNetResEncUNetXLPlans for the capacity A/B
# (planner nnUNetPlannerResEncXL).
```
Verify each: `ls /data/model-factory-nfs/preprocessed/Dataset###_*/*.json` (D087 lesson).
Check generalist preprocessed size vs the 110 GiB tmpfs: `du -sh .../preprocessed/Dataset151_*`.

## Phase D — PARTIAL-LABEL PROTOTYPE GATE (mandatory before D151/D152 full run)

`nnUNetTrainerPartialLabelMLflow` is unvalidated on the host. Before committing
GPU to the full generalists, prototype on a tiny subset and assert correctness:

```bash
# 1. Convert a 60-case partial subset (any clinical spec with partial_label):
#    python -m modelfactory.datasets.convert --spec pelvis_generalist_partial \
#      --source clinical_rtstruct --out /factory --workers 8 --limit 60
# 2. Preprocess it (ResEncL).
# 3. campaign smoke --dataset pelvis_generalist_partial --fold 0 \
#       --trainer nnUNetTrainerPartialLabelMLflow --plans nnUNetResEncUNetLPlans
#    (use a low-epoch subclass for speed, per the smoke convention).
```
Assert, by reading the trainer log + a quick gradient probe:
1. `dataset_json["partial_label_annotations"]` is present post-preprocess.
2. `batch["keys"]` is populated in `train_step` (the trainer logs a WARNING and
   disables masking if not — if you see that warning, masking is OFF and results
   are invalid; fix the dataloader to pass keys before proceeding).
3. A case that did NOT annotate organ C produces **zero gradient** on channel C.
4. `regions_class_order` length == number of organs; output/target are
   `[B, R, *spatial]` (a list across scales under deep supervision).

Only after these pass should the full D151/D152 train. If the trainer needs
fixes, iterate here (cheap) — this is exactly the "prototype-first" gate from
the plan's risk section.

## Phase E — the 14-slot fold-0 wave

```bash
export COLUMNS=10000
cd /data/model-factory
./scripts/precheck_workers.sh --fix       # MANDATORY; rerun until "All workers pass"
```

The wave spans 5 (trainer, plans) recipe pairs → 5 driver jobs sharing the
14-GPU pool. Ray's GPU=1-per-trial accounting makes >14 concurrent impossible;
sizing each driver's `--max-concurrent` to its trial count makes the split
deterministic (Σ = 14). Apply the larger-share drivers first.

```bash
# J1 — PartialLabel / ResEncL  (the headline generalists)
modelfactory campaign run-wave --datasets pelvis_generalist_partial,thorax_generalist_partial \
  --folds 0 --max-concurrent 2 \
  --trainer nnUNetTrainerPartialLabelMLflow --plans nnUNetResEncUNetLPlans \
  --dry-run > wave-j1.yaml

# J2 — PartialLabel / XL  (capacity A/B)
modelfactory campaign run-wave --datasets pelvis_generalist_partial,thorax_generalist_partial \
  --folds 0 --max-concurrent 2 \
  --trainer nnUNetTrainerPartialLabelMLflow --plans nnUNetResEncUNetXLPlans \
  --dry-run > wave-j2.yaml

# J3 — MLflow / ResEncL  (gold female + cluster A/Bs + core_v2)
modelfactory campaign run-wave --datasets pelvis_female_gyn_gold,pelvis_clinical_core_v2,thorax_breast_l_gen \
  --folds 0 --max-concurrent 3 \
  --trainer nnUNetTrainerMLflow --plans nnUNetResEncUNetLPlans \
  --dry-run > wave-j3.yaml

# J4 — SmallStructures / HighRes  (sigmoid quarantine + weak tubular fixes)
modelfactory campaign run-wave --datasets pelvis_colon_sigmoid_gold,thorax_rca_gold,thorax_bronchus_anchored,thorax_coronary_highres \
  --folds 0 --max-concurrent 4 \
  --trainer nnUNetTrainerSmallStructuresMLflow --plans nnUNetResEncUNetLPlans_HighRes \
  --dry-run > wave-j4.yaml

# J5 — SmallStructures / ResEncL  (READY-NOW loss-only A/Bs on existing preprocessed)
#   D142 RCA, D136 female sigmoid, D133 MR thin vessels — all already preprocessed at ResEncL.
modelfactory campaign run-wave --datasets thorax_clinical_coronary,pelvis_ct_female_gyn,pelvis_mr_male_gap \
  --folds 0 --max-concurrent 3 \
  --trainer nnUNetTrainerSmallStructuresMLflow --plans nnUNetResEncUNetLPlans \
  --dry-run > wave-j5.yaml

# 2+2+3+4+3 = 14 trials.  kubectl apply the J* in order (largest share first):
kubectl apply -f wave-j4.yaml -f wave-j3.yaml -f wave-j5.yaml -f wave-j1.yaml -f wave-j2.yaml
```

**Hybrid start:** J5 is ready NOW (existing preprocessed) — submit it while
Phases B/C run, then bring up J1-J4 as their datasets finish preprocessing.

## Phase F — verify + folds 1-4

- 14 slices busy: Ray dashboard (14 RUNNING/0 PENDING), per-worker
  `nvidia-smi --query-compute-apps`, and `metrics.jsonl` freshness under
  `results/Dataset*/.../fold_0/` (the trustworthy signal; driver `Running` is not).
- Per-struct Dice: read `…/fold_0/validation/summary.json` (`mean[label].Dice`,
  `foreground_mean.Dice`), map labels via `dataset.json`; then the QA cross-val report.
- **Decision gates:** D151/D152 per-organ Dice ≥ matching specialists (else fall
  back to cluster ensemble); D158 FG mean > D125's 0.8365 (confirms sigmoid drag);
  D153 Colon_Sigmoid ≥ 0.72; D154 vs D136 sigmoid zero-rate 23%→~5%; D155/D160 vs
  D142 RCA 0.54; D156 vs D145 Bronchus 0.55.
- Folds 1-4 (after winners chosen): `modelfactory schedule enqueue-remaining-folds --folds 1,2,3,4 --priority 30`.

## Risks (see plan for full list)
1. Partial-label trainer is research-grade — Phase D gate is mandatory.
2. Generalist preprocessed > 110 GiB → tmpfs ENOSPC; the NFS fallback needs the
   Phase-A image rebuild.
3. Cold Ray workers silently fail trials — precheck before every submit.
4. CPU oversubscription at 14-way → ~1.3× slower epochs (expected).
