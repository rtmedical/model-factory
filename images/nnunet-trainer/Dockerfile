# syntax=docker/dockerfile:1.7
#
# nnunet-trainer: base image for all training Jobs.
#
# Critical rules (verified against MIC-DKFZ/nnUNet, NVIDIA NGC notes 2026-05):
#   1. NGC pytorch base already ships torch + cuDNN + NCCL tuned for H100.
#      NEVER `pip install torch` on top — it silently overwrites the NGC build
#      and breaks NCCL on multi-GPU.
#   2. Install nnunetv2 with `--no-deps` so it doesn't try to bring in torch.
#      Then install its real deps explicitly.
#   3. batchgenerators v1 + v2 coexist in nnUNetv2 2.5; pin both.
#   4. Run-time requirements (NOT in image): --shm-size=32g, nofile=65535.
#      Set in the Pod spec — see jobs/templates/training-job.yaml.j2.

ARG NGC_TAG=25.04-py3
FROM nvcr.io/nvidia/pytorch:${NGC_TAG} AS wheels

# Stage 1: build wheels in isolation. Keep --no-deps to avoid torch overwrite.
RUN pip install --no-cache-dir --upgrade pip wheel setuptools

# Build wheels for nnUNetv2 + deps (excluding torch). We use a tight version
# pin on nnunetv2 and batchgenerators to avoid silent ABI drift.
#
# rt-utils + opencv-python-headless are for the RTStructDicomSource adapter
# (cancer-archive RTSTRUCT pelvic cohort). rt-utils internally uses
# OpenCV for polygon rasterization; we install the headless variant to
# avoid pulling in X libs and Qt.
RUN pip wheel --no-cache-dir --no-deps -w /wheels \
        nnunetv2==2.5.* \
        batchgenerators==0.25 \
        batchgeneratorsv2>=0.2 \
        SimpleITK==2.4.* \
        nibabel \
        scikit-image \
        scipy \
        pandas \
        tqdm \
        dicom2nifti \
        rt-utils \
        opencv-python-headless \
        connected-components-3d \
        acvl-utils \
        "dynamic-network-architectures>=0.3.1,<0.4" \
        einops \
        graphviz

# SDK deps (mlflow client, k8s client, etc.) — separate layer for cacheability.
#
# mlflow-skinny (no flask/gunicorn) avoids the pyarrow<18 cap that the
# full mlflow client carries and that conflicts with NGC's pinned
# pyarrow==19.0.1. We're file-store-only on the trainer side
# (MLFLOW_TRACKING_URI=file://...), so server-side deps are unnecessary.
#
# torchmetrics is intentionally NOT installed here: its torch upper-bound
# pin conflicts with NGC's nv25.4 torch build, and the only consumer
# (modelfactory.analysis.calibration) is a post-hoc analysis tool that
# runs outside the training pod.
RUN pip wheel --no-cache-dir -w /wheels \
        "mlflow-skinny>=2.20,<3" \
        boto3 \
        psycopg2-binary \
        prometheus-client \
        click \
        pydantic \
        pyyaml \
        jinja2 \
        rich \
        kubernetes

# Ray (default + tune) — orchestration for cross-dataset campaigns.
#
# Ray 2.40 supports Python 3.12 (NGC 25.04-py3) and does NOT depend on torch
# at all when the [train] extra is omitted. ray[default] pulls the dashboard
# + ray-client deps; ray[tune] pulls the Tuner + result-grid deps. Neither
# touches torch, so we let pip resolve transitives normally — no --no-deps
# guard needed here.
#
# pyarrow is intentionally left to NGC's pinned version (19.0.1); Ray 2.40
# accepts pyarrow >= 6.0.1 < 21, so the NGC pin is well inside the band.
RUN pip wheel --no-cache-dir -w /wheels \
        "ray[default,tune]==2.40.*"

# ──────────────────────────────────────────────────────────────────────────
FROM nvcr.io/nvidia/pytorch:${NGC_TAG}

LABEL org.opencontainers.image.title="nnunet-trainer"
LABEL org.opencontainers.image.description="nnUNetv2 + MLflow on NGC pytorch H100-tuned"
LABEL org.opencontainers.image.source="https://internal/model-factory"

# Install wheels without touching torch
COPY --from=wheels /wheels /wheels
# Note: we DO allow PyPI fallback here (no --no-index) so pip can pull
# transitive deps of nnunetv2 (tifffile, joblib, etc.) that we didn't
# build wheels for. The local /wheels dir is preferred via --find-links,
# so torch is never refetched — NGC's tuned build stays put.
RUN pip install --no-cache-dir --find-links=/wheels \
        nnunetv2 batchgenerators batchgeneratorsv2 SimpleITK nibabel \
        scikit-image scipy pandas tqdm dicom2nifti rt-utils opencv-python-headless \
        connected-components-3d \
        acvl-utils "dynamic-network-architectures>=0.3.1,<0.4" einops graphviz \
        mlflow-skinny boto3 psycopg2-binary prometheus-client \
        click pydantic pyyaml jinja2 rich kubernetes \
        "ray[default,tune]==2.40.*" \
    && rm -rf /wheels

# Build-time guard: any future Ray bump that drags torch will break the build
# here rather than silently breaking NCCL at training time. NGC 25.04 ships
# torch 2.7.x; assert we still have that family.
RUN python -c "import torch, sys; \
v = torch.__version__; \
assert v.startswith('2.7'), f'torch was overwritten: {v}'; \
print(f'[guard] torch={v} (NGC build intact)')"

# Install the modelfactory SDK + trainers in editable mode so the image picks
# up code changes when bind-mounted during dev.
COPY pyproject.toml /opt/modelfactory/pyproject.toml
COPY src           /opt/modelfactory/src
RUN pip install --no-cache-dir --no-deps -e /opt/modelfactory

# Bridge: nnUNetv2 2.6 finds trainer classes by walking
# <site-packages>/nnunetv2/training/nnUNetTrainer/ recursively — it does
# NOT honour Python entry-point groups (the `nnunetv2.trainers` group in
# our pyproject.toml is a dead letter as of nnUNet 2.x).
#
# `recursive_find_python_class` uses `pkgutil.iter_modules` to enumerate
# *non-package* modules in each folder. An `__init__.py` is not listed
# as a module by `iter_modules`, so the shim MUST be a separate .py file
# (with a non-empty `__init__.py` to mark the directory as a package).
RUN python -c "import nnunetv2, os; \
d = os.path.join(os.path.dirname(nnunetv2.__file__), 'training', 'nnUNetTrainer', 'variants', 'factory'); \
os.makedirs(d, exist_ok=True); \
open(os.path.join(d, '__init__.py'), 'w').close(); \
open(os.path.join(d, 'mlflow_trainer.py'), 'w').write('from modelfactory.trainers.mlflow_trainer import nnUNetTrainerMLflow\n'); \
open(os.path.join(d, 'hpo_trainer.py'), 'w').write('from modelfactory.trainers.hpo_trainer import nnUNetTrainerHPO\n'); \
open(os.path.join(d, 'small_structures_trainer.py'), 'w').write('from modelfactory.trainers.mlflow_trainer import nnUNetTrainerSmallStructuresMLflow\n'); \
open(os.path.join(d, 'partial_label_trainer.py'), 'w').write('from modelfactory.trainers.mlflow_trainer import nnUNetTrainerPartialLabelMLflow\n'); \
open(os.path.join(d, 'partial_label_balanced_trainer.py'), 'w').write('from modelfactory.trainers.mlflow_trainer import nnUNetTrainerPartialLabelBalancedMLflow\n'); \
p = os.path.join(os.path.dirname(nnunetv2.__file__), 'experiment_planning', 'factory'); \
os.makedirs(p, exist_ok=True); \
open(os.path.join(p, '__init__.py'), 'w').close(); \
open(os.path.join(p, 'highres_planner.py'), 'w').write('from modelfactory.planners.highres_planner import nnUNetPlannerResEncL_HighRes\n')"

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    nnUNet_raw=/factory/datasets \
    nnUNet_preprocessed=/factory/preprocessed \
    nnUNet_results=/factory/results \
    nnUNet_n_proc_DA=18 \
    # MLflow defaults — overridden at Job submission time
    MLFLOW_S3_IGNORE_TLS=true

WORKDIR /factory

# Default to an interactive shell; Jobs override CMD with their training invocation.
CMD ["bash"]
