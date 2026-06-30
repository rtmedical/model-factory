# syntax=docker/dockerfile:1.7
#
# Global ARG — declared before any FROM so it's in scope for every FROM line
# in this multi-stage build. Stage-local re-declarations below pick the same
# default up inside each stage when they need to expand it in RUN/COPY too.
ARG NGC_TAG=25.04-py3
#
# model-qa viewer — single image, single pod.
#
# Three stages:
#   1) web-builder (node)        — Next.js static export → /app/out
#   2) py-wheels   (NGC PyTorch) — build wheels with --no-deps (mirrors the
#                                  nnunet-trainer recipe: never overwrite the
#                                  NGC-tuned torch; install nnunetv2 with
#                                  --no-deps and bring its real deps explicitly)
#   3) runtime     (NGC PyTorch) — installs wheels + the modelfactory SDK,
#                                  copies the web/out static export to
#                                  /opt/qa-viewer/web, runs uvicorn.
#
# CRITICAL RULES (from CLAUDE.md non-negotiable #5):
#   - Never `pip install torch` on top of the NGC base.
#   - Install nnunetv2 with --no-deps; bring real deps explicitly.

# ─────────────────────────── web ─────────────────────────────────────────
FROM node:20-alpine AS web-builder

WORKDIR /app
COPY services/qa-viewer/web/package.json ./
# --legacy-peer-deps: cornerstonejs/tools 1.86 pins peer vtk.js 30.4.1, but
# the project tracks ^32.6.0 for newer rendering APIs. Skip the peer check
# so the major-newer vtk.js installs alongside cornerstonejs.
RUN npm install --no-fund --no-audit --legacy-peer-deps

COPY services/qa-viewer/web/ ./
ENV NEXT_TELEMETRY_DISABLED=1
# `output: 'export'` (set in next.config.ts) writes the static bundle to ./out
RUN npm run build

# ─────────────────────────── wheels ──────────────────────────────────────
FROM nvcr.io/nvidia/pytorch:${NGC_TAG} AS py-wheels

RUN pip install --no-cache-dir --upgrade pip wheel setuptools

# Same package set as nnunet-trainer for the inference path, plus the QA HTTP
# layer (fastapi/uvicorn/nibabel). NOT torch — never torch.
RUN pip wheel --no-cache-dir --no-deps -w /wheels \
        nnunetv2==2.5.* \
        batchgenerators==0.25 \
        batchgeneratorsv2>=0.2 \
        SimpleITK==2.4.* \
        nibabel \
        scikit-image \
        scipy \
        tqdm \
        connected-components-3d \
        acvl-utils \
        "dynamic-network-architectures>=0.3.1,<0.4" \
        einops

RUN pip wheel --no-cache-dir -w /wheels \
        fastapi \
        "uvicorn[standard]" \
        python-multipart \
        click \
        pydantic \
        pyyaml \
        rich \
        "redis>=5.0"

# ─────────────────────────── runtime ─────────────────────────────────────
FROM nvcr.io/nvidia/pytorch:${NGC_TAG}

LABEL org.opencontainers.image.title="model-qa-viewer"
LABEL org.opencontainers.image.description="Unified FastAPI + Next.js QA viewer for model-factory checkpoints"
LABEL org.opencontainers.image.source="https://internal/model-factory"

# Install Python wheels first — letting pip resolve transitives is fine; the
# /wheels dir is preferred via --find-links so torch is never refetched.
COPY --from=py-wheels /wheels /wheels
RUN pip install --no-cache-dir --find-links=/wheels \
        nnunetv2 batchgenerators batchgeneratorsv2 SimpleITK nibabel \
        scikit-image scipy tqdm connected-components-3d acvl-utils \
        "dynamic-network-architectures>=0.3.1,<0.4" einops \
        fastapi "uvicorn[standard]" python-multipart \
        click pydantic pyyaml rich "redis>=5.0" \
    && rm -rf /wheels

# Guard: NGC torch must still be the 2.7.x family.
RUN python -c "import torch; v = torch.__version__; \
assert v.startswith('2.7'), f'torch was overwritten: {v}'; \
print(f'[guard] torch={v}')"

# SDK + the FastAPI app live in the modelfactory package (modelfactory.qa.api).
COPY pyproject.toml /opt/modelfactory/pyproject.toml
COPY src           /opt/modelfactory/src
RUN pip install --no-cache-dir --no-deps -e /opt/modelfactory

# Bridge: nnUNetv2 2.5 finds trainer classes by walking
# <site-packages>/nnunetv2/training/nnUNetTrainer/ recursively (via
# pkgutil.iter_modules + recursive_find_python_class). It does NOT honour
# Python entry-point groups, so the `nnunetv2.trainers` group in
# pyproject.toml is a dead letter. Mirror images/nnunet-trainer/Dockerfile:
# drop a shim .py file under variants/factory/ that imports the trainer
# class — the walker picks it up via `hasattr(module, class_name)`.
#
# mlflow is NOT installed in this image (the qa-viewer never trains, only
# infers). mlflow_trainer.py imports mlflow lazily inside its on_* methods,
# so the shim's `from modelfactory.trainers.mlflow_trainer import ...`
# succeeds and `build_network_architecture` (inherited from nnUNetTrainer)
# is enough to run predictions.
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

# Static frontend — Next.js exports the built site to ./out.
COPY --from=web-builder /app/out /opt/qa-viewer/web

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    QA_FACTORY_ROOT=/factory \
    QA_WEB_STATIC_DIR=/opt/qa-viewer/web \
    QA_PREDICTOR_CACHE_SIZE=3 \
    nnUNet_raw=/factory/datasets \
    nnUNet_preprocessed=/factory/preprocessed \
    nnUNet_results=/factory/results

EXPOSE 8080
WORKDIR /factory

CMD ["uvicorn", "modelfactory.qa.api:app", \
     "--host", "0.0.0.0", "--port", "8080", "--workers", "1", \
     "--log-level", "info"]
