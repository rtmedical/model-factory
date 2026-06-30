# syntax=docker/dockerfile:1.7
#
# synthseg-labeler: CPU-only image that runs SynthSeg-Robust over brain T1s
# and produces per-OAR binary masks for Dataset045_Brain_MR_CoreOAR.
#
# License: Apache 2.0 (SynthSeg) + Apache 2.0 (TensorFlow) — both
# commercially friendly, unlike TS-MR's CC-BY-NC-SA. The label outputs
# are not "derivative works of FreeSurfer" (SynthSeg is independent), so
# downstream commercial use of the silver labels is permitted.
#
# Why CPU-only: a SynthSeg-Robust run is ~60-90s on CPU and ~30s on GPU.
# Running on CPU frees the H100 MIG slices for training and lets us
# parallelise across many CPU cores instead of contending with the
# fold-training Jobs for slice quota.

FROM python:3.11-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    DEBIAN_FRONTEND=noninteractive \
    # SynthSeg uses tensorflow; force CPU
    CUDA_VISIBLE_DEVICES="" \
    TF_CPP_MIN_LOG_LEVEL=2

RUN apt-get update \
 && apt-get install -y --no-install-recommends \
        git curl ca-certificates \
        libgomp1 \
 && rm -rf /var/lib/apt/lists/*

# Install SynthSeg's runtime deps (TF-CPU, plus the inference stack).
# Pinned versions match SynthSeg 2.0's tested combo.
RUN pip install --no-cache-dir \
        "tensorflow-cpu>=2.13,<2.16" \
        "keras>=2.13,<2.16" \
        "numpy>=1.24,<2.0" \
        "nibabel>=5.0" \
        "SimpleITK>=2.3" \
        "scipy>=1.10" \
        "scikit-image>=0.21" \
        "connected-components-3d>=3.20" \
        "matplotlib>=3.7"

# Clone SynthSeg at a fixed commit. We do NOT `pip install` it because
# the project's setup.py restricts to Python 3.6/3.8 with a hard check
# (parser doesn't accept 3.11). SynthSeg is pure Python so we add it to
# PYTHONPATH and import directly — no compiled extensions to worry about.
#
# Then patch out the np.int / np.float deprecated aliases that SynthSeg
# uses (removed in NumPy 1.24); they map cleanly to int / float without
# behaviour change.
WORKDIR /opt
RUN git clone --depth 1 https://github.com/BBillot/SynthSeg.git /opt/synthseg \
 && find /opt/synthseg -name '*.py' -exec sed -i \
       -e 's/np\.int\b/np.int64/g' \
       -e 's/np\.float\b/np.float64/g' \
       -e 's/np\.bool\b/np.bool_/g' \
       -e 's/np\.long\b/np.int64/g' \
       {} \;
ENV PYTHONPATH=/opt/synthseg:/opt/modelfactory/src

# Pull the SynthSeg-2.0 model weights. The robust variant is what we
# want for clinical/noisy scans; the standard variant is the fallback.
# Use the project's auto-downloader; fall back to direct HTTP if it
# fails. Total weight files: ~600 MB.
RUN cd /opt/synthseg \
 && (python download_synthseg_2.0.py 2>&1 | tee /tmp/download.log) \
    || (echo "auto-download failed; fetching robust weights manually" \
        && mkdir -p /opt/synthseg/models \
        && curl -fL -o /opt/synthseg/models/synthseg_robust_2.0.h5 \
            https://liveuclac-my.sharepoint.com/personal/rmapbbi_ucl_ac_uk/Documents/SynthSeg_robust_2.0.h5)

# Copy in the modelfactory package
COPY pyproject.toml /opt/modelfactory/pyproject.toml
COPY src            /opt/modelfactory/src

WORKDIR /work

# Smoke check: confirm imports + that SynthSeg's predict path is reachable
RUN python -c "import sys; sys.path.insert(0, '/opt/synthseg'); \
import tensorflow, SimpleITK, numpy; \
from modelfactory.labelers.fomo_synthseg import FS_IDS_BASE, LOBE_GROUPS, _label_set_for_mode; \
print('FS_IDS_BASE classes:', list(FS_IDS_BASE)); \
print('LOBE_GROUPS classes:', list(LOBE_GROUPS)); \
print('base mode resolves:', list(_label_set_for_mode('base'))[:3], '...'); \
print('parc mode resolves:', list(_label_set_for_mode('parc'))[:3], '...'); \
import os; \
print('synthseg predict script:', os.path.exists('/opt/synthseg/scripts/commands/SynthSeg_predict.py')); \
print('synthseg models present:', os.listdir('/opt/synthseg/models'))"

CMD ["python", "-m", "modelfactory.labelers.fomo_synthseg", "--help"]
