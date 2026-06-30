# syntax=docker/dockerfile:1.7
#
# totalseg-trainer: extends nnunet-trainer with TotalSegmentator.
#
# TotalSegmentator uses nnUNetv2 under the hood (3d_fullres + nnUNetTrainerNoMirroring),
# so the trainer stack is identical. We add:
#   - TotalSegmentator CLI for prediction
#   - Optional pre-staged weights via TOTALSEG_WEIGHTS_PATH env (mounted at runtime,
#     not baked into the image — keeps the image small and avoids re-pulling)
#
# License flag: TotalSegmentator code is Apache-2.0, but weights are NOT uniformly
# licensed. CT weights are CC-BY-4.0 (inherits from TotalSegmentator dataset).
# MR-specific subtask weights are CC-BY-NC-SA → commercial use blocked. See
# docs/licensing.md for the per-subtask audit.

ARG BASE=registry.model-factory.svc:5000/nnunet-trainer:0.1.0
FROM ${BASE}

RUN pip install --no-cache-dir --no-deps TotalSegmentator==2.4.*

ENV TOTALSEG_WEIGHTS_PATH=/factory/weights/totalseg

CMD ["bash"]
