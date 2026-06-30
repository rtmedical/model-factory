"""QA cohort builder + preprocessor for the model-QA web app.

Two responsibilities:
  - cohort.build_cohort: materialize /factory/qa-cohort/ from existing
    datasets (deterministic case selection, hard-copy not symlink).
  - preprocess.preprocess_cohort_for_model: pre-stage nnUNetv2-resampled,
    normalized .npz/.pkl pairs so inference can skip the heavy preproc.
"""
