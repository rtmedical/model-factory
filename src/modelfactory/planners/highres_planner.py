"""Higher-resolution planner for tiny-structure datasets.

Default nnUNet planners pick target spacing from dataset median spacing. For
datasets with thin tubular structures (optic nerves, lenses, vessels), the
auto-derived spacing produces 1-2-voxel cross-sections that the multi-stage
UNet erases at the first downsampling. This planner forces a finer in-plane
target spacing while preserving the source axial resolution — so the upsample
is only in directions where the source actually has the resolution.

Use case: PDDCA H&N CTs are 3 mm axial / 1.0-1.2 mm in-plane. Default L-plans
spacing ~1.5 mm isotropic upsamples axial (fake interpolation) and underuses
the in-plane resolution. This planner produces ~3.0 mm axial / 0.7 mm in-plane
plans — optic nerves become 3-4 voxels thick in cross-section.

Used by datasets tagged `small_structures: true` in SPECS.
"""

from __future__ import annotations

import numpy as np

from nnunetv2.experiment_planning.experiment_planners.residual_unets.residual_encoder_unet_planners import (
    nnUNetPlannerResEncL,
)


class nnUNetPlannerResEncL_HighRes(nnUNetPlannerResEncL):
    """ResEncL variant that pins in-plane spacing to TARGET_INPLANE_SPACING.

    Axial spacing (largest median spacing axis) is left at the auto-derived
    value — upsampling axial from 3 mm slices to 0.7 mm would be fake
    interpolation; in-plane CT is already submillimeter, so the upsample is
    real.

    Writes plans + preprocessed data under the distinct name
    `nnUNetResEncUNetLPlans_HighRes` so coexistence with the default L plans
    is clean (each gets its own plans.json + per-config preprocessed dir).
    """

    TARGET_INPLANE_SPACING: float = 0.7  # mm; tuned for PDDCA optic nerves

    def __init__(self, dataset_name_or_id, gpu_memory_target_in_gb: float = 24,
                 preprocessor_name: str = "DefaultPreprocessor",
                 plans_name: str = "nnUNetResEncUNetLPlans_HighRes",
                 overwrite_target_spacing=None, suppress_transpose: bool = False):
        super().__init__(
            dataset_name_or_id, gpu_memory_target_in_gb, preprocessor_name,
            plans_name, overwrite_target_spacing, suppress_transpose,
        )

    def determine_fullres_target_spacing(self) -> np.ndarray:
        spacing = np.asarray(super().determine_fullres_target_spacing(), dtype=float)
        # Identify the through-slice (axial) axis as the largest median
        # spacing — leave it untouched, only refine in-plane axes.
        axial = int(np.argmax(spacing))
        for i in range(3):
            if i != axial:
                spacing[i] = min(spacing[i], float(self.TARGET_INPLANE_SPACING))
        return spacing
