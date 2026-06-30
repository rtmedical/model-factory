"""Structure-name alias tables for the clinical RTSTRUCT cohorts at
``/data/<cohort>/``, ``/data/<cohort>/`` and ``/data/<cohort>/``.

The internal RT-Medical archive labels structures with a mix of:
  - TG-263 nomenclature (``OpticNrv_L``, ``Glnd_Submand_L``, ``Bone_Mandible``)
  - Portuguese clinical names from the planning system
    (``Cristalino E`` = Lens_L, ``Tronco`` = Brainstem, ``Cóclea D`` = Cochlea_R)
  - Idiosyncratic variants per dosimetrist (``CócleaD`` no-space, ``Cóclea Esq.``)

This module exposes a forward-alias dict ``CLINICAL_RTSTRUCT_ALIASES`` mapping
each known on-disk variant to the *factory canonical* name used in
``StructureMapping.canonical`` across the rest of the repo. The
``clinical_rtstruct`` source adapter applies this dict per-case during
discovery and again when looking up the ROI inside the RTSTRUCT.

Planning-derivative names (PRVs, dose-optimisation surrogates, "z…otm",
"_prv…", "…_0.5", "…_05", "…_3MM") are deliberately **not aliased**: those
are expansions of an OAR, not the OAR itself, and conflating them would
contaminate training masks. Discovery filters them out by inclusion: only
exact alias-table hits count.
"""

from __future__ import annotations

import re


# RTSTRUCT / planning-derivative suffixes that decorate an OAR's base name
# (e.g. "Brain_MR7", "OpticNrv_L_OTM", "Rectum_PRV05", "Esophagus_3mm").
# Stripped before alias lookup so the underlying OAR still resolves.
# Laterality (_L/_R) is NOT a suffix and is preserved.
_PLANNING_SUFFIX_RE = re.compile(r"_(MR\d*|OTM|PRV\d*|\d+MM)$", re.IGNORECASE)


def strip_planning_suffix(name: str) -> str:
    """Remove a trailing RTSTRUCT version/derivative suffix, if present."""
    return _PLANNING_SUFFIX_RE.sub("", name)


def canonical_for(name: str, aliases: dict[str, str]) -> str | None:
    """Resolve an on-disk ROI name to a canonical name via `aliases`.

    Tries the name as-is first, then with a planning/version suffix stripped.
    Returns None if neither resolves (the ROI is not a tracked OAR).
    """
    hit = aliases.get(name)
    if hit:
        return hit
    return aliases.get(strip_planning_suffix(name))


# ── Head & Neck ─────────────────────────────────────────────────────────────
#
# Canonical names chosen to match what's already in
# ``modelfactory.datasets.specs`` (OpticNerve_L not OpticNrv_L, Chiasm not
# OpticChiasm, Parotid_L not Parótida_E). New canonicals introduced for
# structures the repo hadn't seen before are noted inline.
#
# Frequency comments are the case-count from /data/<cohort> metadata.json
# (538 cases total, 537 with RTSTRUCT) as observed 2026-05-22.

HN_CLINICAL_ALIASES: dict[str, str] = {
    # ── Optic apparatus ───────────────────────────────────────────────────
    "OpticNrv_L": "OpticNerve_L",          # 398 cases (TG-263)
    "OpticNrv_R": "OpticNerve_R",          # 398
    "Nervo ótico E": "OpticNerve_L",       # 95
    "Nervo ótico D": "OpticNerve_R",       # 95
    "Nervo Óptico E": "OpticNerve_L",      # 1
    "Nervo Óptico D": "OpticNerve_R",      # 1
    "OpticChiasm": "Chiasm",               # 357
    # ── Eyes / lens ───────────────────────────────────────────────────────
    "Eye_L": "Eye_L",                      # 408
    "Eye_R": "Eye_R",                      # 408
    "Olho E": "Eye_L",                     # 87
    "Olho D": "Eye_R",                     # 87
    "Lens_L": "Lens_L",                    # 403
    "Lens_R": "Lens_R",                    # 402
    "Cristalino E": "Lens_L",              # 96
    "Cristalino D": "Lens_R",              # 96
    # ── Cochlea ───────────────────────────────────────────────────────────
    "Cochlea_L": "Cochlea_L",              # 416
    "Cochlea_R": "Cochlea_R",              # 417
    "Cóclea E": "Cochlea_L",               # 9
    "Cóclea D": "Cochlea_R",               # 10
    "CócleaE": "Cochlea_L",                # 3
    "CócleaD": "Cochlea_R",                # 3
    "Cóclea Esq.": "Cochlea_L",            # 1
    "Cóclea Esquerda": "Cochlea_L",        # 2
    "Cóclea Dir.": "Cochlea_R",            # 1
    "Cóclea Direita": "Cochlea_R",         # 1
    # ── Salivary / thyroid glands ─────────────────────────────────────────
    "Parotid_L": "Parotid_L",              # 264
    "Parotid_R": "Parotid_R",              # 272
    "Parótida E": "Parotid_L",             # 71
    "Parótida D": "Parotid_R",             # 71
    "Parótida dir": "Parotid_R",           # 1
    "Glnd_Submand_L": "Submandibular_L",   # 245 (TG-263 → repo canonical)
    "Glnd_Submand_R": "Submandibular_R",   # 243
    "Submandibular E": "Submandibular_L",  # 59
    "Submandibular D": "Submandibular_R",  # 59
    "Glnd_Thyroid": "Thyroid",             # 255 (new canonical Thyroid)
    "Thyroid": "Thyroid",                  # 1
    "Tireoide": "Thyroid",                 # 1
    # ── Aerodigestive lumen ───────────────────────────────────────────────
    "Cavity_Oral": "OralCavity",           # 266
    "Cavidade oral": "OralCavity",         # 69
    "Larynx": "Larynx",                    # 270
    "Glote": "Larynx_Glottic",             # 63 (per repo's D095 canonical)
    "Musc_Constrict": "Pharynx",           # 266 (per repo's D090 canonical;
    #                                       the segrap PharynxConst alias.
    #                                       OAR-equivalent in our taxonomy.)
    "Esophagus": "Esophagus",              # 285
    "Esôfago": "Esophagus",                # 77
    "Trachea": "Trachea",                  # 282
    "Traqueia": "Trachea",                 # 68
    "Lips": "Lips",                        # 268
    "Lábios": "Lips",                      # — observed in samples
    "Mucosa": "Mucosa",                    # 72 (new canonical)
    "Mucosa_Bucal": "Mucosa",              # 1
    # ── Bone / brachial plexus ────────────────────────────────────────────
    "Bone_Mandible": "Bone_Mandible",      # 277 (new canonical — single
    #                                       bilateral piece, distinct from
    #                                       D091's split Mandible_L / _R)
    "Mandíbula": "Bone_Mandible",          # 76
    "BrachialPlex_L": "BrachialPlex_L",    # 278 (new canonical)
    "BrachialPlex_R": "BrachialPlex_R",    # 276
    # ── CNS / spinal canal ────────────────────────────────────────────────
    "Brainstem": "Brainstem",              # 404
    "Tronco": "Brainstem",                 # 79
    "Tronco cerebral": "Brainstem",        # 18
    "SpinalCanal": "SpinalCanal",          # 403 (new canonical)
    "Canal medular": "SpinalCanal",        # 75
    "Canal Medular": "SpinalCanal",        # 4
    "Pituitary": "Pituitary",              # 122
    "Glândula pituit": "Pituitary",        # — observed in samples
    "Brain": "Brain",                      # 143
    "Encéfalo": "Brain",                   # — observed in samples
    "Cérebro": "Brain",                    # — observed in samples
    "Hippocampus_L": "Hippocampus_L",      # 105
    "Hippocampus_R": "Hippocampus_R",      # 87
    "Hipocampo E": "Hippocampus_L",        # 16
    "Hipocampo D": "Hippocampus_D",        # 17 (typo guard: see note below)
    # ── Neck nodal-level CTVs (TARGETS, not OARs; added 2026-06-02) ────────
    # Elective nodal volumes — the biggest untapped /data/<cohort> coverage. Clean
    # TG-263-style names (self-mapped). Grouped into 3 regional models
    # (D146 lateral II-V, D147 central Ia/Ib/VIa, D148 retropharyngeal VIIa/b).
    "LN_Neck_IA": "LN_Neck_IA",            # 209
    "LN_Neck_IB_L": "LN_Neck_IB_L",        # 178
    "LN_Neck_IB_R": "LN_Neck_IB_R",        # 182
    "LN_Neck_II_L": "LN_Neck_II_L",        # 213
    "LN_Neck_II_R": "LN_Neck_II_R",        # 217
    "LN_Neck_III_L": "LN_Neck_III_L",      # 213
    "LN_Neck_III_R": "LN_Neck_III_R",      # 215
    "LN_Neck_IV_L": "LN_Neck_IV_L",        # 209
    "LN_Neck_IV_R": "LN_Neck_IV_R",        # 214
    "LN_Neck_V_L": "LN_Neck_V_L",          # 199
    "LN_Neck_V_R": "LN_Neck_V_R",          # 207
    "LN_Neck_VIA": "LN_Neck_VIA",          # 176
    "LN_Neck_VIIA_L": "LN_Neck_VIIA_L",    # 188
    "LN_Neck_VIIA_R": "LN_Neck_VIIA_R",    # 195
    "LN_Neck_VIIB_L": "LN_Neck_VIIB_L",    # 202
    "LN_Neck_VIIB_R": "LN_Neck_VIIB_R",    # 208
}

# Typo guard: the "Hipocampo D" entry above was a deliberate ALMOST-bug —
# Portuguese "D" means right (Direita) so the canonical is Hippocampus_R,
# not Hippocampus_D. Correct it here so the lookup stays right.
HN_CLINICAL_ALIASES["Hipocampo D"] = "Hippocampus_R"


# ── Pelvis (frequency survey 2026-05-27 on /data/<cohort>, n=1000 with meta) ─
#
# Canonicals chosen to match TS-derived naming where possible
# (Femur_L/Femur_R, not Femur_RTOG_L). Portuguese variants are observed
# in a small subset of cases authored by older planning workflows; they
# add 10-50 extra cases per OAR.
#
# Planning-derivative names that are intentionally NOT aliased (so they
# don't pollute training masks): Rectum_3mm/Rectum_PRV03 (465+177 cases,
# 3mm expansion PRV), SpinalCanal_0.5/SpinalCanal_PRV05, BODY, Skin,
# CTV*, PTV*, GTV*, _anel*, _a1/a2/a3, _PTV*otm, NS_*, DIBH/FB Contour.
#
# SacralPlex (unsplit, 130 cases) is intentionally NOT aliased — the
# specialist trains on the L/R-split cases (141/140) only. Mixing in
# bilateral-unsplit cases would require a separate canonical the
# spec does not enumerate.
#
# Intestino (34) is ambiguous between SmallBowel and Bowel_Bag in the
# Portuguese planning vocabulary — left unaliased; clinicians who want
# this case-bucket should split it manually in the source.

PELVIS_CLINICAL_ALIASES: dict[str, str] = {
    # ── Genitourinary ────────────────────────────────────────────────────
    "Bladder": "Bladder",                       # 868
    "Bexiga": "Bladder",                        # 34
    "Prostate": "Prostate",                     # 281
    "Próstata": "Prostate",                     # 13
    "SeminalVes": "SeminalVes",                 # 303
    "Vesículas": "SeminalVes",                  # 11
    "PenileBulb": "PenileBulb",                 # 581
    "Bulbo peniano": "PenileBulb",              # 10
    # ── Bowel ────────────────────────────────────────────────────────────
    "Rectum": "Rectum",                         # 755
    "Reto": "Rectum",                           # 34
    "Colon_Sigmoid": "Colon_Sigmoid",           # 744
    "Sigmoide": "Colon_Sigmoid",                # 159
    "Bowel_Bag": "Bowel_Bag",                   # 852
    "SmallBowel": "SmallBowel",                 # 127
    "LargeBowel": "LargeBowel",                 # 116
    # ── Gynaecologic (female) ────────────────────────────────────────────
    # OAR names only. The CTV*/PTV*/MTV* "uterino"/"vagina" variants are
    # planning targets, NOT OARs — deliberately left unaliased (discovery
    # includes only exact alias hits, so they are filtered out). Uterus body
    # folds into the combined utero-cervix OAR used by D136/D154.
    "UteroCervix": "UteroCervix",               # 110 (matches D136 canonical)
    "Uterus": "UteroCervix",                    # 6 (uterus-body variant → combined OAR)
    # ── Skeletal ─────────────────────────────────────────────────────────
    "Femur_RTOG_L": "Femur_L",                  # 859
    "Femur_RTOG_R": "Femur_R",                  # 862
    "Femur E": "Femur_L",                       # 36 (Portuguese E=Esquerda)
    "Femur D": "Femur_R",                       # 36 (Portuguese D=Direita)
    "Bone_Pelvic": "Bone_Pelvic",               # 275
    # ── Neural (pelvis) ──────────────────────────────────────────────────
    "CaudaEquina": "CaudaEquina",               # 813
    "SacralPlex_L": "SacralPlex_L",             # 141
    "SacralPlex_R": "SacralPlex_R",             # 140
    # ── Lymph nodes (pelvic nodal CTV; pure TS gap, D139) ────────────────
    "LN_Pelvics": "LN_Pelvics",                 # 615 (biggest untrained gap)
    "LN_Pélvicos": "LN_Pelvics",                # Portuguese variant guard
    "LN_Pelvic": "LN_Pelvics",                  # singular variant guard
}


# ── Thorax (frequency survey 2026-05-27 on /data/<cohort>, n=1000 with meta) ─
#
# Bilateral-breast cohort (89% C50.x), so Heart/Lungs/Esophagus are dense.
# Carina, A_LAD, Breast_L/Breast_R are the real TS gaps the specs cover.
# Aorta sub-segments (Asc/Dsc/Arch) are too sparse (≤105) and not used in
# any spec; we still alias them so the audit histogram stays informative.
#
# Planning-derivative names NOT aliased: SpinalCanal_0.5/PRV05,
# Esophagus_PRV03/Esophagus_3mm, all CTV*/PTV*/GTV* including the
# Portuguese "CTV mama"/"CTV LEITO"/"PTV Mama"/"PTV mama" variants,
# Cicatriz (scar), Tumor Bed, DIBH/FB Contour, NS_* prosthesis/catheter
# markers, _AVOID, _a1.._a4, _anel*.
#
# BrachialPlexus (60 unsplit) NOT aliased — D131 trains on split-L/R only.

THORAX_CLINICAL_ALIASES: dict[str, str] = {
    # ── Cardiothoracic ───────────────────────────────────────────────────
    "Heart": "Heart",                           # 947
    "Lung_L": "Lung_L",                         # 940
    "Lung_R": "Lung_R",                         # 938
    "Lungs": "Lungs",                           # 925 (kept distinct from L/R)
    "Liver": "Liver",                           # 786
    # ── Airway ───────────────────────────────────────────────────────────
    "Carina": "Carina",                         # 932
    "Bronchus": "Bronchus",                     # 91
    # ── Vascular (great vessels, coronaries) ─────────────────────────────
    "A_LAD": "A_LAD",                           # 725 (left anterior descending)
    "A_Coronary_R": "A_Coronary_R",             # 210
    "A_Aorta": "A_Aorta",                       # 94 (unsegmented whole)
    "A_Aorta_Asc": "A_Aorta_Asc",               # 92 (sparse, not in any spec)
    "A_Aorta_Dsc": "A_Aorta_Dsc",               # 105 (sparse, not in any spec)
    "V_VenaCava_S": "V_VenaCava_S",             # 92 (SVC)
    # ── Breast (bilateral cohort) ────────────────────────────────────────
    "Breast_L": "Breast_L",                     # 371
    "Breast_R": "Breast_R",                     # 414
    # ── Chestwall / ribs ─────────────────────────────────────────────────
    "Chestwall_OAR": "Chestwall_OAR",           # 87
    # ── Lymph nodes (axillary regions; not in any current spec) ──────────
    "LN_Ax_L": "LN_Ax_L",                       # 218
    "LN_Ax_R": "LN_Ax_R",                       # 197
}


# ── Convenience: combined alias dict the clinical_rtstruct adapter loads ───
#
# When a name appears in more than one regional dict, the rightmost dict
# wins. We rely on the regional dicts having disjoint keys; the audit step
# (scripts/inspect_clinical_cohort.py) catches collisions.

CLINICAL_RTSTRUCT_ALIASES: dict[str, str] = {
    **HN_CLINICAL_ALIASES,
    **PELVIS_CLINICAL_ALIASES,
    **THORAX_CLINICAL_ALIASES,
}


# ── TotalSegmentator-MRI RTSTRUCT template (D179–D182) ──────────────────
#
# The upstream TotalSegmentator-MRI image dataset (/data/totalseg-mr, 616 src
# NIfTIs) run through an external auto-segmentation tool, which emits a 58-ROI template
# (every ROI suffixed `_MR`; `strip_planning_suffix` removes the `_MR` before
# this lookup, so keys here are the STRIPPED names). We map ONLY the structures
# that are (a) a TotalSegmentator `total_mr` *gap* — i.e. not one of the 56
# whole-organ classes total_mr already segments — AND (b) populated on enough
# cases to train (≥~25/314 ≈ 8%, measured 2026-06-29). This dict is the
# complementary-model vocabulary: anything NOT listed never resolves and is
# silently ignored at discover time.
#
# DELIBERATELY OMITTED (total_mr already covers the whole organ):
#   Brain, Bladder/HDR_Bladder/Bladder_TRUFI, Prostate/Glnd_Prostate,
#   Bone_Sacrum, Femur_L/Femur_R, SpinalCord_Cerv.
# DELIBERATELY OMITTED (too sparse in THIS cohort to train — case counts in
# parentheses, /314): Lens_L/R (4/4), OpticNrv_L/R (17/12), OpticChiasm (2),
# OpticTract_L/R (4/5), Pituitary (2), Hypo_True (1). These optic/sellar OARs
# are well-covered elsewhere (D138, D045-063) — this is a body/pelvis-heavy MR
# corpus, so dedicate them to those head-MR models, not here.
# DELIBERATELY OMITTED (ambiguous definition): Optics (umbrella of Eye/Lens/
#   Cornea/Retina), HDR_Bowel (203 cases but ill-defined bowel envelope vs
#   small_bowel — revisit as Bowel_Bag once confirmed), HDR_Rec-Canal (40).
#
# Coverage (cases /314, measured 2026-06-29) is noted per entry.

TOTALSEG_MR_TEMPLATE_ALIASES: dict[str, str] = {
    # ── Brain / CNS substructures (total_mr has only whole "brain") ────────
    "Cerebellum": "Cerebellum",            # 201
    "CorpusCallosum": "CorpusCallosum",    # 245 (new canonical)
    "Falx": "Falx",                        # 246 (new — dural reflection)
    "Tentorium": "Tentorium",              # 243 (new — dural reflection)
    "Ventricle_Brain": "Ventricle_Brain",  # 245 (new — ventricular system)
    "Sinuses": "Sinuses",                  # 245 (new — paranasal sinuses)
    "Midbrain": "Midbrain",                # 265 (new — brainstem subpart)
    "Medulla": "Medulla",                  # 201 (new — brainstem subpart)
    "Pons": "Pons",                        # 85  (new — brainstem subpart)
    "Hippocampus_L": "Hippocampus_L",      # 91
    "Hippocampus_R": "Hippocampus_R",      # 101
    "Amygdala_L": "Amygdala_L",            # 117
    "Amygdala_R": "Amygdala_R",            # 79
    "Thalamus": "Thalamus",                # 77  (single ROI here, not L/R)
    # ── Ocular apparatus (total_mr has none) ──────────────────────────────
    "Eye_L": "Eye_L",                      # 242
    "Eye_R": "Eye_R",                      # 204
    "Retina_L": "Retina_L",                # 242 (new)
    "Retina_R": "Retina_R",                # 204 (new)
    "Cornea_L": "Cornea_L",                # 98  (new)
    "Cornea_R": "Cornea_R",                # 59  (new)
    # ── Pelvic functional / RT OARs (TS gaps) ─────────────────────────────
    "Urethra": "Urethra",                  # 292
    "HDR_Urethra": "Urethra",              # 179 (brachy template variant → merge)
    "Bladder_Trigone": "Bladder_Trigone",  # 272
    "A_Pud_Int_L": "A_Pud_Int_L",          # 252 (internal pudendal artery)
    "A_Pud_Int_R": "A_Pud_Int_R",          # 252
    "NVB_L": "NVB_L",                      # 178 (neurovascular bundle)
    "NVB_R": "NVB_R",                      # 183
    "SeminalVes": "SeminalVes",            # 199
    "PenileBulb": "PenileBulb",            # 88
    "PenileBulb_TRUFI": "PenileBulb",      # 182 (TRUFI-sequence variant → merge)
    "Rectal_Spacer": "Rectal_Spacer",      # 222 (SpaceOAR hydrogel)
    "Rectum": "Rectum",                    # 233
    "HDR_Rectum": "Rectum",                # 189 (brachy template variant → merge)
    "Canal_Anal": "Canal_Anal",            # alias of HDR_Canal_Anal below
    "HDR_Canal_Anal": "Canal_Anal",        # 176 (new canonical — anal canal)
    "Colon_Sigmoid": "Colon_Sigmoid",      # 313 (finer than total_mr "colon")
    "Bone_Symphys": "Bone_Symphys",        # 214 (new — pubic symphysis bone)
}


def reverse_alias_index(
    aliases: dict[str, str],
) -> dict[str, list[str]]:
    """Build {canonical → [on-disk variant, …]} for per-case mask lookup.

    The adapter uses this to, given a canonical structure to extract, try
    each known on-disk variant against the case's RTSTRUCT until one hits.
    """
    out: dict[str, list[str]] = {}
    for variant, canonical in aliases.items():
        out.setdefault(canonical, []).append(variant)
    return out


__all__ = [
    "HN_CLINICAL_ALIASES",
    "PELVIS_CLINICAL_ALIASES",
    "THORAX_CLINICAL_ALIASES",
    "CLINICAL_RTSTRUCT_ALIASES",
    "TOTALSEG_MR_TEMPLATE_ALIASES",
    "reverse_alias_index",
    "strip_planning_suffix",
    "canonical_for",
]
