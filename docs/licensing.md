# Licensing audit

This file tracks the upstream license status of every model + dataset + library
that flows through the factory, so you know what is safe to ship in **your
downstream product** commercially. Tag every registered model with
`tags.base_model` and `tags.dataset_license` so lineage stays auditable.

## Models

| Component               | Code license | Weights license   | Commercial use? | Notes |
|-------------------------|--------------|-------------------|-----------------|-------|
| nnUNetv2 (MIC-DKFZ)     | Apache-2.0   | n/a (you train)   | ✅              | Your trained weights are yours |
| TotalSegmentator (code) | Apache-2.0   | n/a               | ✅              | The CLI/lib itself is permissive |
| TotalSegmentator (CT weights)   | n/a   | CC-BY-4.0         | ✅ with attribution | Attribution required in product UI / docs |
| TotalSegmentator (MR weights)   | n/a   | **CC-BY-NC-SA**   | ❌ **commercial blocked** | Cannot redistribute or use commercially. Retrain from licensed data or procure a commercial license. |

## Datasets (training inputs)

| Dataset                 | License            | Commercial OK? |
|-------------------------|--------------------|----------------|
| MSD (Medical Segmentation Decathlon) | CC-BY-SA-4.0 | ⚠️ share-alike contaminates derivative weights — avoid for shippable models |
| SegRap2023 (Datasets 083-088) | CC-BY-SA-4.0 (per release `dataset_task001.json`) | ⚠️ share-alike — same caveat as MSD. NOTE: the challenge page says data is "for the purpose of the challenge" but the authoritative license file in the release is CC-BY-SA-4.0. Confirm with xiangde.luo@uestc.edu.cn before shipping. |
| KiTS21 / KiTS23         | CC-BY-NC-SA-4.0    | ❌ NC clause blocks commercial use |
| AMOS22                  | CC-BY-4.0          | ✅ |
| TCIA collections        | varies per collection | per-collection audit |
| Internal RT planning data | proprietary      | ✅ (we own it) |

## Libraries

| Library         | License      |
|-----------------|--------------|
| PyTorch         | BSD-3        |
| nnUNetv2        | Apache-2.0   |
| batchgenerators | Apache-2.0   |
| SimpleITK       | Apache-2.0   |
| MLflow          | Apache-2.0   |
| Kubernetes Python client | Apache-2.0 |
| Kueue           | Apache-2.0   |

All current dependencies are Apache-2.0 or BSD-3. No GPL leakage.

## Process

- Before adding a new dataset to `/data/model-factory-nfs/datasets/`, update
  the table above. If the license is unclear, **do not register it**.
- Before fine-tuning from a public checkpoint, record the source URL and
  license here.  Released models inherit the most restrictive license of
  any input that contributed to their weights.
- Track derived-model lineage in MLflow tags: every registered model should
  have `tags.base_model` (URL or `from-scratch`) and `tags.dataset_license`.
