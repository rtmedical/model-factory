"""Validate and materialize an nnUNetv2 dataset into the factory NFS root.

A "registered" dataset lives at:
    {nfs_host_root}/datasets/Dataset{ID}_{name}/
        imagesTr/                CASE_0000.nii.gz, CASE_0001.nii.gz, ...
        labelsTr/                CASE.nii.gz
        dataset.json
        [splits_final.json]      optional pre-computed splits

This module:
  - validates the dataset.json schema (matches nnUNetv2 v2.5 contract)
  - asserts every image and label exists with the right channel suffix
  - optionally moves (rather than copies) data into the NFS root for large
    datasets — symlinks are NOT used because the container view rewrites
    /factory and symlinks would dangle
"""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path

from pydantic import BaseModel, Field, ValidationError, field_validator


class DatasetJsonV2(BaseModel):
    """Minimal validating model for nnUNetv2's dataset.json."""

    channel_names: dict[str, str]
    labels: dict[str, int | list[int]]
    file_ending: str = Field(pattern=r"^\.[a-zA-Z0-9.]+$")
    numTraining: int = Field(gt=0)

    @field_validator("channel_names")
    @classmethod
    def keys_are_ints(cls, v: dict[str, str]) -> dict[str, str]:
        for k in v:
            if not k.isdigit():
                raise ValueError(f"channel_names key '{k}' must be a string-encoded integer")
        return v


_DATASET_DIR_RE = re.compile(r"^Dataset(\d{3,})_([A-Za-z0-9][A-Za-z0-9_.-]*)$")


def validate(dataset_dir: Path) -> DatasetJsonV2:
    """Validate the dataset directory layout and return the parsed dataset.json."""
    if not dataset_dir.is_dir():
        raise FileNotFoundError(dataset_dir)
    m = _DATASET_DIR_RE.match(dataset_dir.name)
    if not m:
        raise ValueError(
            f"Directory '{dataset_dir.name}' does not match Dataset{{ID}}_{{name}} pattern"
        )

    ds_json_path = dataset_dir / "dataset.json"
    if not ds_json_path.is_file():
        raise FileNotFoundError(ds_json_path)
    try:
        ds = DatasetJsonV2(**json.loads(ds_json_path.read_text()))
    except ValidationError as e:
        raise ValueError(f"Invalid dataset.json: {e}") from e

    images = dataset_dir / "imagesTr"
    labels = dataset_dir / "labelsTr"
    if not images.is_dir() or not labels.is_dir():
        raise FileNotFoundError(f"Expected {images} and {labels}")

    # Quick sanity: count cases by labels (nnUNet's discovery rule); each must
    # have N channels in imagesTr (one file per channel with _{0000,0001,...} suffix).
    n_channels = len(ds.channel_names)
    label_files = sorted(labels.glob(f"*{ds.file_ending}"))
    if len(label_files) != ds.numTraining:
        raise ValueError(
            f"dataset.json says numTraining={ds.numTraining} but found "
            f"{len(label_files)} label files"
        )
    for lf in label_files:
        case = lf.name.removesuffix(ds.file_ending)
        for ch in range(n_channels):
            img = images / f"{case}_{ch:04d}{ds.file_ending}"
            if not img.is_file():
                raise FileNotFoundError(f"Missing channel image: {img}")

    return ds


def register(
    source: Path,
    nfs_root: Path,
    move: bool = False,
    overwrite: bool = False,
) -> Path:
    """Validate `source` and copy/move it under `nfs_root/datasets/`. Returns dest path."""
    ds = validate(source)
    dest = nfs_root / "datasets" / source.name
    if dest.exists():
        if not overwrite:
            raise FileExistsError(f"{dest} already exists — pass overwrite=True to replace")
        shutil.rmtree(dest)

    dest.parent.mkdir(parents=True, exist_ok=True)
    if move:
        shutil.move(str(source), str(dest))
    else:
        shutil.copytree(source, dest)
    _ = ds  # validated; nothing else to do
    return dest
