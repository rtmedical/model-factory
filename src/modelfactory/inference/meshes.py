"""Pre-compute per-label surface meshes from a segmentation labelmap.

Reads the seg.nii.gz, runs marching cubes on a one-hot mask per non-
background label, and writes a `.vtp` (VTK XML PolyData) file per label.
The QA viewer's 3D fullscreen canvas then fetches those `.vtp` files
directly via vtk.js's `vtkXMLPolyDataReader` instead of running marching
cubes in the browser — turning a 0.5–2 s-per-segment CPU stall into a
network fetch.

No new runtime dependency: the mesh extraction uses scikit-image (already
in the qa-viewer image) and the `.vtp` file is hand-rolled XML with an
appended-raw binary block. The VTK XML PolyData format is well-specified
(https://docs.vtk.org/en/latest/design_documents/VTKFileFormats.html)
and the appended-raw layout vtk.js's reader accepts is:

    <VTKFile type="PolyData" version="1.0" byte_order="LittleEndian"
             header_type="UInt64">
      <PolyData>
        <Piece NumberOfPoints="N" NumberOfPolys="M" ...>
          <Points>
            <DataArray type="Float32" NumberOfComponents="3"
                       format="appended" offset="0"/>
          </Points>
          <PointData>
            <DataArray type="Float32" Name="Normals" NumberOfComponents="3"
                       format="appended" offset="O1"/>
          </PointData>
          <Polys>
            <DataArray type="Int32" Name="connectivity"
                       format="appended" offset="O2"/>
            <DataArray type="Int32" Name="offsets"
                       format="appended" offset="O3"/>
          </Polys>
        </Piece>
      </PolyData>
      <AppendedData encoding="raw">_<header><bytes><header><bytes>...</AppendedData>
    </VTKFile>

Each data block is prefixed with a UInt64 header giving its byte length
(this is what `header_type="UInt64"` declares). Vertices, faces, and
normals are little-endian as declared by `byte_order`.
"""

from __future__ import annotations

import logging
import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class MeshResult:
    out_dir: Path
    by_label: dict[int, Path]
    elapsed_s: float


def precompute_meshes(
    seg_path: Path,
    out_dir: Path,
    label_map: dict[str, int],
) -> MeshResult:
    """Write one `.vtp` per non-background label found in `label_map`.

    Empty labels (no foreground voxels, or a labelmap value the seg
    doesn't actually contain) are skipped — the frontend treats a 404
    on `mesh/{idx}` as "no surface", which is the same outcome.
    """
    import time
    import nibabel as nib  # already in the qa-viewer image
    from skimage.measure import marching_cubes  # already a dep

    from scipy.ndimage import gaussian_filter  # already a runtime dep

    t0 = time.monotonic()
    out_dir.mkdir(parents=True, exist_ok=True)

    img = nib.load(str(seg_path))
    seg = np.asarray(img.dataobj)
    # The nibabel-loaded array dtype can be float32 for older NIfTIs even
    # when the values are integers; cast to int32 for == comparisons.
    seg = seg.astype(np.int32, copy=False)
    # Convert nibabel's mm-spaced affine into per-axis spacing for
    # marching_cubes. We use the diagonal magnitude, which is correct
    # for axis-aligned volumes (the QA cohort is always reoriented to
    # RAS by nnUNet's exporter before write).
    affine = img.affine
    spacing = (
        float(np.linalg.norm(affine[:3, 0])),
        float(np.linalg.norm(affine[:3, 1])),
        float(np.linalg.norm(affine[:3, 2])),
    )

    by_label: dict[int, Path] = {}
    for name, idx in sorted(label_map.items(), key=lambda kv: kv[1]):
        if name == "background" or idx == 0:
            continue
        mask = (seg == idx).astype(np.uint8)
        if not mask.any():
            continue
        # Pre-smooth the binary mask with a small Gaussian. Marching
        # cubes on a hard 0/1 mask produces a stair-stepped surface
        # (especially visible on thin CT structures like Trachea and
        # Esophagus); a sigma ~ 0.75 voxel produces a continuous field
        # that crosses 0.5 along a smoother contour without shrinking
        # the structure noticeably. Cheap (sub-second on 256³).
        mask_f = gaussian_filter(mask.astype(np.float32), sigma=0.75)
        try:
            verts, faces, normals, _ = marching_cubes(
                mask_f,
                level=0.5,
                spacing=spacing,
                allow_degenerate=False,
            )
        except (ValueError, RuntimeError) as exc:
            # marching_cubes raises on degenerate inputs; not fatal —
            # frontend falls back to in-browser MC for this label.
            logger.warning(
                "marching_cubes failed for label %s (%d): %s — skipping",
                name, idx, exc,
            )
            continue
        if verts.shape[0] == 0 or faces.shape[0] == 0:
            continue
        # Taubin smoothing — the vtk.js in-browser path uses
        # WindowedSincPolyDataFilter for the same purpose. Taubin's
        # lambda/mu pair (positive then negative-ish step) preserves
        # the volume of the surface, unlike pure Laplacian smoothing
        # which shrinks it. 20 iterations matches the SMOOTH_ITERATIONS
        # constant on the frontend so the cached path looks identical
        # to the fallback path.
        verts = _taubin_smooth(verts, faces, iterations=20,
                               lam=0.5, mu=-0.53)
        # Recompute normals from the smoothed vertices.
        normals = _compute_vertex_normals(verts, faces)
        out_path = out_dir / f"{idx}.vtp"
        _write_vtp(out_path, verts, faces, normals)
        by_label[idx] = out_path
        logger.info(
            "mesh label %s (%d) → %s  | verts=%d faces=%d",
            name, idx, out_path.name, verts.shape[0], faces.shape[0],
        )

    elapsed = time.monotonic() - t0
    return MeshResult(out_dir=out_dir, by_label=by_label, elapsed_s=elapsed)


def _taubin_smooth(
    verts: np.ndarray,
    faces: np.ndarray,
    *,
    iterations: int,
    lam: float,
    mu: float,
) -> np.ndarray:
    """Volume-preserving mesh smoothing (Taubin λ|μ).

    For each iteration:
      v += λ · (mean(neighbour(v)) − v)   # smoothing pass
      v += μ · (mean(neighbour(v)) − v)   # inverse pass (μ < 0)

    With μ slightly more negative than λ is positive, the high-frequency
    geometric noise is removed without the global shrinkage that a pure
    Laplacian smoother causes. Same algorithm as vtk's
    WindowedSincPolyDataFilter, simpler to implement: a closed-form
    Laplacian using vertex-vertex adjacency derived from the face list.
    """
    n = verts.shape[0]
    # Build vertex-vertex adjacency once (only edges, dedup'd). For a
    # mesh of ~50k verts this is ~150k edges — tiny.
    e1 = np.stack([faces[:, 0], faces[:, 1]], axis=1)
    e2 = np.stack([faces[:, 1], faces[:, 2]], axis=1)
    e3 = np.stack([faces[:, 2], faces[:, 0]], axis=1)
    edges = np.concatenate([e1, e2, e3], axis=0)
    # Make undirected (sort each pair so (a,b) and (b,a) collide) and
    # deduplicate so each edge contributes once to each endpoint.
    edges = np.sort(edges, axis=1)
    edges = np.unique(edges, axis=0)
    # CSR-style: for each vertex, list its neighbours via np.bincount
    # plus an indirection array.
    i = np.concatenate([edges[:, 0], edges[:, 1]])
    j = np.concatenate([edges[:, 1], edges[:, 0]])
    degree = np.bincount(i, minlength=n).astype(np.float32)
    degree[degree == 0] = 1.0  # isolated verts: no movement
    # Build a sort-by-i so neighbours of i are contiguous.
    order = np.argsort(i, kind="stable")
    i_sorted = i[order]
    j_sorted = j[order]
    # Offsets[k] = start of vertex k's neighbours in j_sorted
    offsets = np.zeros(n + 1, dtype=np.int64)
    np.add.at(offsets[1:], i_sorted, 1)
    offsets = np.cumsum(offsets)

    v = verts.astype(np.float32, copy=True)
    for _ in range(iterations):
        # Σ neighbour coords, then divide by degree to get mean(neighbours).
        sums = np.zeros_like(v)
        np.add.at(sums, i_sorted, v[j_sorted])
        means = sums / degree[:, None]
        # First pass (λ): pull toward neighbours.
        v += lam * (means - v)
        sums = np.zeros_like(v)
        np.add.at(sums, i_sorted, v[j_sorted])
        means = sums / degree[:, None]
        # Second pass (μ < 0): push back out — restores volume.
        v += mu * (means - v)
    return v


def _compute_vertex_normals(verts: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Area-weighted vertex normals. Same convention as marching_cubes
    (outward-pointing for `level=0.5` on a 0/1 mask)."""
    v = verts.astype(np.float32, copy=False)
    fnorm = np.cross(
        v[faces[:, 1]] - v[faces[:, 0]],
        v[faces[:, 2]] - v[faces[:, 0]],
    )
    # Don't normalise face normals — keeping the magnitude (∝ 2·area)
    # gives an area-weighted vertex normal when accumulated.
    vnorm = np.zeros_like(v)
    np.add.at(vnorm, faces[:, 0], fnorm)
    np.add.at(vnorm, faces[:, 1], fnorm)
    np.add.at(vnorm, faces[:, 2], fnorm)
    # Normalise (skip zeros so we don't divide by 0 on isolated verts).
    lengths = np.linalg.norm(vnorm, axis=1, keepdims=True)
    lengths[lengths == 0] = 1.0
    return (vnorm / lengths).astype(np.float32, copy=False)


def _write_vtp(
    out_path: Path,
    verts: np.ndarray,
    faces: np.ndarray,
    normals: np.ndarray,
) -> None:
    """Write one VTK XML PolyData (.vtp) file.

    Layout: XML header + one `<AppendedData encoding="raw">_<bytes></AppendedData>`
    block holding the binary payloads in this order:
       1. Points       (float32 verts, N x 3)
       2. Normals      (float32 normals, N x 3)
       3. connectivity (int32 face indices, M x 3 flattened)
       4. offsets      (int32 per-cell vertex counts cumsum, M)
    Each payload is prefixed with a UInt64 length header (the
    `header_type="UInt64"` in the XML declares this).
    """
    verts_f32 = np.ascontiguousarray(verts, dtype="<f4")
    normals_f32 = np.ascontiguousarray(normals, dtype="<f4")
    faces_i32 = np.ascontiguousarray(faces, dtype="<i4")

    n_points = verts_f32.shape[0]
    n_polys = faces_i32.shape[0]
    # VTK's PolyData wants `connectivity` (flat list of vertex indices)
    # plus `offsets` (1-indexed cumulative count: 3, 6, 9, ...). All
    # triangles, so offsets is just 3, 6, 9 ... 3*n_polys.
    connectivity = faces_i32.reshape(-1)
    offsets = (np.arange(1, n_polys + 1, dtype="<i4") * 3)

    blocks: list[bytes] = [
        verts_f32.tobytes(order="C"),
        normals_f32.tobytes(order="C"),
        connectivity.tobytes(order="C"),
        offsets.tobytes(order="C"),
    ]

    # Compute appended-data offsets. Each data block in the AppendedData
    # area is preceded by a UInt64 header containing the block's length;
    # the `offset=` in the XML refers to the start of that header.
    header_size = 8  # bytes per UInt64 length header
    offset_points = 0
    offset_normals = offset_points + header_size + len(blocks[0])
    offset_conn    = offset_normals + header_size + len(blocks[1])
    offset_offsets = offset_conn + header_size + len(blocks[2])

    xml = (
        '<?xml version="1.0"?>\n'
        '<VTKFile type="PolyData" version="1.0" '
        'byte_order="LittleEndian" header_type="UInt64">\n'
        '  <PolyData>\n'
        f'    <Piece NumberOfPoints="{n_points}" NumberOfVerts="0" '
        f'NumberOfLines="0" NumberOfStrips="0" NumberOfPolys="{n_polys}">\n'
        '      <PointData Normals="Normals">\n'
        f'        <DataArray type="Float32" Name="Normals" '
        f'NumberOfComponents="3" format="appended" offset="{offset_normals}"/>\n'
        '      </PointData>\n'
        '      <CellData/>\n'
        '      <Points>\n'
        f'        <DataArray type="Float32" NumberOfComponents="3" '
        f'format="appended" offset="{offset_points}"/>\n'
        '      </Points>\n'
        '      <Verts/>\n'
        '      <Lines/>\n'
        '      <Strips/>\n'
        '      <Polys>\n'
        f'        <DataArray type="Int32" Name="connectivity" '
        f'format="appended" offset="{offset_conn}"/>\n'
        f'        <DataArray type="Int32" Name="offsets" '
        f'format="appended" offset="{offset_offsets}"/>\n'
        '      </Polys>\n'
        '    </Piece>\n'
        '  </PolyData>\n'
        '  <AppendedData encoding="raw">\n'
        '   _'
    ).encode("ascii")

    tail = b'\n  </AppendedData>\n</VTKFile>\n'

    with out_path.open("wb") as f:
        f.write(xml)
        for block in blocks:
            f.write(struct.pack("<Q", len(block)))  # little-endian UInt64
            f.write(block)
        f.write(tail)
