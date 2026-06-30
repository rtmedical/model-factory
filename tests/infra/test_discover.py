"""Unit tests for MIG-slice discovery (no GPU required — parses a fixture)."""

from __future__ import annotations

from pathlib import Path

from modelfactory.infra.discover import ordered_slices, parse_nvidia_smi_l

FIXTURE = Path(__file__).parent / "fixtures" / "nvidia_smi_l.txt"


def test_parse_groups_by_gpu():
    by_gpu = parse_nvidia_smi_l(FIXTURE.read_text())
    # GPU 0 is whole (no MIG lines); GPUs 1-7 each have 2 slices.
    assert by_gpu.get(0, []) == []
    for g in range(1, 8):
        assert len(by_gpu[g]) == 2, f"GPU {g} should have 2 slices"
        assert all(s.profile == "3g.40gb" for s in by_gpu[g])


def test_pool_order_matches_live_mapping():
    by_gpu = parse_nvidia_smi_l(FIXTURE.read_text())
    # poolGpus order [3,4,5,6,7,1,2] must map mig-0 -> GPU3 slice 0, etc.
    slices = ordered_slices(by_gpu, [3, 4, 5, 6, 7, 1, 2])
    assert len(slices) == 14
    assert slices[0].uuid == "MIG-e934d59a-8527-5161-b8fd-989c68836b21"  # mig-0 = GPU3 s0
    assert slices[8].uuid == "MIG-f290aabc-78b2-5f6a-99f4-a1084365c085"  # mig-8 = GPU7 s0 (parked)
    assert slices[10].uuid == "MIG-14d1a71d-7ae5-510a-b984-45adf8572219"  # mig-10 = GPU1 s0
    # GPU index travels with the slice (drives the lean CPU tier for 1/2).
    assert [s.gpu for s in slices] == [3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 1, 1, 2, 2]
