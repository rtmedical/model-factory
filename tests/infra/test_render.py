"""Regression net for the infra generator (the open-sourcing safety proof).

Renders the reference ClusterSpec + the discovered MIG slices and asserts the
resulting RayCluster **pod templates are identical** to what was last applied to
the live cluster. Pod-template equality is exactly the property that determines
whether a future ``kubectl apply`` churns workers — so a green test here means
applying the generated manifest is a no-op for the running 13 workers.

We compare against the ``last-applied-configuration`` annotation (the manifest
that was actually applied), NOT the live object, to avoid server-side defaulting
noise (e.g. ``protocol: TCP`` injected onto ports). The one intended divergence
is mig-8: last-applied recorded replicas 1, but the slice was later parked to 0
directly on the object; the generator renders 0 (correcting the drift), which a
3-way-merge apply resolves to the already-live 0 — still no churn.
"""

from __future__ import annotations

import json
from pathlib import Path

import yaml

from modelfactory.infra.discover import ordered_slices, parse_nvidia_smi_l
from modelfactory.infra.render import (
    render_all,
    render_cluster_queue,
    render_raycluster,
    render_resource_flavor,
)
from modelfactory.infra.spec import ClusterSpec

HERE = Path(__file__).parent
FIXTURES = HERE / "fixtures"
LIVE = HERE / "golden" / "live" / "raycluster.live.yaml"

LAST_APPLIED_KEY = "kubectl.kubernetes.io/last-applied-configuration"


def _spec() -> ClusterSpec:
    return ClusterSpec.load(FIXTURES / "reference_cluster.yaml")


def _slices():
    by_gpu = parse_nvidia_smi_l((FIXTURES / "nvidia_smi_l.txt").read_text())
    return ordered_slices(by_gpu, _spec().gpu.mig.pool_gpus)


def _last_applied() -> dict:
    live = yaml.safe_load(LIVE.read_text())
    return json.loads(live["metadata"]["annotations"][LAST_APPLIED_KEY])


def _by_group(worker_specs: list[dict]) -> dict[str, dict]:
    return {g["groupName"]: g for g in worker_specs}


def test_head_pod_template_matches_live():
    rendered = render_raycluster(_spec(), _slices())
    applied = _last_applied()
    assert rendered["spec"]["headGroupSpec"]["template"]["spec"] == applied["spec"]["headGroupSpec"]["template"]["spec"]
    assert rendered["spec"]["rayVersion"] == applied["spec"]["rayVersion"]
    assert rendered["spec"]["enableInTreeAutoscaling"] is False


def test_worker_pod_templates_match_live():
    rendered = _by_group(render_raycluster(_spec(), _slices())["spec"]["workerGroupSpecs"])
    applied = _by_group(_last_applied()["spec"]["workerGroupSpecs"])

    assert set(rendered) == set(applied), "worker group set must match live exactly"

    for name, applied_group in applied.items():
        # Pod template determines churn — must be byte-equal across ALL groups,
        # including the parked mig-8 (its template is unchanged; only replicas differ).
        assert rendered[name]["template"]["spec"] == applied_group["template"]["spec"], name
        assert rendered[name]["rayStartParams"] == applied_group["rayStartParams"], name


def test_replicas_active_and_parked():
    rendered = _by_group(render_raycluster(_spec(), _slices())["spec"]["workerGroupSpecs"])
    for name, group in rendered.items():
        if name == "mig-8":
            assert group["replicas"] == 0 and group["minReplicas"] == 0  # parked
        else:
            assert group["replicas"] == 1 and group["minReplicas"] == 1
        assert group["maxReplicas"] == 1 and group["numOfHosts"] == 1


def test_cpu_tier_split():
    rendered = _by_group(render_raycluster(_spec(), _slices())["spec"]["workerGroupSpecs"])

    def cpu_req(name):
        return rendered[name]["template"]["spec"]["containers"][0]["resources"]["requests"]["cpu"]

    # GPUs 3-7 (mig-0..9) fat; GPUs 1-2 (mig-10..13) lean.
    assert all(cpu_req(f"mig-{i}") == "18" for i in range(10))
    assert all(cpu_req(f"mig-{i}") == "4" for i in range(10, 14))


def test_per_group_overrides_capture_live_drift():
    """live_cluster.yaml captures the hand-patched mig-5/6/7/10 (300Gi/64Gi)."""
    spec = ClusterSpec.load(FIXTURES / "live_cluster.yaml")
    rendered = _by_group(render_raycluster(spec, _slices())["spec"]["workerGroupSpecs"])

    def mem_limit(name):
        return rendered[name]["template"]["spec"]["containers"][0]["resources"]["limits"]["memory"]

    def dshm(name):
        vols = {v["name"]: v for v in rendered[name]["template"]["spec"]["volumes"]}
        return vols["dshm"]["emptyDir"]["sizeLimit"]

    for name in ("mig-5", "mig-6", "mig-7", "mig-10"):
        assert mem_limit(name) == "300Gi", name
        assert dshm(name) == "64Gi", name
    # Untouched groups keep the defaults.
    assert mem_limit("mig-0") == "160Gi" and dshm("mig-0") == "32Gi"


def test_cluster_queue_and_flavor():
    spec = _spec()
    cq = render_cluster_queue(spec)
    resources = {r["name"]: r["nominalQuota"] for r in cq["spec"]["resourceGroups"][0]["flavors"][0]["resources"]}
    assert resources["nvidia.com/gpu"] == "10"
    assert resources["cpu"] == "196"
    rf = render_resource_flavor(spec)
    assert rf["spec"]["nodeLabels"]["nvidia.com/gpu.product"] == "NVIDIA-H100-80GB-HBM3"
    assert rf["spec"]["nodeLabels"]["factory.io/training"] == "true"


def test_mig_mode_emits_uuid_and_lease_configmaps():
    out = render_all(_spec(), _slices())
    assert "factory-mig-uuids-configmap.yaml" in out
    assert "factory-mig-leases-configmap.yaml" in out
    assert "safe_uuids.env" in out
    uuids = out["factory-mig-uuids-configmap.yaml"]["data"]["uuids.txt"].splitlines()
    assert len(uuids) == 14
    leases = out["factory-mig-leases-configmap.yaml"]["data"]
    assert all(v == "available" for v in leases.values())


# ── whole-GPU mode ──────────────────────────────────────────────────────


def _whole_spec() -> ClusterSpec:
    return ClusterSpec.model_validate(
        {
            "gpu": {"mode": "whole", "whole": {"count": 4}},
            "storage": {"mount": "pvc"},
        }
    )


def test_whole_mode_single_group_with_device_plugin():
    spec = _whole_spec()
    rc = render_raycluster(spec, [])
    groups = rc["spec"]["workerGroupSpecs"]
    assert len(groups) == 1
    g = groups[0]
    assert g["groupName"] == "gpu-workers"
    assert g["replicas"] == 4 and g["maxReplicas"] == 4
    container = g["template"]["spec"]["containers"][0]
    assert container["resources"]["limits"]["nvidia.com/gpu"] == "1"
    # No UUID pinning and no nvidia-legacy runtime in whole mode.
    env_names = [e["name"] for e in container["env"]]
    assert "NVIDIA_VISIBLE_DEVICES" not in env_names
    assert "runtimeClassName" not in g["template"]["spec"]


def test_whole_mode_skips_mig_configmaps():
    out = render_all(_whole_spec(), [])
    assert "factory-mig-uuids-configmap.yaml" not in out
    assert "safe_uuids.env" not in out
    # pvc mount → a PVC manifest is emitted
    assert "factory-pvc.yaml" in out
    assert out["factory-pvc.yaml"]["spec"]["storageClassName"] == "nfs-client"


def test_quota_autosizes_in_whole_mode():
    spec = _whole_spec()
    cq = render_cluster_queue(spec)
    resources = {r["name"]: r["nominalQuota"] for r in cq["spec"]["resourceGroups"][0]["flavors"][0]["resources"]}
    assert resources["nvidia.com/gpu"] == "4"  # auto = whole.count
