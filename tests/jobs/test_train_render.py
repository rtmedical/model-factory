"""Backward-compat + whole-GPU tests for the training-Job renderer.

The default (gpu_mode='mig') render MUST stay semantically identical to the
pre-parametrization template — that is the guarantee that the team's existing
`modelfactory train` / `submit` submissions are unaffected. We compare parsed
YAML (k8s applies semantically, so formatting is irrelevant) against a golden
snapshot captured before the change.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from modelfactory.config import FactoryConfig
from modelfactory.jobs.nnunet import TrainSpec, render

GOLDEN = Path(__file__).parent / "golden" / "train_job_default.yaml"


def _spec() -> TrainSpec:
    return TrainSpec(dataset="Dataset100_Hippocampus", configuration="3d_fullres", fold=0)


def test_default_mig_render_matches_golden():
    out = render(_spec(), FactoryConfig(), leased_uuid="MIG-PLACEHOLDER", job_name="train-fixedname")
    assert yaml.safe_load(out) == yaml.safe_load(GOLDEN.read_text())


def test_whole_mode_requests_device_plugin_gpu():
    cfg = FactoryConfig(gpu_mode="whole", runtime_class="")
    body = yaml.safe_load(render(_spec(), cfg, leased_uuid=None, job_name="train-whole"))
    pod = body["spec"]["template"]["spec"]
    container = pod["containers"][0]
    # device-plugin GPU request, no MIG UUID, no nvidia-legacy runtime
    assert container["resources"]["limits"]["nvidia.com/gpu"] == "1"
    assert "runtimeClassName" not in pod
    assert all(e["name"] != "NVIDIA_VISIBLE_DEVICES" for e in container["env"])


def test_whole_mode_can_keep_a_runtime_class():
    cfg = FactoryConfig(gpu_mode="whole", runtime_class="nvidia")
    pod = yaml.safe_load(render(_spec(), cfg, leased_uuid=None))["spec"]["template"]["spec"]
    assert pod["runtimeClassName"] == "nvidia"


def test_node_selector_is_parametrized():
    cfg = FactoryConfig(node_selector={"gpu-pool": "a100", "zone": "us"})
    pod = yaml.safe_load(render(_spec(), cfg, leased_uuid="MIG-X"))["spec"]["template"]["spec"]
    assert pod["nodeSelector"] == {"gpu-pool": "a100", "zone": "us"}
