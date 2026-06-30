"""Cluster bootstrap & infrastructure rendering for model-factory.

This package turns a single declarative ``cluster.yaml`` (see
``cluster.example.yaml`` at the repo root) into the Kubernetes manifests the
factory needs — a Kueue ClusterQueue/LocalQueue/ResourceFlavor, the KubeRay
RayCluster, the MIG UUID/lease ConfigMaps, the shared PVC, etc. — supporting
both **MIG-partitioned** GPUs (one Ray worker per slice, pinned by UUID) and
**whole-GPU** clusters (the stock NVIDIA device-plugin path).

The generator is intentionally a pure ``dict`` builder + ``yaml.safe_dump`` so
its output is trivially comparable in tests (no Jinja whitespace pitfalls — see
the repo's documented gotcha). ``modelfactory infra render`` writes the result
to ``.render/infra/``; nothing is applied to a cluster unless the operator runs
``modelfactory infra apply``.
"""

from __future__ import annotations

from .spec import ClusterSpec

__all__ = ["ClusterSpec"]
