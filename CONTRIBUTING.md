# Contributing to model-factory

Thanks for your interest in improving model-factory! This project helps teams
train, track, and QA medical-image segmentation models (nnU-Net v2 +
TotalSegmentator) on Kubernetes + GPU clusters at scale.

## Ways to contribute

- **New dataset source adapters** — implement the `DatasetSource` interface in
  `src/modelfactory/datasets/sources/` for a new on-disk format.
- **Trainer / planner variants** — nnU-Net subclasses under
  `src/modelfactory/trainers/` and `src/modelfactory/planners/`.
- **Infra support** — new GPU layouts, cloud storage classes, ingress recipes in
  `src/modelfactory/infra/`.
- **Docs, examples, bug reports.**

## Development setup

```bash
git clone <your fork>
cd model-factory
python -m venv .venv && source .venv/bin/activate
make install-sdk          # pip install -e ".[dev]"
make lint                 # ruff + mypy
make test                 # pytest
```

The SDK core is intentionally torch-free so the CLI runs outside the trainer
container. Anything that imports `torch` / `nnunetv2` must be imported lazily
inside the function that needs it (see `cli.py` for the pattern) and lives
behind the `[trainer]` optional dependency.

## Ground rules

- **No site-specific or proprietary references** in the public tree. CI runs a
  "forbidden-strings" gate; keep hostnames, node names, internal paths, and
  cohort data out of committed code. Site-specific content belongs in a private
  overlay (`overlays/private/`, git-ignored) — see `overlays/README.md`.
- **Never hardcode endpoints.** Hostnames, storage paths, GPU layout, and image
  tags are parameters in `cluster.yaml` / `FactoryConfig`, not literals.
- **Infra changes must keep `tests/infra/` green** — the render tests prove a
  generated manifest applies cleanly (no pod churn) against a reference cluster.
- **Respect model-weight licensing** (see `NOTICE` and `docs/licensing.md`).

## Pull requests

1. Branch from `main`.
2. `make lint && make test` must pass.
3. Describe the change and how you verified it. For infra changes, include the
   relevant `modelfactory infra render` / `kubectl diff` output.
4. By contributing, you agree your contributions are licensed under Apache-2.0.

## Code style

- Python ≥ 3.10, `ruff` (line length 100) + `mypy`.
- Match the surrounding code's conventions and comment density.
