#!/usr/bin/env bash
# Convenience wrapper around `modelfactory infra bootstrap` for operators who
# haven't `pip install -e .`'d the SDK yet. Falls back to running the CLI from
# the source tree via PYTHONPATH.
#
#   ./scripts/bootstrap.sh [--config cluster.yaml] [--mig-create] [--apply]
#
# By default this is render-only + a kubectl diff (it does NOT mutate the
# cluster). Pass --apply to apply, and --mig-create to (re)partition MIG first.
set -euo pipefail
cd "$(dirname "$0")/.."

if command -v modelfactory >/dev/null 2>&1; then
  exec modelfactory infra bootstrap "$@"
else
  echo "modelfactory not on PATH; running from source (PYTHONPATH=src)." >&2
  exec env PYTHONPATH=src python3 -m modelfactory.cli infra bootstrap "$@"
fi
