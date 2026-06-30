"""Static-HTML dashboard renderer for the factory.

Reads per-epoch `metrics.jsonl` files written by
`modelfactory.trainers.mlflow_trainer.nnUNetTrainerMLflow` and produces a
single self-contained HTML page styled with the RT Medical visual
identity. See :mod:`modelfactory.dashboard.render`.
"""

from modelfactory.dashboard.render import render_to_file

__all__ = ["render_to_file"]
