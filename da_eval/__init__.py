"""da-eval: shared domain-adaptation evaluation for object detectors.

- ``da_eval.datasets``  -- user datasets YAML (COCO or YOLOv5), checks, training helpers
- ``da_eval.metrics``   -- framework-agnostic metrics, plots and W&B helpers
- ``da_eval.pipeline``  -- evaluation protocol + ``DetectorAdapter`` interface
- ``da_eval.adapters``  -- model adapters (RT-DETRv4, D-FINE, ...)
- ``da_eval.cli``       -- ``da-eval`` command (evaluation, ``da-eval check``)
- ``da_eval.train``     -- ``da-train`` command (training launcher)
"""

from da_eval.datasets import check_datasets_file, load_dataset_configs
from da_eval.pipeline import (
    DetectorAdapter,
    SplitOutput,
    evaluate_source_against_targets,
    run_all_sources,
    run_eval,
)

__all__ = [
    "DetectorAdapter",
    "SplitOutput",
    "check_datasets_file",
    "evaluate_source_against_targets",
    "load_dataset_configs",
    "run_all_sources",
    "run_eval",
]
