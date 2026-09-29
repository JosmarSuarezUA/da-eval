"""da-eval: shared domain-adaptation evaluation for object detectors.

- ``da_eval.metrics``   -- framework-agnostic metrics, plots and W&B helpers
- ``da_eval.pipeline``  -- evaluation protocol + ``DetectorAdapter`` interface
- ``da_eval.adapters``  -- model adapters (RT-DETRv4, D-FINE, ...)
- ``da_eval.cli``       -- ``da-eval`` command-line entry point
"""

from da_eval.pipeline import (
    DetectorAdapter,
    SplitOutput,
    evaluate_source_against_targets,
    load_dataset_configs,
    run_all_sources,
    run_eval,
)

__all__ = [
    "DetectorAdapter",
    "SplitOutput",
    "evaluate_source_against_targets",
    "load_dataset_configs",
    "run_all_sources",
    "run_eval",
]
