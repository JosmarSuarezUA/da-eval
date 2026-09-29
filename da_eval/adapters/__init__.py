"""Detector adapters, looked up by name and imported lazily.

Each adapter imports its own framework (torch, ultralytics, ...) only when
requested, so one model's environment never needs another model's packages.
"""

from __future__ import annotations

import importlib

# name -> "module:ClassName"
ADAPTERS: dict[str, str] = {
    "rtdetrv4": "da_eval.adapters.yaml_engine:RTDETRv4Adapter",
    "dfine": "da_eval.adapters.yaml_engine:DFINEAdapter",
}


def get_adapter_class(name: str):
    """Return the adapter class registered under ``name``."""
    try:
        module_name, class_name = ADAPTERS[name].split(":")
    except KeyError:
        raise ValueError(f"Unknown adapter '{name}'. Available: {sorted(ADAPTERS)}") from None
    return getattr(importlib.import_module(module_name), class_name)
