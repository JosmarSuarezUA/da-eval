"""
da_eval.adapters.common
=======================
Helpers shared by every adapter: mapping model class indices to dataset
category ids, and capturing / pooling the embeddings of a hooked layer.
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# 1. Category Mapping
# ---------------------------------------------------------------------------

MSCOCO_CATEGORY2LABEL: dict[int, int] = {
    1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7: 6, 8: 7, 9: 8, 10: 9, 11: 10, 13: 11,
    14: 12, 15: 13, 16: 14, 17: 15, 18: 16, 19: 17, 20: 18, 21: 19, 22: 20,
    23: 21, 24: 22, 25: 23, 27: 24, 28: 25, 31: 26, 32: 27, 33: 28, 34: 29,
    35: 30, 36: 31, 37: 32, 38: 33, 39: 34, 40: 35, 41: 36, 42: 37, 43: 38,
    44: 39, 46: 40, 47: 41, 48: 42, 49: 43, 50: 44, 51: 45, 52: 46, 53: 47,
    54: 48, 55: 49, 56: 50, 57: 51, 58: 52, 59: 53, 60: 54, 61: 55, 62: 56,
    63: 57, 64: 58, 65: 59, 67: 60, 70: 61, 72: 62, 73: 63, 74: 64, 75: 65,
    76: 66, 77: 67, 78: 68, 79: 69, 80: 70, 81: 71, 82: 72, 84: 73, 85: 74,
    86: 75, 87: 76, 88: 77, 89: 78, 90: 79,
}
MSCOCO_LABEL2CATEGORY: dict[int, int] = {v: k for k, v in MSCOCO_CATEGORY2LABEL.items()}


def get_label_to_category_map(coco_gt: Any, remap_mscoco: bool = False) -> dict[int, int]:
    """
    Map model predicted class index (0..num_classes-1) to dataset category_id.
    - If remap_mscoco is True and category IDs match MS COCO (80 classes, max 90): uses MSCOCO mapping.
    - Otherwise index i maps to the i-th category in sorted id order, the same order
      ``da-train`` uses when it writes zero-based training annotations.
    """
    cat_ids = sorted(coco_gt.getCatIds())
    if remap_mscoco and len(cat_ids) == 80 and max(cat_ids) == 90:
        return MSCOCO_LABEL2CATEGORY
    return {idx: cat_id for idx, cat_id in enumerate(cat_ids)}


# ---------------------------------------------------------------------------
# 2. Embedding Hook Utilities
# ---------------------------------------------------------------------------

class EmbeddingHook:
    """Forward hook capturing feature representations."""

    def __init__(self):
        self.features: torch.Tensor | None = None

    def __call__(self, module, inp, out):
        self.features = out


def pool_features(feat_output) -> torch.Tensor:
    """Pool arbitrary layer feature outputs to flat [B, C] vectors."""
    feats = feat_output if isinstance(feat_output, (list, tuple)) else [feat_output]
    pooled = []
    for f in feats:
        if isinstance(f, torch.Tensor):
            if f.dim() == 4:  # [B, C, H, W] -> GAP to [B, C]
                pooled.append(F.adaptive_avg_pool2d(f, (1, 1)).flatten(1))
            elif f.dim() == 3:  # [B, N, C] -> Mean to [B, C]
                pooled.append(f.mean(dim=1))
    if not pooled:
        raise ValueError("Could not pool features from hooked layer.")
    return torch.cat(pooled, dim=1)


def feature_levels(feat_output, input_width: int) -> list[dict]:
    """Column range of each part in the ``pool_features`` vector, in the same order.

    Feature maps are named after their pyramid level, ``P<log2(stride)>`` (stride 8 -> P3);
    token sequences, or maps whose names would repeat, are named ``part<i>``.
    """
    feats = feat_output if isinstance(feat_output, (list, tuple)) else [feat_output]
    parts = [f for f in feats if isinstance(f, torch.Tensor) and f.dim() in (3, 4)]
    names = []
    for i, f in enumerate(parts):
        stride = input_width / f.shape[-1] if f.dim() == 4 else 0
        level = math.log2(stride) if stride >= 1 else 0.5
        names.append(f"P{int(level)}" if level.is_integer() else f"part{i}")
    if len(set(names)) < len(names):
        names = [f"part{i}" for i in range(len(parts))]

    levels, start = [], 0
    for name, f in zip(names, parts):
        dim = f.shape[1] if f.dim() == 4 else f.shape[2]
        levels.append({"name": name, "start": start, "stop": start + dim})
        start += dim
    return levels
