"""Section IV: Fuzzy Feature Space Reduction Methods (X -> X_new)."""

from __future__ import annotations

import torch.nn as nn
from torch import Tensor

from labflow.fuzzy_features.bipolar import BipolarSupremalFeaturePooling
from labflow.fuzzy_features.choquet import ChoquetFeaturePooling
from labflow.fuzzy_features.cog import COGFeaturePooling


class LinearFeaturePooling(nn.Module):
    """Standard linear feature aggregation X_new = S^T H."""

    def forward(self, h: Tensor, s: Tensor) -> Tensor:
        return s.t() @ h

    def forward_reference(self, h: Tensor, s: Tensor) -> Tensor:
        return s.t() @ h


__all__ = [
    "COGFeaturePooling",
    "BipolarSupremalFeaturePooling",
    "ChoquetFeaturePooling",
    "LinearFeaturePooling",
]
