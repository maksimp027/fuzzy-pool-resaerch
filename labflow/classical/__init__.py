"""Section I: Classical Algebraic and Spectral Graph Reduction Methods."""

from labflow.classical.diffpool import DiffPoolReducer
from labflow.classical.dmon import DMoNReducer
from labflow.classical.mincut import MinCutPoolReducer

__all__ = [
    "DiffPoolReducer",
    "MinCutPoolReducer",
    "DMoNReducer",
]
