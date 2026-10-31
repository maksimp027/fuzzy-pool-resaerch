"""Section II: Relational Fuzzy Graph Reduction Methods (sup-T compositions)."""

from labflow.fuzzy_relational.goguen import GoguenTopologyReducer
from labflow.fuzzy_relational.hamacher import HamacherTopologyReducer
from labflow.fuzzy_relational.lukasiewicz import LukasiewiczTopologyReducer
from labflow.fuzzy_relational.schweizer_dombi import (
    DombiTopologyReducer,
    SchweizerSklarTopologyReducer,
)
from labflow.fuzzy_relational.zadeh import ZadehTopologyReducer

__all__ = [
    "ZadehTopologyReducer",
    "GoguenTopologyReducer",
    "LukasiewiczTopologyReducer",
    "HamacherTopologyReducer",
    "SchweizerSklarTopologyReducer",
    "DombiTopologyReducer",
]
