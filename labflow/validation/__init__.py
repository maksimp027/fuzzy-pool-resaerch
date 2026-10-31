"""Validation suites: analytical gradcheck, sparse/dense equivalence, and theoretical invariants."""

from labflow.validation.gradcheck_suite import GradCheckRecord, run_full_gradcheck_suite
from labflow.validation.invariants import InvariantRecord, run_invariants_suite
from labflow.validation.sparse_equivalence import (
    SparseEquivalenceRecord,
    run_sparse_equivalence_suite,
)

__all__ = [
    "GradCheckRecord",
    "run_full_gradcheck_suite",
    "SparseEquivalenceRecord",
    "run_sparse_equivalence_suite",
    "InvariantRecord",
    "run_invariants_suite",
]
