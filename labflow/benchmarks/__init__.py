"""Experimental benchmark suite for Labflow graph reduction methods."""

from labflow.benchmarks.runner import (
    ClassificationBenchmarkRecord,
    CliqueBenchmarkRecord,
    SBMBenchmarkRecord,
    run_graph_classification_benchmark,
    run_planted_clique_benchmark,
    run_sbm_community_benchmark,
)

__all__ = [
    "SBMBenchmarkRecord",
    "CliqueBenchmarkRecord",
    "ClassificationBenchmarkRecord",
    "run_sbm_community_benchmark",
    "run_planted_clique_benchmark",
    "run_graph_classification_benchmark",
]
