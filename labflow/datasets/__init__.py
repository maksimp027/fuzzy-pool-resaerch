"""Synthetic, classic real-world, and large-scale sparse graph dataset generators."""

from labflow.datasets.classic import (
    generate_large_sparse_sbm_graph,
    generate_multiscale_noisy_sbm_graph,
    load_classic_network,
)
from labflow.datasets.synthetic import (
    generate_graph_classification_dataset,
    generate_parallel_bridges_graph,
    generate_planted_cliques_graph,
    generate_sbm_graph,
)

__all__ = [
    "generate_sbm_graph",
    "generate_planted_cliques_graph",
    "generate_parallel_bridges_graph",
    "generate_graph_classification_dataset",
    "load_classic_network",
    "generate_multiscale_noisy_sbm_graph",
    "generate_large_sparse_sbm_graph",
]
