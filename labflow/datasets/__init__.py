"""Synthetic, classic real-world, and large-scale TUDataset / OGB graph dataset loaders."""

from labflow.datasets.classic import (
    generate_large_sparse_sbm_graph,
    generate_multiscale_noisy_sbm_graph,
    load_classic_network,
)
from labflow.datasets.real_world import (
    DATASET_SPEC_MAP,
    REAL_WORLD_DATASET_SPECS,
    RealWorldDatasetBundle,
    RealWorldDatasetSpec,
    load_real_world_dataset,
    resolve_dataset_spec,
    split_stratified_train_test,
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
    "REAL_WORLD_DATASET_SPECS",
    "DATASET_SPEC_MAP",
    "RealWorldDatasetSpec",
    "RealWorldDatasetBundle",
    "resolve_dataset_spec",
    "load_real_world_dataset",
    "split_stratified_train_test",
]
