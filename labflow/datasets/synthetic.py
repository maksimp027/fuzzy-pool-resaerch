"""Synthetic graph generators for testing and benchmarking graph reduction methods."""

from __future__ import annotations

from typing import List, Sequence, Tuple

import networkx as nx
import numpy as np
import torch

from labflow.types import GraphData, dense_to_sparse


def generate_sbm_graph(
    block_sizes: Sequence[int] = (15, 15, 15),
    p_in: float = 0.65,
    p_out: float = 0.05,
    feature_dim: int = 16,
    feature_sep: float = 2.0,
    weighted: bool = True,
    seed: int = 42,
    dtype: torch.dtype = torch.float32,
) -> GraphData:
    """Generate a Stochastic Block Model (SBM) community graph with block-correlated features."""
    rng = np.random.default_rng(seed)
    k = len(block_sizes)
    probs = [[p_in if i == j else p_out for j in range(k)] for i in range(k)]
    g_nx = nx.stochastic_block_model(list(block_sizes), probs, seed=seed)

    n = sum(block_sizes)
    adj_np = nx.to_numpy_array(g_nx, dtype=np.float64)
    if weighted:
        weights = rng.uniform(0.5, 1.0, size=(n, n))
        weights = 0.5 * (weights + weights.T)
        adj_np = adj_np * weights

    labels: List[int] = []
    for cluster_id, sz in enumerate(block_sizes):
        labels.extend([cluster_id] * sz)
    labels_np = np.array(labels, dtype=np.int64)

    centers = rng.normal(0.0, feature_sep, size=(k, feature_dim))
    x_np = centers[labels_np] + rng.normal(0.0, 0.6, size=(n, feature_dim))

    adj = torch.tensor(adj_np, dtype=dtype)
    x = torch.tensor(x_np, dtype=dtype)
    edge_index, edge_weight = dense_to_sparse(adj, threshold=0.0)
    return GraphData(
        x=x,
        adj=adj,
        edge_index=edge_index,
        edge_weight=edge_weight,
        node_labels=torch.tensor(labels_np, dtype=torch.long),
        metadata={"kind": "sbm", "num_blocks": k, "p_in": p_in, "p_out": p_out},
    )


def generate_planted_cliques_graph(
    num_cliques: int = 3,
    clique_size: int = 6,
    num_noise_nodes: int = 12,
    noise_edge_prob: float = 0.12,
    feature_dim: int = 12,
    seed: int = 7,
    dtype: torch.dtype = torch.float32,
) -> GraphData:
    """Generate planted complete cliques embedded in a sparse noisy background graph.

    Node labels: 0..(num_cliques - 1) for clique members, num_cliques for background noise nodes.
    """
    rng = np.random.default_rng(seed)
    n_clique_total = num_cliques * clique_size
    n = n_clique_total + num_noise_nodes
    adj_np = np.zeros((n, n), dtype=np.float64)
    labels_np = np.full((n,), num_cliques, dtype=np.int64)

    # Plant complete cliques
    for c in range(num_cliques):
        start = c * clique_size
        end = start + clique_size
        labels_np[start:end] = c
        adj_np[start:end, start:end] = 1.0
        np.fill_diagonal(adj_np[start:end, start:end], 0.0)

    # Sparse background & inter-clique bridge edges
    for i in range(n):
        for j in range(i + 1, n):
            same_clique = (
                i < n_clique_total
                and j < n_clique_total
                and labels_np[i] == labels_np[j]
            )
            if not same_clique and rng.random() < noise_edge_prob:
                adj_np[i, j] = adj_np[j, i] = 1.0

    centers = rng.normal(0.0, 1.8, size=(num_cliques + 1, feature_dim))
    x_np = centers[labels_np] + rng.normal(0.0, 0.45, size=(n, feature_dim))

    adj = torch.tensor(adj_np, dtype=dtype)
    x = torch.tensor(x_np, dtype=dtype)
    edge_index, edge_weight = dense_to_sparse(adj, threshold=0.0)
    return GraphData(
        x=x,
        adj=adj,
        edge_index=edge_index,
        edge_weight=edge_weight,
        node_labels=torch.tensor(labels_np, dtype=torch.long),
        metadata={
            "kind": "planted_cliques",
            "num_cliques": num_cliques,
            "clique_size": clique_size,
            "num_noise_nodes": num_noise_nodes,
        },
    )


def generate_parallel_bridges_graph(
    cluster_size: int = 8,
    num_bridges: int = 5,
    bridge_weight: float = 0.4,
    dtype: torch.dtype = torch.float64,
) -> Tuple[GraphData, torch.Tensor]:
    """Generate two clusters connected by exactly `num_bridges` parallel edges of weight `bridge_weight`.

    Returns (graph, crisp_membership_S) so parallel-path accumulation can be measured analytically.
    """
    n = 2 * cluster_size
    adj = torch.zeros((n, n), dtype=dtype)

    # Intra-cluster cliques of weight 1.0
    for i in range(cluster_size):
        for j in range(i + 1, cluster_size):
            adj[i, j] = adj[j, i] = 1.0
            adj[cluster_size + i, cluster_size + j] = adj[cluster_size + j, cluster_size + i] = 1.0

    # Parallel bridges between cluster 0 and cluster 1
    max_bridges = cluster_size * cluster_size
    if num_bridges > max_bridges:
        raise ValueError(f"num_bridges ({num_bridges}) exceeds cluster_size^2 ({max_bridges}).")

    count = 0
    for i in range(cluster_size):
        for j in range(cluster_size):
            if count >= num_bridges:
                break
            u = i
            v = cluster_size + j
            adj[u, v] = adj[v, u] = bridge_weight
            count += 1
        if count >= num_bridges:
            break

    s = torch.zeros((n, 2), dtype=dtype)
    s[:cluster_size, 0] = 1.0
    s[cluster_size:, 1] = 1.0

    x = torch.cat(
        [
            torch.ones((cluster_size, 4), dtype=dtype),
            -torch.ones((cluster_size, 4), dtype=dtype),
        ],
        dim=0,
    )
    edge_index, edge_weight = dense_to_sparse(adj, threshold=0.0)
    graph = GraphData(
        x=x,
        adj=adj,
        edge_index=edge_index,
        edge_weight=edge_weight,
        metadata={"num_bridges": num_bridges, "bridge_weight": bridge_weight},
    )
    return graph, s


def generate_graph_classification_dataset(
    num_graphs_per_class: int = 24,
    feature_dim: int = 8,
    seed: int = 123,
    dtype: torch.dtype = torch.float32,
) -> List[GraphData]:
    """Generate a 4-class synthetic structural+feature graph classification dataset.

    Classes:
        0: Ring of 3 cliques (triangle of dense 5-cliques)
        1: Two dense clusters joined by a chain bridge (barbell-like)
        2: Multi-star hub communities (3 stars linked at hubs)
        3: Bipartite / 2D grid lattice
    """
    rng = np.random.default_rng(seed)
    dataset: List[GraphData] = []

    for class_id in range(4):
        for idx in range(num_graphs_per_class):
            local_seed = int(seed * 1000 + class_id * 100 + idx)
            if class_id == 0:
                g = nx.ring_of_cliques(3, 5)
            elif class_id == 1:
                g = nx.barbell_graph(6, 3)
            elif class_id == 2:
                g = nx.Graph()
                hubs = [0, 5, 10]
                for h_node in hubs:
                    for leaf in range(h_node + 1, h_node + 5):
                        g.add_edge(h_node, leaf)
                g.add_edge(0, 5)
                g.add_edge(5, 10)
                g.add_edge(0, 10)
            else:
                g = nx.convert_node_labels_to_integers(nx.grid_2d_graph(3, 5))

            # Random mild edge perturbation
            n = g.number_of_nodes()
            adj_np = nx.to_numpy_array(g, dtype=np.float64)
            if rng.random() < 0.4:
                u, v = rng.choice(n, size=2, replace=False)
                adj_np[u, v] = adj_np[v, u] = 1.0

            # Structural + noisy signed node features (degree, clustering, signed motif signal)
            deg = adj_np.sum(axis=1, keepdims=True) / max(float(n - 1), 1.0)
            clust_dict = nx.clustering(nx.from_numpy_array(adj_np))
            clust = np.array([[clust_dict[i]] for i in range(n)], dtype=np.float64)
            noise = rng.normal(0.0, 0.45, size=(n, feature_dim - 2))
            # Add class-correlated signed extremes on 2 nodes so feature pooling operators differentiate
            sign_pattern = np.array([1.0, -1.0, 0.5, -0.5])[class_id]
            noise[0, 0] += 1.5 * sign_pattern
            noise[-1, 1] -= 1.5 * sign_pattern
            x_np = np.concatenate([deg, clust, noise], axis=1)

            adj = torch.tensor(adj_np, dtype=dtype)
            x = torch.tensor(x_np, dtype=dtype)
            edge_index, edge_weight = dense_to_sparse(adj, threshold=0.0)
            dataset.append(
                GraphData(
                    x=x,
                    adj=adj,
                    edge_index=edge_index,
                    edge_weight=edge_weight,
                    y=torch.tensor(class_id, dtype=torch.long),
                    metadata={"class_id": class_id, "sample_seed": local_seed},
                )
            )

    rng.shuffle(dataset)
    return dataset
