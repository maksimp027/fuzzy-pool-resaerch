"""Classic real-world network benchmarks and multi-scale / true-sparse graph generators."""

from __future__ import annotations

from typing import List, Sequence

import networkx as nx
import numpy as np
import torch

from labflow.types import GraphData, dense_to_sparse


def _compute_structural_node_features(
    g_nx: nx.Graph,
    adj_np: np.ndarray,
    feature_dim: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Build rich structural + spectral node embeddings without label leakage."""
    n = adj_np.shape[0]
    deg = adj_np.sum(axis=1, keepdims=True)
    deg_norm = deg / max(float(deg.max()), 1.0)

    clust_map = nx.clustering(g_nx, weight="weight")
    clust = np.array([[float(clust_map[i])] for i in range(n)], dtype=np.float64)

    # Pure NumPy power-iteration PageRank (no scipy dependency required)
    row_sums = adj_np.sum(axis=1)
    pr_vec = np.full((n,), 1.0 / max(n, 1), dtype=np.float64)
    alpha_pr = 0.85
    for _ in range(60):
        dangling_mass = float(pr_vec[row_sums <= 1e-12].sum())
        safe_contrib = np.where(row_sums > 1e-12, pr_vec / np.maximum(row_sums, 1e-12), 0.0)
        pr_next = alpha_pr * (adj_np.T @ safe_contrib + dangling_mass / max(n, 1)) + (1.0 - alpha_pr) / max(n, 1)
        if float(np.abs(pr_next - pr_vec).max()) < 1e-9:
            pr_vec = pr_next
            break
        pr_vec = pr_next
    pr = (pr_vec * float(n)).reshape(n, 1)

    g_simple = nx.Graph(g_nx)
    g_simple.remove_edges_from(nx.selfloop_edges(g_simple))
    core_map = nx.core_number(g_simple) if g_simple.number_of_edges() > 0 else {i: 0 for i in range(n)}
    max_core = max(max(core_map.values(), default=1), 1)
    core = np.array([[float(core_map[i]) / float(max_core)] for i in range(n)], dtype=np.float64)

    # Normalized Laplacian lowest non-trivial eigenvectors
    d_vec = adj_np.sum(axis=1)
    inv_sqrt_d = np.where(d_vec > 1e-12, 1.0 / np.sqrt(np.maximum(d_vec, 1e-12)), 0.0)
    l_sym = np.eye(n, dtype=np.float64) - (inv_sqrt_d[:, None] * adj_np * inv_sqrt_d[None, :])
    _, eigvecs = np.linalg.eigh(l_sym)
    n_spec = min(max(feature_dim - 4, 2), max(n - 1, 1))
    spec_feats = eigvecs[:, 1 : 1 + n_spec] if n > 1 else np.zeros((n, 1), dtype=np.float64)

    raw = np.concatenate([deg_norm, clust, pr, core, spec_feats], axis=1)
    if raw.shape[1] >= feature_dim:
        return raw[:, :feature_dim]

    pad_cols = feature_dim - raw.shape[1]
    proj = rng.normal(0.0, 0.35, size=(raw.shape[1], pad_cols))
    extra = np.tanh(raw @ proj)
    return np.concatenate([raw, extra], axis=1)


def load_classic_network(
    name: str = "karate",
    feature_dim: int = 16,
    seed: int = 42,
    dtype: torch.dtype = torch.float32,
) -> GraphData:
    """Load a classic real-world social/historical network with ground-truth or modularity communities.

    Supported networks:
        - 'karate': Zachary's Karate Club (N=34, K=2 ground-truth factions)
        - 'les_miserables': Hugo's Les Miserables character co-appearance network (N=77, weighted)
        - 'florentine': Renaissance Florentine Families marriage network (N=15)
        - 'davis': Davis Southern Women social network (N=32, K=2 bipartite groups)
    """
    rng = np.random.default_rng(seed)
    key = name.lower().strip()

    if key == "karate":
        g_raw = nx.karate_club_graph()
        g_nx = nx.convert_node_labels_to_integers(g_raw)
        clubs = [g_nx.nodes[i].get("club", "Mr. Hi") for i in range(g_nx.number_of_nodes())]
        labels_np = np.array([0 if c == "Mr. Hi" else 1 for c in clubs], dtype=np.int64)
        num_clusters = 2
    elif key == "les_miserables":
        g_raw = nx.les_miserables_graph()
        g_nx = nx.convert_node_labels_to_integers(g_raw)
        comms = list(nx.community.greedy_modularity_communities(g_nx, weight="weight"))
        labels_np = np.zeros((g_nx.number_of_nodes(),), dtype=np.int64)
        for cid, nodes in enumerate(comms):
            for u in nodes:
                labels_np[int(u)] = cid
        num_clusters = len(comms)
    elif key == "florentine":
        g_raw = nx.florentine_families_graph()
        g_nx = nx.convert_node_labels_to_integers(g_raw)
        comms = list(nx.community.greedy_modularity_communities(g_nx))
        labels_np = np.zeros((g_nx.number_of_nodes(),), dtype=np.int64)
        for cid, nodes in enumerate(comms):
            for u in nodes:
                labels_np[int(u)] = cid
        num_clusters = len(comms)
    elif key == "davis":
        g_raw = nx.davis_southern_women_graph()
        g_nx = nx.convert_node_labels_to_integers(g_raw)
        labels_np = np.array(
            [int(g_nx.nodes[i].get("bipartite", 0)) for i in range(g_nx.number_of_nodes())],
            dtype=np.int64,
        )
        num_clusters = 2
    else:
        raise ValueError(
            f"Unknown classic network '{name}'. Choose from: 'karate', 'les_miserables', 'florentine', 'davis'."
        )

    adj_np = nx.to_numpy_array(g_nx, dtype=np.float64, weight="weight")
    np.fill_diagonal(adj_np, 0.0)
    max_w = float(adj_np.max())
    if max_w > 0.0:
        adj_np = adj_np / max_w

    x_np = _compute_structural_node_features(g_nx, adj_np, feature_dim=feature_dim, rng=rng)
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
            "kind": "classic_network",
            "name": key,
            "num_clusters": num_clusters,
            "num_nodes": int(adj_np.shape[0]),
            "num_edges": int(edge_weight.numel()),
        },
    )


def generate_multiscale_noisy_sbm_graph(
    block_sizes: Sequence[int] = (8, 16, 28, 12),
    p_in_range: tuple[float, float] = (0.45, 0.78),
    p_out: float = 0.06,
    feature_dim: int = 16,
    feature_noise: float = 0.90,
    seed: int = 99,
    dtype: torch.dtype = torch.float32,
) -> GraphData:
    """Generate an unbalanced multi-scale noisy SBM graph with heterogeneous block densities."""
    rng = np.random.default_rng(seed)
    k = len(block_sizes)
    p_ins = np.linspace(p_in_range[0], p_in_range[1], num=k)
    probs = [[float(p_ins[i]) if i == j else float(p_out) for j in range(k)] for i in range(k)]
    g_nx = nx.stochastic_block_model(list(block_sizes), probs, seed=seed)

    n = sum(block_sizes)
    adj_np = nx.to_numpy_array(g_nx, dtype=np.float64)
    weights = rng.uniform(0.35, 1.0, size=(n, n))
    weights = 0.5 * (weights + weights.T)
    adj_np = adj_np * weights

    labels: List[int] = []
    for cid, sz in enumerate(block_sizes):
        labels.extend([cid] * sz)
    labels_np = np.array(labels, dtype=np.int64)

    centers = rng.normal(0.0, 1.75, size=(k, feature_dim))
    x_np = centers[labels_np] + rng.normal(0.0, feature_noise, size=(n, feature_dim))

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
            "kind": "multiscale_noisy_sbm",
            "block_sizes": list(block_sizes),
            "num_blocks": k,
            "p_out": p_out,
        },
    )


def generate_large_sparse_sbm_graph(
    num_nodes: int = 2000,
    num_blocks: int = 8,
    avg_degree: float = 12.0,
    out_in_ratio: float = 0.12,
    feature_dim: int = 16,
    seed: int = 2026,
    dtype: torch.dtype = torch.float32,
) -> GraphData:
    """Generate a large sparse SBM GraphData in strict O(N + M) memory (NEVER allocates N x N adjacency)."""
    rng = np.random.default_rng(seed)
    base_sz = num_nodes // num_blocks
    rem = num_nodes % num_blocks
    sizes = [base_sz + (1 if b < rem else 0) for b in range(num_blocks)]
    offsets = np.cumsum([0] + sizes)

    # Solve for p_in, p_out such that expected degree ~= avg_degree
    mean_b = float(num_nodes) / float(num_blocks)
    denom = max((mean_b - 1.0) + (num_nodes - mean_b) * out_in_ratio, 1.0)
    p_in = min(0.95, avg_degree / denom)
    p_out = min(0.50, p_in * out_in_ratio)

    u_chunks: List[np.ndarray] = []
    v_chunks: List[np.ndarray] = []

    for b in range(num_blocks):
        s_b = sizes[b]
        lo_b = int(offsets[b])
        max_pairs = s_b * (s_b - 1) // 2
        if max_pairs > 0:
            n_e = int(rng.binomial(max_pairs, p_in))
            if n_e > 0:
                u_loc = rng.integers(0, s_b, size=n_e * 2)
                v_loc = rng.integers(0, s_b, size=n_e * 2)
                mask = u_loc < v_loc
                u_loc = u_loc[mask][:n_e] + lo_b
                v_loc = v_loc[mask][:n_e] + lo_b
                if u_loc.size > 0:
                    u_chunks.append(u_loc)
                    v_chunks.append(v_loc)

        for b2 in range(b + 1, num_blocks):
            s_b2 = sizes[b2]
            lo_b2 = int(offsets[b2])
            max_inter = s_b * s_b2
            if max_inter > 0:
                n_inter = int(rng.binomial(max_inter, p_out))
                if n_inter > 0:
                    u_inter = rng.integers(lo_b, lo_b + s_b, size=n_inter)
                    v_inter = rng.integers(lo_b2, lo_b2 + s_b2, size=n_inter)
                    u_chunks.append(u_inter)
                    v_chunks.append(v_inter)

    if u_chunks:
        u_all = np.concatenate(u_chunks)
        v_all = np.concatenate(v_chunks)
        keys = u_all.astype(np.int64) * num_nodes + v_all.astype(np.int64)
        uniq_keys = np.unique(keys)
        u_u = uniq_keys // num_nodes
        v_u = uniq_keys - u_u * num_nodes
        w_half = rng.uniform(0.45, 1.0, size=u_u.shape[0])

        src_np = np.concatenate([u_u, v_u])
        dst_np = np.concatenate([v_u, u_u])
        ew_np = np.concatenate([w_half, w_half])
    else:
        src_np = np.empty((0,), dtype=np.int64)
        dst_np = np.empty((0,), dtype=np.int64)
        ew_np = np.empty((0,), dtype=np.float64)

    labels_list: List[int] = []
    for cid, sz in enumerate(sizes):
        labels_list.extend([cid] * sz)
    labels_np = np.array(labels_list, dtype=np.int64)

    centers = rng.normal(0.0, 1.8, size=(num_blocks, feature_dim))
    x_np = centers[labels_np] + rng.normal(0.0, 0.7, size=(num_nodes, feature_dim))

    edge_index = torch.tensor(np.stack([src_np, dst_np], axis=0), dtype=torch.long)
    edge_weight = torch.tensor(ew_np, dtype=dtype)
    x = torch.tensor(x_np, dtype=dtype)

    return GraphData(
        x=x,
        adj=None,  # Strictly sparse! Never allocates N x N
        edge_index=edge_index,
        edge_weight=edge_weight,
        node_labels=torch.tensor(labels_np, dtype=torch.long),
        metadata={
            "kind": "large_sparse_sbm",
            "num_nodes": num_nodes,
            "num_blocks": num_blocks,
            "num_edges": int(edge_weight.numel()),
            "avg_degree_target": avg_degree,
        },
    )
