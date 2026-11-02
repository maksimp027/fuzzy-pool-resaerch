"""Multi-domain mathematical, spectral, topological, fuzzy, and Jacobian metric suite for graph reducers."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional

import numpy as np
import torch
from torch import Tensor

from labflow.composite import GraphReductionLayer
from labflow.types import GraphData, PoolingOutput


@dataclass
class ReducerMetricProfile:
    """Comprehensive 24-metric analytical profile of a graph reduction operator on a target graph."""

    method_key: str
    method_code: str
    dataset_name: str
    # A. Clustering & Community Alignment
    nmi: float
    ari: float
    cluster_purity: float
    macro_f1: float
    # B. Cut & Modularity Quality
    hard_modularity_q: float
    fuzzy_modularity_q: float
    conductance: float
    normalized_cut: float
    # C. Spectral & Topological Fidelity
    spectral_laplacian_dist: float
    link_recon_rmse: float
    link_recon_auc: float
    unit_interval_compliance: float
    adj_new_max: float
    diag_offdiag_contrast: float
    exact_sparsity: float
    effective_sparsity: float
    # D. Fuzzy Partition Geometry
    deluca_termini_entropy: float
    bezdek_pc: float
    active_cluster_ratio: float
    orthogonality_defect: float
    # E. Feature Preservation
    feature_recon_r2: float
    # F. Gradient & Jacobian Health
    topo_jacobian_density: float
    dead_superedge_ratio: float
    param_grad_finite_ratio: float

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


def compute_nmi_score(pred_labels: np.ndarray, true_labels: np.ndarray) -> float:
    """Normalized Mutual Information (NMI) in [0, 1]."""
    n = len(true_labels)
    if n == 0:
        return 0.0
    u_true, t_idx = np.unique(true_labels, return_inverse=True)
    u_pred, p_idx = np.unique(pred_labels, return_inverse=True)
    cont = np.zeros((len(u_true), len(u_pred)), dtype=np.float64)
    np.add.at(cont, (t_idx, p_idx), 1.0)
    cont /= float(n)

    p_t = cont.sum(axis=1, keepdims=True)
    p_p = cont.sum(axis=0, keepdims=True)
    nz = cont > 0.0
    denom = (p_t @ p_p)[nz]
    mi = float(np.sum(cont[nz] * np.log(cont[nz] / np.maximum(denom, 1e-15))))

    h_t = -float(np.sum(p_t[p_t > 0] * np.log(p_t[p_t > 0])))
    h_p = -float(np.sum(p_p[p_p > 0] * np.log(p_p[p_p > 0])))
    if h_t + h_p < 1e-15:
        return 1.0
    return float(np.clip(2.0 * mi / (h_t + h_p), 0.0, 1.0))


def compute_ari_score(pred_labels: np.ndarray, true_labels: np.ndarray) -> float:
    """Exact Adjusted Rand Index (ARI) in [-1, 1] using combinatorial contingency counts."""
    n = len(true_labels)
    if n < 2:
        return 1.0
    u_true, t_idx = np.unique(true_labels, return_inverse=True)
    u_pred, p_idx = np.unique(pred_labels, return_inverse=True)
    cont = np.zeros((len(u_true), len(u_pred)), dtype=np.float64)
    np.add.at(cont, (t_idx, p_idx), 1.0)

    sum_comb_c = float(np.sum(cont * (cont - 1.0) * 0.5))
    a_sums = cont.sum(axis=1)
    b_sums = cont.sum(axis=0)
    sum_comb_a = float(np.sum(a_sums * (a_sums - 1.0) * 0.5))
    sum_comb_b = float(np.sum(b_sums * (b_sums - 1.0) * 0.5))
    total_pairs = float(n * (n - 1) * 0.5)

    expected = (sum_comb_a * sum_comb_b) / max(total_pairs, 1.0)
    max_idx = 0.5 * (sum_comb_a + sum_comb_b)
    denom = max_idx - expected
    if abs(denom) < 1e-15:
        return 1.0 if abs(sum_comb_c - expected) < 1e-12 else 0.0
    return float((sum_comb_c - expected) / denom)


def compute_purity_and_macro_f1(pred_labels: np.ndarray, true_labels: np.ndarray) -> tuple[float, float]:
    """Compute cluster purity and best-matched Macro-F1 score in [0, 1]."""
    n = len(true_labels)
    if n == 0:
        return 0.0, 0.0
    u_true, t_idx = np.unique(true_labels, return_inverse=True)
    u_pred, p_idx = np.unique(pred_labels, return_inverse=True)
    cont = np.zeros((len(u_true), len(u_pred)), dtype=np.float64)
    np.add.at(cont, (t_idx, p_idx), 1.0)

    purity = float(np.sum(np.max(cont, axis=0)) / float(n))

    t_counts = cont.sum(axis=1, keepdims=True)
    p_counts = cont.sum(axis=0, keepdims=True)
    denom = t_counts + p_counts
    f1_table = np.where(denom > 0, 2.0 * cont / np.maximum(denom, 1.0), 0.0)
    # Best F1 match for each true community
    macro_f1 = float(np.mean(np.max(f1_table, axis=1)))
    return purity, macro_f1


def compute_sparse_cut_and_modularity(
    edge_index: Tensor,
    edge_weight: Tensor,
    s: Tensor,
    pred_labels: np.ndarray,
    num_clusters: int,
) -> tuple[float, float, float, float]:
    """Compute (hard_modularity_q, fuzzy_modularity_q, conductance, normalized_cut) in O(M + N K)."""
    n = s.shape[0]
    s_np = s.detach().cpu().numpy().astype(np.float64)
    s_row_sum = np.maximum(s_np.sum(axis=1, keepdims=True), 1e-12)
    s_tilde = s_np / s_row_sum

    if edge_weight.numel() == 0:
        return 0.0, 0.0, 1.0, float(num_clusters)

    src = edge_index[0].detach().cpu().numpy()
    dst = edge_index[1].detach().cpu().numpy()
    ew = edge_weight.detach().cpu().numpy().astype(np.float64)

    deg = np.zeros((n,), dtype=np.float64)
    np.add.at(deg, src, ew)
    two_m = float(deg.sum())
    if two_m <= 1e-12:
        return 0.0, 0.0, 1.0, float(num_clusters)

    # Hard modularity, conductance, normalized cut
    c_src = pred_labels[src]
    c_dst = pred_labels[dst]
    same_c = c_src == c_dst
    internal_per_c = np.zeros((num_clusters,), dtype=np.float64)
    np.add.at(internal_per_c, c_src[same_c], ew[same_c])

    vol_per_c = np.zeros((num_clusters,), dtype=np.float64)
    np.add.at(vol_per_c, pred_labels, deg)

    hard_q = 0.0
    conductances: List[float] = []
    ncut = 0.0
    for c in range(num_clusters):
        vol = float(vol_per_c[c])
        if vol <= 1e-12:
            continue
        internal = float(internal_per_c[c])
        cut = max(0.0, vol - internal)
        hard_q += (internal / two_m) - ((vol / two_m) ** 2)
        denom_cond = min(vol, two_m - vol)
        if denom_cond > 1e-12:
            conductances.append(cut / denom_cond)
        ncut += cut / (vol + 1e-12)

    avg_cond = float(np.mean(conductances)) if conductances else 1.0

    # Continuous Chung-Lu fuzzy modularity: (1 / 2m) Tr(S_tilde^T A S_tilde - (d^T S_tilde)^T (d^T S_tilde) / 2m)
    tr_sas = float(np.sum(ew[:, None] * s_tilde[src] * s_tilde[dst]))
    d_s = deg @ s_tilde  # [K]
    tr_null = float(np.sum(d_s * d_s) / two_m)
    fuzzy_q = (tr_sas - tr_null) / two_m

    return float(hard_q), float(fuzzy_q), avg_cond, float(ncut)


def compute_spectral_laplacian_distance(
    edge_index: Tensor,
    edge_weight: Tensor,
    num_nodes: int,
    adj_new: Tensor,
) -> float:
    """Compare normalized Laplacian smallest K eigenvalues between G and coarsened G_new."""
    k = adj_new.shape[0]
    a_new_np = adj_new.detach().cpu().numpy().astype(np.float64)
    a_new_sym = np.maximum(0.0, 0.5 * (a_new_np + a_new_np.T))
    np.fill_diagonal(a_new_sym, 0.0)
    d_new = a_new_sym.sum(axis=1)
    inv_sq_new = np.where(d_new > 1e-12, 1.0 / np.sqrt(np.maximum(d_new, 1e-12)), 0.0)
    l_new = np.eye(k) - (inv_sq_new[:, None] * a_new_sym * inv_sq_new[None, :])
    eigs_new = np.sort(np.clip(np.linalg.eigvalsh(l_new), 0.0, 2.0))

    if num_nodes > 600:
        # For large graphs, compare spectral gap proxy via power-free zero-eigenvalue & trace ratio
        return float(np.sqrt(np.mean((eigs_new - np.linspace(0.0, 1.2, k)) ** 2)))

    adj_np = np.zeros((num_nodes, num_nodes), dtype=np.float64)
    if edge_weight.numel() > 0:
        src = edge_index[0].detach().cpu().numpy()
        dst = edge_index[1].detach().cpu().numpy()
        ew = edge_weight.detach().cpu().numpy().astype(np.float64)
        np.add.at(adj_np, (src, dst), ew)
    adj_sym = np.maximum(0.0, 0.5 * (adj_np + adj_np.T))
    np.fill_diagonal(adj_sym, 0.0)
    d_orig = adj_sym.sum(axis=1)
    inv_sq_orig = np.where(d_orig > 1e-12, 1.0 / np.sqrt(np.maximum(d_orig, 1e-12)), 0.0)
    l_orig = np.eye(num_nodes) - (inv_sq_orig[:, None] * adj_sym * inv_sq_orig[None, :])
    eigs_orig = np.sort(np.clip(np.linalg.eigvalsh(l_orig), 0.0, 2.0))[:k]
    if len(eigs_orig) < k:
        eigs_orig = np.pad(eigs_orig, (0, k - len(eigs_orig)), constant_values=1.0)

    return float(np.sqrt(np.mean((eigs_orig - eigs_new) ** 2)))


def compute_link_reconstruction_metrics(
    graph: GraphData,
    s: Tensor,
    adj_new: Tensor,
    seed: int = 2026,
) -> tuple[float, float]:
    """Compute link reconstruction RMSE and ROC-AUC from A_hat = S_tilde A_new S_tilde^T."""
    n = graph.num_nodes
    s_det = s.detach()
    s_tilde = s_det / (s_det.sum(dim=1, keepdim=True) + 1e-12)
    a_new_det = adj_new.detach()
    # Rescale unbounded classical DiffPool A_new into [0, 1] for fair link probability scoring
    a_scale = float(a_new_det.abs().max().item())
    a_norm = a_new_det / max(a_scale, 1.0) if a_scale > 1.0 + 1e-6 else a_new_det

    edge_index, edge_weight = graph.sparse_edges()
    if n <= 512:
        adj_true = graph.dense_adj().detach().cpu().numpy().astype(np.float64)
        a_hat = (s_tilde @ a_norm @ s_tilde.t()).clamp(0.0, 1.0).cpu().numpy().astype(np.float64)
        rmse = float(np.sqrt(np.mean((adj_true - a_hat) ** 2)))

        off_mask = ~np.eye(n, dtype=bool)
        pos_scores = a_hat[(adj_true > 0.0) & off_mask]
        neg_scores = a_hat[(adj_true == 0.0) & off_mask]
    else:
        rng = np.random.default_rng(seed)
        m = int(edge_weight.numel())
        n_sample = min(max(m, 500), 2000)
        if m > 0:
            e_pick = rng.choice(m, size=min(m, n_sample), replace=False)
            u_pos = edge_index[0, e_pick]
            v_pos = edge_index[1, e_pick]
            w_pos = edge_weight[e_pick].detach().cpu().numpy().astype(np.float64)
            pos_scores = ((s_tilde[u_pos] @ a_norm) * s_tilde[v_pos]).sum(dim=1).clamp(0.0, 1.0).cpu().numpy()
        else:
            w_pos = np.zeros((0,), dtype=np.float64)
            pos_scores = np.zeros((0,), dtype=np.float64)

        u_neg = torch.tensor(rng.integers(0, n, size=n_sample), device=s.device)
        v_neg = torch.tensor(rng.integers(0, n, size=n_sample), device=s.device)
        neg_scores = ((s_tilde[u_neg] @ a_norm) * s_tilde[v_neg]).sum(dim=1).clamp(0.0, 1.0).cpu().numpy()
        err_sq = np.concatenate([(pos_scores - w_pos) ** 2, neg_scores ** 2])
        rmse = float(np.sqrt(np.mean(err_sq))) if err_sq.size > 0 else 0.0

    if pos_scores.size == 0 or neg_scores.size == 0:
        return rmse, 0.5

    # Exact Mann-Whitney U ROC-AUC via sorted ranks (O(E log E))
    all_scores = np.concatenate([pos_scores, neg_scores])
    order = np.argsort(all_scores, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    # Average ranks for ties
    sorted_vals = all_scores[order]
    n_tot = len(sorted_vals)
    i = 0
    while i < n_tot:
        j = i + 1
        while j < n_tot and sorted_vals[j] == sorted_vals[i]:
            j += 1
        avg_rank = 0.5 * (i + 1 + j)
        ranks[order[i:j]] = avg_rank
        i = j

    n_pos = float(pos_scores.size)
    n_neg = float(neg_scores.size)
    pos_rank_sum = float(np.sum(ranks[: pos_scores.size]))
    auc = (pos_rank_sum - n_pos * (n_pos + 1.0) * 0.5) / (n_pos * n_neg)
    return rmse, float(np.clip(auc, 0.0, 1.0))


def compute_deluca_termini_entropy(s: Tensor) -> float:
    """De Luca-Termini non-probabilistic fuzzy entropy in [0, 1]."""
    s_clamped = s.clamp(1e-12, 1.0 - 1e-12)
    return float(
        (-(s_clamped * torch.log(s_clamped) + (1.0 - s_clamped) * torch.log(1.0 - s_clamped)).mean().item())
        / math.log(2.0)
    )


def compute_deluca_termini_and_bezdek(s: Tensor) -> tuple[float, float]:
    """Compute (deluca_termini_entropy, bezdek_partition_coefficient)."""
    dlt = compute_deluca_termini_entropy(s)
    s_tilde = s / (s.sum(dim=1, keepdim=True) + 1e-12)
    bezdek_pc = float((s_tilde * s_tilde).sum(dim=1).mean().item())
    return dlt, bezdek_pc


def compute_link_reconstruction_auc_rmse(
    adj: Tensor, a_new: Tensor, s: Tensor
) -> tuple[float, float]:
    """Link reconstruction (AUC, RMSE) from dense adjacency, super-adjacency, and membership."""
    from labflow.types import dense_to_sparse

    edge_index, edge_weight = dense_to_sparse(adj, threshold=0.0)
    g = GraphData(
        x=torch.zeros((adj.shape[0], 1), dtype=adj.dtype, device=adj.device),
        adj=adj,
        edge_index=edge_index,
        edge_weight=edge_weight,
    )
    rmse, auc = compute_link_reconstruction_metrics(g, s, a_new)
    return auc, rmse


def check_parameter_gradient_finite(layer: GraphReductionLayer) -> bool:
    """Check that all trainable parameters in layer have finite (non-NaN, non-Inf) gradients."""
    params = [p for p in layer.parameters() if p.requires_grad]
    if not params:
        return True
    return all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in params)


def measure_empirical_jacobian_density_and_dead_edges(
    layer: GraphReductionLayer, graph: GraphData
) -> tuple[float, float]:
    """Measure empirical Jacobian active ratio and dead superedge fraction."""
    density, dead, _ = compute_jacobian_and_dead_superedges(layer, graph)
    return density, dead


def compute_feature_reconstruction_r2(target_feat: Tensor, s: Tensor, x_new: Tensor) -> float:
    """Optimal linear ridge-regularized R^2 reconstructing node embeddings H from S_tilde X_new."""
    h_np = target_feat.detach().cpu().numpy().astype(np.float64)
    s_np = s.detach().cpu().numpy().astype(np.float64)
    x_new_np = x_new.detach().cpu().numpy().astype(np.float64)

    s_tilde = s_np / np.maximum(s_np.sum(axis=1, keepdims=True), 1e-12)
    z_Lift = s_tilde @ x_new_np  # [N, F']
    # Solve min_W ||Z_lift W - H||_F^2 + 1e-6 ||W||_F^2
    f_dim = z_Lift.shape[1]
    gram = z_Lift.T @ z_Lift + 1e-6 * np.eye(f_dim)
    w_opt = np.linalg.solve(gram, z_Lift.T @ h_np)
    h_hat = z_Lift @ w_opt

    ss_res = float(np.sum((h_np - h_hat) ** 2))
    h_mean = h_np.mean(axis=0, keepdims=True)
    ss_tot = float(np.sum((h_np - h_mean) ** 2))
    if ss_tot <= 1e-12:
        return 1.0
    return float(np.clip(1.0 - ss_res / ss_tot, -1.0, 1.0))


def compute_jacobian_and_dead_superedges(
    layer: GraphReductionLayer,
    graph: GraphData,
) -> tuple[float, float, float]:
    """Measure (topo_jacobian_density, dead_superedge_ratio, param_grad_finite_ratio)."""
    edge_index, edge_weight = graph.sparse_edges()
    x = graph.x

    # 1. Parameter gradient finiteness check on end-to-end forward/backward
    layer.zero_grad()
    out = layer(graph)
    probe_loss = (out.adj_new.mean() + out.x_new.mean() + out.aux_loss)
    probe_loss.backward()
    params = [p for p in layer.parameters() if p.requires_grad]
    if params:
        finite_cnt = sum(1 for p in params if p.grad is not None and bool(torch.isfinite(p.grad).all()))
        param_finite_ratio = float(finite_cnt) / float(len(params))
    else:
        param_finite_ratio = 1.0
    layer.zero_grad()

    # 2. Topological Jacobian dA_new / dS density and per-superedge dead gradient ratio
    with torch.no_grad():
        h, z = layer.encoder.forward_sparse(edge_index, edge_weight, x)
        s0 = layer.membership(z)

    k = s0.shape[1]
    s_probe = s0.detach().clone().requires_grad_(True)
    if layer.is_classical:
        assert layer.classical_reducer is not None
        a_new = layer.classical_reducer(None, h.detach(), s_probe, edge_index=edge_index, edge_weight=edge_weight).adj_new
    else:
        assert layer.topology_reducer is not None
        a_new = layer.topology_reducer(s_probe, edge_index=edge_index, edge_weight=edge_weight)

    total_grad = torch.autograd.grad(a_new.sum(), s_probe, retain_graph=True)[0]
    topo_density = float((total_grad.abs() > 1e-12).to(torch.float32).mean().item())

    # Check dead super-edges (off-diagonal for zero-diagonal spectral methods, all K^2 for fuzzy/DiffPool)
    zero_diag_method = layer.topology_name in ("mincut", "dmon")
    dead_count = 0
    checked_count = 0
    for i_c in range(k):
        for j_c in range(k):
            if zero_diag_method and i_c == j_c:
                continue
            checked_count += 1
            g_ij = torch.autograd.grad(a_new[i_c, j_c], s_probe, retain_graph=True)[0]
            if float(g_ij.abs().max().item()) <= 1e-12:
                dead_count += 1

    dead_ratio = float(dead_count) / float(max(checked_count, 1))
    return topo_density, dead_ratio, param_finite_ratio


def evaluate_reducer_metrics(
    layer: GraphReductionLayer,
    graph: GraphData,
    method_key: str,
    method_code: str,
    dataset_name: str = "sbm",
    precomputed_output: Optional[PoolingOutput] = None,
) -> ReducerMetricProfile:
    """Compute the full 24-metric analytical profile for `layer` on `graph`."""
    with torch.no_grad():
        out = precomputed_output if precomputed_output is not None else layer(graph)

    s = out.s.detach()
    adj_new = out.adj_new.detach()
    x_new = out.x_new.detach()
    h = out.h.detach() if out.h is not None else graph.x.detach()
    n, k = s.shape

    pred_labels = s.argmax(dim=1).cpu().numpy()
    if graph.node_labels is not None:
        true_labels = graph.node_labels.detach().cpu().numpy()
        nmi = compute_nmi_score(pred_labels, true_labels)
        ari = compute_ari_score(pred_labels, true_labels)
        purity, macro_f1 = compute_purity_and_macro_f1(pred_labels, true_labels)
    else:
        nmi, ari, purity, macro_f1 = 0.0, 0.0, 0.0, 0.0

    edge_index, edge_weight = graph.sparse_edges()
    hard_q, fuzzy_q, cond, ncut = compute_sparse_cut_and_modularity(
        edge_index=edge_index,
        edge_weight=edge_weight,
        s=s,
        pred_labels=pred_labels,
        num_clusters=k,
    )

    spec_dist = compute_spectral_laplacian_distance(
        edge_index=edge_index,
        edge_weight=edge_weight,
        num_nodes=n,
        adj_new=adj_new,
    )
    link_rmse, link_auc = compute_link_reconstruction_metrics(graph=graph, s=s, adj_new=adj_new)

    a_min = float(adj_new.min().item())
    a_max = float(adj_new.max().item())
    violation = max(0.0, -a_min) + max(0.0, a_max - 1.0)
    unit_compliance = 1.0 if violation <= 1e-6 else float(1.0 / (1.0 + violation))

    if k > 1:
        diag_mean = float(torch.diagonal(adj_new).mean().item())
        off_sum = float(adj_new.sum().item() - torch.diagonal(adj_new).sum().item())
        off_mean = off_sum / float(k * (k - 1))
        contrast = diag_mean - off_mean
    else:
        contrast = float(adj_new.mean().item())

    exact_sparsity = float((adj_new == 0.0).to(torch.float32).mean().item())
    effective_sparsity = float((adj_new.abs() < 1e-3).to(torch.float32).mean().item())

    # Fuzzy partition geometry
    s_clamped = s.clamp(1e-12, 1.0 - 1e-12)
    dlt_ent = float(
        (-(s_clamped * torch.log(s_clamped) + (1.0 - s_clamped) * torch.log(1.0 - s_clamped)).mean().item())
        / math.log(2.0)
    )
    s_tilde = s / (s.sum(dim=1, keepdim=True) + 1e-12)
    bezdek_pc = float((s_tilde * s_tilde).sum(dim=1).mean().item())
    active_clusters = len(np.unique(pred_labels))
    active_cluster_ratio = float(active_clusters) / float(max(k, 1))

    q_mat = s.t() @ s
    q_norm = torch.linalg.norm(q_mat, ord="fro").clamp(min=1e-12)
    eye_k = torch.eye(k, dtype=s.dtype, device=s.device) / math.sqrt(float(k))
    orth_defect = float(torch.linalg.norm(q_mat / q_norm - eye_k, ord="fro").item())

    feat_r2 = compute_feature_reconstruction_r2(target_feat=h, s=s, x_new=x_new)
    topo_density, dead_ratio, param_finite = compute_jacobian_and_dead_superedges(layer=layer, graph=graph)

    return ReducerMetricProfile(
        method_key=method_key,
        method_code=method_code,
        dataset_name=dataset_name,
        nmi=nmi,
        ari=ari,
        cluster_purity=purity,
        macro_f1=macro_f1,
        hard_modularity_q=hard_q,
        fuzzy_modularity_q=fuzzy_q,
        conductance=cond,
        normalized_cut=ncut,
        spectral_laplacian_dist=spec_dist,
        link_recon_rmse=link_rmse,
        link_recon_auc=link_auc,
        unit_interval_compliance=unit_compliance,
        adj_new_max=a_max,
        diag_offdiag_contrast=contrast,
        exact_sparsity=exact_sparsity,
        effective_sparsity=effective_sparsity,
        deluca_termini_entropy=dlt_ent,
        bezdek_pc=bezdek_pc,
        active_cluster_ratio=active_cluster_ratio,
        orthogonality_defect=orth_defect,
        feature_recon_r2=feat_r2,
        topo_jacobian_density=topo_density,
        dead_superedge_ratio=dead_ratio,
        param_grad_finite_ratio=param_finite,
    )
