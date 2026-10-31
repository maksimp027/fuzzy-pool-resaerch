"""Shared vectorized O(M K + N K^2) two-step sparse decomposition utilities for sup-T fuzzy pooling."""

from __future__ import annotations

from typing import Callable, Tuple

import torch
from torch import Tensor


def sparse_two_step_max_with_winners(
    s: Tensor,
    edge_index: Tensor,
    u_ej: Tensor,
    step2_combine: Callable[[Tensor, Tensor], Tensor],
) -> Tuple[Tensor, Tensor, Tensor]:
    """Execute 2-step sparse max-T decomposition and return (a_new, t_ij_winning, e_star).

    Args:
        s: Membership matrix [N, K].
        edge_index: COO edge indices [2, M].
        u_ej: Step-1 edge-to-cluster values T(A_{u_e, v_e}, S_{v_e, J}) of shape [M, K].
        step2_combine: Vectorized binary operator (s_ni1 [N, K, 1], t_n1j [N, 1, K]) -> [N, K, K].

    Returns:
        a_new: Reduced topology [K, K].
        t_star: Intermediate Step-1 value T_{i*_{IJ}, J} at winning node i*_{IJ}, shape [K, K].
        e_star: Winning edge index e*_{IJ} in {0, ..., M - 1} for each pair (I, J), shape [K, K].
    """
    n, k = s.shape
    m = int(edge_index.shape[1])
    device = s.device
    dtype = s.dtype

    if m == 0:
        zeros_kk = torch.zeros((k, k), dtype=dtype, device=device)
        neg_one = torch.full((k, k), -1, dtype=torch.long, device=device)
        return zeros_kk, zeros_kk, neg_one

    src = edge_index[0]
    src_idx = src.unsqueeze(1).expand(m, k)

    # Step 1: T_{iJ} = max_{j in N(i)} U_{e=(i,j), J}
    t_ij = torch.full((n, k), -torch.inf, dtype=dtype, device=device)
    t_ij = t_ij.scatter_reduce(0, src_idx, u_ej, reduce="amax", include_self=True)

    # Deterministic tie-breaking: smallest edge index e achieving T_{u_e, J}
    t_at_src = t_ij[src]
    is_max = (u_ej == t_at_src) & torch.isfinite(t_at_src)
    e_ids = torch.arange(m, dtype=torch.long, device=device).unsqueeze(1).expand(m, k)
    cand_e = torch.where(is_max, e_ids, torch.full_like(e_ids, m))
    best_e_ij = torch.full((n, k), m, dtype=torch.long, device=device)
    best_e_ij = best_e_ij.scatter_reduce(0, src_idx, cand_e, reduce="amin", include_self=True)

    valid_ij = best_e_ij < m  # [N, K]
    t_safe = torch.where(valid_ij, t_ij, torch.zeros_like(t_ij))

    # Step 2: A_new(I, J) = max_{i} T(S_{iI}, T_{iJ}) over nodes i with N(i) != empty
    v_nij = step2_combine(s.unsqueeze(2), t_safe.unsqueeze(1))  # [N, K, K]
    v_masked = torch.where(valid_ij.unsqueeze(1), v_nij, torch.full_like(v_nij, -torch.inf))
    a_new, i_star = v_masked.max(dim=0)  # [K, K], [K, K]

    e_star = best_e_ij.gather(0, i_star)  # e_star[I, J] = best_e_ij[i_star[I, J], J]
    t_star = t_safe.gather(0, i_star)  # t_star[I, J] = t_safe[i_star[I, J], J]
    return a_new, t_star, e_star


def accumulate_winner_gradients(
    num_nodes: int,
    num_clusters: int,
    num_edges: int,
    edge_index: Tensor,
    e_star: Tensor,
    contrib_left: Tensor,
    contrib_right: Tensor,
    contrib_edge: Tensor | None = None,
) -> Tuple[Tensor, Tensor | None]:
    """Accumulate O(K^2) winning-edge analytical partial derivatives into grad_S and grad_edge_weight.

    Implements:
        grad_S[k, L] = sum_J contrib_left[L, J] delta_{k, i*_{LJ}}
                     + sum_I contrib_right[I, L] delta_{k, j*_{IL}}
    """
    dtype = contrib_left.dtype
    device = contrib_left.device
    grad_s = torch.zeros((num_nodes, num_clusters), dtype=dtype, device=device)
    grad_ew = (
        torch.zeros((num_edges,), dtype=dtype, device=device)
        if contrib_edge is not None
        else None
    )

    if num_edges == 0:
        return grad_s, grad_ew

    src = edge_index[0]
    dst = edge_index[1]
    i_star = src[e_star]  # [K, K]
    j_star = dst[e_star]  # [K, K]

    grad_s.scatter_add_(0, i_star.t(), contrib_left.t())
    grad_s.scatter_add_(0, j_star, contrib_right)

    if grad_ew is not None and contrib_edge is not None:
        grad_ew.scatter_add_(0, e_star.reshape(-1), contrib_edge.reshape(-1))

    return grad_s, grad_ew
