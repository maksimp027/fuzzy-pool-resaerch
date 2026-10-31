"""Method 11: Bandler-Kohout Implicative Infimum-Residual FuzzyPool (inf-implication reduction).

Universal quantifier clique pooling over all pairs (i, j) in {1..N}^2:
    A_new^{BK}(I, J) = min_{i, j in {1..N}} (T(S_{iI}, S_{jJ}) =>_T A_{ij})

Three canonical variants (all dual to existential max-T on complement graph A_bar = 1 1^T - A_refl):
1. Lukasiewicz:   min_{i,j} min(1, 2 - S_{iI} - S_{jJ} + A_{ij}) = 1 - max_{i,j} max(0, S_{iI} + A_bar_{ij} + S_{jJ} - 2)
2. Kleene-Dienes: min_{i,j} max(1 - min(S_{iI}, S_{jJ}), A_{ij}) = 1 - max_{i,j} min(S_{iI}, A_bar_{ij}, S_{jJ})
3. Reichenbach:   min_{i,j} (1 - S_{iI} S_{jJ} (1 - A_{ij}))    = 1 - max_{i,j} (S_{iI} (1 - A_{ij}) S_{jJ})

Matrix-Free Exact Sparse Decomposition O((M + N) K log((M + N) K) + N K^2):
    Eliminates N x N complement matrix materialization (A_bar) by partitioning {1..N} per row i into:
    - Sparse active edges j in N_refl(i): evaluated directly on COO edges in O(M K).
    - Non-edges j not in N_refl(i) where A_bar_{ij} == 1: exact maximum found in O(M K log(M K))
      via the Minimum Excluded Rank (MEX) theorem on descending-sorted columns of S!
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


def _prepare_sparse_reflexive_edges(
    edge_index: Tensor,
    edge_weight: Tensor,
    num_nodes: int,
    ensure_self_loops: bool,
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Coalesce sparse COO edges and optionally merge reflexive self-loops max(A_{ii}, 1.0).

    Returns:
        uniq_keys: Sorted 1D keys `src_u * N + dst_u` of shape [M_u].
        src_u: Sorted source indices [M_u].
        dst_u: Destination indices [M_u].
        ew_u: Reflexive edge weights [M_u].
        edge_to_u: Mapping from input edge `e in 0..M-1` to index in `uniq_keys` [M].
    """
    device = edge_weight.device
    dtype = edge_weight.dtype
    m = int(edge_weight.numel())

    if m > 0:
        in_keys = edge_index[0] * num_nodes + edge_index[1]
        in_keys_s, in_order = torch.sort(in_keys)
        has_dups = (m > 1) and bool((in_keys_s[1:] == in_keys_s[:-1]).any())
        if has_dups:
            c_keys, inv_in = torch.unique_consecutive(in_keys_s, return_inverse=True)
            c_ew = torch.zeros((c_keys.numel(),), dtype=dtype, device=device)
            c_ew.index_add_(0, inv_in, edge_weight[in_order])
        else:
            c_keys = in_keys_s
            c_ew = edge_weight[in_order]
    else:
        in_keys = torch.empty((0,), dtype=torch.long, device=device)
        c_keys = in_keys
        c_ew = edge_weight

    if ensure_self_loops and num_nodes > 0:
        diag_nodes = torch.arange(num_nodes, dtype=torch.long, device=device)
        diag_keys = diag_nodes * (num_nodes + 1)
        diag_ew = torch.ones((num_nodes,), dtype=dtype, device=device)
        if c_keys.numel() > 0:
            all_keys = torch.cat([c_keys, diag_keys], dim=0)
            all_ew = torch.cat([c_ew, diag_ew], dim=0)
            s_keys, s_ord = torch.sort(all_keys, stable=True)
            s_ew = all_ew[s_ord]
            uniq_keys, inv_u = torch.unique_consecutive(s_keys, return_inverse=True)
            ew_u = torch.full((uniq_keys.numel(),), -torch.inf, dtype=dtype, device=device)
            ew_u.scatter_reduce_(0, inv_u, s_ew, reduce="amax", include_self=True)
        else:
            uniq_keys = diag_keys
            ew_u = diag_ew
    else:
        uniq_keys = c_keys
        ew_u = c_ew

    src_u = torch.div(uniq_keys, num_nodes, rounding_mode="floor")
    dst_u = uniq_keys - src_u * num_nodes
    edge_to_u = torch.searchsorted(uniq_keys, in_keys) if m > 0 else in_keys
    return uniq_keys, src_u, dst_u, ew_u, edge_to_u


def _sparse_bk_step1(
    s: Tensor,
    src_u: Tensor,
    dst_u: Tensor,
    ew_u: Tensor,
    implication: str,
) -> Tuple[Tensor, Tensor]:
    """Compute exact Step-1 complement max T_bar_{iJ} = max_{j in 1..N} f(1 - A_refl_{ij}, S_{jJ}) without N x N matrix."""
    n, k = s.shape
    m_u = int(ew_u.numel())
    device = s.device
    dtype = s.dtype

    # 1. Active sparse edges evaluation
    t_edge = torch.full((n, k), -torch.inf, dtype=dtype, device=device)
    j_edge_best = torch.full((n, k), n, dtype=torch.long, device=device)

    if m_u > 0:
        a_bar_e = 1.0 - ew_u  # [M_u]
        s_dst = s[dst_u]  # [M_u, K]
        if implication == "reichenbach":
            u_edge = a_bar_e.unsqueeze(1) * s_dst
        elif implication == "lukasiewicz":
            u_edge = a_bar_e.unsqueeze(1) + s_dst
        else:  # kleene_dienes
            u_edge = torch.minimum(a_bar_e.unsqueeze(1), s_dst)

        src_exp = src_u.unsqueeze(1).expand(m_u, k)
        t_edge.scatter_reduce_(0, src_exp, u_edge, reduce="amax", include_self=True)

        t_at_src = t_edge[src_u]
        is_max_e = (u_edge == t_at_src) & torch.isfinite(t_at_src)
        cand_j = torch.where(is_max_e, dst_u.unsqueeze(1).expand(m_u, k), torch.full((m_u, k), n, dtype=torch.long, device=device))
        j_edge_best.scatter_reduce_(0, src_exp, cand_j, reduce="amin", include_self=True)

    # 2. Non-edges evaluation via exact Minimum Excluded Rank (MEX) theorem on descending-sorted columns of S
    perm = torch.argsort(-s, dim=0, stable=True)  # [N, K], ties broken by smallest node index j
    ranks = torch.arange(n, dtype=torch.long, device=device).unsqueeze(1).expand(n, k)
    inv_perm = torch.empty((n, k), dtype=torch.long, device=device)
    inv_perm.scatter_(0, perm, ranks)

    if m_u > 0:
        deg_u = torch.bincount(src_u, minlength=n)  # [N]
        k_ids = torch.arange(k, dtype=torch.long, device=device).unsqueeze(0)
        flat_seg = (src_u.unsqueeze(1) * k + k_ids).reshape(-1)  # [M_u * K]
        flat_r = inv_perm[dst_u].reshape(-1)  # [M_u * K]

        sort_key = flat_seg * n + flat_r
        sorted_key = torch.sort(sort_key)[0]
        sorted_seg = torch.div(sorted_key, n, rounding_mode="floor")
        sorted_r = sorted_key - sorted_seg * n

        seg_lengths = deg_u.unsqueeze(1).expand(n, k).reshape(-1)  # [N * K]
        seg_starts = torch.cumsum(seg_lengths, dim=0) - seg_lengths
        local_idx = torch.arange(m_u * k, dtype=torch.long, device=device) - seg_starts[sorted_seg]

        is_prefix = (sorted_r == local_idx).to(torch.long)
        mex_flat = torch.zeros((n * k,), dtype=torch.long, device=device)
        mex_flat.scatter_add_(0, sorted_seg, is_prefix)
        mex_rank = mex_flat.view(n, k)
    else:
        mex_rank = torch.zeros((n, k), dtype=torch.long, device=device)

    has_nonedge = mex_rank < n
    safe_rank = mex_rank.clamp(max=max(0, n - 1))
    j_nonedge_cand = perm.gather(0, safe_rank)  # [N, K]
    s_nonedge = s.gather(0, j_nonedge_cand)  # [N, K]

    if implication == "reichenbach":
        val_nonedge = s_nonedge
    elif implication == "lukasiewicz":
        val_nonedge = 1.0 + s_nonedge
    else:  # kleene_dienes
        val_nonedge = torch.minimum(torch.ones_like(s_nonedge), s_nonedge)

    t_nonedge = torch.where(has_nonedge, val_nonedge, torch.full_like(val_nonedge, -torch.inf))
    j_nonedge = torch.where(has_nonedge, j_nonedge_cand, torch.full_like(j_nonedge_cand, n))

    # 3. Combine active-edge max and non-edge max with smallest-j tie-breaking
    t_bar = torch.maximum(t_edge, t_nonedge)
    j_best = torch.where(
        t_edge > t_nonedge,
        j_edge_best,
        torch.where(t_nonedge > t_edge, j_nonedge, torch.minimum(j_edge_best, j_nonedge)),
    )
    return t_bar, j_best


class _BandlerKohoutSparseFn(torch.autograd.Function):
    """Exact Matrix-Free O((M+N) K log((M+N) K) + N K^2) forward & analytical backward for Bandler-Kohout."""

    @staticmethod
    def forward(
        ctx,
        s: Tensor,
        edge_index: Tensor,
        edge_weight: Tensor,
        ensure_self_loops: bool,
        implication: str,
    ) -> Tensor:
        n, k = s.shape
        uniq_keys, src_u, dst_u, ew_u, _ = _prepare_sparse_reflexive_edges(
            edge_index=edge_index,
            edge_weight=edge_weight,
            num_nodes=n,
            ensure_self_loops=ensure_self_loops,
        )
        t_bar, j_best = _sparse_bk_step1(s, src_u, dst_u, ew_u, implication=implication)

        if implication == "reichenbach":
            v_nik = s.unsqueeze(2) * t_bar.unsqueeze(1)
            max_violation, i_dag = v_nik.max(dim=0)
            a_new = 1.0 - max_violation
            u_active = torch.ones_like(max_violation)
        elif implication == "lukasiewicz":
            v_nik = s.unsqueeze(2) + t_bar.unsqueeze(1) - 2.0
            max_violation, i_dag = v_nik.max(dim=0)
            u_active = (max_violation > 0.0).to(s.dtype)
            a_new = 1.0 - F.relu(max_violation)
        else:  # kleene_dienes
            v_nik = torch.minimum(s.unsqueeze(2), t_bar.unsqueeze(1))
            max_violation, i_dag = v_nik.max(dim=0)
            a_new = 1.0 - max_violation
            u_active = torch.ones_like(max_violation)

        j_dag = j_best.gather(0, i_dag)

        # Look up A_bar(i_dag, j_dag) in O(K^2 log M_u) without any N x N matrix!
        dag_keys = i_dag * n + j_dag
        if uniq_keys.numel() > 0:
            pos = torch.searchsorted(uniq_keys, dag_keys).clamp(max=uniq_keys.numel() - 1)
            found = uniq_keys[pos] == dag_keys
            a_refl_dag = torch.where(found, ew_u[pos], torch.zeros_like(a_new))
        else:
            a_refl_dag = torch.zeros_like(a_new)
        a_bar_val = 1.0 - a_refl_dag

        ctx.save_for_backward(s, edge_index, edge_weight, i_dag, j_dag, a_bar_val, max_violation, u_active)
        ctx.ensure_self_loops = ensure_self_loops
        ctx.implication = implication
        return a_new

    @staticmethod
    def backward(
        ctx,
        grad_a_new: Tensor,
    ) -> Tuple[Optional[Tensor], None, Optional[Tensor], None, None]:
        s, edge_index, edge_weight, i_dag, j_dag, a_bar_val, max_violation, u_active = ctx.saved_tensors
        implication = ctx.implication
        ensure_self_loops = ctx.ensure_self_loops
        n, k = s.shape
        m = int(edge_weight.numel())

        i_idx = torch.arange(k, device=s.device).unsqueeze(1).expand(k, k)
        j_idx = torch.arange(k, device=s.device).unsqueeze(0).expand(k, k)
        s_left = s[i_dag, i_idx]
        s_right = s[j_dag, j_idx]

        if implication == "reichenbach":
            w_edge = s_left * s_right
            contrib_left = -grad_a_new * (a_bar_val * s_right)
            contrib_right = -grad_a_new * (s_left * a_bar_val)
        elif implication == "lukasiewicz":
            w_edge = u_active
            active_grad = grad_a_new * u_active
            contrib_left = -active_grad
            contrib_right = -active_grad
        else:  # kleene_dienes
            is_left = (s_left == max_violation).to(s.dtype)
            is_edge = (a_bar_val == max_violation).to(s.dtype)
            is_right = (s_right == max_violation).to(s.dtype)
            tie_count = (is_left + is_edge + is_right).clamp(min=1.0)
            w_edge = is_edge / tie_count
            contrib_left = -grad_a_new * (is_left / tie_count)
            contrib_right = -grad_a_new * (is_right / tie_count)

        grad_s = None
        if ctx.needs_input_grad[0]:
            grad_s = torch.zeros((n, k), dtype=s.dtype, device=s.device)
            grad_s.scatter_add_(0, i_dag.t(), contrib_left.t())
            grad_s.scatter_add_(0, j_dag, contrib_right)

        grad_ew = None
        if ctx.needs_input_grad[2]:
            grad_ew = torch.zeros((m,), dtype=s.dtype, device=s.device)
            if m > 0:
                # Map winning pairs (i_dag, j_dag) back to input edges if present
                in_keys = edge_index[0] * n + edge_index[1]
                in_keys_s, in_order = torch.sort(in_keys)
                dag_flat = (i_dag * n + j_dag).reshape(-1)
                g_edge_flat = (grad_a_new * w_edge).reshape(-1)
                if ensure_self_loops:
                    # On diagonal with ensure_self_loops=True, max(A_{ii}, 1.0) has derivative 0 when A_{ii} <= 1.0
                    not_clamped_diag = (i_dag != j_dag).reshape(-1) | ((1.0 - a_bar_val).reshape(-1) > 1.0)
                    g_edge_flat = torch.where(not_clamped_diag, g_edge_flat, torch.zeros_like(g_edge_flat))

                # Accumulate across all K^2 pairs into unique dag keys, then match against input edges
                pos_l = torch.searchsorted(in_keys_s, dag_flat, right=False)
                pos_r = torch.searchsorted(in_keys_s, dag_flat, right=True)
                for idx_pair in range(dag_flat.numel()):
                    lo = int(pos_l[idx_pair].item())
                    hi = int(pos_r[idx_pair].item())
                    if hi > lo:
                        grad_ew[in_order[lo:hi]] += g_edge_flat[idx_pair]

        return grad_s, None, grad_ew, None, None


class _BandlerKohoutReichenbachFn(torch.autograd.Function):
    """Exact analytical forward/backward for dense Reichenbach Bandler-Kohout reduction (Section 11.3)."""

    @staticmethod
    def forward(ctx, adj_refl: Tensor, s: Tensor) -> Tensor:
        a_bar = 1.0 - adj_refl
        u_njk = a_bar.unsqueeze(2) * s.unsqueeze(0)
        t_bar, j_best = u_njk.max(dim=1)
        v_nik = s.unsqueeze(2) * t_bar.unsqueeze(1)
        max_violation, i_dag = v_nik.max(dim=0)
        j_dag = j_best.gather(0, i_dag)
        a_new = 1.0 - max_violation

        ctx.save_for_backward(a_bar, s, i_dag, j_dag)
        return a_new

    @staticmethod
    def backward(ctx, grad_a_new: Tensor) -> Tuple[Optional[Tensor], Optional[Tensor]]:
        a_bar, s, i_dag, j_dag = ctx.saved_tensors
        n, k = s.shape

        i_idx = torch.arange(k, device=s.device).unsqueeze(1).expand(k, k)
        j_idx = torch.arange(k, device=s.device).unsqueeze(0).expand(k, k)

        s_left = s[i_dag, i_idx]
        s_right = s[j_dag, j_idx]
        a_bar_val = a_bar[i_dag, j_dag]

        grad_adj_refl = None
        if ctx.needs_input_grad[0]:
            grad_flat = torch.zeros((n * n,), dtype=s.dtype, device=s.device)
            flat_idx = (i_dag * n + j_dag).reshape(-1)
            grad_flat.scatter_add_(0, flat_idx, (grad_a_new * s_left * s_right).reshape(-1))
            grad_adj_refl = grad_flat.view(n, n)

        grad_s = None
        if ctx.needs_input_grad[1]:
            grad_s = torch.zeros((n, k), dtype=s.dtype, device=s.device)
            contrib_left = -grad_a_new * (a_bar_val * s_right)
            contrib_right = -grad_a_new * (s_left * a_bar_val)
            grad_s.scatter_add_(0, i_dag.t(), contrib_left.t())
            grad_s.scatter_add_(0, j_dag, contrib_right)

        return grad_adj_refl, grad_s


class _BandlerKohoutLukasiewiczFn(torch.autograd.Function):
    """Exact analytical forward/subgradient for dense Lukasiewicz Bandler-Kohout reduction."""

    @staticmethod
    def forward(ctx, adj_refl: Tensor, s: Tensor) -> Tensor:
        a_bar = 1.0 - adj_refl
        u_njk = a_bar.unsqueeze(2) + s.unsqueeze(0)
        t_bar, j_best = u_njk.max(dim=1)
        v_nik = s.unsqueeze(2) + t_bar.unsqueeze(1) - 2.0
        raw_max, i_dag = v_nik.max(dim=0)
        j_dag = j_best.gather(0, i_dag)

        u_active = (raw_max > 0.0).to(s.dtype)
        a_new = 1.0 - F.relu(raw_max)
        ctx.save_for_backward(s, i_dag, j_dag, u_active)
        return a_new

    @staticmethod
    def backward(ctx, grad_a_new: Tensor) -> Tuple[Optional[Tensor], Optional[Tensor]]:
        s, i_dag, j_dag, u_active = ctx.saved_tensors
        n, k = s.shape
        active_grad = grad_a_new * u_active

        grad_adj_refl = None
        if ctx.needs_input_grad[0]:
            grad_flat = torch.zeros((n * n,), dtype=s.dtype, device=s.device)
            flat_idx = (i_dag * n + j_dag).reshape(-1)
            grad_flat.scatter_add_(0, flat_idx, active_grad.reshape(-1))
            grad_adj_refl = grad_flat.view(n, n)

        grad_s = None
        if ctx.needs_input_grad[1]:
            grad_s = torch.zeros((n, k), dtype=s.dtype, device=s.device)
            grad_s.scatter_add_(0, i_dag.t(), (-active_grad).t())
            grad_s.scatter_add_(0, j_dag, -active_grad)

        return grad_adj_refl, grad_s


class _BandlerKohoutKleeneDienesFn(torch.autograd.Function):
    """Exact analytical forward/subgradient for dense Kleene-Dienes Bandler-Kohout reduction."""

    @staticmethod
    def forward(ctx, adj_refl: Tensor, s: Tensor) -> Tensor:
        a_bar = 1.0 - adj_refl
        u_njk = torch.minimum(a_bar.unsqueeze(2), s.unsqueeze(0))
        t_bar, j_best = u_njk.max(dim=1)
        v_nik = torch.minimum(s.unsqueeze(2), t_bar.unsqueeze(1))
        max_violation, i_dag = v_nik.max(dim=0)
        j_dag = j_best.gather(0, i_dag)
        a_new = 1.0 - max_violation

        ctx.save_for_backward(a_bar, s, max_violation, i_dag, j_dag)
        return a_new

    @staticmethod
    def backward(ctx, grad_a_new: Tensor) -> Tuple[Optional[Tensor], Optional[Tensor]]:
        a_bar, s, max_violation, i_dag, j_dag = ctx.saved_tensors
        n, k = s.shape

        i_idx = torch.arange(k, device=s.device).unsqueeze(1).expand(k, k)
        j_idx = torch.arange(k, device=s.device).unsqueeze(0).expand(k, k)

        s_left = s[i_dag, i_idx]
        s_right = s[j_dag, j_idx]
        a_bar_val = a_bar[i_dag, j_dag]

        is_left = (s_left == max_violation).to(s.dtype)
        is_edge = (a_bar_val == max_violation).to(s.dtype)
        is_right = (s_right == max_violation).to(s.dtype)
        tie_count = (is_left + is_edge + is_right).clamp(min=1.0)

        grad_adj_refl = None
        if ctx.needs_input_grad[0]:
            grad_flat = torch.zeros((n * n,), dtype=s.dtype, device=s.device)
            flat_idx = (i_dag * n + j_dag).reshape(-1)
            grad_flat.scatter_add_(0, flat_idx, (grad_a_new * (is_edge / tie_count)).reshape(-1))
            grad_adj_refl = grad_flat.view(n, n)

        grad_s = None
        if ctx.needs_input_grad[1]:
            grad_s = torch.zeros((n, k), dtype=s.dtype, device=s.device)
            contrib_left = -grad_a_new * (is_left / tie_count)
            contrib_right = -grad_a_new * (is_right / tie_count)
            grad_s.scatter_add_(0, i_dag.t(), contrib_left.t())
            grad_s.scatter_add_(0, j_dag, contrib_right)

        return grad_adj_refl, grad_s


class BandlerKohoutTopologyReducer(nn.Module):
    """Method 11: Bandler-Kohout implicative infimum-residual FuzzyPool (universal quantifier).

    Uses exact Matrix-Free O((M+N) K log((M+N) K) + N K^2) sparse MEX decomposition whenever
    `edge_index` and `edge_weight` are provided (avoiding O(N^2) complement materialization).
    """

    def __init__(
        self,
        implication: str = "reichenbach",
        ensure_self_loops: bool = True,
    ) -> None:
        super().__init__()
        implication = implication.lower()
        if implication not in ("reichenbach", "lukasiewicz", "kleene_dienes"):
            raise ValueError("`implication` must be 'reichenbach', 'lukasiewicz', or 'kleene_dienes'.")
        self.implication = implication
        self.ensure_self_loops = ensure_self_loops

    def _prepare_adj(self, adj: Tensor) -> Tensor:
        if not self.ensure_self_loops:
            return adj
        eye = torch.eye(adj.shape[0], dtype=adj.dtype, device=adj.device)
        return torch.maximum(adj, eye)

    def forward(
        self,
        s: Tensor,
        adj: Optional[Tensor] = None,
        edge_index: Optional[Tensor] = None,
        edge_weight: Optional[Tensor] = None,
    ) -> Tensor:
        if edge_index is not None and edge_weight is not None and (adj is None or not adj.requires_grad):
            return _BandlerKohoutSparseFn.apply(
                s,
                edge_index,
                edge_weight,
                self.ensure_self_loops,
                self.implication,
            )

        if adj is None:
            raise ValueError("Provide either `adj` or `(edge_index, edge_weight)`.")
        adj_refl = self._prepare_adj(adj)

        if self.implication == "reichenbach":
            return _BandlerKohoutReichenbachFn.apply(adj_refl, s)
        if self.implication == "lukasiewicz":
            return _BandlerKohoutLukasiewiczFn.apply(adj_refl, s)
        return _BandlerKohoutKleeneDienesFn.apply(adj_refl, s)

    def forward_direct_implication_reference(self, adj: Tensor, s: Tensor) -> Tensor:
        """Direct universal-quantifier min_{i,j} (T(S_{iI}, S_{jJ}) => A_{ij}) reference from Section 11.1."""
        adj_refl = self._prepare_adj(adj)
        s_i = s[:, None, :, None]
        s_j = s[None, :, None, :]
        a_ij = adj_refl[:, :, None, None]

        if self.implication == "lukasiewicz":
            impl = torch.clamp(2.0 - s_i - s_j + a_ij, max=1.0)
        elif self.implication == "kleene_dienes":
            impl = torch.maximum(1.0 - torch.minimum(s_i, s_j), a_ij)
        else:  # reichenbach
            impl = 1.0 - s_i * s_j * (1.0 - a_ij)
        return impl.amin(dim=(0, 1))
