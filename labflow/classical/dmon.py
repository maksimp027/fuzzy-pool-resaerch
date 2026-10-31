"""Method 3: DMoN (Deep Modularity Networks — Chung-Lu Modular Graph Reduction).

Forward Pass:
    d     = A 1_N,  2m = 1_N^T d
    B     = A - (d d^T) / (2m)  (never materialized in memory; rank-1 decomposition)
    X_new = D_S^{-1} S^T H,  D_S = diag(sum_i S_{i1}, ..., sum_i S_{iK})
    A_new = MinCutPool spectral zero-diagonal normalized topology reduction
Loss:
    L_DMoN = -(1 / (2m)) Tr(S^T A S - d_S d_S^T / (2m)) + (sqrt(K) / N) ||c||_2 - 1

Supports both dense O(N^2) adjacency and true sparse O(M K + N K^2) COO edges without N x N allocation.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from labflow.classical.mincut import mincut_topology_backward
from labflow.gnn import sparse_symmetric_normalize_adjacency, symmetric_normalize_adjacency
from labflow.types import PoolingOutput


class _DMoNFn(torch.autograd.Function):
    """Exact analytical forward/backward for DMoN without materializing N x N modularity matrix B."""

    @staticmethod
    def forward(
        ctx,
        adj: Tensor,
        a_tilde: Tensor,
        h: Tensor,
        s: Tensor,
        eps: float,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        n, k = s.shape

        c = s.sum(dim=0)
        c_eps = c + eps
        x_new = (s.t() @ h) / c_eps.unsqueeze(1)

        a_pool = s.t() @ a_tilde @ s
        off_diag = 1.0 - torch.eye(k, dtype=s.dtype, device=s.device)
        a_hat_new = a_pool * off_diag
        d_new = a_hat_new.sum(dim=1) + eps
        inv_sqrt_d_new = torch.rsqrt(d_new)
        adj_new = inv_sqrt_d_new.unsqueeze(1) * a_hat_new * inv_sqrt_d_new.unsqueeze(0)

        d_row = adj.sum(dim=1)
        d_col = adj.sum(dim=0)
        two_m = d_row.sum() + eps
        as_mat = adj @ s
        tr_sas = (s * as_mat).sum()
        d_s_row = s.t() @ d_row
        d_s_col = s.t() @ d_col
        tr_null = (d_s_row * d_s_col).sum() / two_m
        l_mod = -(tr_sas - tr_null) / two_m

        c_norm = torch.sqrt((c * c).sum() + eps * eps)
        l_collapse = (math.sqrt(float(k)) / float(n)) * c_norm - 1.0

        ctx.save_for_backward(
            adj,
            a_tilde,
            h,
            s,
            x_new,
            adj_new,
            c,
            c_eps,
            c_norm,
            d_new,
            inv_sqrt_d_new,
            off_diag,
            d_row,
            d_col,
            two_m,
            tr_sas,
            tr_null,
            d_s_row,
            d_s_col,
            as_mat,
        )
        return x_new, adj_new, l_mod, l_collapse

    @staticmethod
    def backward(
        ctx,
        grad_x_new: Tensor,
        grad_adj_new: Tensor,
        grad_l_mod: Tensor,
        grad_l_collapse: Tensor,
    ) -> Tuple[Optional[Tensor], Optional[Tensor], Optional[Tensor], Optional[Tensor], None]:
        (
            adj,
            a_tilde,
            h,
            s,
            x_new,
            adj_new,
            c,
            c_eps,
            c_norm,
            d_new,
            inv_sqrt_d_new,
            off_diag,
            d_row,
            d_col,
            two_m,
            tr_sas,
            tr_null,
            d_s_row,
            d_s_col,
            as_mat,
        ) = ctx.saved_tensors
        n, k = s.shape

        g_a_hat = mincut_topology_backward(grad_adj_new, adj_new, d_new, inv_sqrt_d_new, off_diag)

        grad_adj = None
        if ctx.needs_input_grad[0]:
            outer_ds = (s @ d_s_col).unsqueeze(1) + (s @ d_s_row).unsqueeze(0)
            d_tr_null_d_adj = (outer_ds * two_m - (d_s_row * d_s_col).sum()) / (two_m * two_m)
            d_tr_b_d_adj = (s @ s.t()) - d_tr_null_d_adj
            tr_b = tr_sas - tr_null
            grad_adj = -grad_l_mod * (d_tr_b_d_adj * two_m - tr_b) / (two_m * two_m)

        grad_a_tilde = None
        if ctx.needs_input_grad[1]:
            grad_a_tilde = s @ g_a_hat @ s.t()

        grad_h = None
        if ctx.needs_input_grad[2]:
            grad_h = s @ (grad_x_new / c_eps.unsqueeze(1))

        grad_s = None
        if ctx.needs_input_grad[3]:
            row_dot = (grad_x_new * x_new).sum(dim=1)
            grad_s_feat = (h @ grad_x_new.t() - row_dot.unsqueeze(0)) / c_eps.unsqueeze(0)
            grad_s_topo = (a_tilde @ s) @ g_a_hat.t() + (a_tilde.t() @ s) @ g_a_hat

            ats_mat = as_mat if torch.equal(d_row, d_col) and torch.allclose(adj, adj.t()) else (adj.t() @ s)
            null_term = (
                d_row.unsqueeze(1) * d_s_col.unsqueeze(0)
                + d_col.unsqueeze(1) * d_s_row.unsqueeze(0)
            ) / two_m
            grad_s_mod = -grad_l_mod * ((as_mat + ats_mat) - null_term) / two_m

            scale_col = grad_l_collapse * (math.sqrt(float(k)) / (float(n) * c_norm))
            grad_s_collapse = scale_col * c.unsqueeze(0).expand(n, k)

            grad_s = grad_s_feat + grad_s_topo + grad_s_mod + grad_s_collapse

        return grad_adj, grad_a_tilde, grad_h, grad_s, None


class _DMoNSparseFn(torch.autograd.Function):
    """Exact O(M K + N K^2) analytical forward/backward for sparse DMoN without any N x N matrix."""

    @staticmethod
    def forward(
        ctx,
        edge_index: Tensor,
        ew: Tensor,
        ew_tilde: Tensor,
        h: Tensor,
        s: Tensor,
        eps: float,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        n, k = s.shape

        c = s.sum(dim=0)
        c_eps = c + eps
        x_new = (s.t() @ h) / c_eps.unsqueeze(1)

        d_row = torch.zeros((n,), dtype=s.dtype, device=s.device)
        d_col = torch.zeros((n,), dtype=s.dtype, device=s.device)
        as_mat = torch.zeros((n, k), dtype=s.dtype, device=s.device)
        ats_mat = torch.zeros((n, k), dtype=s.dtype, device=s.device)
        at_s = torch.zeros((n, k), dtype=s.dtype, device=s.device)
        att_s = torch.zeros((n, k), dtype=s.dtype, device=s.device)

        if ew.numel() > 0:
            src, dst = edge_index[0], edge_index[1]
            d_row.index_add_(0, src, ew)
            d_col.index_add_(0, dst, ew)
            as_mat.index_add_(0, src, ew.unsqueeze(1) * s[dst])
            ats_mat.index_add_(0, dst, ew.unsqueeze(1) * s[src])
            at_s.index_add_(0, src, ew_tilde.unsqueeze(1) * s[dst])
            att_s.index_add_(0, dst, ew_tilde.unsqueeze(1) * s[src])

        a_pool = s.t() @ at_s
        off_diag = 1.0 - torch.eye(k, dtype=s.dtype, device=s.device)
        a_hat_new = a_pool * off_diag
        d_new = a_hat_new.sum(dim=1) + eps
        inv_sqrt_d_new = torch.rsqrt(d_new)
        adj_new = inv_sqrt_d_new.unsqueeze(1) * a_hat_new * inv_sqrt_d_new.unsqueeze(0)

        two_m = d_row.sum() + eps
        tr_sas = (s * as_mat).sum()
        d_s_row = s.t() @ d_row
        d_s_col = s.t() @ d_col
        tr_null = (d_s_row * d_s_col).sum() / two_m
        l_mod = -(tr_sas - tr_null) / two_m

        c_norm = torch.sqrt((c * c).sum() + eps * eps)
        l_collapse = (math.sqrt(float(k)) / float(n)) * c_norm - 1.0

        ctx.save_for_backward(
            edge_index,
            ew,
            ew_tilde,
            h,
            s,
            x_new,
            adj_new,
            c,
            c_eps,
            c_norm,
            d_new,
            inv_sqrt_d_new,
            off_diag,
            d_row,
            d_col,
            two_m,
            tr_sas,
            tr_null,
            d_s_row,
            d_s_col,
            as_mat,
            ats_mat,
            at_s,
            att_s,
        )
        return x_new, adj_new, l_mod, l_collapse

    @staticmethod
    def backward(
        ctx,
        grad_x_new: Tensor,
        grad_adj_new: Tensor,
        grad_l_mod: Tensor,
        grad_l_collapse: Tensor,
    ) -> Tuple[None, Optional[Tensor], Optional[Tensor], Optional[Tensor], Optional[Tensor], None]:
        (
            edge_index,
            ew,
            ew_tilde,
            h,
            s,
            x_new,
            adj_new,
            c,
            c_eps,
            c_norm,
            d_new,
            inv_sqrt_d_new,
            off_diag,
            d_row,
            d_col,
            two_m,
            tr_sas,
            tr_null,
            d_s_row,
            d_s_col,
            as_mat,
            ats_mat,
            at_s,
            att_s,
        ) = ctx.saved_tensors
        n, k = s.shape

        g_a_hat = mincut_topology_backward(grad_adj_new, adj_new, d_new, inv_sqrt_d_new, off_diag)

        grad_ew = None
        if ctx.needs_input_grad[1]:
            if ew.numel() > 0:
                src, dst = edge_index[0], edge_index[1]
                s_ds_col = s @ d_s_col
                s_ds_row = s @ d_s_row
                outer_e = s_ds_col[src] + s_ds_row[dst]
                d_tr_null_de = (outer_e * two_m - (d_s_row * d_s_col).sum()) / (two_m * two_m)
                d_tr_b_de = (s[src] * s[dst]).sum(dim=1) - d_tr_null_de
                tr_b = tr_sas - tr_null
                grad_ew = -grad_l_mod * (d_tr_b_de * two_m - tr_b) / (two_m * two_m)
            else:
                grad_ew = torch.zeros_like(ew)

        grad_ew_tilde = None
        if ctx.needs_input_grad[2]:
            if ew_tilde.numel() > 0:
                src, dst = edge_index[0], edge_index[1]
                grad_ew_tilde = ((s[src] @ g_a_hat) * s[dst]).sum(dim=1)
            else:
                grad_ew_tilde = torch.zeros_like(ew_tilde)

        grad_h = s @ (grad_x_new / c_eps.unsqueeze(1)) if ctx.needs_input_grad[3] else None

        grad_s = None
        if ctx.needs_input_grad[4]:
            row_dot = (grad_x_new * x_new).sum(dim=1)
            grad_s_feat = (h @ grad_x_new.t() - row_dot.unsqueeze(0)) / c_eps.unsqueeze(0)
            grad_s_topo = at_s @ g_a_hat.t() + att_s @ g_a_hat

            null_term = (
                d_row.unsqueeze(1) * d_s_col.unsqueeze(0)
                + d_col.unsqueeze(1) * d_s_row.unsqueeze(0)
            ) / two_m
            grad_s_mod = -grad_l_mod * ((as_mat + ats_mat) - null_term) / two_m

            scale_col = grad_l_collapse * (math.sqrt(float(k)) / (float(n) * c_norm))
            grad_s_collapse = scale_col * c.unsqueeze(0).expand(n, k)

            grad_s = grad_s_feat + grad_s_topo + grad_s_mod + grad_s_collapse

        return None, grad_ew, grad_ew_tilde, grad_h, grad_s, None


class DMoNReducer(nn.Module):
    """Method 3: Deep Modularity Networks (DMoN) graph reduction operator."""

    def __init__(
        self,
        mod_weight: float = 1.0,
        collapse_weight: float = 1.0,
        eps: float = 1e-12,
    ) -> None:
        super().__init__()
        self.mod_weight = mod_weight
        self.collapse_weight = collapse_weight
        self.eps = eps

    def forward(
        self,
        adj: Optional[Tensor],
        h: Tensor,
        s: Tensor,
        edge_index: Optional[Tensor] = None,
        edge_weight: Optional[Tensor] = None,
    ) -> PoolingOutput:
        if adj is None:
            if edge_index is None or edge_weight is None:
                raise ValueError("Provide either `adj` or `(edge_index, edge_weight)`.")
            ew_tilde = sparse_symmetric_normalize_adjacency(edge_index, edge_weight, num_nodes=s.shape[0], eps=self.eps)
            x_new, adj_new, l_mod, l_collapse = _DMoNSparseFn.apply(edge_index, edge_weight, ew_tilde, h, s, self.eps)
        else:
            a_tilde = symmetric_normalize_adjacency(adj, eps=self.eps)
            x_new, adj_new, l_mod, l_collapse = _DMoNFn.apply(adj, a_tilde, h, s, self.eps)

        aux_loss = self.mod_weight * l_mod + self.collapse_weight * l_collapse
        return PoolingOutput(
            x_new=x_new,
            adj_new=adj_new,
            s=s,
            h=h,
            aux_loss=aux_loss,
            loss_components={"l_mod": l_mod, "l_collapse": l_collapse, "l_dmon": l_mod + l_collapse},
            diagnostics={
                "spectral_modularity_Q": float((-l_mod.detach()).item()),
                "l_collapse": float(l_collapse.detach().item()),
            },
        )

    def forward_reference(self, adj: Tensor, h: Tensor, s: Tensor) -> PoolingOutput:
        """Pure PyTorch autograd reference (explicitly forming B = A - d d^T / 2m for cross-validation)."""
        n, k = s.shape
        a_tilde = symmetric_normalize_adjacency(adj, eps=self.eps)

        c = s.sum(dim=0)
        x_new = (s.t() @ h) / (c + self.eps).unsqueeze(1)

        a_pool = s.t() @ a_tilde @ s
        off_diag = 1.0 - torch.eye(k, dtype=s.dtype, device=s.device)
        a_hat_new = a_pool * off_diag
        d_new = a_hat_new.sum(dim=1) + self.eps
        inv_sqrt_d_new = torch.rsqrt(d_new)
        adj_new = inv_sqrt_d_new.unsqueeze(1) * a_hat_new * inv_sqrt_d_new.unsqueeze(0)

        d_row = adj.sum(dim=1)
        d_col = adj.sum(dim=0)
        two_m = d_row.sum() + self.eps
        b_mat = adj - torch.outer(d_row, d_col) / two_m
        l_mod = -torch.diagonal(s.t() @ b_mat @ s).sum() / two_m

        c_norm = torch.sqrt((c * c).sum() + self.eps * self.eps)
        l_collapse = (math.sqrt(float(k)) / float(n)) * c_norm - 1.0
        aux_loss = self.mod_weight * l_mod + self.collapse_weight * l_collapse
        return PoolingOutput(
            x_new=x_new,
            adj_new=adj_new,
            s=s,
            h=h,
            aux_loss=aux_loss,
            loss_components={"l_mod": l_mod, "l_collapse": l_collapse, "l_dmon": l_mod + l_collapse},
        )
