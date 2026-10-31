"""Method 1: DiffPool (Differentiable Bilinear Probabilistic Graph Reduction).

Forward Pass:
    X_new = S^T H
    A_new = S^T A S
Regularization:
    L_LP  = (1 / N^2) ||A - S S^T||_F^2 = (1 / N^2) (||A||_F^2 - 2 Tr(S^T A S) + ||S^T S||_F^2)
    L_ent = -(1 / N) sum_{i,I} S_{iI} ln(S_{iI} + eps)

Supports both dense O(N^2) adjacency and true sparse O(M K + N K^2) COO edges without N x N allocation.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from labflow.types import PoolingOutput


class _DiffPoolFn(torch.autograd.Function):
    """Exact analytical forward and backward pass for dense DiffPool."""

    @staticmethod
    def forward(
        ctx,
        adj: Tensor,
        h: Tensor,
        s: Tensor,
        eps: float,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        n = adj.shape[0]
        x_new = s.t() @ h
        adj_new = s.t() @ adj @ s

        recon_err = adj - (s @ s.t())
        l_lp = (recon_err * recon_err).sum() / float(n * n)
        l_ent = -(s * torch.log(s + eps)).sum() / float(n)

        ctx.save_for_backward(adj, h, s, recon_err)
        ctx.eps = eps
        return x_new, adj_new, l_lp, l_ent

    @staticmethod
    def backward(
        ctx,
        grad_x_new: Tensor,
        grad_adj_new: Tensor,
        grad_l_lp: Tensor,
        grad_l_ent: Tensor,
    ) -> Tuple[Optional[Tensor], Optional[Tensor], Optional[Tensor], None]:
        adj, h, s, recon_err = ctx.saved_tensors
        eps = ctx.eps
        n = float(adj.shape[0])

        grad_adj = None
        if ctx.needs_input_grad[0]:
            grad_adj = s @ grad_adj_new @ s.t() + grad_l_lp * (2.0 / (n * n)) * recon_err

        grad_h = None
        if ctx.needs_input_grad[1]:
            grad_h = s @ grad_x_new

        grad_s = None
        if ctx.needs_input_grad[2]:
            grad_s_main = (
                h @ grad_x_new.t()
                + (adj @ s) @ grad_adj_new.t()
                + (adj.t() @ s) @ grad_adj_new
            )
            grad_s_lp = -grad_l_lp * (2.0 / (n * n)) * ((recon_err + recon_err.t()) @ s)
            grad_s_ent = -grad_l_ent * (1.0 / n) * (torch.log(s + eps) + s / (s + eps))
            grad_s = grad_s_main + grad_s_lp + grad_s_ent

        return grad_adj, grad_h, grad_s, None


class _DiffPoolSparseFn(torch.autograd.Function):
    """Exact O(M K + N K^2) analytical forward/backward for sparse DiffPool via Frobenius trace identity."""

    @staticmethod
    def forward(
        ctx,
        edge_index: Tensor,
        edge_weight: Tensor,
        h: Tensor,
        s: Tensor,
        eps: float,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        n, k = s.shape
        m = int(edge_weight.numel())

        # Coalesce duplicate edges if any so ||A||_F^2 = sum_e w_e^2 is exact
        if m > 1:
            keys = edge_index[0] * n + edge_index[1]
            uniq_keys, inv = torch.unique(keys, sorted=True, return_inverse=True)
            if uniq_keys.numel() < m:
                src_u = torch.div(uniq_keys, n, rounding_mode="floor")
                dst_u = uniq_keys - src_u * n
                ew_u = torch.zeros((uniq_keys.numel(),), dtype=edge_weight.dtype, device=edge_weight.device)
                ew_u.index_add_(0, inv, edge_weight)
            else:
                src_u, dst_u, ew_u, inv = edge_index[0], edge_index[1], edge_weight, None
        elif m == 1:
            src_u, dst_u, ew_u, inv = edge_index[0], edge_index[1], edge_weight, None
        else:
            src_u = edge_index[0]
            dst_u = edge_index[1]
            ew_u = edge_weight
            inv = None

        x_new = s.t() @ h
        as_mat = torch.zeros((n, k), dtype=s.dtype, device=s.device)
        ats_mat = torch.zeros((n, k), dtype=s.dtype, device=s.device)
        if ew_u.numel() > 0:
            as_mat.index_add_(0, src_u, ew_u.unsqueeze(1) * s[dst_u])
            ats_mat.index_add_(0, dst_u, ew_u.unsqueeze(1) * s[src_u])

        adj_new = s.t() @ as_mat
        sts = s.t() @ s  # [K, K]

        # Exact identity: ||A - S S^T||_F^2 = ||A||_F^2 - 2 Tr(S^T A S) + ||S^T S||_F^2
        norm_a_sq = (ew_u * ew_u).sum()
        tr_sas = torch.diagonal(adj_new).sum()
        norm_sst_sq = (sts * sts).sum()
        l_lp = (norm_a_sq - 2.0 * tr_sas + norm_sst_sq).clamp(min=0.0) / float(n * n)
        l_ent = -(s * torch.log(s + eps)).sum() / float(n)

        if inv is not None:
            ctx.save_for_backward(src_u, dst_u, ew_u, h, s, as_mat, ats_mat, sts, inv)
            ctx.has_inv = True
        else:
            ctx.save_for_backward(src_u, dst_u, ew_u, h, s, as_mat, ats_mat, sts)
            ctx.has_inv = False
        ctx.eps = eps
        return x_new, adj_new, l_lp, l_ent

    @staticmethod
    def backward(
        ctx,
        grad_x_new: Tensor,
        grad_adj_new: Tensor,
        grad_l_lp: Tensor,
        grad_l_ent: Tensor,
    ) -> Tuple[None, Optional[Tensor], Optional[Tensor], Optional[Tensor], None]:
        saved = ctx.saved_tensors
        if ctx.has_inv:
            src_u, dst_u, ew_u, h, s, as_mat, ats_mat, sts, inv = saved
        else:
            src_u, dst_u, ew_u, h, s, as_mat, ats_mat, sts = saved
            inv = None
        eps = ctx.eps
        n = float(s.shape[0])

        grad_ew = None
        if ctx.needs_input_grad[1]:
            if ew_u.numel() > 0:
                s_src = s[src_u]
                s_dst = s[dst_u]
                g_topo = ((s_src @ grad_adj_new) * s_dst).sum(dim=1)
                g_lp = grad_l_lp * (2.0 / (n * n)) * (ew_u - (s_src * s_dst).sum(dim=1))
                grad_ew_u = g_topo + g_lp
                grad_ew = grad_ew_u[inv] if inv is not None else grad_ew_u
            else:
                grad_ew = torch.zeros_like(ew_u)

        grad_h = s @ grad_x_new if ctx.needs_input_grad[2] else None

        grad_s = None
        if ctx.needs_input_grad[3]:
            grad_s_main = (
                h @ grad_x_new.t()
                + as_mat @ grad_adj_new.t()
                + ats_mat @ grad_adj_new
            )
            grad_s_lp = -grad_l_lp * (2.0 / (n * n)) * ((as_mat + ats_mat) - 2.0 * (s @ sts))
            grad_s_ent = -grad_l_ent * (1.0 / n) * (torch.log(s + eps) + s / (s + eps))
            grad_s = grad_s_main + grad_s_lp + grad_s_ent

        return None, grad_ew, grad_h, grad_s, None


class DiffPoolReducer(nn.Module):
    """Method 1: DiffPool graph reduction operator with LP and entropy regularization."""

    def __init__(
        self,
        lp_weight: float = 1.0,
        ent_weight: float = 1.0,
        eps: float = 1e-12,
    ) -> None:
        super().__init__()
        self.lp_weight = lp_weight
        self.ent_weight = ent_weight
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
            x_new, adj_new, l_lp, l_ent = _DiffPoolSparseFn.apply(edge_index, edge_weight, h, s, self.eps)
            mass_in = edge_weight.detach().sum()
        else:
            x_new, adj_new, l_lp, l_ent = _DiffPoolFn.apply(adj, h, s, self.eps)
            mass_in = adj.detach().sum()

        aux_loss = self.lp_weight * l_lp + self.ent_weight * l_ent
        mass_out = adj_new.detach().sum()
        return PoolingOutput(
            x_new=x_new,
            adj_new=adj_new,
            s=s,
            h=h,
            aux_loss=aux_loss,
            loss_components={"l_lp": l_lp, "l_ent": l_ent},
            diagnostics={
                "edge_mass_in": float(mass_in.item()),
                "edge_mass_out": float(mass_out.item()),
                "edge_mass_abs_diff": float((mass_out - mass_in).abs().item()),
            },
        )

    def forward_reference(self, adj: Tensor, h: Tensor, s: Tensor) -> PoolingOutput:
        """Pure PyTorch autograd reference implementation."""
        n = float(adj.shape[0])
        x_new = s.t() @ h
        adj_new = s.t() @ adj @ s
        recon_err = adj - (s @ s.t())
        l_lp = (recon_err * recon_err).sum() / (n * n)
        l_ent = -(s * torch.log(s + self.eps)).sum() / n
        aux_loss = self.lp_weight * l_lp + self.ent_weight * l_ent
        return PoolingOutput(
            x_new=x_new,
            adj_new=adj_new,
            s=s,
            h=h,
            aux_loss=aux_loss,
            loss_components={"l_lp": l_lp, "l_ent": l_ent},
        )
