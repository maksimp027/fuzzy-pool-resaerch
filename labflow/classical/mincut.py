"""Method 2: MinCutPool (Spectral Minimum Cut Graph Reduction).

Forward Pass:
    A_tilde   = D^{-1/2} A D^{-1/2}
    A_pool    = S^T A_tilde S
    A_hat_new = A_pool - diag(A_pool)
    A_new     = D_new^{-1/2} A_hat_new D_new^{-1/2},  (D_new)_{II} = sum_J A_hat_new(I, J) + eps
    X_new     = S^T H
Spectral Loss:
    L_cut  = -Tr(S^T A_tilde S) / Tr(S^T D_tilde S)
    L_orth = || S^T S / ||S^T S||_F - I_K / sqrt(K) ||_F

Supports both dense O(N^2) adjacency and true sparse O(M K + N K^2) COO edges without N x N allocation.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from labflow.gnn import sparse_symmetric_normalize_adjacency, symmetric_normalize_adjacency
from labflow.types import PoolingOutput


def mincut_topology_backward(
    grad_adj_new: Tensor,
    adj_new: Tensor,
    d_new: Tensor,
    inv_sqrt_d_new: Tensor,
    off_diag: Tensor,
) -> Tensor:
    """Compute exact dL / d(A_pool) through spectral normalization and zero-diagonal projection."""
    direct = inv_sqrt_d_new.unsqueeze(1) * grad_adj_new * inv_sqrt_d_new.unsqueeze(0)
    row_term = -0.5 * (grad_adj_new * adj_new).sum(dim=1) / d_new
    col_term = -0.5 * (adj_new * grad_adj_new).sum(dim=0) / d_new
    grad_d_new = row_term + col_term
    return off_diag * (direct + grad_d_new.unsqueeze(1))


class _MinCutPoolFn(torch.autograd.Function):
    """Exact analytical forward and backward pass for dense MinCutPool given A_tilde, H, S."""

    @staticmethod
    def forward(
        ctx,
        a_tilde: Tensor,
        h: Tensor,
        s: Tensor,
        eps: float,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        k = s.shape[1]
        x_new = s.t() @ h
        a_pool = s.t() @ a_tilde @ s

        off_diag = 1.0 - torch.eye(k, dtype=s.dtype, device=s.device)
        a_hat_new = a_pool * off_diag
        d_new = a_hat_new.sum(dim=1) + eps
        inv_sqrt_d_new = torch.rsqrt(d_new)
        adj_new = inv_sqrt_d_new.unsqueeze(1) * a_hat_new * inv_sqrt_d_new.unsqueeze(0)

        d_tilde_vec = a_tilde.sum(dim=1)
        num = torch.diagonal(a_pool).sum()
        den = ((s * s) * d_tilde_vec.unsqueeze(1)).sum() + eps
        l_cut = -num / den

        q = s.t() @ s
        q_norm = torch.linalg.norm(q, ord="fro").clamp(min=eps)
        q_hat = q / q_norm
        m_mat = q_hat - torch.eye(k, dtype=s.dtype, device=s.device) / math.sqrt(float(k))
        l_orth = torch.linalg.norm(m_mat, ord="fro")

        ctx.save_for_backward(
            a_tilde,
            h,
            s,
            adj_new,
            d_new,
            inv_sqrt_d_new,
            off_diag,
            d_tilde_vec,
            num,
            den,
            q_norm,
            q_hat,
            m_mat,
            l_orth,
        )
        ctx.eps = eps
        return x_new, adj_new, l_cut, l_orth

    @staticmethod
    def backward(
        ctx,
        grad_x_new: Tensor,
        grad_adj_new: Tensor,
        grad_l_cut: Tensor,
        grad_l_orth: Tensor,
    ) -> Tuple[Optional[Tensor], Optional[Tensor], Optional[Tensor], None]:
        (
            a_tilde,
            h,
            s,
            adj_new,
            d_new,
            inv_sqrt_d_new,
            off_diag,
            d_tilde_vec,
            num,
            den,
            q_norm,
            q_hat,
            m_mat,
            l_orth,
        ) = ctx.saved_tensors
        eps = ctx.eps

        g_a_hat = mincut_topology_backward(grad_adj_new, adj_new, d_new, inv_sqrt_d_new, off_diag)

        grad_a_tilde = None
        if ctx.needs_input_grad[0]:
            grad_a_pool = s @ g_a_hat @ s.t()
            s_sq_row = (s * s).sum(dim=1, keepdim=True).expand_as(a_tilde)
            grad_a_cut = -grad_l_cut * ((s @ s.t()) * den - s_sq_row * num) / (den * den)
            grad_a_tilde = grad_a_pool + grad_a_cut

        grad_h = s @ grad_x_new if ctx.needs_input_grad[1] else None

        grad_s = None
        if ctx.needs_input_grad[2]:
            grad_s_feat = h @ grad_x_new.t()
            grad_s_topo = (a_tilde @ s) @ g_a_hat.t() + (a_tilde.t() @ s) @ g_a_hat

            grad_num_s = (a_tilde + a_tilde.t()) @ s
            grad_den_s = 2.0 * d_tilde_vec.unsqueeze(1) * s
            grad_s_cut = -grad_l_cut * (grad_num_s * den - grad_den_s * num) / (den * den)

            if float(l_orth.detach().item()) > eps:
                tr_mq = (m_mat * q_hat).sum()
                grad_s_orth = (
                    grad_l_orth
                    * (2.0 / (q_norm * l_orth))
                    * (s @ (m_mat - tr_mq * q_hat))
                )
            else:
                grad_s_orth = torch.zeros_like(s)

            grad_s = grad_s_feat + grad_s_topo + grad_s_cut + grad_s_orth

        return grad_a_tilde, grad_h, grad_s, None


class _MinCutPoolSparseFn(torch.autograd.Function):
    """Exact O(M K + N K^2) analytical forward/backward for sparse MinCutPool without N x N matrices."""

    @staticmethod
    def forward(
        ctx,
        edge_index: Tensor,
        ew_tilde: Tensor,
        h: Tensor,
        s: Tensor,
        eps: float,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        n, k = s.shape
        x_new = s.t() @ h

        at_s = torch.zeros((n, k), dtype=s.dtype, device=s.device)
        att_s = torch.zeros((n, k), dtype=s.dtype, device=s.device)
        d_tilde_vec = torch.zeros((n,), dtype=s.dtype, device=s.device)
        if ew_tilde.numel() > 0:
            src, dst = edge_index[0], edge_index[1]
            at_s.index_add_(0, src, ew_tilde.unsqueeze(1) * s[dst])
            att_s.index_add_(0, dst, ew_tilde.unsqueeze(1) * s[src])
            d_tilde_vec.index_add_(0, src, ew_tilde)

        a_pool = s.t() @ at_s
        off_diag = 1.0 - torch.eye(k, dtype=s.dtype, device=s.device)
        a_hat_new = a_pool * off_diag
        d_new = a_hat_new.sum(dim=1) + eps
        inv_sqrt_d_new = torch.rsqrt(d_new)
        adj_new = inv_sqrt_d_new.unsqueeze(1) * a_hat_new * inv_sqrt_d_new.unsqueeze(0)

        num = torch.diagonal(a_pool).sum()
        den = ((s * s) * d_tilde_vec.unsqueeze(1)).sum() + eps
        l_cut = -num / den

        q = s.t() @ s
        q_norm = torch.linalg.norm(q, ord="fro").clamp(min=eps)
        q_hat = q / q_norm
        m_mat = q_hat - torch.eye(k, dtype=s.dtype, device=s.device) / math.sqrt(float(k))
        l_orth = torch.linalg.norm(m_mat, ord="fro")

        ctx.save_for_backward(
            edge_index,
            ew_tilde,
            h,
            s,
            at_s,
            att_s,
            adj_new,
            d_new,
            inv_sqrt_d_new,
            off_diag,
            d_tilde_vec,
            num,
            den,
            q_norm,
            q_hat,
            m_mat,
            l_orth,
        )
        ctx.eps = eps
        return x_new, adj_new, l_cut, l_orth

    @staticmethod
    def backward(
        ctx,
        grad_x_new: Tensor,
        grad_adj_new: Tensor,
        grad_l_cut: Tensor,
        grad_l_orth: Tensor,
    ) -> Tuple[None, Optional[Tensor], Optional[Tensor], Optional[Tensor], None]:
        (
            edge_index,
            ew_tilde,
            h,
            s,
            at_s,
            att_s,
            adj_new,
            d_new,
            inv_sqrt_d_new,
            off_diag,
            d_tilde_vec,
            num,
            den,
            q_norm,
            q_hat,
            m_mat,
            l_orth,
        ) = ctx.saved_tensors
        eps = ctx.eps

        g_a_hat = mincut_topology_backward(grad_adj_new, adj_new, d_new, inv_sqrt_d_new, off_diag)

        grad_ew_tilde = None
        if ctx.needs_input_grad[1]:
            if ew_tilde.numel() > 0:
                src, dst = edge_index[0], edge_index[1]
                s_src = s[src]
                s_dst = s[dst]
                g_pool = ((s_src @ g_a_hat) * s_dst).sum(dim=1)
                g_cut = -grad_l_cut * (((s_src * s_dst).sum(dim=1)) * den - ((s_src * s_src).sum(dim=1)) * num) / (den * den)
                grad_ew_tilde = g_pool + g_cut
            else:
                grad_ew_tilde = torch.zeros_like(ew_tilde)

        grad_h = s @ grad_x_new if ctx.needs_input_grad[2] else None

        grad_s = None
        if ctx.needs_input_grad[3]:
            grad_s_feat = h @ grad_x_new.t()
            grad_s_topo = at_s @ g_a_hat.t() + att_s @ g_a_hat

            grad_num_s = at_s + att_s
            grad_den_s = 2.0 * d_tilde_vec.unsqueeze(1) * s
            grad_s_cut = -grad_l_cut * (grad_num_s * den - grad_den_s * num) / (den * den)

            if float(l_orth.detach().item()) > eps:
                tr_mq = (m_mat * q_hat).sum()
                grad_s_orth = (
                    grad_l_orth
                    * (2.0 / (q_norm * l_orth))
                    * (s @ (m_mat - tr_mq * q_hat))
                )
            else:
                grad_s_orth = torch.zeros_like(s)

            grad_s = grad_s_feat + grad_s_topo + grad_s_cut + grad_s_orth

        return None, grad_ew_tilde, grad_h, grad_s, None


class MinCutPoolReducer(nn.Module):
    """Method 2: MinCutPool spectral minimum cut graph reduction operator."""

    def __init__(
        self,
        cut_weight: float = 1.0,
        orth_weight: float = 1.0,
        prenormalized: bool = False,
        eps: float = 1e-12,
    ) -> None:
        super().__init__()
        self.cut_weight = cut_weight
        self.orth_weight = orth_weight
        self.prenormalized = prenormalized
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
            ew_tilde = (
                edge_weight
                if self.prenormalized
                else sparse_symmetric_normalize_adjacency(edge_index, edge_weight, num_nodes=s.shape[0], eps=self.eps)
            )
            x_new, adj_new, l_cut, l_orth = _MinCutPoolSparseFn.apply(edge_index, ew_tilde, h, s, self.eps)
        else:
            a_tilde = adj if self.prenormalized else symmetric_normalize_adjacency(adj, eps=self.eps)
            x_new, adj_new, l_cut, l_orth = _MinCutPoolFn.apply(a_tilde, h, s, self.eps)

        aux_loss = self.cut_weight * l_cut + self.orth_weight * l_orth
        return PoolingOutput(
            x_new=x_new,
            adj_new=adj_new,
            s=s,
            h=h,
            aux_loss=aux_loss,
            loss_components={"l_cut": l_cut, "l_orth": l_orth},
            diagnostics={
                "diag_max_abs": float(torch.diagonal(adj_new.detach()).abs().max().item()),
                "l_cut": float(l_cut.detach().item()),
                "l_orth": float(l_orth.detach().item()),
            },
        )

    def forward_reference(self, adj: Tensor, h: Tensor, s: Tensor) -> PoolingOutput:
        """Pure PyTorch autograd reference implementation."""
        a_tilde = adj if self.prenormalized else symmetric_normalize_adjacency(adj, eps=self.eps)
        k = s.shape[1]
        x_new = s.t() @ h
        a_pool = s.t() @ a_tilde @ s
        off_diag = 1.0 - torch.eye(k, dtype=s.dtype, device=s.device)
        a_hat_new = a_pool * off_diag
        d_new = a_hat_new.sum(dim=1) + self.eps
        inv_sqrt_d_new = torch.rsqrt(d_new)
        adj_new = inv_sqrt_d_new.unsqueeze(1) * a_hat_new * inv_sqrt_d_new.unsqueeze(0)

        d_tilde_vec = a_tilde.sum(dim=1)
        num = torch.diagonal(a_pool).sum()
        den = ((s * s) * d_tilde_vec.unsqueeze(1)).sum() + self.eps
        l_cut = -num / den

        q = s.t() @ s
        q_norm = torch.linalg.norm(q, ord="fro").clamp(min=self.eps)
        q_hat = q / q_norm
        m_mat = q_hat - torch.eye(k, dtype=s.dtype, device=s.device) / math.sqrt(float(k))
        l_orth = torch.linalg.norm(m_mat, ord="fro")

        aux_loss = self.cut_weight * l_cut + self.orth_weight * l_orth
        return PoolingOutput(
            x_new=x_new,
            adj_new=adj_new,
            s=s,
            h=h,
            aux_loss=aux_loss,
            loss_components={"l_cut": l_cut, "l_orth": l_orth},
        )
