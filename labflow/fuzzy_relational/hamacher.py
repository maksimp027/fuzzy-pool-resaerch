"""Method 7: Rational Hamacher-Einstein FuzzyPool (max-T_{H, gamma} composition).

Forward Pass:
    T_{H, gamma}(a, b) = (a * b) / (gamma + (1 - gamma) * (a + b - a * b)),  gamma > 0
    Special case gamma = 2: Relativistic Einstein t-norm.

Analytical Backward Pass:
    D_{H, gamma}(a, b) = d T_{H, gamma}(a, b) / da = b * (gamma + (1 - gamma) * b) / denom^2
    G_{H, gamma}(a, b) = d T_{H, gamma}(a, b) / dgamma = -a * b * (1 - a) * (1 - b) / denom^2
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from labflow.fuzzy_relational.sparse_utils import (
    accumulate_winner_gradients,
    sparse_two_step_max_with_winners,
)
from labflow.memberships import inverse_softplus
from labflow.types import dense_to_sparse


def hamacher_tnorm(a: Tensor, b: Tensor, gamma: Tensor | float, eps: float = 1e-12) -> Tensor:
    """Binary Hamacher t-norm T_{H, gamma}(a, b)."""
    denom = gamma + (1.0 - gamma) * (a + b - a * b)
    return (a * b) / denom.clamp(min=eps)


def hamacher_da(a: Tensor, b: Tensor, gamma: Tensor | float, eps: float = 1e-12) -> Tensor:
    """Analytical partial derivative D_{H, gamma}(a, b) = d T_{H, gamma}(a, b) / da."""
    denom = (gamma + (1.0 - gamma) * (a + b - a * b)).clamp(min=eps)
    return (b * (gamma + (1.0 - gamma) * b)) / (denom * denom)


def hamacher_dgamma(a: Tensor, b: Tensor, gamma: Tensor | float, eps: float = 1e-12) -> Tensor:
    """Analytical partial derivative G_{H, gamma}(a, b) = d T_{H, gamma}(a, b) / dgamma."""
    denom = (gamma + (1.0 - gamma) * (a + b - a * b)).clamp(min=eps)
    return -(a * b * (1.0 - a) * (1.0 - b)) / (denom * denom)


def hamacher_ternary_closed_form(
    a: Tensor,
    b: Tensor,
    c: Tensor,
    gamma: Tensor | float,
    eps: float = 1e-12,
) -> Tensor:
    """Exact symmetric closed-form ternary Hamacher t-norm T_{H, gamma}(a, T_{H, gamma}(b, c))."""
    abc = a * b * c
    denom = (
        gamma * gamma
        + gamma * (1.0 - gamma) * (a + b + c)
        + (1.0 - gamma) * (1.0 - gamma) * (a * b + a * c + b * c)
        - (1.0 - gamma) * (2.0 - gamma) * abc
    )
    return abc / denom.clamp(min=eps)


class _HamacherScalarSparseFn(torch.autograd.Function):
    """Exact O(M K + N K^2) 2-step sparse forward & analytical backward for scalar gamma."""

    @staticmethod
    def forward(
        ctx,
        s: Tensor,
        edge_index: Tensor,
        edge_weight: Tensor,
        gamma_tilde: Tensor,
        eps: float,
    ) -> Tensor:
        gamma = F.softplus(gamma_tilde)
        m = int(edge_weight.numel())
        dst = edge_index[1]
        u_ej = (
            hamacher_tnorm(edge_weight.unsqueeze(1), s[dst], gamma, eps=eps)
            if m > 0
            else s.new_empty((0, s.shape[1]))
        )
        a_new, t_star, e_star = sparse_two_step_max_with_winners(
            s=s,
            edge_index=edge_index,
            u_ej=u_ej,
            step2_combine=lambda s_ni1, t_n1j: hamacher_tnorm(s_ni1, t_n1j, gamma, eps=eps),
        )
        ctx.save_for_backward(s, edge_index, edge_weight, t_star, e_star, gamma, gamma_tilde)
        ctx.eps = eps
        return a_new

    @staticmethod
    def backward(
        ctx,
        grad_a_new: Tensor,
    ) -> Tuple[Optional[Tensor], None, Optional[Tensor], Optional[Tensor], None]:
        s, edge_index, edge_weight, t_star, e_star, gamma, gamma_tilde = ctx.saved_tensors
        eps = ctx.eps
        n, k = s.shape
        m = int(edge_weight.numel())

        if m == 0:
            grad_s = torch.zeros_like(s) if ctx.needs_input_grad[0] else None
            grad_ew = torch.zeros_like(edge_weight) if ctx.needs_input_grad[2] else None
            grad_gt = torch.zeros_like(gamma_tilde) if ctx.needs_input_grad[3] else None
            return grad_s, None, grad_ew, grad_gt, None

        src, dst = edge_index[0], edge_index[1]
        i_star = src[e_star]
        j_star = dst[e_star]

        i_idx = torch.arange(k, device=s.device).unsqueeze(1).expand(k, k)
        j_idx = torch.arange(k, device=s.device).unsqueeze(0).expand(k, k)

        s_left = s[i_star, i_idx]
        s_right = s[j_star, j_idx]
        a_val = edge_weight[e_star]

        # Chain rule via D_{H, gamma} and G_{H, gamma}
        d_a_s_left = hamacher_da(s_left, t_star, gamma, eps=eps)
        d_a_t_star = hamacher_da(t_star, s_left, gamma, eps=eps)
        d_t_s_right = hamacher_da(s_right, a_val, gamma, eps=eps)

        contrib_left = grad_a_new * d_a_s_left
        contrib_right = grad_a_new * (d_a_t_star * d_t_s_right)
        contrib_edge = (
            grad_a_new * (d_a_t_star * hamacher_da(a_val, s_right, gamma, eps=eps))
            if ctx.needs_input_grad[2]
            else None
        )

        grad_s_calc, grad_ew = accumulate_winner_gradients(
            num_nodes=n,
            num_clusters=k,
            num_edges=m,
            edge_index=edge_index,
            e_star=e_star,
            contrib_left=contrib_left,
            contrib_right=contrib_right,
            contrib_edge=contrib_edge,
        )
        grad_s = grad_s_calc if ctx.needs_input_grad[0] else None

        grad_gamma_tilde = None
        if ctx.needs_input_grad[3]:
            d_a_dgamma = hamacher_dgamma(s_left, t_star, gamma, eps=eps) + d_a_t_star * hamacher_dgamma(
                a_val, s_right, gamma, eps=eps
            )
            grad_gamma = (grad_a_new * d_a_dgamma).sum()
            grad_gamma_tilde = (grad_gamma * torch.sigmoid(gamma_tilde)).reshape_as(gamma_tilde)

        return grad_s, None, grad_ew, grad_gamma_tilde, None


class _HamacherMatrixFn(torch.autograd.Function):
    """Exact analytical forward/backward when Gamma in R_{>0}^{K x K} is a symmetric cluster-pair matrix."""

    @staticmethod
    def forward(
        ctx,
        s: Tensor,
        edge_index: Tensor,
        edge_weight: Tensor,
        gamma_tilde_mat: Tensor,
        eps: float,
    ) -> Tensor:
        k = s.shape[1]
        m = int(edge_weight.numel())
        sym_raw = 0.5 * (gamma_tilde_mat + gamma_tilde_mat.t())
        gamma_mat = F.softplus(sym_raw)  # [K, K]

        if m == 0:
            a_new = torch.zeros((k, k), dtype=s.dtype, device=s.device)
            e_star = torch.full((k, k), -1, dtype=torch.long, device=s.device)
            t_star = torch.zeros((k, k), dtype=s.dtype, device=s.device)
            ctx.save_for_backward(s, edge_index, edge_weight, t_star, e_star, gamma_mat, sym_raw)
            ctx.eps = eps
            return a_new

        src, dst = edge_index[0], edge_index[1]
        s_src = s[src].unsqueeze(2)  # [M, K, 1]
        s_dst = s[dst].unsqueeze(1)  # [M, 1, K]
        a_e = edge_weight[:, None, None]  # [M, 1, 1]
        g_kk = gamma_mat.unsqueeze(0)  # [1, K, K]

        t_mkk = hamacher_tnorm(a_e, s_dst, g_kk, eps=eps)
        val_mkk = hamacher_tnorm(s_src, t_mkk, g_kk, eps=eps)
        a_new, e_star = val_mkk.max(dim=0)
        t_star = t_mkk.gather(0, e_star.unsqueeze(0)).squeeze(0)

        ctx.save_for_backward(s, edge_index, edge_weight, t_star, e_star, gamma_mat, sym_raw)
        ctx.eps = eps
        return a_new

    @staticmethod
    def backward(
        ctx,
        grad_a_new: Tensor,
    ) -> Tuple[Optional[Tensor], None, Optional[Tensor], Optional[Tensor], None]:
        s, edge_index, edge_weight, t_star, e_star, gamma_mat, sym_raw = ctx.saved_tensors
        eps = ctx.eps
        n, k = s.shape
        m = int(edge_weight.numel())

        if m == 0:
            grad_s = torch.zeros_like(s) if ctx.needs_input_grad[0] else None
            grad_ew = torch.zeros_like(edge_weight) if ctx.needs_input_grad[2] else None
            grad_gt = torch.zeros_like(sym_raw) if ctx.needs_input_grad[3] else None
            return grad_s, None, grad_ew, grad_gt, None

        src, dst = edge_index[0], edge_index[1]
        i_star = src[e_star]
        j_star = dst[e_star]

        i_idx = torch.arange(k, device=s.device).unsqueeze(1).expand(k, k)
        j_idx = torch.arange(k, device=s.device).unsqueeze(0).expand(k, k)

        s_left = s[i_star, i_idx]
        s_right = s[j_star, j_idx]
        a_val = edge_weight[e_star]

        d_a_s_left = hamacher_da(s_left, t_star, gamma_mat, eps=eps)
        d_a_t_star = hamacher_da(t_star, s_left, gamma_mat, eps=eps)
        d_t_s_right = hamacher_da(s_right, a_val, gamma_mat, eps=eps)

        contrib_left = grad_a_new * d_a_s_left
        contrib_right = grad_a_new * (d_a_t_star * d_t_s_right)
        contrib_edge = (
            grad_a_new * (d_a_t_star * hamacher_da(a_val, s_right, gamma_mat, eps=eps))
            if ctx.needs_input_grad[2]
            else None
        )

        grad_s_calc, grad_ew = accumulate_winner_gradients(
            num_nodes=n,
            num_clusters=k,
            num_edges=m,
            edge_index=edge_index,
            e_star=e_star,
            contrib_left=contrib_left,
            contrib_right=contrib_right,
            contrib_edge=contrib_edge,
        )
        grad_s = grad_s_calc if ctx.needs_input_grad[0] else None

        grad_gamma_tilde = None
        if ctx.needs_input_grad[3]:
            d_a_dgamma = hamacher_dgamma(s_left, t_star, gamma_mat, eps=eps) + d_a_t_star * hamacher_dgamma(
                a_val, s_right, gamma_mat, eps=eps
            )
            grad_sym = grad_a_new * d_a_dgamma * torch.sigmoid(sym_raw)
            grad_gamma_tilde = 0.5 * (grad_sym + grad_sym.t())

        return grad_s, None, grad_ew, grad_gamma_tilde, None


class HamacherTopologyReducer(nn.Module):
    """Method 7: Rational Hamacher-Einstein FuzzyPool (supports scalar gamma, Einstein gamma=2, or matrix Gamma)."""

    def __init__(
        self,
        init_gamma: float = 2.0,
        learnable_gamma: bool = True,
        matrix_gamma: bool = False,
        num_clusters: Optional[int] = None,
        eps: float = 1e-12,
    ) -> None:
        super().__init__()
        self.matrix_gamma = matrix_gamma
        self.eps = eps
        raw_val = inverse_softplus(init_gamma)

        if matrix_gamma:
            if num_clusters is None:
                raise ValueError("`num_clusters` is required when `matrix_gamma=True`.")
            raw_tensor = torch.full((num_clusters, num_clusters), raw_val, dtype=torch.float32)
        else:
            raw_tensor = torch.tensor(raw_val, dtype=torch.float32)

        if learnable_gamma:
            self.gamma_tilde = nn.Parameter(raw_tensor)
        else:
            self.register_buffer("gamma_tilde", raw_tensor)

    @property
    def gamma(self) -> Tensor:
        if self.matrix_gamma:
            return F.softplus(0.5 * (self.gamma_tilde + self.gamma_tilde.t()))
        return F.softplus(self.gamma_tilde)

    def forward(
        self,
        s: Tensor,
        adj: Optional[Tensor] = None,
        edge_index: Optional[Tensor] = None,
        edge_weight: Optional[Tensor] = None,
    ) -> Tensor:
        if edge_index is None or edge_weight is None:
            if adj is None:
                raise ValueError("Provide either `adj` or `(edge_index, edge_weight)`.")
            edge_index, edge_weight = dense_to_sparse(adj, threshold=0.0)
        gt = self.gamma_tilde.to(dtype=s.dtype)
        if self.matrix_gamma:
            return _HamacherMatrixFn.apply(s, edge_index, edge_weight, gt, self.eps)
        return _HamacherScalarSparseFn.apply(s, edge_index, edge_weight, gt, self.eps)

    def forward_closed_form_reference(
        self,
        s: Tensor,
        edge_index: Tensor,
        edge_weight: Tensor,
    ) -> Tensor:
        """Closed-form ternary formula reference from Section 7.1 over edges E."""
        k = s.shape[1]
        if edge_weight.numel() == 0:
            return torch.zeros((k, k), dtype=s.dtype, device=s.device)
        src, dst = edge_index[0], edge_index[1]
        g = self.gamma.to(dtype=s.dtype)
        if self.matrix_gamma:
            g = g.unsqueeze(0)
        tri = hamacher_ternary_closed_form(
            s[src].unsqueeze(2),
            edge_weight[:, None, None],
            s[dst].unsqueeze(1),
            g,
            eps=self.eps,
        )
        return tri.amax(dim=0)
