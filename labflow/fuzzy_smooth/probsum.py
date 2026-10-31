"""Method 10: Probabilistic Sum FuzzyPool (ProbSum-T reduction) with Scale-Invariant Normalization.

Forward Pass (Log-Space Numerically Stable Accumulation):
    omega_{ij}^{(I, J)} = S_{iI} * A_{ij} * S_{jJ}
    L_{IJ} = sum_{(i, j) in E} ln((1 - omega_{ij}^{(I, J)} + eps) / (1 + eps))
    A_new(I, J) = 1 - exp(gamma_M * L_{IJ}) in [0, 1]

where scale exponent gamma_M is:
    - 'none':       gamma_M = 1.0  (classical ProbSum)
    - 'mean_field': gamma_M = min(1.0, K^2 / max(1, M))  (prevents A_new -> 1 clique saturation at large M)
    - 'sqrt_edges': gamma_M = 1.0 / sqrt(max(1.0, M / K^2))

Analytical Backward Pass:
    dA_new(I, J) / d(omega_{ij}^{(I, J)}) = gamma_M * (1 - A_new(I, J)) / (1 - omega_{ij}^{(I, J)} + eps)
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from labflow.types import dense_to_sparse


def _probsum_scale_exponent(num_edges: int, num_clusters: int, normalization: str) -> float:
    if normalization == "none" or num_edges <= 0:
        return 1.0
    m_f = float(max(1, num_edges))
    k_f = float(max(1, num_clusters))
    if normalization == "mean_field":
        return min(1.0, k_f / m_f)
    if normalization == "geometric":
        return 1.0 / m_f
    if normalization == "sqrt_edges":
        return 1.0 / math.sqrt(m_f)
    raise ValueError(f"Unsupported ProbSum normalization: {normalization}")


class _ProbSumFn(torch.autograd.Function):
    """Exact analytical forward and backward pass for Probabilistic Sum FuzzyPool."""

    @staticmethod
    def forward(
        ctx,
        s: Tensor,
        edge_index: Tensor,
        edge_weight: Tensor,
        eps: float,
        normalization: str = "none",
    ) -> Tensor:
        k = s.shape[1]
        m = int(edge_weight.numel())
        gamma_m = _probsum_scale_exponent(m, k, normalization)

        if m == 0:
            a_new = torch.zeros((k, k), dtype=s.dtype, device=s.device)
            omega = s.new_empty((0, k, k))
            ctx.save_for_backward(s, edge_index, edge_weight, omega, a_new)
            ctx.eps = eps
            ctx.gamma_m = gamma_m
            return a_new

        src, dst = edge_index[0], edge_index[1]
        omega = s[src].unsqueeze(2) * edge_weight[:, None, None] * s[dst].unsqueeze(1)  # [M, K, K]
        log_terms = torch.log((1.0 - omega + eps) / (1.0 + eps))
        l_ij = log_terms.sum(dim=0)
        a_new = 1.0 - torch.exp(gamma_m * l_ij)

        ctx.save_for_backward(s, edge_index, edge_weight, omega, a_new)
        ctx.eps = eps
        ctx.gamma_m = gamma_m
        return a_new

    @staticmethod
    def backward(
        ctx,
        grad_a_new: Tensor,
    ) -> Tuple[Optional[Tensor], None, Optional[Tensor], None, None]:
        s, edge_index, edge_weight, omega, a_new = ctx.saved_tensors
        eps = ctx.eps
        gamma_m = ctx.gamma_m
        n, k = s.shape
        m = int(edge_weight.numel())

        if m == 0:
            grad_s = torch.zeros_like(s) if ctx.needs_input_grad[0] else None
            grad_ew = torch.zeros_like(edge_weight) if ctx.needs_input_grad[2] else None
            return grad_s, None, grad_ew, None, None

        src, dst = edge_index[0], edge_index[1]
        # dA_new(I, J) / d(omega_e) = gamma_M * (1 - A_new(I, J)) / (1 - omega_e + eps)
        m_eij = (gamma_m * grad_a_new * (1.0 - a_new)).unsqueeze(0) / (1.0 - omega + eps)  # [M, K, K]

        grad_s = None
        if ctx.needs_input_grad[0]:
            grad_s = torch.zeros((n, k), dtype=s.dtype, device=s.device)
            c_left = edge_weight.unsqueeze(1) * (m_eij * s[dst].unsqueeze(1)).sum(dim=2)
            c_right = edge_weight.unsqueeze(1) * (m_eij * s[src].unsqueeze(2)).sum(dim=1)
            grad_s.scatter_add_(0, src.unsqueeze(1).expand(m, k), c_left)
            grad_s.scatter_add_(0, dst.unsqueeze(1).expand(m, k), c_right)

        grad_ew = None
        if ctx.needs_input_grad[2]:
            grad_ew = (m_eij * s[src].unsqueeze(2) * s[dst].unsqueeze(1)).sum(dim=(1, 2))

        return grad_s, None, grad_ew, None, None


class ProbSumTopologyReducer(nn.Module):
    """Method 10: Probabilistic Sum FuzzyPool with optional scale-invariant anti-saturation normalization."""

    def __init__(self, eps: float = 1e-12, normalization: str = "none") -> None:
        super().__init__()
        if normalization not in ("none", "mean_field", "geometric", "sqrt_edges"):
            raise ValueError("`normalization` must be 'none', 'mean_field', 'geometric', or 'sqrt_edges'.")
        self.eps = eps
        self.normalization = normalization

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
        return _ProbSumFn.apply(s, edge_index, edge_weight, self.eps, self.normalization)

    def forward_reference(
        self,
        s: Tensor,
        edge_index: Tensor,
        edge_weight: Tensor,
    ) -> Tensor:
        """Pure PyTorch autograd reference implementation."""
        k = s.shape[1]
        m = int(edge_weight.numel())
        if m == 0:
            return torch.zeros((k, k), dtype=s.dtype, device=s.device)
        gamma_m = _probsum_scale_exponent(m, k, self.normalization)
        src, dst = edge_index[0], edge_index[1]
        omega = s[src].unsqueeze(2) * edge_weight[:, None, None] * s[dst].unsqueeze(1)
        l_ij = torch.log((1.0 - omega + self.eps) / (1.0 + self.eps)).sum(dim=0)
        return 1.0 - torch.exp(gamma_m * l_ij)
