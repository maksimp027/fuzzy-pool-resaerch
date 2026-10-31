"""Method 9: Energetic Boltzmann FuzzyPool (Smooth conorm-tnorm reduction).

Forward Pass (Global Edge Smooth Max, Sections 9.1 & 9.3):
    omega_e^{(I, J)} = S_{iI} * A_{ij} * S_{jJ}   for e = (i, j) in E
    tau_eff          = tau_c / c_M,  where c_M = 1 + ln(max(1, M / K^2)) if scale_by_log_m else 1
    pi_e^{(I, J)}    = exp(omega_e^{(I, J)} / tau_eff) / sum_{e' in E} exp(omega_{e'}^{(I, J)} / tau_eff)
    A_new(I, J)      = sum_{e in E} pi_e^{(I, J)} * omega_e^{(I, J)}

Two-Step Decomposed Smooth Max (Section 9.2):
    Step 1: T_{iJ}      = sum_{j in N(i)} Softmax_{j in N(i)}(A_{ij} S_{jJ} / tau_eff) * (A_{ij} S_{jJ})
    Step 2: A_new(I, J) = sum_{i=1..N} Softmax_{i}(S_{iI} T_{iJ} / tau_eff) * (S_{iI} T_{iJ})

Analytical Backward Pass:
    dA_new(I, J) / d(omega_e^{(I, J)}) = pi_e^{(I, J)} * [1 + (omega_e^{(I, J)} - A_new(I, J)) / tau_eff]
    dA_new(I, J) / d(tau_c)            = -(1 / (c_M * tau_eff^2)) * Var_{pi^{(I, J)}}(omega^{(I, J)}) <= 0
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from labflow.memberships import inverse_softplus
from labflow.types import dense_to_sparse


def _boltzmann_log_scale_factor(num_edges: int, num_clusters: int, scale_by_log_m: bool) -> float:
    if not scale_by_log_m or num_edges <= 0:
        return 1.0
    ratio = max(1.0, float(num_edges) / float(max(1, num_clusters * num_clusters)))
    return 1.0 + math.log(ratio)


class _BoltzmannGlobalEdgeFn(torch.autograd.Function):
    """Exact analytical forward/backward for global-edge Boltzmann smooth-max FuzzyPool."""

    @staticmethod
    def forward(
        ctx,
        s: Tensor,
        edge_index: Tensor,
        edge_weight: Tensor,
        tau_c_tilde: Tensor,
        scale_by_log_m: bool = False,
    ) -> Tensor:
        tau_c = F.softplus(tau_c_tilde)
        k = s.shape[1]
        m = int(edge_weight.numel())
        c_m = _boltzmann_log_scale_factor(m, k, scale_by_log_m)
        tau_eff = tau_c / c_m

        if m == 0:
            a_new = torch.zeros((k, k), dtype=s.dtype, device=s.device)
            omega = s.new_empty((0, k, k))
            pi = s.new_empty((0, k, k))
            ctx.save_for_backward(s, edge_index, edge_weight, omega, pi, a_new, tau_eff, tau_c_tilde)
            ctx.c_m = c_m
            return a_new

        src, dst = edge_index[0], edge_index[1]
        omega = s[src].unsqueeze(2) * edge_weight[:, None, None] * s[dst].unsqueeze(1)  # [M, K, K]
        pi = F.softmax(omega / tau_eff, dim=0)  # [M, K, K]
        a_new = (pi * omega).sum(dim=0)  # [K, K]

        ctx.save_for_backward(s, edge_index, edge_weight, omega, pi, a_new, tau_eff, tau_c_tilde)
        ctx.c_m = c_m
        return a_new

    @staticmethod
    def backward(
        ctx,
        grad_a_new: Tensor,
    ) -> Tuple[Optional[Tensor], None, Optional[Tensor], Optional[Tensor], None]:
        s, edge_index, edge_weight, omega, pi, a_new, tau_eff, tau_c_tilde = ctx.saved_tensors
        c_m = ctx.c_m
        n, k = s.shape
        m = int(edge_weight.numel())

        if m == 0:
            grad_s = torch.zeros_like(s) if ctx.needs_input_grad[0] else None
            grad_ew = torch.zeros_like(edge_weight) if ctx.needs_input_grad[2] else None
            grad_tt = torch.zeros_like(tau_c_tilde) if ctx.needs_input_grad[3] else None
            return grad_s, None, grad_ew, grad_tt, None

        src, dst = edge_index[0], edge_index[1]
        diff = omega - a_new.unsqueeze(0)  # [M, K, K]
        d_a_domega = pi * (1.0 + diff / tau_eff)
        m_eij = grad_a_new.unsqueeze(0) * d_a_domega  # [M, K, K]

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

        grad_tau_c_tilde = None
        if ctx.needs_input_grad[3]:
            var_omega = (pi * (diff * diff)).sum(dim=0)
            d_a_dtau_eff = -var_omega / (tau_eff * tau_eff)
            grad_tau_c = (grad_a_new * d_a_dtau_eff).sum() / c_m
            grad_tau_c_tilde = (grad_tau_c * torch.sigmoid(tau_c_tilde)).reshape_as(tau_c_tilde)

        return grad_s, None, grad_ew, grad_tau_c_tilde, None


class _BoltzmannTwoStepFn(torch.autograd.Function):
    """Exact analytical O(M K + N K^2) 2-step decomposed Boltzmann smooth-max FuzzyPool (Section 9.2)."""

    @staticmethod
    def forward(
        ctx,
        s: Tensor,
        edge_index: Tensor,
        edge_weight: Tensor,
        tau_c_tilde: Tensor,
        scale_by_log_m: bool = False,
    ) -> Tensor:
        tau_c = F.softplus(tau_c_tilde)
        n, k = s.shape
        m = int(edge_weight.numel())
        c_m = _boltzmann_log_scale_factor(m, k, scale_by_log_m)
        tau_eff = tau_c / c_m

        if m == 0:
            a_new = torch.zeros((k, k), dtype=s.dtype, device=s.device)
            empty_mk = s.new_empty((0, k))
            empty_nk = torch.zeros((n, k), dtype=s.dtype, device=s.device)
            empty_nkk = torch.zeros((n, k, k), dtype=s.dtype, device=s.device)
            ctx.save_for_backward(
                s, edge_index, edge_weight, empty_mk, empty_mk, empty_nk, empty_nkk, empty_nkk, a_new, tau_eff, tau_c_tilde
            )
            ctx.c_m = c_m
            return a_new

        src, dst = edge_index[0], edge_index[1]
        src_idx = src.unsqueeze(1).expand(m, k)

        # Step 1: Segmented Boltzmann smooth max over j in N(i)
        u_ej = edge_weight.unsqueeze(1) * s[dst]  # [M, K]
        u_max = torch.full((n, k), -torch.inf, dtype=s.dtype, device=s.device)
        u_max = u_max.scatter_reduce(0, src_idx, u_ej, reduce="amax", include_self=True)
        u_max_safe = torch.where(torch.isfinite(u_max), u_max, torch.zeros_like(u_max))

        exp_1 = torch.exp((u_ej - u_max_safe[src]) / tau_eff)
        denom_1 = torch.zeros((n, k), dtype=s.dtype, device=s.device)
        denom_1.scatter_add_(0, src_idx, exp_1)
        pi_1 = exp_1 / denom_1[src].clamp(min=1e-30)

        t_ij = torch.zeros((n, k), dtype=s.dtype, device=s.device)
        t_ij.scatter_add_(0, src_idx, pi_1 * u_ej)

        # Step 2: Boltzmann smooth max over i in {1..N}
        v_nij = s.unsqueeze(2) * t_ij.unsqueeze(1)  # [N, K, K]
        pi_2 = F.softmax(v_nij / tau_eff, dim=0)  # [N, K, K]
        a_new = (pi_2 * v_nij).sum(dim=0)  # [K, K]

        ctx.save_for_backward(
            s, edge_index, edge_weight, u_ej, pi_1, t_ij, v_nij, pi_2, a_new, tau_eff, tau_c_tilde
        )
        ctx.c_m = c_m
        return a_new

    @staticmethod
    def backward(
        ctx,
        grad_a_new: Tensor,
    ) -> Tuple[Optional[Tensor], None, Optional[Tensor], Optional[Tensor], None]:
        (
            s,
            edge_index,
            edge_weight,
            u_ej,
            pi_1,
            t_ij,
            v_nij,
            pi_2,
            a_new,
            tau_eff,
            tau_c_tilde,
        ) = ctx.saved_tensors
        c_m = ctx.c_m
        n, k = s.shape
        m = int(edge_weight.numel())

        if m == 0:
            grad_s = torch.zeros_like(s) if ctx.needs_input_grad[0] else None
            grad_ew = torch.zeros_like(edge_weight) if ctx.needs_input_grad[2] else None
            grad_tt = torch.zeros_like(tau_c_tilde) if ctx.needs_input_grad[3] else None
            return grad_s, None, grad_ew, grad_tt, None

        src, dst = edge_index[0], edge_index[1]

        # Step 2 backward
        diff_2 = v_nij - a_new.unsqueeze(0)
        dv_nij = grad_a_new.unsqueeze(0) * pi_2 * (1.0 + diff_2 / tau_eff)
        grad_s_step2 = (dv_nij * t_ij.unsqueeze(1)).sum(dim=2)
        grad_t_ij = (dv_nij * s.unsqueeze(2)).sum(dim=1)

        # Step 1 backward
        diff_1 = u_ej - t_ij[src]
        du_ej = grad_t_ij[src] * pi_1 * (1.0 + diff_1 / tau_eff)

        grad_s = None
        if ctx.needs_input_grad[0]:
            grad_s = grad_s_step2.clone()
            grad_s.scatter_add_(0, dst.unsqueeze(1).expand(m, k), du_ej * edge_weight.unsqueeze(1))

        grad_ew = None
        if ctx.needs_input_grad[2]:
            grad_ew = (du_ej * s[dst]).sum(dim=1)

        grad_tau_c_tilde = None
        if ctx.needs_input_grad[3]:
            dtau_2 = -(grad_a_new * (pi_2 * (diff_2 * diff_2)).sum(dim=0)).sum() / (tau_eff * tau_eff)
            dtau_1 = -(grad_t_ij[src] * pi_1 * (diff_1 * diff_1)).sum() / (tau_eff * tau_eff)
            grad_tau_c = (dtau_1 + dtau_2) / c_m
            grad_tau_c_tilde = (grad_tau_c * torch.sigmoid(tau_c_tilde)).reshape_as(tau_c_tilde)

        return grad_s, None, grad_ew, grad_tau_c_tilde, None


class BoltzmannTopologyReducer(nn.Module):
    """Method 9: Energetic Boltzmann Smooth FuzzyPool (supports 'global_edge', 'two_step', and log-M scaling)."""

    def __init__(
        self,
        init_tau_c: float = 0.25,
        learnable_tau_c: bool = True,
        mode: str = "global_edge",
        scale_by_log_m: bool = False,
    ) -> None:
        super().__init__()
        if mode not in ("global_edge", "two_step"):
            raise ValueError("`mode` must be 'global_edge' or 'two_step'.")
        self.mode = mode
        self.scale_by_log_m = bool(scale_by_log_m)
        raw = torch.tensor(inverse_softplus(init_tau_c), dtype=torch.float32)
        if learnable_tau_c:
            self.tau_c_tilde = nn.Parameter(raw)
        else:
            self.register_buffer("tau_c_tilde", raw)

    @property
    def tau_c(self) -> Tensor:
        return F.softplus(self.tau_c_tilde)

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
        tt = self.tau_c_tilde.to(dtype=s.dtype)
        if self.mode == "global_edge":
            return _BoltzmannGlobalEdgeFn.apply(s, edge_index, edge_weight, tt, self.scale_by_log_m)
        return _BoltzmannTwoStepFn.apply(s, edge_index, edge_weight, tt, self.scale_by_log_m)

    def forward_reference(
        self,
        s: Tensor,
        edge_index: Tensor,
        edge_weight: Tensor,
    ) -> Tensor:
        """Pure PyTorch autograd reference implementation."""
        n, k = s.shape
        m = int(edge_weight.numel())
        if m == 0:
            return torch.zeros((k, k), dtype=s.dtype, device=s.device)
        c_m = _boltzmann_log_scale_factor(m, k, self.scale_by_log_m)
        tau_eff = F.softplus(self.tau_c_tilde.to(dtype=s.dtype)) / c_m
        src, dst = edge_index[0], edge_index[1]

        if self.mode == "global_edge":
            omega = s[src].unsqueeze(2) * edge_weight[:, None, None] * s[dst].unsqueeze(1)
            pi = F.softmax(omega / tau_eff, dim=0)
            return (pi * omega).sum(dim=0)

        t_rows = []
        for i in range(n):
            mask = src == i
            if not mask.any():
                t_rows.append(torch.zeros((k,), dtype=s.dtype, device=s.device))
            else:
                u_i = edge_weight[mask].unsqueeze(1) * s[dst[mask]]
                pi_i = F.softmax(u_i / tau_eff, dim=0)
                t_rows.append((pi_i * u_i).sum(dim=0))
        t_ij = torch.stack(t_rows, dim=0)
        v_nij = s.unsqueeze(2) * t_ij.unsqueeze(1)
        pi_2 = F.softmax(v_nij / tau_eff, dim=0)
        return (pi_2 * v_nij).sum(dim=0)
