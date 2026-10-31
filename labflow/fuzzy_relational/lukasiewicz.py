"""Method 6: Boundary Sparse Lukasiewicz FuzzyPool (max-T_L composition).

Forward Pass:
    Hard clamped mode (`clamp_unit=True`, `smooth_tau=0.0`):
        A_new(I, J) = min(1, max_{(i, j) in E} max(0, S_{iI} + A_{ij} + S_{jJ} - eta_{IJ})) in [0, 1]
    Unclamped legacy mode (`clamp_unit=False`, `smooth_tau=0.0`):
        A_new(I, J) = max_{(i, j) in E} max(0, S_{iI} + A_{ij} + S_{jJ} - eta_{IJ})
    Smooth Softplus-windowed mode (`smooth_tau > 0`, eliminates Dead Super-Edges):
        phi_tau(u) = tau * (Softplus(u / tau) - Softplus((u - 1) / tau)) in (0, 1)
        A_new(I, J) = phi_tau(max_{(i, j) in E} (S_{iI} + A_{ij} + S_{jJ}) - eta_{IJ})

Two-Step Sparse Decomposition O(M K + N K^2):
    Since phi(u) is monotonically non-decreasing in all modes:
        Step 1: T_raw_{iJ}  = max_{j in N(i)} (A_{ij} + S_{jJ})
        Step 2: raw_max_{IJ} = max_{i: N(i)!=empty} (S_{iI} + T_raw_{iJ})
        Output: A_new(I, J)  = phi(raw_max_{IJ} - eta_{IJ})
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from labflow.fuzzy_relational.sparse_utils import (
    accumulate_winner_gradients,
    sparse_two_step_max_with_winners,
)
from labflow.types import dense_to_sparse


def _inverse_sigmoid(y: float) -> float:
    y_clamped = min(max(y, 1e-6), 1.0 - 1e-6)
    return math.log(y_clamped / (1.0 - y_clamped))


def lukasiewicz_activation(diff: Tensor, clamp_unit: bool = True, smooth_tau: float = 0.0) -> Tuple[Tensor, Tensor]:
    """Evaluate bounded/smooth Lukasiewicz activation and its exact analytical derivative wrt `diff`."""
    if smooth_tau > 0.0:
        u1 = diff / smooth_tau
        if clamp_unit:
            u2 = (diff - 1.0) / smooth_tau
            val = smooth_tau * (F.softplus(u1) - F.softplus(u2))
            deriv = torch.sigmoid(u1) - torch.sigmoid(u2)
        else:
            val = smooth_tau * F.softplus(u1)
            deriv = torch.sigmoid(u1)
        return val, deriv

    if clamp_unit:
        val = diff.clamp(min=0.0, max=1.0)
        deriv = ((diff > 0.0) & (diff < 1.0)).to(diff.dtype)
    else:
        val = F.relu(diff)
        deriv = (diff > 0.0).to(diff.dtype)
    return val, deriv


class _LukasiewiczSparseFn(torch.autograd.Function):
    """Exact analytical forward and subgradient/smooth backward pass for Lukasiewicz max-T_L pooling."""

    @staticmethod
    def forward(
        ctx,
        s: Tensor,
        edge_index: Tensor,
        edge_weight: Tensor,
        eta_tilde: Optional[Tensor],
        fixed_eta: float,
        clamp_unit: bool = True,
        smooth_tau: float = 0.0,
    ) -> Tensor:
        k = s.shape[1]
        m = int(edge_weight.numel())
        if eta_tilde is not None:
            sig_eta = torch.sigmoid(eta_tilde)
            eta = 1.0 + sig_eta  # [K, K] in (1, 2)
        else:
            sig_eta = None
            eta = torch.full((k, k), fixed_eta, dtype=s.dtype, device=s.device)

        if m == 0:
            a_new = torch.zeros((k, k), dtype=s.dtype, device=s.device)
            e_star = torch.full((k, k), -1, dtype=torch.long, device=s.device)
            u_active = torch.zeros((k, k), dtype=s.dtype, device=s.device)
            if sig_eta is not None:
                ctx.save_for_backward(s, edge_index, edge_weight, e_star, u_active, sig_eta)
            else:
                ctx.save_for_backward(s, edge_index, edge_weight, e_star, u_active)
            ctx.has_eta_tilde = eta_tilde is not None
            return a_new

        dst = edge_index[1]
        # Raw additive Step 1: T_raw_{iJ} = max_{j in N(i)} (A_{ij} + S_{jJ})
        u_ej = edge_weight.unsqueeze(1) + s[dst]
        raw_max, _, e_star = sparse_two_step_max_with_winners(
            s=s,
            edge_index=edge_index,
            u_ej=u_ej,
            step2_combine=torch.add,
        )
        diff = raw_max - eta
        a_new, u_active = lukasiewicz_activation(diff, clamp_unit=clamp_unit, smooth_tau=smooth_tau)
        # Mask out cluster pairs where no edges exist at all (raw_max == -inf)
        valid_pairs = e_star >= 0
        if not bool(valid_pairs.all()):
            a_new = torch.where(valid_pairs, a_new, torch.zeros_like(a_new))
            u_active = torch.where(valid_pairs, u_active, torch.zeros_like(u_active))

        if sig_eta is not None:
            ctx.save_for_backward(s, edge_index, edge_weight, e_star, u_active, sig_eta)
        else:
            ctx.save_for_backward(s, edge_index, edge_weight, e_star, u_active)
        ctx.has_eta_tilde = eta_tilde is not None
        return a_new

    @staticmethod
    def backward(
        ctx,
        grad_a_new: Tensor,
    ) -> Tuple[Optional[Tensor], None, Optional[Tensor], Optional[Tensor], None, None, None]:
        saved = ctx.saved_tensors
        if ctx.has_eta_tilde:
            s, edge_index, edge_weight, e_star, u_active, sig_eta = saved
        else:
            s, edge_index, edge_weight, e_star, u_active = saved
            sig_eta = None

        n, k = s.shape
        m = int(edge_weight.numel())

        active_grad = grad_a_new * u_active  # G_A(I, J) * phi'(diff_{IJ})

        if m == 0:
            grad_s = torch.zeros_like(s) if ctx.needs_input_grad[0] else None
            grad_ew = torch.zeros_like(edge_weight) if ctx.needs_input_grad[2] else None
            grad_eta_tilde = torch.zeros_like(sig_eta) if (ctx.has_eta_tilde and ctx.needs_input_grad[3]) else None
            return grad_s, None, grad_ew, grad_eta_tilde, None, None, None

        contrib_edge = active_grad if ctx.needs_input_grad[2] else None
        grad_s_calc, grad_ew = accumulate_winner_gradients(
            num_nodes=n,
            num_clusters=k,
            num_edges=m,
            edge_index=edge_index,
            e_star=e_star,
            contrib_left=active_grad,
            contrib_right=active_grad,
            contrib_edge=contrib_edge,
        )
        grad_s = grad_s_calc if ctx.needs_input_grad[0] else None

        grad_eta_tilde = None
        if ctx.has_eta_tilde and ctx.needs_input_grad[3]:
            assert sig_eta is not None
            grad_eta_tilde = -active_grad * sig_eta * (1.0 - sig_eta)

        return grad_s, None, grad_ew, grad_eta_tilde, None, None, None


class LukasiewiczTopologyReducer(nn.Module):
    """Method 6: Boundary sparse Lukasiewicz FuzzyPool with [0, 1] clamping and optional smooth window."""

    def __init__(
        self,
        num_clusters: int,
        learnable_eta: bool = True,
        init_eta: float = 1.5,
        clamp_unit: bool = True,
        smooth_tau: float = 0.0,
    ) -> None:
        super().__init__()
        self.num_clusters = num_clusters
        self.learnable_eta = learnable_eta
        self.fixed_eta = float(init_eta)
        self.clamp_unit = bool(clamp_unit)
        self.smooth_tau = float(smooth_tau)

        if learnable_eta:
            if not (1.0 < init_eta < 2.0):
                init_eta = 1.5
            raw_val = _inverse_sigmoid(init_eta - 1.0)
            self.eta_tilde = nn.Parameter(
                torch.full((num_clusters, num_clusters), raw_val, dtype=torch.float32)
            )
        else:
            self.register_parameter("eta_tilde", None)

    @property
    def eta(self) -> Tensor:
        if self.eta_tilde is not None:
            return 1.0 + torch.sigmoid(self.eta_tilde)
        return torch.full((self.num_clusters, self.num_clusters), self.fixed_eta)

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
        eta_tilde = self.eta_tilde.to(dtype=s.dtype) if self.eta_tilde is not None else None
        return _LukasiewiczSparseFn.apply(
            s,
            edge_index,
            edge_weight,
            eta_tilde,
            self.fixed_eta,
            self.clamp_unit,
            self.smooth_tau,
        )

    def forward_classical_two_step(
        self,
        s: Tensor,
        edge_index: Tensor,
        edge_weight: Tensor,
    ) -> Tensor:
        """Classical eta = 2 two-step sparse formula:
        T_{iJ} = max_{j in N(i)} max(0, A_{ij} + S_{jJ} - 1),
        A_new(I, J) = max_i max(0, S_{iI} + T_{iJ} - 1).
        """
        n, k = s.shape
        m = int(edge_weight.numel())
        if m == 0:
            return torch.zeros((k, k), dtype=s.dtype, device=s.device)
        src, dst = edge_index[0], edge_index[1]
        u_ej = F.relu(edge_weight.unsqueeze(1) + s[dst] - 1.0)
        t_ij = torch.zeros((n, k), dtype=s.dtype, device=s.device)
        t_ij = t_ij.scatter_reduce(0, src.unsqueeze(1).expand(m, k), u_ej, reduce="amax", include_self=True)
        out = F.relu(s.unsqueeze(2) + t_ij.unsqueeze(1) - 1.0).amax(dim=0)
        return out.clamp(max=1.0) if self.clamp_unit else out

    def forward_edge_reference(
        self,
        s: Tensor,
        edge_index: Tensor,
        edge_weight: Tensor,
    ) -> Tensor:
        """Direct O(M K^2) autograd reference over edges E."""
        k = s.shape[1]
        if edge_weight.numel() == 0:
            return torch.zeros((k, k), dtype=s.dtype, device=s.device)
        eta = (
            1.0 + torch.sigmoid(self.eta_tilde.to(dtype=s.dtype))
            if self.eta_tilde is not None
            else torch.full((k, k), self.fixed_eta, dtype=s.dtype, device=s.device)
        )
        src, dst = edge_index[0], edge_index[1]
        diff = (
            s[src].unsqueeze(2)
            + edge_weight[:, None, None]
            + s[dst].unsqueeze(1)
            - eta.unsqueeze(0)
        )
        val, _ = lukasiewicz_activation(diff, clamp_unit=self.clamp_unit, smooth_tau=self.smooth_tau)
        return val.amax(dim=0)
