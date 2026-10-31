"""Method 8: Continuously Parameterized Schweizer-Sklar & Dombi FuzzyPool (Safe Log-Domain Implementation).

Method 8A (Schweizer-Sklar negative parameterization p = -q, q = Softplus(q_tilde) > 0):
    omega_{ij}^{(I, J)} = (S_{iI}^{-q} + A_{ij}^{-q} + S_{jJ}^{-q} - 2)^{-1/q}
    Evaluated in overflow-safe Log-Sum-Exp domain:
        L*_{IJ} = ln(S_{i*I}^{-q} + A_{i*j*}^{-q} + S_{j*J}^{-q} - 2)
        A_new(I, J) = exp(-L*_{IJ} / q) in (0, 1]
    Eliminates float32 overflow for arbitrarily large q > 12 (e.g. q = 50..1000).

Method 8B (Dombi parameterization lambda = Softplus(lambda_tilde) > 0):
    phi(x) = (1 - x) / (x + eps)
    omega_{ij}^{(I, J)} = (1 + [ phi(S_{iI})^lambda + phi(A_{ij})^lambda + phi(S_{jJ})^lambda ]^{1/lambda})^{-1}
    Evaluated in overflow-safe Log-Sum-Exp + Sigmoid domain:
        L*_{IJ} = ln(phi(S_{i*I})^lambda + phi(A_{i*j*})^lambda + phi(S_{j*J})^lambda)
        A_new(I, J) = sigmoid(-L*_{IJ} / lambda) in (0, 1]
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

_DOMBI_LOG_ZERO = -1.0e9


def _ss_log_combine(log_term_a: Tensor, log_term_b: Tensor, eps: float = 1e-12) -> Tensor:
    """Compute ln(exp(a) + exp(b) - 1) safely without overflow for arbitrarily large positive a, b."""
    m = torch.maximum(torch.maximum(log_term_a, log_term_b), torch.zeros_like(log_term_a + log_term_b))
    inner = (torch.exp(log_term_a - m) + torch.exp(log_term_b - m) - torch.exp(-m)).clamp(min=eps)
    return m + torch.log(inner)


def _dombi_log_combine(log_term_a: Tensor, log_term_b: Tensor) -> Tensor:
    """Compute ln(exp(a) + exp(b)) safely without overflow."""
    m = torch.maximum(log_term_a, log_term_b)
    return m + torch.log(torch.exp(log_term_a - m) + torch.exp(log_term_b - m))


class _SchweizerSklarSparseFn(torch.autograd.Function):
    """Exact O(M K + N K^2) overflow-safe log-domain 2-step generator forward & analytical backward."""

    @staticmethod
    def forward(
        ctx,
        s: Tensor,
        edge_index: Tensor,
        edge_weight: Tensor,
        q_tilde: Tensor,
        eps: float,
    ) -> Tensor:
        q = F.softplus(q_tilde)
        k = s.shape[1]
        m = int(edge_weight.numel())

        if m == 0:
            a_new = torch.zeros((k, k), dtype=s.dtype, device=s.device)
            l_star = torch.zeros((k, k), dtype=s.dtype, device=s.device)
            e_star = torch.full((k, k), -1, dtype=torch.long, device=s.device)
            ctx.save_for_backward(s, edge_index, edge_weight, a_new, l_star, e_star, q, q_tilde)
            ctx.eps = eps
            return a_new

        s_c = s.clamp(min=eps)
        ew_c = edge_weight.clamp(min=eps)

        # Positive log-potentials: q * (-ln x)
        q_neg_ln_s = -q * torch.log(s_c)  # [N, K]
        q_neg_ln_ew = -q * torch.log(ew_c)  # [M]

        dst = edge_index[1]
        # Step 1: L1_{e, J} = ln(A_{ij}^{-q} + S_{jJ}^{-q} - 1); maximize -L1_{e, J}
        l1_ej = _ss_log_combine(q_neg_ln_ew.unsqueeze(1), q_neg_ln_s[dst], eps=eps)
        neg_l_star, _, e_star = sparse_two_step_max_with_winners(
            s=q_neg_ln_s,
            edge_index=edge_index,
            u_ej=-l1_ej,
            step2_combine=lambda q_nl_s_ni1, neg_l1_n1j: -_ss_log_combine(q_nl_s_ni1, -neg_l1_n1j, eps=eps),
        )
        l_star = -neg_l_star  # ln(U*_{IJ})
        valid_pairs = e_star >= 0
        a_new = torch.where(valid_pairs, torch.exp(-l_star / q), torch.zeros_like(l_star))

        ctx.save_for_backward(s_c, edge_index, ew_c, a_new, l_star, e_star, q, q_tilde)
        ctx.eps = eps
        return a_new

    @staticmethod
    def backward(
        ctx,
        grad_a_new: Tensor,
    ) -> Tuple[Optional[Tensor], None, Optional[Tensor], Optional[Tensor], None]:
        s_c, edge_index, ew_c, a_new, l_star, e_star, q, q_tilde = ctx.saved_tensors
        n, k = s_c.shape
        m = int(ew_c.numel())

        if m == 0:
            grad_s = torch.zeros_like(s_c) if ctx.needs_input_grad[0] else None
            grad_ew = torch.zeros_like(ew_c) if ctx.needs_input_grad[2] else None
            grad_qt = torch.zeros_like(q_tilde) if ctx.needs_input_grad[3] else None
            return grad_s, None, grad_ew, grad_qt, None

        valid_pairs = e_star >= 0
        e_star_safe = e_star.clamp(min=0)

        src, dst = edge_index[0], edge_index[1]
        i_star = src[e_star_safe]
        j_star = dst[e_star_safe]

        i_idx = torch.arange(k, device=s_c.device).unsqueeze(1).expand(k, k)
        j_idx = torch.arange(k, device=s_c.device).unsqueeze(0).expand(k, k)

        s_left = s_c[i_star, i_idx]
        s_right = s_c[j_star, j_idx]
        a_val = ew_c[e_star_safe]

        ln_s_left = torch.log(s_left)
        ln_s_right = torch.log(s_right)
        ln_a_val = torch.log(a_val)
        ln_a_new = -l_star / q

        # Overflow-free log-ratio derivative: (A_new / x)^{q + 1} = exp((q + 1) * (ln A_new - ln x))
        exp_qp1 = q + 1.0
        g_valid = torch.where(valid_pairs, grad_a_new, torch.zeros_like(grad_a_new))
        contrib_left = g_valid * torch.exp(exp_qp1 * (ln_a_new - ln_s_left))
        contrib_right = g_valid * torch.exp(exp_qp1 * (ln_a_new - ln_s_right))
        contrib_edge = (
            g_valid * torch.exp(exp_qp1 * (ln_a_new - ln_a_val))
            if ctx.needs_input_grad[2]
            else None
        )

        # Mask invalid e_star entries before scatter
        if not bool(valid_pairs.all()):
            contrib_left = torch.where(valid_pairs, contrib_left, torch.zeros_like(contrib_left))
            contrib_right = torch.where(valid_pairs, contrib_right, torch.zeros_like(contrib_right))
            if contrib_edge is not None:
                contrib_edge = torch.where(valid_pairs, contrib_edge, torch.zeros_like(contrib_edge))

        grad_s_calc, grad_ew = accumulate_winner_gradients(
            num_nodes=n,
            num_clusters=k,
            num_edges=m,
            edge_index=edge_index,
            e_star=e_star_safe,
            contrib_left=contrib_left,
            contrib_right=contrib_right,
            contrib_edge=contrib_edge,
        )
        grad_s = grad_s_calc if ctx.needs_input_grad[0] else None

        grad_q_tilde = None
        if ctx.needs_input_grad[3]:
            # Normalized weights x^{-q} / U* = exp(-q ln(x) - L*) in (0, 1], never overflows!
            w_left = torch.exp(-q * ln_s_left - l_star)
            w_edge = torch.exp(-q * ln_a_val - l_star)
            w_right = torch.exp(-q * ln_s_right - l_star)
            weighted_ln = w_left * ln_s_left + w_edge * ln_a_val + w_right * ln_s_right
            d_a_dq = (a_new / q) * ((l_star / q) + weighted_ln)
            grad_q = (g_valid * d_a_dq).sum()
            grad_q_tilde = (grad_q * torch.sigmoid(q_tilde)).reshape_as(q_tilde)

        return grad_s, None, grad_ew, grad_q_tilde, None


class SchweizerSklarTopologyReducer(nn.Module):
    """Method 8A: Continuously parameterized Schweizer-Sklar FuzzyPool (p = -q < 0, overflow-safe log-domain)."""

    def __init__(
        self,
        init_q: float = 1.0,
        learnable_q: bool = True,
        eps: float = 1e-12,
    ) -> None:
        super().__init__()
        self.eps = eps
        raw = torch.tensor(inverse_softplus(init_q), dtype=torch.float32)
        if learnable_q:
            self.q_tilde = nn.Parameter(raw)
        else:
            self.register_buffer("q_tilde", raw)

    @property
    def q(self) -> Tensor:
        return F.softplus(self.q_tilde)

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
        return _SchweizerSklarSparseFn.apply(
            s,
            edge_index,
            edge_weight,
            self.q_tilde.to(dtype=s.dtype),
            self.eps,
        )

    def forward_edge_reference(
        self,
        s: Tensor,
        edge_index: Tensor,
        edge_weight: Tensor,
    ) -> Tensor:
        """Direct O(M K^2) overflow-safe log-domain autograd reference over edges E."""
        k = s.shape[1]
        if edge_weight.numel() == 0:
            return torch.zeros((k, k), dtype=s.dtype, device=s.device)
        q = F.softplus(self.q_tilde.to(dtype=s.dtype))
        s_c = s.clamp(min=self.eps)
        ew_c = edge_weight.clamp(min=self.eps)
        src, dst = edge_index[0], edge_index[1]

        a = (-q * torch.log(s_c[src])).unsqueeze(2)
        b = (-q * torch.log(ew_c))[:, None, None]
        c = (-q * torch.log(s_c[dst])).unsqueeze(1)
        m = torch.maximum(torch.maximum(torch.maximum(a, b), c), torch.zeros_like(a + b + c))
        log_u = m + torch.log((torch.exp(a - m) + torch.exp(b - m) + torch.exp(c - m) - 2.0 * torch.exp(-m)).clamp(min=self.eps))
        omega = torch.exp(-log_u / q)
        return omega.amax(dim=0)

    @staticmethod
    def forward_general_p(
        s: Tensor,
        edge_index: Tensor,
        edge_weight: Tensor,
        p: float,
        eps: float = 1e-12,
    ) -> Tensor:
        """General Schweizer-Sklar formula for any p != 0 (p > 0 nilpotent, p < 0 strict log-safe)."""
        k = s.shape[1]
        if edge_weight.numel() == 0:
            return torch.zeros((k, k), dtype=s.dtype, device=s.device)
        if abs(p) < 1e-12:
            raise ValueError("p must be non-zero (p -> 0 corresponds to Goguen product).")
        src, dst = edge_index[0], edge_index[1]
        s_c = s.clamp(min=eps)
        ew_c = edge_weight.clamp(min=eps)
        if p > 0:
            inner = s_c[src].pow(p).unsqueeze(2) + ew_c.pow(p)[:, None, None] + s_c[dst].pow(p).unsqueeze(1) - 2.0
            omega = F.relu(inner).pow(1.0 / p)
        else:
            q = -p
            a = (-q * torch.log(s_c[src])).unsqueeze(2)
            b = (-q * torch.log(ew_c))[:, None, None]
            c = (-q * torch.log(s_c[dst])).unsqueeze(1)
            m = torch.maximum(torch.maximum(torch.maximum(a, b), c), torch.zeros_like(a + b + c))
            log_u = m + torch.log((torch.exp(a - m) + torch.exp(b - m) + torch.exp(c - m) - 2.0 * torch.exp(-m)).clamp(min=eps))
            omega = torch.exp(-log_u / q)
        return omega.amax(dim=0)


class _DombiSparseFn(torch.autograd.Function):
    """Exact O(M K + N K^2) overflow-safe log-domain 2-step generator forward & analytical backward for Dombi."""

    @staticmethod
    def forward(
        ctx,
        s: Tensor,
        edge_index: Tensor,
        edge_weight: Tensor,
        lam_tilde: Tensor,
        eps: float,
    ) -> Tensor:
        lam = F.softplus(lam_tilde)
        k = s.shape[1]
        m = int(edge_weight.numel())

        if m == 0:
            a_new = torch.zeros((k, k), dtype=s.dtype, device=s.device)
            l_star = torch.zeros((k, k), dtype=s.dtype, device=s.device)
            e_star = torch.full((k, k), -1, dtype=torch.long, device=s.device)
            ctx.save_for_backward(s, edge_index, edge_weight, a_new, l_star, e_star, lam, lam_tilde)
            ctx.eps = eps
            return a_new

        phi_s = ((1.0 - s).clamp(min=0.0)) / (s + eps)
        phi_ew = ((1.0 - edge_weight).clamp(min=0.0)) / (edge_weight + eps)

        ln_phi_s = torch.where(
            phi_s > 0.0,
            torch.log(phi_s.clamp(min=1e-38)),
            torch.full_like(phi_s, _DOMBI_LOG_ZERO),
        )
        ln_phi_ew = torch.where(
            phi_ew > 0.0,
            torch.log(phi_ew.clamp(min=1e-38)),
            torch.full_like(phi_ew, _DOMBI_LOG_ZERO),
        )
        psi_s = lam * ln_phi_s  # [N, K]
        psi_ew = lam * ln_phi_ew  # [M]

        dst = edge_index[1]
        l1_ej = _dombi_log_combine(psi_ew.unsqueeze(1), psi_s[dst])
        neg_l_star, _, e_star = sparse_two_step_max_with_winners(
            s=psi_s,
            edge_index=edge_index,
            u_ej=-l1_ej,
            step2_combine=lambda ps_ni1, neg_l1_n1j: -_dombi_log_combine(ps_ni1, -neg_l1_n1j),
        )
        l_star = -neg_l_star  # ln(V*_{IJ}) = lam * ln(R*_{IJ})
        valid_pairs = e_star >= 0
        # 1 / (1 + R*) == sigmoid(-ln(R*)) == sigmoid(-L* / lam)
        a_new = torch.where(valid_pairs, torch.sigmoid(-l_star / lam), torch.zeros_like(l_star))

        ctx.save_for_backward(s, edge_index, edge_weight, a_new, l_star, e_star, lam, lam_tilde)
        ctx.eps = eps
        return a_new

    @staticmethod
    def backward(
        ctx,
        grad_a_new: Tensor,
    ) -> Tuple[Optional[Tensor], None, Optional[Tensor], Optional[Tensor], None]:
        s, edge_index, edge_weight, a_new, l_star, e_star, lam, lam_tilde = ctx.saved_tensors
        eps = ctx.eps
        n, k = s.shape
        m = int(edge_weight.numel())

        if m == 0:
            grad_s = torch.zeros_like(s) if ctx.needs_input_grad[0] else None
            grad_ew = torch.zeros_like(edge_weight) if ctx.needs_input_grad[2] else None
            grad_lt = torch.zeros_like(lam_tilde) if ctx.needs_input_grad[3] else None
            return grad_s, None, grad_ew, grad_lt, None

        valid_pairs = e_star >= 0
        e_star_safe = e_star.clamp(min=0)

        src, dst = edge_index[0], edge_index[1]
        i_star = src[e_star_safe]
        j_star = dst[e_star_safe]

        i_idx = torch.arange(k, device=s.device).unsqueeze(1).expand(k, k)
        j_idx = torch.arange(k, device=s.device).unsqueeze(0).expand(k, k)

        s_left = s[i_star, i_idx]
        s_right = s[j_star, j_idx]
        a_val = edge_weight[e_star_safe]

        phi_l = ((1.0 - s_left).clamp(min=0.0)) / (s_left + eps)
        phi_a = ((1.0 - a_val).clamp(min=0.0)) / (a_val + eps)
        phi_r = ((1.0 - s_right).clamp(min=0.0)) / (s_right + eps)
        v_pos = valid_pairs & ((phi_l > 0.0) | (phi_a > 0.0) | (phi_r > 0.0))

        sig_factor = a_new * (1.0 - a_new)  # == a_new^2 * R*

        def _d_omega_dx(x: Tensor, phi_x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
            active = v_pos & (phi_x > 0.0)
            ln_phi_x = torch.where(active, torch.log(phi_x.clamp(min=1e-38)), torch.zeros_like(phi_x))
            w_x = torch.where(active, torch.exp(lam * ln_phi_x - l_star), torch.zeros_like(phi_x))
            denom_x = ((1.0 - x).clamp(min=1e-38)) * (x + eps)
            deriv = torch.where(
                active,
                sig_factor * w_x * ((1.0 + eps) / denom_x),
                torch.zeros_like(x),
            )
            return deriv, w_x, ln_phi_x

        d_left, w_l, ln_phi_l = _d_omega_dx(s_left, phi_l)
        d_right, w_r, ln_phi_r = _d_omega_dx(s_right, phi_r)
        d_edge, w_a, ln_phi_a = _d_omega_dx(a_val, phi_a)

        g_valid = torch.where(valid_pairs, grad_a_new, torch.zeros_like(grad_a_new))
        contrib_left = g_valid * d_left
        contrib_right = g_valid * d_right
        contrib_edge = g_valid * d_edge if ctx.needs_input_grad[2] else None

        grad_s_calc, grad_ew = accumulate_winner_gradients(
            num_nodes=n,
            num_clusters=k,
            num_edges=m,
            edge_index=edge_index,
            e_star=e_star_safe,
            contrib_left=contrib_left,
            contrib_right=contrib_right,
            contrib_edge=contrib_edge,
        )
        grad_s = grad_s_calc if ctx.needs_input_grad[0] else None

        grad_lam_tilde = None
        if ctx.needs_input_grad[3]:
            z = l_star / lam
            weighted_ln_phi = w_l * ln_phi_l + w_a * ln_phi_a + w_r * ln_phi_r
            d_omega_dlam = torch.where(
                v_pos,
                -sig_factor * (weighted_ln_phi - z) / lam,
                torch.zeros_like(a_new),
            )
            grad_lam = (g_valid * d_omega_dlam).sum()
            grad_lam_tilde = (grad_lam * torch.sigmoid(lam_tilde)).reshape_as(lam_tilde)

        return grad_s, None, grad_ew, grad_lam_tilde, None


class DombiTopologyReducer(nn.Module):
    """Method 8B: Continuously parameterized Dombi FuzzyPool (overflow-safe log-domain)."""

    def __init__(
        self,
        init_lambda: float = 1.0,
        learnable_lambda: bool = True,
        eps: float = 1e-12,
    ) -> None:
        super().__init__()
        self.eps = eps
        raw = torch.tensor(inverse_softplus(init_lambda), dtype=torch.float32)
        if learnable_lambda:
            self.lam_tilde = nn.Parameter(raw)
        else:
            self.register_buffer("lam_tilde", raw)

    @property
    def lam(self) -> Tensor:
        return F.softplus(self.lam_tilde)

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
        return _DombiSparseFn.apply(
            s,
            edge_index,
            edge_weight,
            self.lam_tilde.to(dtype=s.dtype),
            self.eps,
        )

    def forward_edge_reference(
        self,
        s: Tensor,
        edge_index: Tensor,
        edge_weight: Tensor,
    ) -> Tensor:
        """Direct O(M K^2) overflow-safe log-domain autograd reference over edges E."""
        k = s.shape[1]
        if edge_weight.numel() == 0:
            return torch.zeros((k, k), dtype=s.dtype, device=s.device)
        lam = F.softplus(self.lam_tilde.to(dtype=s.dtype))
        phi_s = ((1.0 - s).clamp(min=0.0)) / (s + self.eps)
        phi_ew = ((1.0 - edge_weight).clamp(min=0.0)) / (edge_weight + self.eps)
        ln_phi_s = torch.where(phi_s > 0.0, torch.log(phi_s.clamp(min=1e-38)), torch.full_like(phi_s, _DOMBI_LOG_ZERO))
        ln_phi_ew = torch.where(phi_ew > 0.0, torch.log(phi_ew.clamp(min=1e-38)), torch.full_like(phi_ew, _DOMBI_LOG_ZERO))
        src, dst = edge_index[0], edge_index[1]
        a = (lam * ln_phi_s[src]).unsqueeze(2)
        b = (lam * ln_phi_ew)[:, None, None]
        c = (lam * ln_phi_s[dst]).unsqueeze(1)
        l_v = _dombi_log_combine(_dombi_log_combine(a, b), c)
        omega = torch.sigmoid(-l_v / lam)
        return omega.amax(dim=0)
