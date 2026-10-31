"""Method 13B: Discrete Fuzzy Choquet Integral Feature Pooling.

Forward Pass:
    Sort features per column f: H_{pi_f(1), f} <= ... <= H_{pi_f(N), f}
    Tail cumulative membership: C_i^{(I, f)} = sum_{k=i}^N S_{pi_f(k), I},  C_{N+1} = 0
    Distorted fuzzy measure: g_1 = 1,  g_i^{(I, f)} = (C_i^{(I, f)} / (C_1^{(I, f)} + eps))^{alpha_I}  for i >= 2
    Abel summation: X_new^{Choquet}(I, f) = H_{pi_f(1), f} + sum_{i=2}^N (H_{pi_f(i), f} - H_{pi_f(i-1), f}) g_i^{(I, f)}

Analytical Backward Pass:
    dX_new(I, f) / dH_{pi_f(i), f} = g_i^{(I, f)} - g_{i+1}^{(I, f)}
    dX_new(I, f) / dS_{pi_f(k), I} = alpha_I * [ sum_{i=2}^k B_i / C_i - (sum_{i=2}^N B_i) / (C_1 + eps) ]
        where B_i = (H_{pi_f(i), f} - H_{pi_f(i-1), f}) * g_i^{(I, f)}
    dX_new(I, f) / d(alpha_I)      = sum_{i=2}^N B_i * ln(C_i / (C_1 + eps))
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from labflow.memberships import inverse_softplus


class _ChoquetFeaturePoolFn(torch.autograd.Function):
    """Exact O(N K F') analytical forward and backward pass for Discrete Choquet Feature Pooling."""

    @staticmethod
    def forward(
        ctx,
        h: Tensor,
        s: Tensor,
        alpha_tilde: Tensor,
        eps: float,
    ) -> Tensor:
        n, f_dim = h.shape
        k = s.shape[1]
        alpha = F.softplus(alpha_tilde)  # [K]

        if n == 1:
            x_new = h.expand(k, f_dim).clone()
            empty = h.new_empty((0,))
            ctx.save_for_backward(h, s, alpha, alpha_tilde, empty, empty, empty, empty, empty)
            ctx.eps = eps
            return x_new

        h_sorted, perm = torch.sort(h, dim=0, stable=True)  # [N, F']
        perm_exp = perm.unsqueeze(1).expand(n, k, f_dim)
        s_sorted = torch.gather(s.unsqueeze(2).expand(n, k, f_dim), dim=0, index=perm_exp)  # [N, K, F']

        # Tail cumulative sums C_i^{(I, f)} = sum_{m=i}^{N-1} S_{pi_f(m), I}
        c_tail = torch.flip(torch.cumsum(torch.flip(s_sorted, dims=[0]), dim=0), dims=[0])  # [N, K, F']
        c_1 = c_tail[0]  # [K, F']
        c_sub = c_tail[1:]  # [N - 1, K, F'] (corresponds to i = 2..N in 1-based indexing)

        ratio = c_sub / (c_1.unsqueeze(0) + eps)
        g_sub = torch.where(
            ratio > 0.0,
            ratio.clamp(min=1e-30).pow(alpha.view(1, k, 1)),
            torch.zeros_like(ratio),
        )  # [N - 1, K, F']

        dh = h_sorted[1:] - h_sorted[:-1]  # [N - 1, F']
        x_new = h_sorted[0].unsqueeze(0) + (dh.unsqueeze(1) * g_sub).sum(dim=0)  # [K, F']

        ctx.save_for_backward(h, s, alpha, alpha_tilde, perm, c_1, c_sub, ratio, g_sub, dh)
        ctx.eps = eps
        return x_new

    @staticmethod
    def backward(
        ctx,
        grad_x_new: Tensor,
    ) -> Tuple[Optional[Tensor], Optional[Tensor], Optional[Tensor], None]:
        saved = ctx.saved_tensors
        h, s, alpha, alpha_tilde = saved[0], saved[1], saved[2], saved[3]
        n, f_dim = h.shape
        k = s.shape[1]
        eps = ctx.eps

        if n == 1:
            grad_h = grad_x_new.sum(dim=0, keepdim=True) if ctx.needs_input_grad[0] else None
            grad_s = torch.zeros_like(s) if ctx.needs_input_grad[1] else None
            grad_at = torch.zeros_like(alpha_tilde) if ctx.needs_input_grad[2] else None
            return grad_h, grad_s, grad_at, None

        perm, c_1, c_sub, ratio, g_sub, dh = saved[4], saved[5], saved[6], saved[7], saved[8], saved[9]

        grad_h = None
        if ctx.needs_input_grad[0]:
            # g_1 = 1, g_{2..N} = g_sub, g_{N+1} = 0
            ones_top = torch.ones((1, k, f_dim), dtype=h.dtype, device=h.device)
            zeros_bot = torch.zeros((1, k, f_dim), dtype=h.dtype, device=h.device)
            g_full = torch.cat([ones_top, g_sub, zeros_bot], dim=0)  # [N + 1, K, F']
            delta_g = g_full[:-1] - g_full[1:]  # [N, K, F']
            grad_h_sorted = (grad_x_new.unsqueeze(0) * delta_g).sum(dim=1)  # [N, F']
            grad_h = torch.zeros_like(h)
            grad_h.scatter_(0, perm, grad_h_sorted)

        # B_i^{(I, f)} = (H_{pi_f(i), f} - H_{pi_f(i-1), f}) * g_i^{(I, f)} for i = 2..N
        b_sub = dh.unsqueeze(1) * g_sub  # [N - 1, K, F']

        grad_s = None
        if ctx.needs_input_grad[1]:
            # Exact O(N K F') prefix-sum evaluation of Section 13B.2 Formula 3:
            # dX_new(I, f) / dS_{pi_f(k), I} = alpha_I * (sum_{i=2}^k B_i / C_i - (sum_{i=2}^N B_i) / (C_1 + eps))
            b_over_c = torch.where(
                c_sub > 0.0,
                b_sub / c_sub.clamp(min=1e-30),
                torch.zeros_like(b_sub),
            )
            prefix_bc = torch.cumsum(b_over_c, dim=0)  # [N - 1, K, F'] for k = 2..N
            zero_k1 = torch.zeros((1, k, f_dim), dtype=h.dtype, device=h.device)
            prefix_all = torch.cat([zero_k1, prefix_bc], dim=0)  # [N, K, F'] for k = 1..N

            total_b_over_c1 = b_sub.sum(dim=0, keepdim=True) / (c_1.unsqueeze(0) + eps)  # [1, K, F']
            dx_ds_sorted = alpha.view(1, k, 1) * (prefix_all - total_b_over_c1)  # [N, K, F']
            grad_s_sorted = grad_x_new.unsqueeze(0) * dx_ds_sorted  # [N, K, F']

            perm_exp = perm.unsqueeze(1).expand(n, k, f_dim)
            grad_s_3d = torch.zeros((n, k, f_dim), dtype=h.dtype, device=h.device)
            grad_s_3d.scatter_add_(0, perm_exp, grad_s_sorted)
            grad_s = grad_s_3d.sum(dim=2)

        grad_alpha_tilde = None
        if ctx.needs_input_grad[2]:
            ln_ratio = torch.where(
                ratio > 0.0,
                torch.log(ratio.clamp(min=1e-30)),
                torch.zeros_like(ratio),
            )
            dx_dalpha = (b_sub * ln_ratio).sum(dim=0)  # [K, F']
            grad_alpha = (grad_x_new * dx_dalpha).sum(dim=1)  # [K]
            grad_alpha_tilde = grad_alpha * torch.sigmoid(alpha_tilde)

        return grad_h, grad_s, grad_alpha_tilde, None


class ChoquetFeaturePooling(nn.Module):
    """Method 13B: Discrete fuzzy Choquet integral feature pooling with learnable distortion alpha_I."""

    def __init__(
        self,
        num_clusters: int,
        init_alpha: float = 1.0,
        learnable_alpha: bool = True,
        eps: float = 1e-12,
    ) -> None:
        super().__init__()
        self.num_clusters = num_clusters
        self.eps = eps
        raw = torch.full((num_clusters,), inverse_softplus(init_alpha), dtype=torch.float32)
        if learnable_alpha:
            self.alpha_tilde = nn.Parameter(raw)
        else:
            self.register_buffer("alpha_tilde", raw)

    @property
    def alpha(self) -> Tensor:
        return F.softplus(self.alpha_tilde)

    def forward(self, h: Tensor, s: Tensor) -> Tensor:
        return _ChoquetFeaturePoolFn.apply(
            h,
            s,
            self.alpha_tilde.to(dtype=h.dtype),
            self.eps,
        )

    def forward_reference(self, h: Tensor, s: Tensor) -> Tensor:
        """Pure PyTorch autograd reference implementation."""
        n, f_dim = h.shape
        k = s.shape[1]
        if n == 1:
            return h.expand(k, f_dim).clone()
        alpha = F.softplus(self.alpha_tilde.to(dtype=h.dtype))
        h_sorted, perm = torch.sort(h, dim=0, stable=True)
        perm_exp = perm.unsqueeze(1).expand(n, k, f_dim)
        s_sorted = torch.gather(s.unsqueeze(2).expand(n, k, f_dim), dim=0, index=perm_exp)
        c_tail = torch.flip(torch.cumsum(torch.flip(s_sorted, dims=[0]), dim=0), dims=[0])
        c_1 = c_tail[0]
        c_sub = c_tail[1:]
        ratio = c_sub / (c_1.unsqueeze(0) + self.eps)
        g_sub = torch.where(
            ratio > 0.0,
            ratio.clamp(min=1e-30).pow(alpha.view(1, k, 1)),
            torch.zeros_like(ratio),
        )
        dh = h_sorted[1:] - h_sorted[:-1]
        return h_sorted[0].unsqueeze(0) + (dh.unsqueeze(1) * g_sub).sum(dim=0)
