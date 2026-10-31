"""Method 12: Power Centroid Defuzzification (Fuzzy C-Means / Mamdani COG Feature Pooling).

Forward Pass:
    m = 1 + Softplus(m_tilde) in (1, inf)
    W_I = sum_{j=1}^N S_{jI}^m + eps
    X_new^{COG}(I, f) = (sum_{i=1}^N S_{iI}^m H_{if}) / W_I

Analytical Backward Pass:
    dL / dH_{if} = sum_I (S_{iI}^m / W_I) G_X(I, f)
    (dL / dS_{iI})_{feat} = (m S_{iI}^{m - 1} / W_I) sum_f G_X(I, f) (H_{if} - X_new^{COG}(I, f))
    dL / dm = sum_I (1 / W_I) sum_i S_{iI}^m ln(S_{iI}) sum_f G_X(I, f) (H_{if} - X_new^{COG}(I, f))
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from labflow.memberships import inverse_softplus


class _COGFeaturePoolFn(torch.autograd.Function):
    """Exact analytical forward and backward pass for Power Centroid Defuzzification (COG)."""

    @staticmethod
    def forward(
        ctx,
        h: Tensor,
        s: Tensor,
        m_tilde: Tensor,
        eps: float,
    ) -> Tensor:
        m = 1.0 + F.softplus(m_tilde)
        s_pos = s > 0.0
        s_safe = s.clamp(min=1e-30)
        s_pow_m = torch.where(s_pos, s_safe.pow(m), torch.zeros_like(s))
        w = s_pow_m.sum(dim=0) + eps  # [K]
        x_new = (s_pow_m.t() @ h) / w.unsqueeze(1)  # [K, F']

        ctx.save_for_backward(h, s, s_pow_m, w, x_new, m, m_tilde)
        return x_new

    @staticmethod
    def backward(
        ctx,
        grad_x_new: Tensor,
    ) -> Tuple[Optional[Tensor], Optional[Tensor], Optional[Tensor], None]:
        h, s, s_pow_m, w, x_new, m, m_tilde = ctx.saved_tensors
        s_pos = s > 0.0
        s_safe = s.clamp(min=1e-30)

        grad_h = None
        if ctx.needs_input_grad[0]:
            grad_h = s_pow_m @ (grad_x_new / w.unsqueeze(1))

        # Centered deviation contraction: Delta_{iI} = sum_f G_X(I, f) (H_{if} - X_new(I, f))
        need_s = ctx.needs_input_grad[1]
        need_m = ctx.needs_input_grad[2]
        delta = None
        if need_s or need_m:
            row_dot = (grad_x_new * x_new).sum(dim=1)  # [K]
            delta = (h @ grad_x_new.t()) - row_dot.unsqueeze(0)  # [N, K]

        grad_s = None
        if need_s:
            assert delta is not None
            s_pow_mm1 = torch.where(s_pos, s_safe.pow(m - 1.0), torch.zeros_like(s))
            grad_s = (m * s_pow_mm1 / w.unsqueeze(0)) * delta

        grad_m_tilde = None
        if need_m:
            assert delta is not None
            s_pow_m_ln = torch.where(s_pos, s_pow_m * torch.log(s_safe), torch.zeros_like(s))
            grad_m = ((s_pow_m_ln * delta).sum(dim=0) / w).sum()
            grad_m_tilde = (grad_m * torch.sigmoid(m_tilde)).reshape_as(m_tilde)

        return grad_h, grad_s, grad_m_tilde, None


class COGFeaturePooling(nn.Module):
    """Method 12: Fuzzy C-Means / Mamdani Center-of-Gravity feature space reduction."""

    def __init__(
        self,
        init_m: float = 2.0,
        learnable_m: bool = True,
        eps: float = 1e-12,
    ) -> None:
        super().__init__()
        if init_m <= 1.0:
            raise ValueError("Fuzzifier `init_m` must be strictly greater than 1.0.")
        self.eps = eps
        raw = torch.tensor(inverse_softplus(init_m - 1.0), dtype=torch.float32)
        if learnable_m:
            self.m_tilde = nn.Parameter(raw)
        else:
            self.register_buffer("m_tilde", raw)

    @property
    def m(self) -> Tensor:
        return 1.0 + F.softplus(self.m_tilde)

    def forward(self, h: Tensor, s: Tensor) -> Tensor:
        return _COGFeaturePoolFn.apply(h, s, self.m_tilde.to(dtype=h.dtype), self.eps)

    def forward_reference(self, h: Tensor, s: Tensor) -> Tensor:
        m = 1.0 + F.softplus(self.m_tilde.to(dtype=h.dtype))
        s_pow_m = torch.where(s > 0.0, s.clamp(min=1e-30).pow(m), torch.zeros_like(s))
        w = s_pow_m.sum(dim=0) + self.eps
        return (s_pow_m.t() @ h) / w.unsqueeze(1)
