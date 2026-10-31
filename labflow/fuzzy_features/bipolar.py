"""Method 13A: Bipolar Supremal t-norm Feature Pooling.

Forward Pass:
    H^+ = max(0, H),  H^- = max(0, -H)
    X_new^{Bipolar}(I, f) = max_i (S_{iI} * H^+_{if}) - max_j (S_{jI} * H^-_{jf})

Analytical Backward Pass:
    dL / dH_{kf} = sum_I G_X(I, f) S_{kI} (delta_{k, i^+_{If}} I(H_{kf} > 0) + delta_{k, i^-_{If}} I(H_{kf} < 0))
    (dL / dS_{kI})_{feat} = sum_f G_X(I, f) (delta_{k, i^+_{If}} H^+_{kf} - delta_{k, i^-_{If}} H^-_{kf})
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class _BipolarSupremalFeaturePoolFn(torch.autograd.Function):
    """Exact analytical forward and backward pass for Bipolar Supremal Feature Pooling."""

    @staticmethod
    def forward(ctx, h: Tensor, s: Tensor) -> Tensor:
        h_pos = F.relu(h)  # [N, F']
        h_neg = F.relu(-h)  # [N, F']

        # [N, K, 1] * [N, 1, F'] -> [N, K, F']
        p_tensor = s.unsqueeze(2) * h_pos.unsqueeze(1)
        n_tensor = s.unsqueeze(2) * h_neg.unsqueeze(1)

        m_pos, i_pos = p_tensor.max(dim=0)  # [K, F'], [K, F']
        m_neg, i_neg = n_tensor.max(dim=0)  # [K, F'], [K, F']

        ctx.save_for_backward(h, s, h_pos, h_neg, i_pos, i_neg)
        return m_pos - m_neg

    @staticmethod
    def backward(ctx, grad_x_new: Tensor) -> Tuple[Optional[Tensor], Optional[Tensor]]:
        h, s, h_pos, h_neg, i_pos, i_neg = ctx.saved_tensors
        n, f_dim = h.shape
        k = s.shape[1]

        f_idx = torch.arange(f_dim, device=h.device).unsqueeze(0).expand(k, f_dim)
        i_idx = torch.arange(k, device=h.device).unsqueeze(1).expand(k, f_dim)

        grad_h = None
        if ctx.needs_input_grad[0]:
            grad_h = torch.zeros_like(h)
            h_at_pos = h[i_pos, f_idx]
            s_at_pos = s[i_pos, i_idx]
            c_h_pos = grad_x_new * s_at_pos * (h_at_pos > 0.0).to(h.dtype)

            h_at_neg = h[i_neg, f_idx]
            s_at_neg = s[i_neg, i_idx]
            c_h_neg = grad_x_new * s_at_neg * (h_at_neg < 0.0).to(h.dtype)

            grad_h.scatter_add_(0, i_pos, c_h_pos)
            grad_h.scatter_add_(0, i_neg, c_h_neg)

        grad_s = None
        if ctx.needs_input_grad[1]:
            grad_s = torch.zeros_like(s)
            hp_at_pos = h_pos[i_pos, f_idx]
            hn_at_neg = h_neg[i_neg, f_idx]

            grad_s.scatter_add_(0, i_pos.t(), (grad_x_new * hp_at_pos).t())
            grad_s.scatter_add_(0, i_neg.t(), (-grad_x_new * hn_at_neg).t())

        return grad_h, grad_s


class BipolarSupremalFeaturePooling(nn.Module):
    """Method 13A: Orthogonal cone bipolar supremal t-norm feature pooling."""

    def forward(self, h: Tensor, s: Tensor) -> Tensor:
        return _BipolarSupremalFeaturePoolFn.apply(h, s)

    def forward_reference(self, h: Tensor, s: Tensor) -> Tensor:
        h_pos = F.relu(h)
        h_neg = F.relu(-h)
        m_pos = (s.unsqueeze(2) * h_pos.unsqueeze(1)).amax(dim=0)
        m_neg = (s.unsqueeze(2) * h_neg.unsqueeze(1)).amax(dim=0)
        return m_pos - m_neg
