"""Unit tests for Section IV: Fuzzy Feature Space Reduction Methods (12, 13A, 13B)."""

from __future__ import annotations

import torch
from torch.autograd import gradcheck

from labflow.fuzzy_features import (
    BipolarSupremalFeaturePooling,
    ChoquetFeaturePooling,
    COGFeaturePooling,
)
from labflow.memberships import inverse_softplus


def test_cog_feature_pooling_centered_gradient_and_gradcheck() -> None:
    gen = torch.Generator().manual_seed(91)
    h = torch.randn(7, 4, dtype=torch.float64, generator=gen, requires_grad=True)
    s = (torch.rand(7, 3, dtype=torch.float64, generator=gen) * 0.8 + 0.1).requires_grad_(True)
    cog = COGFeaturePooling(init_m=2.2, learnable_m=True).double()

    out = cog(h, s)
    assert torch.allclose(out, cog.forward_reference(h.detach(), s.detach()), atol=1e-12)
    assert gradcheck(lambda hin, sin: cog(hin, sin), (h, s), eps=1e-6, atol=1e-6)

    # Constant node features => H_{if} == X_new^{COG}(I, f) => dL/dS == 0 and dL/dm == 0!
    h_const = torch.ones((7, 4), dtype=torch.float64) * 2.5
    s_c = s.detach().clone().requires_grad_(True)
    cog.zero_grad()
    cog(h_const, s_c).sum().backward()
    assert s_c.grad is not None and float(s_c.grad.abs().max()) < 1e-10
    assert cog.m_tilde.grad is not None and float(cog.m_tilde.grad.abs()) < 1e-10


def test_bipolar_supremal_cones_and_gradcheck() -> None:
    gen = torch.Generator().manual_seed(92)
    raw_h = torch.linspace(-1.7, 1.9, 24, dtype=torch.float64)
    raw_h = torch.where(raw_h.abs() < 0.15, raw_h + 0.3, raw_h)
    h = raw_h[torch.randperm(24, generator=gen)].view(6, 4).requires_grad_(True)
    s = torch.linspace(0.15, 0.92, 18, dtype=torch.float64)[torch.randperm(18, generator=gen)].view(6, 3).requires_grad_(True)

    bip = BipolarSupremalFeaturePooling()
    out = bip(h, s)
    assert torch.allclose(out, bip.forward_reference(h.detach(), s.detach()), atol=1e-12)
    assert torch.allclose(bip(-h.detach(), s.detach()), -out.detach(), atol=1e-12)
    assert gradcheck(lambda hin, sin: bip(hin, sin), (h, s), eps=1e-6, atol=1e-6)


def test_choquet_integral_interpolation_single_node_and_gradcheck() -> None:
    gen = torch.Generator().manual_seed(93)
    h = torch.linspace(-1.8, 2.1, 24, dtype=torch.float64)[torch.randperm(24, generator=gen)].view(6, 4).requires_grad_(True)
    s = torch.linspace(0.16, 0.91, 18, dtype=torch.float64)[torch.randperm(18, generator=gen)].view(6, 3).requires_grad_(True)

    choq = ChoquetFeaturePooling(num_clusters=3, init_alpha=1.4, learnable_alpha=True).double()
    out = choq(h, s)
    assert torch.allclose(out, choq.forward_reference(h.detach(), s.detach()), atol=1e-12)
    assert gradcheck(lambda hin, sin: choq(hin, sin), (h, s), eps=1e-6, atol=1e-6)

    # alpha = 1 recovers exact weighted average
    choq_1 = ChoquetFeaturePooling(num_clusters=3, init_alpha=1.0, learnable_alpha=False).double()
    choq_1.alpha_tilde.fill_(inverse_softplus(1.0))
    mean_expected = (s.detach().t() @ h.detach()) / s.detach().sum(dim=0, keepdim=True).t()
    assert torch.allclose(choq_1(h.detach(), s.detach()), mean_expected, atol=1e-11)

    # Single-node graph edge case (N = 1)
    h1 = torch.tensor([[1.2, -0.5, 3.1]], dtype=torch.float64, requires_grad=True)
    s1 = torch.tensor([[0.7, 0.4]], dtype=torch.float64, requires_grad=True)
    choq_n1 = ChoquetFeaturePooling(num_clusters=2, init_alpha=1.5).double()
    out1 = choq_n1(h1, s1)
    assert out1.shape == (2, 3)
    assert torch.allclose(out1[0], h1[0]) and torch.allclose(out1[1], h1[0])
