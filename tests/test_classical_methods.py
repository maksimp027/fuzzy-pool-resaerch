"""Unit tests for Section I: Classical Algebraic & Spectral Graph Reduction Methods (1..3)."""

from __future__ import annotations

import math

import torch
from torch.autograd import gradcheck

from labflow.classical import DiffPoolReducer, DMoNReducer, MinCutPoolReducer
from labflow.datasets import generate_sbm_graph


def test_diffpool_mass_conservation_and_gradients() -> None:
    g = generate_sbm_graph(block_sizes=(6, 6, 6), seed=10, dtype=torch.float64)
    adj = g.dense_adj()
    h = g.x[:, :5].clone().requires_grad_(True)
    s = torch.softmax(torch.randn(18, 3, dtype=torch.float64), dim=-1).requires_grad_(True)

    dp = DiffPoolReducer()
    out = dp(adj, h, s)

    # Edge mass conservation: sum(A_new) == sum(A)
    assert torch.allclose(out.adj_new.sum(), adj.sum(), atol=1e-11)
    assert float(out.adj_new.detach().max()) > 1.0

    # Gradcheck wrt H and S
    assert gradcheck(
        lambda hin, sin: (dp(adj, hin, sin).x_new, dp(adj, hin, sin).adj_new, dp(adj, hin, sin).aux_loss),
        (h, s),
        eps=1e-6,
        atol=1e-5,
    )


def test_mincutpool_spectral_bounds_and_zero_diagonal() -> None:
    g = generate_sbm_graph(block_sizes=(5, 5, 5), seed=20, dtype=torch.float64)
    adj = g.dense_adj()
    h = g.x[:, :4].clone().requires_grad_(True)
    s = torch.softmax(torch.randn(15, 3, dtype=torch.float64), dim=-1).requires_grad_(True)

    mc = MinCutPoolReducer()
    out = mc(adj, h, s)

    assert float(torch.diagonal(out.adj_new.detach()).abs().max()) == 0.0
    l_cut = float(out.loss_components["l_cut"].detach())
    l_orth = float(out.loss_components["l_orth"].detach())
    assert -1.0 - 1e-9 <= l_cut <= 1e-9
    assert 0.0 <= l_orth <= math.sqrt(2.0) + 1e-9

    assert gradcheck(
        lambda hin, sin: (mc(adj, hin, sin).x_new, mc(adj, hin, sin).adj_new, mc(adj, hin, sin).aux_loss),
        (h, s),
        eps=1e-6,
        atol=1e-5,
    )


def test_dmon_rank1_modularity_and_gradients() -> None:
    g = generate_sbm_graph(block_sizes=(5, 6, 5), seed=30, dtype=torch.float64)
    adj = g.dense_adj()
    h = g.x[:, :4].clone().requires_grad_(True)
    s = torch.softmax(torch.randn(16, 3, dtype=torch.float64), dim=-1).requires_grad_(True)

    dmon = DMoNReducer()
    out = dmon(adj, h, s)
    ref = dmon.forward_reference(adj, h.detach(), s.detach())

    assert torch.allclose(out.x_new, ref.x_new, atol=1e-12)
    assert torch.allclose(out.adj_new, ref.adj_new, atol=1e-12)
    assert torch.allclose(out.loss_components["l_mod"], ref.loss_components["l_mod"], atol=1e-12)
    assert torch.allclose(out.loss_components["l_collapse"], ref.loss_components["l_collapse"], atol=1e-12)

    assert gradcheck(
        lambda hin, sin: (dmon(adj, hin, sin).x_new, dmon(adj, hin, sin).adj_new, dmon(adj, hin, sin).aux_loss),
        (h, s),
        eps=1e-6,
        atol=1e-5,
    )
