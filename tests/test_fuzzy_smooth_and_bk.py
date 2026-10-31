"""Unit tests for Section III: Smooth Conorm & Implicative Quantifier Fuzzy Reduction (9..11)."""

from __future__ import annotations

import torch
from torch.autograd import gradcheck

from labflow.datasets import generate_parallel_bridges_graph, generate_planted_cliques_graph
from labflow.fuzzy_relational import GoguenTopologyReducer, LukasiewiczTopologyReducer, ZadehTopologyReducer
from labflow.fuzzy_smooth import (
    BandlerKohoutTopologyReducer,
    BoltzmannTopologyReducer,
    ProbSumTopologyReducer,
)
from labflow.types import dense_to_sparse


def test_boltzmann_both_modes_and_variance_derivative() -> None:
    gen = torch.Generator().manual_seed(81)
    adj = torch.linspace(0.2, 0.9, 25, dtype=torch.float64)[torch.randperm(25, generator=gen)].view(5, 5)
    adj.fill_diagonal_(0.0)
    ei, ew = dense_to_sparse(adj, threshold=0.0)
    s = torch.linspace(0.15, 0.88, 15, dtype=torch.float64)[torch.randperm(15, generator=gen)].view(5, 3)

    for mode in ("global_edge", "two_step"):
        boltz = BoltzmannTopologyReducer(init_tau_c=0.3, learnable_tau_c=True, mode=mode).double()
        out = boltz(s, edge_index=ei, edge_weight=ew)
        ref = boltz.forward_reference(s, ei, ew)
        assert torch.allclose(out, ref, atol=1e-12)

        # Temperature derivative is strictly non-positive (-Var / tau_c^2 <= 0)
        boltz.zero_grad()
        out.sum().backward()
        assert boltz.tau_c_tilde.grad is not None
        assert float(boltz.tau_c_tilde.grad.item()) <= 0.0

        s_req = s.clone().requires_grad_(True)
        assert gradcheck(lambda sin: boltz(sin, edge_index=ei, edge_weight=ew), (s_req,), eps=1e-6, atol=1e-6)


def test_probsum_parallel_bridges_and_large_edge_stability() -> None:
    g, s = generate_parallel_bridges_graph(cluster_size=6, num_bridges=8, bridge_weight=0.3, dtype=torch.float64)
    ei, ew = g.sparse_edges()
    ps = ProbSumTopologyReducer()
    a_new = ps(s, edge_index=ei, edge_weight=ew)

    expected = 1.0 - (1.0 - 0.3) ** 8
    assert abs(float(a_new[0, 1].item()) - expected) < 1e-11
    assert bool((a_new >= 0.0).all() and (a_new <= 1.0).all())

    # Stability test with 2000 edges (direct product would underflow, log-space stays stable)
    src = torch.zeros(2000, dtype=torch.long)
    dst = torch.ones(2000, dtype=torch.long)
    ei_big = torch.stack([src, dst], dim=0)
    ew_big = torch.full((2000,), 0.5, dtype=torch.float64)
    s_big = torch.tensor([[0.8, 0.2], [0.2, 0.8]], dtype=torch.float64)
    a_big = ps(s_big, edge_index=ei_big, edge_weight=ew_big)
    assert torch.isfinite(a_big).all()
    assert abs(float(a_big[0, 1].item()) - 1.0) < 1e-12


def test_bandler_kohout_demorgan_dualities_and_clique_pooling() -> None:
    gen = torch.Generator().manual_seed(82)
    adj = torch.linspace(0.05, 0.95, 36, dtype=torch.float64)[torch.randperm(36, generator=gen)].view(6, 6)
    adj.fill_diagonal_(1.0)
    a_bar = 1.0 - adj
    ei_bar, ew_bar = dense_to_sparse(a_bar, threshold=0.0)
    s = torch.linspace(0.18, 0.91, 18, dtype=torch.float64)[torch.randperm(18, generator=gen)].view(6, 3)

    # All 3 Bandler-Kohout variants equal 1 - existential dual on complement graph A_bar
    bk_r = BandlerKohoutTopologyReducer(implication="reichenbach", ensure_self_loops=False)
    bk_l = BandlerKohoutTopologyReducer(implication="lukasiewicz", ensure_self_loops=False)
    bk_kd = BandlerKohoutTopologyReducer(implication="kleene_dienes", ensure_self_loops=False)

    assert torch.allclose(bk_r(s, adj=adj), 1.0 - GoguenTopologyReducer()(s, edge_index=ei_bar, edge_weight=ew_bar), atol=1e-12)
    assert torch.allclose(
        bk_l(s, adj=adj),
        1.0 - LukasiewiczTopologyReducer(3, learnable_eta=False, init_eta=2.0).double()(s, edge_index=ei_bar, edge_weight=ew_bar),
        atol=1e-12,
    )
    assert torch.allclose(bk_kd(s, adj=adj), 1.0 - ZadehTopologyReducer()(s, edge_index=ei_bar, edge_weight=ew_bar), atol=1e-12)

    # Clique pooling theorem
    g_clq = generate_planted_cliques_graph(num_cliques=2, clique_size=4, num_noise_nodes=3, noise_edge_prob=0.0, dtype=torch.float64)
    s_clq = torch.zeros((11, 2), dtype=torch.float64)
    s_clq[:4, 0] = 0.9  # exact 4-clique
    s_clq[4:7, 1] = 0.9
    s_clq[9, 1] = 0.9  # disconnected node breaks clique
    a_clq_new = BandlerKohoutTopologyReducer(implication="reichenbach", ensure_self_loops=True)(s_clq, adj=g_clq.dense_adj())
    assert abs(float(a_clq_new[0, 0].item()) - 1.0) < 1e-12
    assert float(a_clq_new[1, 1].item()) < 0.25
