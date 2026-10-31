"""Unit & mathematical verification tests for all AUD.md algorithmic fixes."""

from __future__ import annotations

import torch
from torch.autograd import gradcheck

from labflow.analytics import run_audit_stress_suite
from labflow.classical import DiffPoolReducer, DMoNReducer, MinCutPoolReducer
from labflow.composite import CANONICAL_METHOD_SPECS, GraphReductionLayer
from labflow.datasets import generate_large_sparse_sbm_graph, generate_sbm_graph
from labflow.fuzzy_relational import (
    DombiTopologyReducer,
    LukasiewiczTopologyReducer,
    SchweizerSklarTopologyReducer,
    ZadehTopologyReducer,
)
from labflow.fuzzy_smooth import (
    BandlerKohoutTopologyReducer,
    BoltzmannTopologyReducer,
    ProbSumTopologyReducer,
)
from labflow.gnn import BaseGraphEncoder
from labflow.types import dense_to_sparse


def test_true_sparse_gnn_and_classical_reducers_exact_parity() -> None:
    graph = generate_sbm_graph(block_sizes=(10, 12, 8), feature_dim=10, seed=303, dtype=torch.float64)
    adj = graph.dense_adj()
    ei, ew = graph.sparse_edges()
    x = graph.x

    torch.manual_seed(303)
    enc = BaseGraphEncoder(in_features=10, embed_dim=8, pool_dim=3).double()
    x_dn = x.clone().requires_grad_(True)
    x_sp = x.clone().requires_grad_(True)

    h_dn, z_dn = enc(adj, x_dn)
    h_sp, z_sp = enc.forward_sparse(ei, ew, x_sp)
    assert torch.allclose(h_dn, h_sp, atol=1e-13)
    assert torch.allclose(z_dn, z_sp, atol=1e-13)

    (h_dn.sum() + z_dn.sum()).backward()
    enc_grads_dn = [p.grad.clone() for p in enc.parameters() if p.grad is not None]
    enc.zero_grad()
    (h_sp.sum() + z_sp.sum()).backward()
    enc_grads_sp = [p.grad.clone() for p in enc.parameters() if p.grad is not None]
    assert x_dn.grad is not None and x_sp.grad is not None
    assert torch.allclose(x_dn.grad, x_sp.grad, atol=1e-13)
    for g1, g2 in zip(enc_grads_dn, enc_grads_sp):
        assert torch.allclose(g1, g2, atol=1e-13)

    s_base = torch.softmax(z_dn.detach(), dim=-1)
    h_base = h_dn.detach()
    for reducer in (DiffPoolReducer(), MinCutPoolReducer(), DMoNReducer()):
        s_a = s_base.clone().requires_grad_(True)
        h_a = h_base.clone().requires_grad_(True)
        out_a = reducer(adj, h_a, s_a)
        (out_a.x_new.sum() + out_a.adj_new.sum() + out_a.aux_loss).backward()

        s_b = s_base.clone().requires_grad_(True)
        h_b = h_base.clone().requires_grad_(True)
        out_b = reducer(None, h_b, s_b, edge_index=ei, edge_weight=ew)
        (out_b.x_new.sum() + out_b.adj_new.sum() + out_b.aux_loss).backward()

        assert torch.allclose(out_a.x_new, out_b.x_new, atol=1e-13)
        assert torch.allclose(out_a.adj_new, out_b.adj_new, atol=1e-13)
        assert torch.allclose(out_a.aux_loss, out_b.aux_loss, atol=1e-13)
        assert s_a.grad is not None and s_b.grad is not None
        assert h_a.grad is not None and h_b.grad is not None
        assert torch.allclose(s_a.grad, s_b.grad, atol=1e-12)
        assert torch.allclose(h_a.grad, h_b.grad, atol=1e-12)


def test_matrix_free_sparse_mex_bandler_kohout_edge_cases() -> None:
    """Test Matrix-Free Sparse MEX Bandler-Kohout on sparse graphs, complete graphs, and duplicate COO edges."""
    gen = torch.Generator().manual_seed(919)
    n, k = 14, 4
    s = (torch.rand((n, k), dtype=torch.float64, generator=gen) * 0.8 + 0.1)

    # 1. Sparse random graph + duplicate COO edge test
    g = generate_sbm_graph(block_sizes=(5, 5, 4), p_in=0.6, p_out=0.1, seed=919, dtype=torch.float64)
    adj = g.dense_adj()
    ei, ew = g.sparse_edges()
    # Split edge 0 into two duplicate half-weight COO entries to test coalescing
    ei_dup = torch.cat([ei, ei[:, :1]], dim=1)
    ew_dup = ew.clone()
    ew_dup[0] *= 0.4
    ew_dup = torch.cat([ew_dup, ew[:1] * 0.6], dim=0)

    # 2. Complete graph (mex_rank == N everywhere -> has_nonedge is False!)
    adj_complete = torch.rand((6, 6), dtype=torch.float64, generator=gen) * 0.7 + 0.2
    adj_complete.fill_diagonal_(0.0)
    ei_comp, ew_comp = dense_to_sparse(adj_complete, threshold=0.0)
    s_comp = torch.rand((6, 3), dtype=torch.float64, generator=gen) * 0.8 + 0.1

    for impl in ("reichenbach", "lukasiewicz", "kleene_dienes"):
        bk = BandlerKohoutTopologyReducer(implication=impl, ensure_self_loops=True)
        ref = bk.forward_direct_implication_reference(adj, s)
        out_sp = bk(s, edge_index=ei, edge_weight=ew)
        out_dup = bk(s, edge_index=ei_dup, edge_weight=ew_dup)
        assert torch.allclose(out_sp, ref, atol=1e-13)
        assert torch.allclose(out_dup, ref, atol=1e-13)

        ref_comp = bk.forward_direct_implication_reference(adj_complete, s_comp)
        out_comp = bk(s_comp, edge_index=ei_comp, edge_weight=ew_comp)
        assert torch.allclose(out_comp, ref_comp, atol=1e-13)


def test_lukasiewicz_unit_clamping_and_smooth_window_gradients() -> None:
    ei = torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long)
    ew = torch.tensor([0.92, 0.88, 0.95], dtype=torch.float64)
    s_high = torch.full((3, 2), 0.96, dtype=torch.float64)

    luk_clamp = LukasiewiczTopologyReducer(2, learnable_eta=False, init_eta=1.15, clamp_unit=True).double()
    a_clamp = luk_clamp(s_high, edge_index=ei, edge_weight=ew)
    assert bool((a_clamp >= 0.0).all() and (a_clamp <= 1.0).all())
    assert torch.allclose(a_clamp, torch.ones_like(a_clamp))

    # Sub-threshold cold start: hard ReLU has zero grad, smooth window has positive grad & passes gradcheck
    s_low = torch.tensor([[0.15, 0.22], [0.19, 0.14], [0.25, 0.17]], dtype=torch.float64, requires_grad=True)
    ew_low = torch.tensor([0.31, 0.28, 0.35], dtype=torch.float64, requires_grad=True)
    luk_sm = LukasiewiczTopologyReducer(2, learnable_eta=True, init_eta=1.65, clamp_unit=True, smooth_tau=0.14).double()
    a_sm = luk_sm(s_low, edge_index=ei, edge_weight=ew_low)
    assert bool((a_sm > 0.0).all() and (a_sm < 1.0).all())
    assert gradcheck(
        lambda sin, ewin: luk_sm(sin, edge_index=ei, edge_weight=ewin),
        (s_low, ew_low),
        eps=1e-6,
        atol=1e-6,
    )


def test_log_domain_schweizer_sklar_and_dombi_float32_extreme_exponents() -> None:
    gen = torch.Generator().manual_seed(1234)
    s32 = (torch.rand((10, 3), generator=gen, dtype=torch.float32) * 0.95 + 0.02).requires_grad_(True)
    ei = torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7, 8], [1, 2, 3, 4, 5, 6, 7, 8, 9]], dtype=torch.long)
    ew32 = (torch.rand((9,), generator=gen, dtype=torch.float32) * 0.90 + 0.05).requires_grad_(True)

    ss50 = SchweizerSklarTopologyReducer(init_q=50.0, learnable_q=True)
    out_ss = ss50(s32, edge_index=ei, edge_weight=ew32)
    out_ss.sum().backward()
    assert torch.isfinite(out_ss).all()
    assert s32.grad is not None and torch.isfinite(s32.grad).all()
    assert ew32.grad is not None and torch.isfinite(ew32.grad).all()
    assert ss50.q_tilde.grad is not None and torch.isfinite(ss50.q_tilde.grad).all()

    # Large q=50 closely approximates Zadeh max-min
    zadeh_ref = ZadehTopologyReducer()(s32.detach(), edge_index=ei, edge_weight=ew32.detach())
    assert float((out_ss.detach() - zadeh_ref).abs().max().item()) < 0.03

    s32_d = s32.detach().clone().requires_grad_(True)
    ew32_d = ew32.detach().clone().requires_grad_(True)
    dom50 = DombiTopologyReducer(init_lambda=50.0, learnable_lambda=True)
    out_dom = dom50(s32_d, edge_index=ei, edge_weight=ew32_d)
    out_dom.sum().backward()
    assert torch.isfinite(out_dom).all()
    assert s32_d.grad is not None and torch.isfinite(s32_d.grad).all()
    assert ew32_d.grad is not None and torch.isfinite(ew32_d.grad).all()
    assert dom50.lam_tilde.grad is not None and torch.isfinite(dom50.lam_tilde.grad).all()


def test_scale_normalized_probsum_and_adaptive_boltzmann() -> None:
    g = generate_large_sparse_sbm_graph(num_nodes=500, num_blocks=4, avg_degree=12.0, seed=616)
    ei, ew = g.sparse_edges()
    s = torch.sigmoid(torch.randn((500, 4), generator=torch.Generator().manual_seed(616)))

    for norm_mode in ("mean_field", "geometric"):
        s_req = s.clone().requires_grad_(True)
        ps = ProbSumTopologyReducer(normalization=norm_mode)
        a_ps = ps(s_req, edge_index=ei, edge_weight=ew)
        a_ps.sum().backward()
        assert bool((a_ps > 0.01).all() and (a_ps < 0.99).all())
        assert s_req.grad is not None and float(s_req.grad.abs().mean().item()) > 1e-5

    boltz_ad = BoltzmannTopologyReducer(init_tau_c=0.3, learnable_tau_c=True, scale_by_log_m=True)
    s_b = s.clone().requires_grad_(True)
    a_b = boltz_ad(s_b, edge_index=ei, edge_weight=ew)
    a_b.sum().backward()
    assert torch.isfinite(a_b).all()
    assert s_b.grad is not None and torch.isfinite(s_b.grad).all()
    assert boltz_ad.tau_c_tilde.grad is not None and float(boltz_ad.tau_c_tilde.grad.item()) <= 0.0


def test_audit_stress_suite_passes_all_checks() -> None:
    records = run_audit_stress_suite()
    assert len(records) == 8
    assert all(r.passed for r in records), [r for r in records if not r.passed]
    assert len(CANONICAL_METHOD_SPECS) == 22
    for key in ("lukasiewicz_smooth", "boltzmann_adaptive", "probsum_norm"):
        layer = GraphReductionLayer.from_spec(key, in_features=16, out_features=12, num_clusters=4)
        assert layer is not None
