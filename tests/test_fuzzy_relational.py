"""Unit tests for Section II: Relational Fuzzy Graph Reduction Methods (4..8)."""

from __future__ import annotations

import torch
from torch.autograd import gradcheck

from labflow.fuzzy_relational import (
    DombiTopologyReducer,
    GoguenTopologyReducer,
    HamacherTopologyReducer,
    LukasiewiczTopologyReducer,
    SchweizerSklarTopologyReducer,
    ZadehTopologyReducer,
)
from labflow.types import PoolingOutput, dense_to_sparse


def _distinct_graph_and_s(n: int = 6, k: int = 3, seed: int = 71) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    gen = torch.Generator().manual_seed(seed)
    base_a = torch.linspace(0.21, 0.93, n * n, dtype=torch.float64)[torch.randperm(n * n, generator=gen)].view(n, n)
    base_a.fill_diagonal_(0.0)
    base_a[0, 5] = base_a[5, 0] = 0.0
    ei, ew = dense_to_sparse(base_a, threshold=0.0)
    s = torch.linspace(0.15, 0.92, n * k, dtype=torch.float64)[torch.randperm(n * k, generator=gen)].view(n, k)
    return base_a, ei, ew, s


def test_zadeh_height_bottleneck_and_empty_graph() -> None:
    adj, ei, ew, s = _distinct_graph_and_s()
    zadeh = ZadehTopologyReducer()

    a_sparse = zadeh(s, edge_index=ei, edge_weight=ew)
    a_dense = zadeh.forward_dense_reference(adj, s)
    assert torch.allclose(a_sparse, a_dense, atol=1e-12)

    # Reflexive graph diagonal == fuzzy set height
    adj_refl = torch.maximum(adj, torch.eye(adj.shape[0], dtype=torch.float64))
    a_refl_new = zadeh(s, adj=adj_refl)
    assert torch.allclose(torch.diagonal(a_refl_new), s.amax(dim=0), atol=1e-12)

    # Edge bottleneck masking: A_{01} = 0.2 < min(S_{0,0}=0.8, S_{1,1}=0.9) => dA_new(0,1)/dS == 0
    adj_b = torch.tensor([[0.0, 0.2], [0.2, 0.0]], dtype=torch.float64)
    s_b = torch.tensor([[0.8, 0.1], [0.1, 0.9]], dtype=torch.float64, requires_grad=True)
    out_b = zadeh(s_b, adj=adj_b)
    out_b[0, 1].backward()
    assert s_b.grad is not None and float(s_b.grad.abs().max()) == 0.0

    # Empty graph edge case
    empty_adj = torch.zeros((4, 4), dtype=torch.float64)
    s_empty = torch.rand((4, 2), dtype=torch.float64, requires_grad=True)
    out_empty = zadeh(s_empty, adj=empty_adj)
    assert float(out_empty.detach().abs().max()) == 0.0
    out_empty.sum().backward()
    assert s_empty.grad is not None and float(s_empty.grad.abs().max()) == 0.0


def test_goguen_tropical_isomorphism_and_gradcheck() -> None:
    adj, ei, ew, s = _distinct_graph_and_s(seed=72)
    goguen = GoguenTopologyReducer()
    a_new = goguen(s, edge_index=ei, edge_weight=ew)
    assert torch.allclose(a_new, goguen.forward_dense_reference(adj, s), atol=1e-12)

    # Tropical shortest-path isomorphism
    trop = (-torch.log(s[ei[0]]).unsqueeze(2) - torch.log(ew)[:, None, None] - torch.log(s[ei[1]]).unsqueeze(1)).amin(dim=0)
    assert torch.allclose(-torch.log(a_new), trop, atol=1e-12)

    s_req = s.clone().requires_grad_(True)
    ew_req = ew.clone().requires_grad_(True)
    assert gradcheck(lambda sin, ewin: goguen(sin, edge_index=ei, edge_weight=ewin), (s_req, ew_req), eps=1e-6, atol=1e-6)


def test_lukasiewicz_exact_sparsity_and_gradcheck() -> None:
    _, ei, ew, s = _distinct_graph_and_s(seed=73)
    luk = LukasiewiczTopologyReducer(num_clusters=3, learnable_eta=True, init_eta=1.7).double()
    assert luk.eta_tilde is not None

    # Ensure well-separated active and inactive pairs
    s_scaled = s.clone()
    s_scaled[:, 0] *= 0.4
    a_new = luk(s_scaled, edge_index=ei, edge_weight=ew)
    assert torch.allclose(a_new, luk.forward_edge_reference(s_scaled, ei, ew), atol=1e-12)
    assert bool((a_new == 0.0).any() and (a_new > 0.0).any())

    # Lossless sparse COO & CSR export
    po = PoolingOutput(x_new=torch.zeros(3, 2, dtype=torch.float64), adj_new=a_new, s=s_scaled)
    assert torch.equal(po.to_sparse_coo().to_dense(), a_new)
    assert torch.equal(po.to_sparse_csr().to_dense(), a_new)


def test_hamacher_and_schweizer_dombi_gradchecks() -> None:
    _, ei, ew, s = _distinct_graph_and_s(seed=74)
    s_req = s.clone().requires_grad_(True)
    ew_req = ew.clone().requires_grad_(True)

    ham = HamacherTopologyReducer(init_gamma=1.6, learnable_gamma=True).double()
    assert torch.allclose(
        ham(s, edge_index=ei, edge_weight=ew),
        ham.forward_closed_form_reference(s, ei, ew),
        atol=1e-12,
    )
    assert gradcheck(lambda sin, ewin: ham(sin, edge_index=ei, edge_weight=ewin), (s_req, ew_req), eps=1e-6, atol=1e-6)

    ss = SchweizerSklarTopologyReducer(init_q=1.3, learnable_q=True).double()
    assert torch.allclose(
        ss(s, edge_index=ei, edge_weight=ew),
        ss.forward_edge_reference(s, ei, ew),
        atol=1e-12,
    )
    assert gradcheck(lambda sin, ewin: ss(sin, edge_index=ei, edge_weight=ewin), (s_req, ew_req), eps=1e-6, atol=1e-6)

    dombi = DombiTopologyReducer(init_lambda=1.4, learnable_lambda=True).double()
    assert torch.allclose(
        dombi(s, edge_index=ei, edge_weight=ew),
        dombi.forward_edge_reference(s, ei, ew),
        atol=1e-12,
    )
    assert gradcheck(lambda sin, ewin: dombi(sin, edge_index=ei, edge_weight=ewin), (s_req, ew_req), eps=1e-6, atol=1e-6)
