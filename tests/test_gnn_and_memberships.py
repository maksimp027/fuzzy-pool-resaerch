"""Unit tests for base GraphSAGE preprocessing operators and membership mappings Z -> S."""

from __future__ import annotations

import torch
from torch.autograd import gradcheck

from labflow.gnn import (
    BaseGraphEncoder,
    GraphSAGEOperator,
    symmetric_normalize_adjacency,
    symmetric_normalize_with_self_loops,
)
from labflow.memberships import (
    AnisotropicSigmoidMembership,
    GaussianRBFMembership,
    TemperatureSoftmaxMembership,
)


def test_symmetric_normalizations_spectrum() -> None:
    torch.manual_seed(0)
    raw = torch.rand(8, 8, dtype=torch.float64)
    adj = 0.5 * (raw + raw.t())
    adj.fill_diagonal_(0.0)

    a_hat = symmetric_normalize_with_self_loops(adj)
    a_tilde = symmetric_normalize_adjacency(adj)

    assert torch.allclose(a_hat, a_hat.t(), atol=1e-12)
    assert torch.allclose(a_tilde, a_tilde.t(), atol=1e-12)

    eig_hat = torch.linalg.eigvalsh(a_hat)
    eig_tilde = torch.linalg.eigvalsh(a_tilde)
    assert float(eig_hat.min()) >= -1.0 - 1e-9 and float(eig_hat.max()) <= 1.0 + 1e-9
    assert float(eig_tilde.min()) >= -1.0 - 1e-9 and float(eig_tilde.max()) <= 1.0 + 1e-9


def test_graphsage_and_encoder_analytical_vs_reference() -> None:
    torch.manual_seed(1)
    adj = torch.rand(7, 7, dtype=torch.float64)
    adj = 0.5 * (adj + adj.t())
    x = torch.randn(7, 5, dtype=torch.float64, requires_grad=True)
    x_ref = x.detach().clone().requires_grad_(True)

    enc = BaseGraphEncoder(in_features=5, embed_dim=6, pool_dim=3, embed_activation="elu").double()
    h, z = enc(adj, x)
    (h.sum() + 0.5 * z.sum()).backward()
    grads_custom = [p.grad.detach().clone() for p in enc.parameters() if p.grad is not None]
    gx_custom = x.grad.detach().clone()  # type: ignore[union-attr]

    enc.zero_grad()
    h_r, z_r = enc.forward_reference(adj, x_ref)
    (h_r.sum() + 0.5 * z_r.sum()).backward()
    grads_ref = [p.grad.detach().clone() for p in enc.parameters() if p.grad is not None]
    gx_ref = x_ref.grad.detach().clone()  # type: ignore[union-attr]

    assert torch.allclose(h, h_r, atol=1e-12)
    assert torch.allclose(z, z_r, atol=1e-12)
    assert torch.allclose(gx_custom, gx_ref, atol=1e-11)
    for gc, gr in zip(grads_custom, grads_ref):
        assert torch.allclose(gc, gr, atol=1e-11)


def test_memberships_domains_and_gradcheck() -> None:
    torch.manual_seed(2)
    z = torch.randn(6, 4, dtype=torch.float64, requires_grad=True)

    # 1. Temperature Softmax: simplex Delta^{K-1}
    m_soft = TemperatureSoftmaxMembership(init_tau=0.75, learnable_tau=True).double()
    s_soft = m_soft(z)
    assert torch.allclose(s_soft.sum(dim=1), torch.ones(6, dtype=torch.float64), atol=1e-12)
    assert gradcheck(lambda zin: m_soft(zin), (z,), eps=1e-6, atol=1e-6)

    # 2. Anisotropic Sigmoid: hypercube (0, 1)^{N x K}, unnormalized rows
    m_sig = AnisotropicSigmoidMembership(num_clusters=4, init_gamma=1.4, init_beta=-0.2).double()
    s_sig = m_sig(z)
    assert bool((s_sig > 0.0).all() and (s_sig < 1.0).all())
    assert not torch.allclose(s_sig.sum(dim=1), torch.ones(6, dtype=torch.float64))
    assert gradcheck(lambda zin: m_sig(zin), (z,), eps=1e-6, atol=1e-6)

    # 3. Gaussian RBF: prototype peak S_{iI} == 1 when Z_i == mu_I
    m_rbf = GaussianRBFMembership(num_clusters=4, latent_dim=3, init_sigma=0.8).double()
    z_lat = torch.randn(5, 3, dtype=torch.float64, requires_grad=True)
    s_rbf = m_rbf(z_lat)
    assert bool((s_rbf > 0.0).all() and (s_rbf <= 1.0).all())
    # Evaluate right at prototype mu_0
    s_at_proto = m_rbf(m_rbf.mu.detach())
    assert torch.allclose(torch.diagonal(s_at_proto), torch.ones(4, dtype=torch.float64), atol=1e-12)
    assert gradcheck(lambda zin: m_rbf(zin), (z_lat,), eps=1e-6, atol=1e-6)
