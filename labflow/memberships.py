"""Membership parameterizations Z -> S and their exact analytical Jacobians.

Implements:
1. TemperatureSoftmax: S = Softmax(Z / tau), tau = Softplus(tau_tilde) > 0 (simplex Delta^{K-1})
2. AnisotropicSigmoid: S_{iI} = sigma(gamma_I (Z_{iI} - beta_I)), gamma_I = Softplus(gamma_tilde_I) > 0 ([0, 1]^{N x K})
3. GaussianRBF: S_{iI} = exp(-sum_d (Z_{id} - mu_{Id})^2 / (2 (sigma_{Id}^2 + eps))), sigma_{Id} = Softplus(sigma_tilde_{Id})
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


def inverse_softplus(value: float) -> float:
    """Compute x such that Softplus(x) == value for value > 0."""
    if value <= 0.0:
        raise ValueError("inverse_softplus requires value > 0")
    if value > 20.0:
        return value
    return math.log(math.expm1(value))


class _TemperatureSoftmaxFn(torch.autograd.Function):
    """Exact analytical Jacobian for S_{iI} = exp(Z_{iI}/tau) / sum_J exp(Z_{iJ}/tau), tau = Softplus(tau_tilde)."""

    @staticmethod
    def forward(ctx, z: Tensor, tau_tilde: Tensor) -> Tensor:
        tau = F.softplus(tau_tilde)
        s = F.softmax(z / tau, dim=-1)
        ctx.save_for_backward(z, s, tau, tau_tilde)
        return s

    @staticmethod
    def backward(ctx, grad_s: Tensor) -> Tuple[Optional[Tensor], Optional[Tensor]]:
        z, s, tau, tau_tilde = ctx.saved_tensors
        # dL/dZ_{iI} = (1 / tau) * S_{iI} * (dL/dS_{iI} - sum_J S_{iJ} dL/dS_{iJ})
        dot = (s * grad_s).sum(dim=-1, keepdim=True)
        grad_z_calc = (s * (grad_s - dot)) / tau

        grad_z = grad_z_calc if ctx.needs_input_grad[0] else None
        grad_tau_tilde = None
        if ctx.needs_input_grad[1]:
            # d(Z / tau)/dtau = -Z / tau^2  =>  dL/dtau = -(1 / tau) * sum_{i,I} (dL/dZ_{iI}) * Z_{iI}
            grad_tau = -(grad_z_calc * z).sum() / tau
            grad_tau_tilde = grad_tau * torch.sigmoid(tau_tilde)
            if tau_tilde.ndim > 0:
                grad_tau_tilde = grad_tau_tilde.reshape_as(tau_tilde)
        return grad_z, grad_tau_tilde


class TemperatureSoftmaxMembership(nn.Module):
    """Row-stochastic simplex projection S = Softmax(Z / tau)."""

    def __init__(self, init_tau: float = 1.0, learnable_tau: bool = True) -> None:
        super().__init__()
        raw = torch.tensor(inverse_softplus(init_tau), dtype=torch.float32)
        if learnable_tau:
            self.tau_tilde = nn.Parameter(raw)
        else:
            self.register_buffer("tau_tilde", raw)

    @property
    def tau(self) -> Tensor:
        return F.softplus(self.tau_tilde)

    def forward(self, z: Tensor) -> Tensor:
        return _TemperatureSoftmaxFn.apply(z, self.tau_tilde.to(dtype=z.dtype))

    def forward_reference(self, z: Tensor) -> Tensor:
        tau = F.softplus(self.tau_tilde.to(dtype=z.dtype))
        return F.softmax(z / tau, dim=-1)


class _AnisotropicSigmoidFn(torch.autograd.Function):
    """Exact analytical Jacobian for S_{iI} = sigma(gamma_I * (Z_{iI} - beta_I))."""

    @staticmethod
    def forward(ctx, z: Tensor, gamma_tilde: Tensor, beta: Tensor) -> Tensor:
        gamma = F.softplus(gamma_tilde)
        diff = z - beta.unsqueeze(0)
        s = torch.sigmoid(gamma.unsqueeze(0) * diff)
        ctx.save_for_backward(s, diff, gamma, gamma_tilde)
        return s

    @staticmethod
    def backward(ctx, grad_s: Tensor) -> Tuple[Optional[Tensor], Optional[Tensor], Optional[Tensor]]:
        s, diff, gamma, gamma_tilde = ctx.saved_tensors
        # ds = S_{iI} * (1 - S_{iI})
        delta = grad_s * s * (1.0 - s)

        grad_z = delta * gamma.unsqueeze(0) if ctx.needs_input_grad[0] else None
        grad_gamma_tilde = None
        if ctx.needs_input_grad[1]:
            grad_gamma_tilde = torch.sigmoid(gamma_tilde) * (diff * delta).sum(dim=0)
        grad_beta = None
        if ctx.needs_input_grad[2]:
            grad_beta = -gamma * delta.sum(dim=0)
        return grad_z, grad_gamma_tilde, grad_beta


class AnisotropicSigmoidMembership(nn.Module):
    """Fuzzy hypercube mapping S_{iI} = sigma(gamma_I (Z_{iI} - beta_I)) in (0, 1)^{N x K}."""

    def __init__(self, num_clusters: int, init_gamma: float = 1.0, init_beta: float = 0.0) -> None:
        super().__init__()
        self.num_clusters = num_clusters
        self.gamma_tilde = nn.Parameter(
            torch.full((num_clusters,), inverse_softplus(init_gamma), dtype=torch.float32)
        )
        self.beta = nn.Parameter(torch.full((num_clusters,), init_beta, dtype=torch.float32))

    @property
    def gamma(self) -> Tensor:
        return F.softplus(self.gamma_tilde)

    def forward(self, z: Tensor) -> Tensor:
        return _AnisotropicSigmoidFn.apply(
            z,
            self.gamma_tilde.to(dtype=z.dtype),
            self.beta.to(dtype=z.dtype),
        )

    def forward_reference(self, z: Tensor) -> Tensor:
        gamma = F.softplus(self.gamma_tilde.to(dtype=z.dtype))
        beta = self.beta.to(dtype=z.dtype)
        return torch.sigmoid(gamma.unsqueeze(0) * (z - beta.unsqueeze(0)))


class _GaussianRBFFn(torch.autograd.Function):
    """Exact analytical Jacobian for Gaussian prototype RBF membership mapping."""

    @staticmethod
    def forward(ctx, z: Tensor, mu: Tensor, sigma_tilde: Tensor, eps: float) -> Tensor:
        sigma = F.softplus(sigma_tilde)
        denom = sigma * sigma + eps  # [K, D]
        # diff: [N, K, D]
        diff = z.unsqueeze(1) - mu.unsqueeze(0)
        exponent = -0.5 * ((diff * diff) / denom.unsqueeze(0)).sum(dim=-1)
        s = torch.exp(exponent)
        ctx.save_for_backward(s, diff, sigma, sigma_tilde, denom)
        return s

    @staticmethod
    def backward(ctx, grad_s: Tensor) -> Tuple[Optional[Tensor], Optional[Tensor], Optional[Tensor], None]:
        s, diff, sigma, sigma_tilde, denom = ctx.saved_tensors
        # w: [N, K, 1] = G_S * S
        w = (grad_s * s).unsqueeze(-1)
        scaled = w * (diff / denom.unsqueeze(0))  # [N, K, D]

        grad_z = -scaled.sum(dim=1) if ctx.needs_input_grad[0] else None
        grad_mu = scaled.sum(dim=0) if ctx.needs_input_grad[1] else None
        grad_sigma_tilde = None
        if ctx.needs_input_grad[2]:
            # dS/dsigma_tilde_{Id} = S_{iI} * ((Z_{id} - mu_{Id})^2 * sigma_{Id} / (sigma_{Id}^2 + eps)^2) * sigmoid(sigma_tilde_{Id})
            term = (w * (diff * diff)).sum(dim=0) * sigma / (denom * denom)
            grad_sigma_tilde = term * torch.sigmoid(sigma_tilde)

        return grad_z, grad_mu, grad_sigma_tilde, None


class GaussianRBFMembership(nn.Module):
    """Gaussian prototype (RBF) fuzzy membership mapping Z in R^{N x D} -> S in (0, 1]^{N x K}."""

    def __init__(
        self,
        num_clusters: int,
        latent_dim: Optional[int] = None,
        init_sigma: float = 1.0,
        eps: float = 1e-12,
    ) -> None:
        super().__init__()
        self.num_clusters = num_clusters
        self.latent_dim = latent_dim or num_clusters
        self.eps = eps

        self.mu = nn.Parameter(torch.randn(num_clusters, self.latent_dim) * 0.5)
        self.sigma_tilde = nn.Parameter(
            torch.full((num_clusters, self.latent_dim), inverse_softplus(init_sigma), dtype=torch.float32)
        )

    @property
    def sigma(self) -> Tensor:
        return F.softplus(self.sigma_tilde)

    def forward(self, z: Tensor) -> Tensor:
        return _GaussianRBFFn.apply(
            z,
            self.mu.to(dtype=z.dtype),
            self.sigma_tilde.to(dtype=z.dtype),
            self.eps,
        )

    def forward_reference(self, z: Tensor) -> Tensor:
        mu = self.mu.to(dtype=z.dtype)
        sigma = F.softplus(self.sigma_tilde.to(dtype=z.dtype))
        denom = sigma * sigma + self.eps
        diff = z.unsqueeze(1) - mu.unsqueeze(0)
        return torch.exp(-0.5 * ((diff * diff) / denom.unsqueeze(0)).sum(dim=-1))
