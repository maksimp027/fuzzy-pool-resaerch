"""Base preprocessing pipeline and GraphSAGE/GCN operators for H and Z generation.

Implements both dense O(N^2) and true sparse O(M F + N F F') pipelines:
    A_hat = D_tilde^{-1/2} (A + I_N) D_tilde^{-1/2},  D_tilde_{ii} = 1 + sum_j A_{ij}
    H = psi(A_hat X W_{msg,e} + X W_{self,e} + 1_N b_e^T)
    Z = A_hat X W_{msg,p} + X W_{self,p} + 1_N b_p^T
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


def symmetric_normalize_with_self_loops(adj: Tensor, eps: float = 1e-12) -> Tensor:
    """Compute dense A_hat = D_tilde^{-1/2} (A + I_N) D_tilde^{-1/2}, D_tilde_{ii} = 1 + sum_j A_{ij}."""
    n = adj.shape[0]
    eye = torch.eye(n, dtype=adj.dtype, device=adj.device)
    a_tilde = adj + eye
    deg_tilde = 1.0 + adj.sum(dim=1)
    inv_sqrt_deg = torch.rsqrt(deg_tilde.clamp(min=eps))
    return inv_sqrt_deg.unsqueeze(1) * a_tilde * inv_sqrt_deg.unsqueeze(0)


def symmetric_normalize_adjacency(adj: Tensor, eps: float = 1e-12) -> Tensor:
    """Compute dense A_tilde = D^{-1/2} A D^{-1/2} without adding self-loops (used by MinCutPool)."""
    deg = adj.sum(dim=1)
    inv_sqrt_deg = torch.where(deg > 0.0, torch.rsqrt(deg.clamp(min=eps)), torch.zeros_like(deg))
    return inv_sqrt_deg.unsqueeze(1) * adj * inv_sqrt_deg.unsqueeze(0)


def sparse_symmetric_normalize_with_self_loops(
    edge_index: Tensor,
    edge_weight: Tensor,
    num_nodes: int,
    eps: float = 1e-12,
) -> Tuple[Tensor, Tensor]:
    """Compute sparse COO (ei_hat, ew_hat) of A_hat = D_tilde^{-1/2} (A + I_N) D_tilde^{-1/2} in O(M + N)."""
    device = edge_weight.device
    dtype = edge_weight.dtype

    deg = torch.zeros((num_nodes,), dtype=dtype, device=device)
    if edge_weight.numel() > 0:
        deg.index_add_(0, edge_index[0], edge_weight)
    deg_tilde = 1.0 + deg
    inv_sqrt_deg = torch.rsqrt(deg_tilde.clamp(min=eps))

    self_nodes = torch.arange(num_nodes, dtype=torch.long, device=device)
    self_ei = torch.stack([self_nodes, self_nodes], dim=0)
    self_ew = torch.ones((num_nodes,), dtype=dtype, device=device)

    if edge_weight.numel() > 0:
        ei_tilde = torch.cat([edge_index, self_ei], dim=1)
        ew_tilde = torch.cat([edge_weight, self_ew], dim=0)
    else:
        ei_tilde = self_ei
        ew_tilde = self_ew

    ew_hat = inv_sqrt_deg[ei_tilde[0]] * ew_tilde * inv_sqrt_deg[ei_tilde[1]]
    return ei_tilde, ew_hat


def sparse_symmetric_normalize_adjacency(
    edge_index: Tensor,
    edge_weight: Tensor,
    num_nodes: int,
    eps: float = 1e-12,
) -> Tensor:
    """Compute normalized edge weights ew_tilde for A_tilde = D^{-1/2} A D^{-1/2} in O(M + N)."""
    device = edge_weight.device
    dtype = edge_weight.dtype
    if edge_weight.numel() == 0:
        return edge_weight.clone()

    deg = torch.zeros((num_nodes,), dtype=dtype, device=device)
    deg.index_add_(0, edge_index[0], edge_weight)
    inv_sqrt_deg = torch.where(deg > 0.0, torch.rsqrt(deg.clamp(min=eps)), torch.zeros_like(deg))
    return inv_sqrt_deg[edge_index[0]] * edge_weight * inv_sqrt_deg[edge_index[1]]


class _GraphSAGELinearFn(torch.autograd.Function):
    """Exact analytical forward/backward for dense U = A_hat X W_msg + X W_self + 1_N b^T."""

    @staticmethod
    def forward(
        ctx,
        a_hat: Tensor,
        x: Tensor,
        w_msg: Tensor,
        w_self: Tensor,
        bias: Tensor,
    ) -> Tensor:
        ax = a_hat @ x
        u = ax @ w_msg + x @ w_self + bias.unsqueeze(0)
        ctx.save_for_backward(a_hat, x, ax, w_msg, w_self)
        return u

    @staticmethod
    def backward(ctx, grad_u: Tensor) -> Tuple[Optional[Tensor], ...]:
        a_hat, x, ax, w_msg, w_self = ctx.saved_tensors
        grad_a_hat = grad_x = grad_w_msg = grad_w_self = grad_bias = None

        grad_msg_x = grad_u @ w_msg.t()
        if ctx.needs_input_grad[0]:
            grad_a_hat = grad_msg_x @ x.t()
        if ctx.needs_input_grad[1]:
            grad_x = a_hat.t() @ grad_msg_x + grad_u @ w_self.t()
        if ctx.needs_input_grad[2]:
            grad_w_msg = ax.t() @ grad_u
        if ctx.needs_input_grad[3]:
            grad_w_self = x.t() @ grad_u
        if ctx.needs_input_grad[4]:
            grad_bias = grad_u.sum(dim=0)

        return grad_a_hat, grad_x, grad_w_msg, grad_w_self, grad_bias


class _SparseGraphSAGELinearFn(torch.autograd.Function):
    """Exact O(M F + N F F') analytical forward/backward for sparse U = A_hat X W_msg + X W_self + 1_N b^T."""

    @staticmethod
    def forward(
        ctx,
        ei_hat: Tensor,
        ew_hat: Tensor,
        x: Tensor,
        w_msg: Tensor,
        w_self: Tensor,
        bias: Tensor,
    ) -> Tensor:
        n, f_in = x.shape
        ax = torch.zeros((n, f_in), dtype=x.dtype, device=x.device)
        if ew_hat.numel() > 0:
            src, dst = ei_hat[0], ei_hat[1]
            ax.index_add_(0, src, ew_hat.unsqueeze(1) * x[dst])
        u = ax @ w_msg + x @ w_self + bias.unsqueeze(0)
        ctx.save_for_backward(ei_hat, ew_hat, x, ax, w_msg, w_self)
        return u

    @staticmethod
    def backward(ctx, grad_u: Tensor) -> Tuple[None, Optional[Tensor], Optional[Tensor], Optional[Tensor], Optional[Tensor], Optional[Tensor]]:
        ei_hat, ew_hat, x, ax, w_msg, w_self = ctx.saved_tensors
        grad_ew_hat = grad_x = grad_w_msg = grad_w_self = grad_bias = None

        grad_msg_x = grad_u @ w_msg.t()
        if ctx.needs_input_grad[1]:
            if ew_hat.numel() > 0:
                src, dst = ei_hat[0], ei_hat[1]
                grad_ew_hat = (grad_msg_x[src] * x[dst]).sum(dim=1)
            else:
                grad_ew_hat = torch.zeros_like(ew_hat)

        if ctx.needs_input_grad[2]:
            at_grad = torch.zeros_like(x)
            if ew_hat.numel() > 0:
                src, dst = ei_hat[0], ei_hat[1]
                at_grad.index_add_(0, dst, ew_hat.unsqueeze(1) * grad_msg_x[src])
            grad_x = at_grad + grad_u @ w_self.t()

        if ctx.needs_input_grad[3]:
            grad_w_msg = ax.t() @ grad_u
        if ctx.needs_input_grad[4]:
            grad_w_self = x.t() @ grad_u
        if ctx.needs_input_grad[5]:
            grad_bias = grad_u.sum(dim=0)

        return None, grad_ew_hat, grad_x, grad_w_msg, grad_w_self, grad_bias


class GraphSAGEOperator(nn.Module):
    """GraphSAGE / GCN message+self operator: psi(A_hat X W_msg + X W_self + 1_N b^T)."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        activation: str = "relu",
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.activation_name = activation.lower()

        self.w_msg = nn.Parameter(torch.empty(in_features, out_features))
        self.w_self = nn.Parameter(torch.empty(in_features, out_features))
        self.bias = nn.Parameter(torch.zeros(out_features))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.w_msg)
        nn.init.xavier_uniform_(self.w_self)
        nn.init.zeros_(self.bias)

    def _apply_activation(self, u: Tensor) -> Tensor:
        if self.activation_name in ("none", "identity", "linear"):
            return u
        if self.activation_name == "relu":
            return F.relu(u)
        if self.activation_name == "elu":
            return F.elu(u)
        if self.activation_name == "leaky_relu":
            return F.leaky_relu(u, negative_slope=0.1)
        if self.activation_name == "tanh":
            return torch.tanh(u)
        if self.activation_name == "silu":
            return F.silu(u)
        raise ValueError(f"Unsupported activation: {self.activation_name}")

    def forward(
        self,
        adj: Optional[Tensor],
        x: Tensor,
        prenormalized: bool = False,
        edge_index: Optional[Tensor] = None,
        edge_weight: Optional[Tensor] = None,
    ) -> Tensor:
        if adj is None:
            if edge_index is None or edge_weight is None:
                raise ValueError("Provide either dense `adj` or sparse `(edge_index, edge_weight)`.")
            return self.forward_sparse(edge_index, edge_weight, x, prenormalized=prenormalized)
        a_hat = adj if prenormalized else symmetric_normalize_with_self_loops(adj)
        u = _GraphSAGELinearFn.apply(a_hat, x, self.w_msg, self.w_self, self.bias)
        return self._apply_activation(u)

    def forward_sparse(
        self,
        edge_index: Tensor,
        edge_weight: Tensor,
        x: Tensor,
        prenormalized: bool = False,
    ) -> Tensor:
        """True O(M F + N F F') sparse message passing without materializing N x N adjacency."""
        n = x.shape[0]
        if prenormalized:
            ei_hat, ew_hat = edge_index, edge_weight
        else:
            ei_hat, ew_hat = sparse_symmetric_normalize_with_self_loops(edge_index, edge_weight, num_nodes=n)
        u = _SparseGraphSAGELinearFn.apply(ei_hat, ew_hat, x, self.w_msg, self.w_self, self.bias)
        return self._apply_activation(u)

    def forward_reference(self, adj: Tensor, x: Tensor, prenormalized: bool = False) -> Tensor:
        """Pure PyTorch autograd reference forward pass."""
        a_hat = adj if prenormalized else symmetric_normalize_with_self_loops(adj)
        u = (a_hat @ x) @ self.w_msg + x @ self.w_self + self.bias.unsqueeze(0)
        return self._apply_activation(u)

    def forward_sparse_reference(
        self,
        edge_index: Tensor,
        edge_weight: Tensor,
        x: Tensor,
        prenormalized: bool = False,
    ) -> Tensor:
        """Pure PyTorch autograd sparse reference forward pass."""
        n, f_in = x.shape
        if prenormalized:
            ei_hat, ew_hat = edge_index, edge_weight
        else:
            ei_hat, ew_hat = sparse_symmetric_normalize_with_self_loops(edge_index, edge_weight, num_nodes=n)
        ax = torch.zeros((n, f_in), dtype=x.dtype, device=x.device)
        if ew_hat.numel() > 0:
            ax = ax.index_add(0, ei_hat[0], ew_hat.unsqueeze(1) * x[ei_hat[1]])
        u = ax @ self.w_msg + x @ self.w_self + self.bias.unsqueeze(0)
        return self._apply_activation(u)


class BaseGraphEncoder(nn.Module):
    """Dual operator generating H = GNN_embed(A, X) and Z = GNN_pool(A, X)."""

    def __init__(
        self,
        in_features: int,
        embed_dim: int,
        pool_dim: int,
        embed_activation: str = "elu",
    ) -> None:
        super().__init__()
        self.gnn_embed = GraphSAGEOperator(in_features, embed_dim, activation=embed_activation)
        self.gnn_pool = GraphSAGEOperator(in_features, pool_dim, activation="identity")

    def forward(
        self,
        adj: Optional[Tensor],
        x: Tensor,
        edge_index: Optional[Tensor] = None,
        edge_weight: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        if adj is None:
            if edge_index is None or edge_weight is None:
                raise ValueError("Provide either dense `adj` or sparse `(edge_index, edge_weight)`.")
            return self.forward_sparse(edge_index, edge_weight, x)
        a_hat = symmetric_normalize_with_self_loops(adj)
        h = self.gnn_embed(a_hat, x, prenormalized=True)
        z = self.gnn_pool(a_hat, x, prenormalized=True)
        return h, z

    def forward_sparse(self, edge_index: Tensor, edge_weight: Tensor, x: Tensor) -> Tuple[Tensor, Tensor]:
        """True O(M F + N F F') sparse dual encoder without materializing N x N adjacency."""
        ei_hat, ew_hat = sparse_symmetric_normalize_with_self_loops(edge_index, edge_weight, num_nodes=x.shape[0])
        h = self.gnn_embed.forward_sparse(ei_hat, ew_hat, x, prenormalized=True)
        z = self.gnn_pool.forward_sparse(ei_hat, ew_hat, x, prenormalized=True)
        return h, z

    def forward_reference(self, adj: Tensor, x: Tensor) -> Tuple[Tensor, Tensor]:
        a_hat = symmetric_normalize_with_self_loops(adj)
        h = self.gnn_embed.forward_reference(a_hat, x, prenormalized=True)
        z = self.gnn_pool.forward_reference(a_hat, x, prenormalized=True)
        return h, z
