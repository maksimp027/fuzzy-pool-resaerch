"""Method 5: Algebraic Goguen FuzzyPool (max-product composition).

Forward Pass (2-step sparse decomposition):
    Step 1: T_{iJ}       = max_{j in N(i)} (A_{ij} * S_{jJ})
    Step 2: A_new(I, J)  = max_{i in {1..N}} (S_{iI} * T_{iJ})

Analytical Backward Pass:
    dA_new(I, J) / dS_{kL} = delta_{IL} delta_{k, i*_{IJ}} (A_{i* j*} S_{j* J})
                           + delta_{JL} delta_{k, j*_{IJ}} (S_{i* I} A_{i* j*})
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from labflow.fuzzy_relational.sparse_utils import (
    accumulate_winner_gradients,
    sparse_two_step_max_with_winners,
)
from labflow.types import dense_to_sparse


class _GoguenMaxProdSparseFn(torch.autograd.Function):
    """Exact analytical forward and backward pass for Goguen max-product topology reduction."""

    @staticmethod
    def forward(
        ctx,
        s: Tensor,
        edge_index: Tensor,
        edge_weight: Tensor,
    ) -> Tensor:
        dst = edge_index[1]
        u_ej = (edge_weight.unsqueeze(1) * s[dst]) if edge_weight.numel() > 0 else s.new_empty((0, s.shape[1]))
        a_new, _, e_star = sparse_two_step_max_with_winners(
            s=s,
            edge_index=edge_index,
            u_ej=u_ej,
            step2_combine=torch.mul,
        )
        ctx.save_for_backward(s, edge_index, edge_weight, e_star)
        return a_new

    @staticmethod
    def backward(
        ctx,
        grad_a_new: Tensor,
    ) -> Tuple[Optional[Tensor], None, Optional[Tensor]]:
        s, edge_index, edge_weight, e_star = ctx.saved_tensors
        n, k = s.shape
        m = int(edge_weight.numel())

        if m == 0:
            grad_s = torch.zeros_like(s) if ctx.needs_input_grad[0] else None
            grad_ew = torch.zeros_like(edge_weight) if ctx.needs_input_grad[2] else None
            return grad_s, None, grad_ew

        src = edge_index[0]
        dst = edge_index[1]
        i_star = src[e_star]
        j_star = dst[e_star]

        i_idx = torch.arange(k, device=s.device).unsqueeze(1).expand(k, k)
        j_idx = torch.arange(k, device=s.device).unsqueeze(0).expand(k, k)

        s_left = s[i_star, i_idx]
        s_right = s[j_star, j_idx]
        a_val = edge_weight[e_star]

        contrib_left = grad_a_new * (a_val * s_right)
        contrib_right = grad_a_new * (s_left * a_val)
        contrib_edge = grad_a_new * (s_left * s_right) if ctx.needs_input_grad[2] else None

        grad_s_calc, grad_ew = accumulate_winner_gradients(
            num_nodes=n,
            num_clusters=k,
            num_edges=m,
            edge_index=edge_index,
            e_star=e_star,
            contrib_left=contrib_left,
            contrib_right=contrib_right,
            contrib_edge=contrib_edge,
        )
        grad_s = grad_s_calc if ctx.needs_input_grad[0] else None
        return grad_s, None, grad_ew


class GoguenTopologyReducer(nn.Module):
    """Method 5: Goguen max-product fuzzy relational topology reducer."""

    def forward(
        self,
        s: Tensor,
        adj: Optional[Tensor] = None,
        edge_index: Optional[Tensor] = None,
        edge_weight: Optional[Tensor] = None,
    ) -> Tensor:
        if edge_index is None or edge_weight is None:
            if adj is None:
                raise ValueError("Provide either `adj` or `(edge_index, edge_weight)`.")
            edge_index, edge_weight = dense_to_sparse(adj, threshold=0.0)
        return _GoguenMaxProdSparseFn.apply(s, edge_index, edge_weight)

    def forward_dense_reference(self, adj: Tensor, s: Tensor) -> Tensor:
        """Dense O(N^2 K^2) reference: max_{i,j} (S_{iI} * A_{ij} * S_{jJ})."""
        s_i = s[:, None, :, None]
        a_ij = adj[:, :, None, None]
        s_j = s[None, :, None, :]
        tri = s_i * a_ij * s_j
        return tri.amax(dim=(0, 1))
