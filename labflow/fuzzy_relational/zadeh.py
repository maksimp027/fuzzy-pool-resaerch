"""Method 4: Classical Zadeh-Godel FuzzyPool (max-min composition).

Forward Pass (2-step sparse decomposition):
    Step 1: T_{iJ}       = max_{j in N(i)} min(A_{ij}, S_{jJ})
    Step 2: A_new(I, J)  = max_{i in {1..N}} min(S_{iI}, T_{iJ})

Analytical Subgradient:
    For winning edge (i*_{IJ}, j*_{IJ}) = argmax_{(i,j) in E} min(S_{iI}, A_{ij}, S_{jJ}),
    active lower-bound indicator masks out non-bottleneck terms (Edge Bottleneck Masking),
    with symmetric tie-splitting when S_{i* I} == S_{j* J} (e.g., diagonal self-loop height).
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


class _ZadehMaxMinSparseFn(torch.autograd.Function):
    """Exact analytical forward and subgradient backward pass for Zadeh-Godel max-min topology reduction."""

    @staticmethod
    def forward(
        ctx,
        s: Tensor,
        edge_index: Tensor,
        edge_weight: Tensor,
    ) -> Tensor:
        dst = edge_index[1]
        u_ej = torch.minimum(edge_weight.unsqueeze(1), s[dst]) if edge_weight.numel() > 0 else s.new_empty((0, s.shape[1]))
        a_new, _, e_star = sparse_two_step_max_with_winners(
            s=s,
            edge_index=edge_index,
            u_ej=u_ej,
            step2_combine=torch.minimum,
        )
        ctx.save_for_backward(s, edge_index, edge_weight, a_new, e_star)
        return a_new

    @staticmethod
    def backward(
        ctx,
        grad_a_new: Tensor,
    ) -> Tuple[Optional[Tensor], None, Optional[Tensor]]:
        s, edge_index, edge_weight, a_new, e_star = ctx.saved_tensors
        n, k = s.shape
        m = int(edge_weight.numel())

        if m == 0:
            grad_s = torch.zeros_like(s) if ctx.needs_input_grad[0] else None
            grad_ew = torch.zeros_like(edge_weight) if ctx.needs_input_grad[2] else None
            return grad_s, None, grad_ew

        src = edge_index[0]
        dst = edge_index[1]
        i_star = src[e_star]  # [K, K]
        j_star = dst[e_star]  # [K, K]

        i_idx = torch.arange(k, device=s.device).unsqueeze(1).expand(k, k)
        j_idx = torch.arange(k, device=s.device).unsqueeze(0).expand(k, k)

        s_left = s[i_star, i_idx]
        s_right = s[j_star, j_idx]
        a_val = edge_weight[e_star]

        # Active lower-bound indicators; when unique, coincides with strict inequality
        # I(S_{i* I} < min(A_{i* j*}, S_{j* J})), while splitting ties (e.g., i*=j* on diagonal self-loops)
        is_left = (s_left == a_new).to(s.dtype)
        is_edge = (a_val == a_new).to(s.dtype)
        is_right = (s_right == a_new).to(s.dtype)
        tie_count = (is_left + is_edge + is_right).clamp(min=1.0)

        w_left = is_left / tie_count
        w_edge = is_edge / tie_count
        w_right = is_right / tie_count

        contrib_left = grad_a_new * w_left
        contrib_right = grad_a_new * w_right
        contrib_edge = grad_a_new * w_edge if ctx.needs_input_grad[2] else None

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


class ZadehTopologyReducer(nn.Module):
    """Method 4: Zadeh-Godel max-min fuzzy relational topology reducer."""

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
        return _ZadehMaxMinSparseFn.apply(s, edge_index, edge_weight)

    def forward_dense_reference(self, adj: Tensor, s: Tensor) -> Tensor:
        """Dense O(N^2 K^2) reference: max_{i,j} min(S_{iI}, A_{ij}, S_{jJ})."""
        # [N, 1, K, 1], [N, N, 1, 1], [1, N, 1, K] -> [N, N, K, K]
        s_i = s[:, None, :, None]
        a_ij = adj[:, :, None, None]
        s_j = s[None, :, None, :]
        tri = torch.minimum(torch.minimum(s_i, a_ij), s_j)
        return tri.amax(dim=(0, 1))
