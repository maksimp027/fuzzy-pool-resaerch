"""Core data structures and utility conversions for graph reduction in Labflow."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import torch
from torch import Tensor


@dataclass
class GraphData:
    """Container for an input or reduced graph G = (V, E, A, X).

    Supports both dense adjacency matrices A in [0, 1]^{N x N} and sparse COO
    representations (edge_index, edge_weight). Conversions are cached lazily.
    """

    x: Tensor
    adj: Optional[Tensor] = None
    edge_index: Optional[Tensor] = None
    edge_weight: Optional[Tensor] = None
    y: Optional[Tensor] = None
    node_labels: Optional[Tensor] = None
    metadata: Dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.adj is None and (self.edge_index is None or self.edge_weight is None):
            raise ValueError("Either `adj` or both `edge_index` and `edge_weight` must be provided.")

    @property
    def num_nodes(self) -> int:
        return int(self.x.shape[0])

    @property
    def num_features(self) -> int:
        return int(self.x.shape[1])

    @property
    def num_edges(self) -> int:
        if self.edge_weight is not None:
            return int(self.edge_weight.numel())
        assert self.adj is not None
        return int((self.adj > 0.0).sum().item())

    @property
    def device(self) -> torch.device:
        return self.x.device

    @property
    def dtype(self) -> torch.dtype:
        return self.x.dtype

    def dense_adj(self) -> Tensor:
        """Return dense adjacency matrix A in R^{N x N}."""
        if self.adj is not None:
            return self.adj
        assert self.edge_index is not None and self.edge_weight is not None
        n = self.num_nodes
        adj = torch.zeros((n, n), dtype=self.edge_weight.dtype, device=self.device)
        adj.index_put_((self.edge_index[0], self.edge_index[1]), self.edge_weight, accumulate=True)
        return adj

    def sparse_edges(self, threshold: float = 0.0) -> Tuple[Tensor, Tensor]:
        """Return (edge_index [2, M], edge_weight [M]) for edges with weight > threshold."""
        if self.edge_index is not None and self.edge_weight is not None and threshold <= 0.0:
            return self.edge_index, self.edge_weight
        adj = self.dense_adj()
        return dense_to_sparse(adj, threshold=threshold)

    def to(self, device: Optional[torch.device | str] = None, dtype: Optional[torch.dtype] = None) -> "GraphData":
        x = self.x.to(device=device, dtype=dtype)
        adj = self.adj.to(device=device, dtype=dtype) if self.adj is not None else None
        edge_index = self.edge_index.to(device=device) if self.edge_index is not None else None
        edge_weight = self.edge_weight.to(device=device, dtype=dtype) if self.edge_weight is not None else None
        y = self.y.to(device=device) if self.y is not None else None
        node_labels = self.node_labels.to(device=device) if self.node_labels is not None else None
        return GraphData(
            x=x,
            adj=adj,
            edge_index=edge_index,
            edge_weight=edge_weight,
            y=y,
            node_labels=node_labels,
            metadata=dict(self.metadata),
        )


@dataclass
class PoolingOutput:
    """Standardized output of any graph reduction / pooling operator."""

    x_new: Tensor
    adj_new: Tensor
    s: Tensor
    h: Optional[Tensor] = None
    z: Optional[Tensor] = None
    aux_loss: Tensor = field(default_factory=lambda: torch.tensor(0.0))
    loss_components: Dict[str, Tensor] = field(default_factory=dict)
    diagnostics: Dict[str, object] = field(default_factory=dict)

    def to_sparse_coo(self) -> Tensor:
        """Convert exact-sparse reduced adjacency (e.g., Lukasiewicz) to PyTorch sparse COO tensor."""
        import warnings

        edge_index, edge_weight = dense_to_sparse(self.adj_new, threshold=0.0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            return torch.sparse_coo_tensor(
                edge_index,
                edge_weight,
                size=self.adj_new.shape,
                dtype=self.adj_new.dtype,
                device=self.adj_new.device,
            ).coalesce()

    def to_sparse_csr(self) -> Tensor:
        """Convert reduced adjacency to PyTorch sparse CSR tensor without arbitrary thresholding."""
        import warnings

        coo = self.to_sparse_coo()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            return coo.to_sparse_csr()


def dense_to_sparse(adj: Tensor, threshold: float = 0.0) -> Tuple[Tensor, Tensor]:
    """Extract (edge_index, edge_weight) in lexicographical (i, j) order where adj > threshold."""
    mask = adj > threshold
    indices = mask.nonzero(as_tuple=False)  # [M, 2], lexicographically sorted by row then col
    if indices.numel() == 0:
        empty_idx = torch.empty((2, 0), dtype=torch.long, device=adj.device)
        empty_val = torch.empty((0,), dtype=adj.dtype, device=adj.device)
        return empty_idx, empty_val
    edge_index = indices.t().contiguous()
    edge_weight = adj[edge_index[0], edge_index[1]]
    return edge_index, edge_weight


def sparse_to_dense(edge_index: Tensor, edge_weight: Tensor, num_nodes: int) -> Tensor:
    """Materialize dense N x N adjacency from COO (edge_index, edge_weight)."""
    adj = torch.zeros((num_nodes, num_nodes), dtype=edge_weight.dtype, device=edge_weight.device)
    if edge_weight.numel() > 0:
        adj.index_put_((edge_index[0], edge_index[1]), edge_weight, accumulate=True)
    return adj
