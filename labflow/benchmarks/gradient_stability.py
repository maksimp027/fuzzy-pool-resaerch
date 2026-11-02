"""Gradient stability, Jacobian regularity, and dead super-edge benchmark test for graph reduction methods.

Evaluates backpropagation dynamics, active topological Jacobian entries, elimination of
Dead Super-Edges (e.g. Softplus vs hard ReLU in Lukasiewicz), and gradient finiteness.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Sequence

import torch

from labflow.analytics.metrics import (
    check_parameter_gradient_finite,
    measure_empirical_jacobian_density_and_dead_edges,
)
from labflow.composite import CANONICAL_METHOD_SPECS, METHOD_SPEC_MAP, GraphReductionLayer
from labflow.datasets.synthetic import generate_sbm_graph
from labflow.types import GraphData


@dataclass
class GradientStabilityBenchmarkRecord:
    """Scientific record for Gradient Dynamics & Optimization Health Test."""

    key: str
    code: str
    title: str
    topo_jacobian_density: float
    dead_superedge_ratio: float
    param_grad_finite_ratio: float
    mean_gradient_norm: float
    gradient_vanishing_resistance: float
    ms_per_step: float

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


def run_gradient_stability_benchmark(
    method_keys: Optional[Sequence[str]] = None,
    seed: int = 42,
    graph: Optional[GraphData] = None,
) -> List[GradientStabilityBenchmarkRecord]:
    """Execute dedicated Gradient Dynamics & Optimization Health Benchmark Test."""
    keys = list(method_keys) if method_keys is not None else [s.key for s in CANONICAL_METHOD_SPECS]

    if graph is None:
        graph = generate_sbm_graph(
            block_sizes=(15, 15, 15),
            p_in=0.68,
            p_out=0.04,
            feature_dim=16,
            seed=seed,
        )

    k = 3
    f_dim = graph.num_features

    records: List[GradientStabilityBenchmarkRecord] = []
    for key in keys:
        spec = METHOD_SPEC_MAP[key]
        torch.manual_seed(seed)
        layer = GraphReductionLayer.from_spec(
            key,
            in_features=f_dim,
            out_features=f_dim,
            num_clusters=k,
        )

        t0 = time.perf_counter()
        # Measure empirical Jacobian density and dead super-edges
        j_density, dead_ratio = measure_empirical_jacobian_density_and_dead_edges(layer, graph)

        # Execute full forward-backward pass to check gradient finiteness and norm
        layer.zero_grad()
        out = layer(graph)
        loss = out.x_new.sum() + out.adj_new.sum() + out.aux_loss
        loss.backward()
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        grad_finite = check_parameter_gradient_finite(layer)

        # Gradient norms across parameters
        grads = [p.grad for p in layer.parameters() if p.grad is not None and torch.isfinite(p.grad).all()]
        if grads:
            grad_norms = [torch.linalg.norm(g).item() for g in grads]
            mean_norm = float(sum(grad_norms) / len(grad_norms))
            min_norm = float(min(grad_norms))
            max_norm = float(max(grad_norms))
            # Resistance to vanishing: ratio of min grad to max grad (clipped to [0, 1])
            vanishing_res = float(min_norm / max(max_norm, 1e-12))
        else:
            mean_norm = 0.0
            vanishing_res = 0.0

        records.append(
            GradientStabilityBenchmarkRecord(
                key=key,
                code=spec.code,
                title=spec.title,
                topo_jacobian_density=j_density,
                dead_superedge_ratio=dead_ratio,
                param_grad_finite_ratio=1.0 if grad_finite else 0.0,
                mean_gradient_norm=mean_norm,
                gradient_vanishing_resistance=vanishing_res,
                ms_per_step=elapsed_ms,
            )
        )

    return records
