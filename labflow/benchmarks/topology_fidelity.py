"""Topological fidelity, spectral preservation, and sparsity benchmark test for graph reduction methods.

Evaluates how faithfully a reduced graph G_new preserves original topology,
link connectivity, Laplacian spectral structure, unit interval compliance, and sparsity.
"""

from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from labflow.analytics.metrics import (
    compute_link_reconstruction_auc_rmse,
    compute_spectral_laplacian_distance,
)
from labflow.composite import CANONICAL_METHOD_SPECS, METHOD_SPEC_MAP, GraphReductionLayer
from labflow.datasets.synthetic import generate_sbm_graph
from labflow.types import GraphData


@dataclass
class TopologyFidelityBenchmarkRecord:
    """Scientific record for Topological Fidelity & Sparsity Test."""

    key: str
    code: str
    title: str
    link_recon_auc: float
    link_recon_rmse: float
    spectral_laplacian_dist: float
    exact_sparsity: float
    effective_sparsity: float
    unit_interval_compliance: float
    adj_new_max: float
    diag_offdiag_contrast: float
    ms_per_step: float

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


def run_topology_fidelity_benchmark(
    method_keys: Optional[Sequence[str]] = None,
    epochs: int = 70,
    lr: float = 0.025,
    seed: int = 42,
    graph: Optional[GraphData] = None,
) -> List[TopologyFidelityBenchmarkRecord]:
    """Execute dedicated Topological Fidelity & Sparsity Benchmark Test."""
    keys = list(method_keys) if method_keys is not None else [s.key for s in CANONICAL_METHOD_SPECS]

    if graph is None:
        graph = generate_sbm_graph(
            block_sizes=(15, 15, 15),
            p_in=0.68,
            p_out=0.04,
            feature_dim=16,
            seed=seed,
        )

    adj = graph.dense_adj()
    x = graph.x
    k = 3
    f_dim = graph.num_features

    records: List[TopologyFidelityBenchmarkRecord] = []
    for key in keys:
        spec = METHOD_SPEC_MAP[key]
        torch.manual_seed(seed)
        layer = GraphReductionLayer.from_spec(
            key,
            in_features=f_dim,
            out_features=f_dim,
            num_clusters=k,
        )
        dec_x = nn.Linear(f_dim, f_dim, bias=False)
        optimizer = torch.optim.Adam(list(layer.parameters()) + list(dec_x.parameters()), lr=lr)

        t0 = time.perf_counter()
        for _ in range(epochs):
            optimizer.zero_grad()
            out = layer(graph)
            s = out.s
            s_row_norm = s / (s.sum(dim=1, keepdim=True) + 1e-8)

            x_hat = s_row_norm @ dec_x(out.x_new)
            loss_feat = F.mse_loss(x_hat, x)

            a_scale = out.adj_new.detach().abs().max().clamp(min=1.0)
            a_norm = out.adj_new / a_scale if layer.topology_name == "diffpool" else out.adj_new
            a_hat = (s_row_norm @ a_norm @ s_row_norm.t()).clamp(0.0, 1.0)
            loss_topo = F.mse_loss(a_hat, adj)

            q_mat = s.t() @ s
            q_norm = torch.linalg.norm(q_mat, ord="fro").clamp(min=1e-8)
            eye_k = torch.eye(k, dtype=s.dtype, device=s.device) / math.sqrt(float(k))
            loss_orth = torch.linalg.norm(q_mat / q_norm - eye_k, ord="fro")

            if layer.is_classical:
                loss = out.aux_loss + 0.4 * loss_feat + 0.2 * loss_topo
            else:
                loss = loss_topo + 0.5 * loss_feat + 0.35 * loss_orth

            loss.backward()
            optimizer.step()

        elapsed_ms = (time.perf_counter() - t0) * 1000.0 / max(epochs, 1)

        with torch.no_grad():
            final_out = layer(graph)
            a_new = final_out.adj_new.detach()
            s = final_out.s.detach()

            # Link reconstruction and spectral fidelity
            edge_index, edge_weight = graph.sparse_edges()
            auc_val, rmse_val = compute_link_reconstruction_auc_rmse(adj, a_new, s)
            lap_dist = compute_spectral_laplacian_distance(
                edge_index=edge_index,
                edge_weight=edge_weight,
                num_nodes=graph.num_nodes,
                adj_new=a_new,
            )

            # Sparsity metrics
            exact_sp = float((a_new == 0.0).to(torch.float32).mean().item())
            eff_sp = float((a_new < 1e-3).to(torch.float32).mean().item())

            # Unit interval [0, 1] compliance
            in_unit = (a_new >= -1e-6) & (a_new <= 1.0 + 1e-6)
            unit_compliance = float(in_unit.to(torch.float32).mean().item())
            adj_max = float(a_new.max().item())

            # Contrast between diagonal self-loops and off-diagonal super-edges
            diag = torch.diagonal(a_new)
            off_mask = ~torch.eye(k, dtype=torch.bool, device=a_new.device)
            diag_mean = float(diag.mean().item())
            off_mean = float(a_new[off_mask].mean().item()) if off_mask.any() else 1e-6
            contrast = float(diag_mean / max(off_mean, 1e-6))

        records.append(
            TopologyFidelityBenchmarkRecord(
                key=key,
                code=spec.code,
                title=spec.title,
                link_recon_auc=auc_val,
                link_recon_rmse=rmse_val,
                spectral_laplacian_dist=lap_dist,
                exact_sparsity=exact_sp,
                effective_sparsity=eff_sp,
                unit_interval_compliance=unit_compliance,
                adj_new_max=adj_max,
                diag_offdiag_contrast=contrast,
                ms_per_step=elapsed_ms,
            )
        )

    return records
