"""Community detection and clustering benchmark test for graph reduction methods.

Evaluates unsupervised cluster discovery, modularity, conductance, and partition alignment
on structured Stochastic Block Model (SBM) and classic real-world graphs.
"""

from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from labflow.analytics.metrics import (
    compute_ari_score,
    compute_nmi_score,
    compute_purity_and_macro_f1,
    compute_sparse_cut_and_modularity,
)
from labflow.composite import CANONICAL_METHOD_SPECS, METHOD_SPEC_MAP, GraphReductionLayer
from labflow.datasets.synthetic import generate_sbm_graph
from labflow.types import GraphData


@dataclass
class CommunityBenchmarkRecord:
    """Scientific record for Community Detection & Clustering Test."""

    key: str
    code: str
    title: str
    nmi: float
    ari: float
    modularity_q: float
    fuzzy_modularity_q: float
    conductance: float
    normalized_cut: float
    cluster_purity: float
    macro_f1: float
    adj_new_max: float
    ms_per_step: float

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


def run_community_detection_benchmark(
    method_keys: Optional[Sequence[str]] = None,
    epochs: int = 80,
    lr: float = 0.025,
    seed: int = 42,
    graph: Optional[GraphData] = None,
) -> List[CommunityBenchmarkRecord]:
    """Execute dedicated Community Detection & Partition Alignment Benchmark Test.

    Evaluates:
    - Homophily preservation (NMI, ARI, Purity, Macro-F1)
    - Structural cut quality (Discrete Modularity Q, Fuzzy Modularity, Conductance, Normalized Cut)
    """
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
    edge_index, edge_weight = graph.sparse_edges()
    assert graph.node_labels is not None
    true_labels = graph.node_labels.cpu().numpy()
    k = len(np.unique(true_labels))
    f_dim = graph.num_features

    records: List[CommunityBenchmarkRecord] = []
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

            # Reconstruct node features and relational topology
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
            pred = final_out.s.argmax(dim=1).cpu().numpy()

            nmi = compute_nmi_score(pred, true_labels)
            ari = compute_ari_score(pred, true_labels)
            purity, macro_f1 = compute_purity_and_macro_f1(pred, true_labels)
            hard_q, fuzzy_q, cond, ncut = compute_sparse_cut_and_modularity(
                edge_index, edge_weight, final_out.s, pred, num_clusters=k
            )
            adj_max = float(final_out.adj_new.max().item())

        records.append(
            CommunityBenchmarkRecord(
                key=key,
                code=spec.code,
                title=spec.title,
                nmi=nmi,
                ari=ari,
                modularity_q=hard_q,
                fuzzy_modularity_q=fuzzy_q,
                conductance=cond,
                normalized_cut=ncut,
                cluster_purity=purity,
                macro_f1=macro_f1,
                adj_new_max=adj_max,
                ms_per_step=elapsed_ms,
            )
        )

    return records
