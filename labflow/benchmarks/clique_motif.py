"""Structural motif and planted clique isolation benchmark test for graph reduction methods.

Evaluates universal quantifier (inf-implication / Bandler-Kohout) against existential
(sup-composition) and classical bilinear aggregation in detecting complete dense subgraphs.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch

from labflow.analytics.metrics import compute_deluca_termini_and_bezdek
from labflow.composite import CANONICAL_METHOD_SPECS, METHOD_SPEC_MAP, GraphReductionLayer
from labflow.datasets.synthetic import generate_planted_cliques_graph
from labflow.types import GraphData


@dataclass
class CliqueMotifBenchmarkRecord:
    """Scientific record for Structural Motif & Planted Clique Isolation Test."""

    key: str
    code: str
    title: str
    internal_clique_density: float
    clique_recovery_f1: float
    diagonal_reflexivity: float
    motif_contrast_ratio: float
    deluca_termini_entropy: float
    ms_per_step: float

    # Backward compatibility alias
    @property
    def mean_diag_a_new(self) -> float:
        return self.diagonal_reflexivity

    def to_dict(self) -> Dict[str, object]:
        d = asdict(self)
        d["mean_diag_a_new"] = self.diagonal_reflexivity
        return d


def run_clique_motif_benchmark(
    method_keys: Optional[Sequence[str]] = None,
    epochs: int = 120,
    lr: float = 0.05,
    seed: int = 7,
    clique_size: int = 6,
    num_cliques: int = 3,
    num_noise_nodes: int = 12,
    noise_edge_prob: float = 0.10,
) -> List[CliqueMotifBenchmarkRecord]:
    """Execute dedicated Structural Motif & Planted Clique Isolation Benchmark Test.

    Tests the universal quantifier Bandler-Kohout implication theorem:
        A_new(I, I) = 1.0 <=> supp(S_{:, I}) forms a complete clique.
    """
    default_keys = [
        "bk_kleene_dienes",
        "bk_lukasiewicz",
        "bk_reichenbach",
        "goguen",
        "zadeh",
        "lukasiewicz",
        "lukasiewicz_smooth",
        "boltzmann",
        "boltzmann_adaptive",
        "probsum",
        "probsum_norm",
        "mincut",
        "dmon",
    ]
    compare_keys = list(method_keys) if method_keys is not None else default_keys

    graph_raw = generate_planted_cliques_graph(
        num_cliques=num_cliques,
        clique_size=clique_size,
        num_noise_nodes=num_noise_nodes,
        noise_edge_prob=noise_edge_prob,
        feature_dim=12,
        seed=seed,
    )
    n_clq = graph_raw.num_nodes

    # Pure topological discovery using identity node features (no feature leakage)
    graph = GraphData(
        x=torch.eye(n_clq, dtype=torch.float32),
        adj=graph_raw.adj,
        edge_index=graph_raw.edge_index,
        edge_weight=graph_raw.edge_weight,
        node_labels=graph_raw.node_labels,
        metadata=graph_raw.metadata,
    )
    adj = graph.dense_adj()
    assert graph.node_labels is not None
    true_labels = graph.node_labels.cpu().numpy()
    planted_sets = [set(np.where(true_labels == c)[0].tolist()) for c in range(num_cliques)]

    records: List[CliqueMotifBenchmarkRecord] = []
    for key in compare_keys:
        spec = METHOD_SPEC_MAP[key]
        torch.manual_seed(seed)
        layer = GraphReductionLayer.from_spec(
            key,
            in_features=n_clq,
            out_features=12,
            num_clusters=num_cliques,
        )
        optimizer = torch.optim.Adam(layer.parameters(), lr=lr)

        t0 = time.perf_counter()
        for _ in range(epochs):
            optimizer.zero_grad()
            out = layer(graph)
            s = out.s

            if layer.is_classical:
                loss = out.aux_loss
            else:
                diag_clique_score = torch.diagonal(out.adj_new).mean()
                cluster_sizes = s.sum(dim=0)
                size_penalty = ((cluster_sizes - float(clique_size)) ** 2).mean() / float(clique_size ** 2)
                normality_penalty = (1.0 - s.amax(dim=0)).mean()
                fuzziness_penalty = (4.0 * s * (1.0 - s)).mean()
                overlap = s.t() @ s
                off_diag = overlap - torch.diag(torch.diagonal(overlap))
                overlap_penalty = (off_diag * off_diag).mean() / float(clique_size ** 2)
                loss = (
                    -diag_clique_score
                    + 1.2 * size_penalty
                    + 0.8 * normality_penalty
                    + 0.35 * fuzziness_penalty
                    + 1.0 * overlap_penalty
                )

            loss.backward()
            optimizer.step()

        elapsed_ms = (time.perf_counter() - t0) * 1000.0 / max(epochs, 1)

        with torch.no_grad():
            out = layer(graph)
            s_np = out.s.cpu().numpy()
            densities: List[float] = []
            f1_scores: List[float] = []
            adj_np = adj.cpu().numpy()

            for c in range(num_cliques):
                top_nodes = np.argsort(-s_np[:, c])[:clique_size]
                sub = adj_np[np.ix_(top_nodes, top_nodes)]
                max_edges = clique_size * (clique_size - 1)
                densities.append(float(sub.sum()) / float(max(max_edges, 1)))

                pred_set = set(top_nodes.tolist())
                best_f1 = 0.0
                for p_set in planted_sets:
                    inter = len(pred_set & p_set)
                    if inter > 0:
                        prec = inter / len(pred_set)
                        rec = inter / len(p_set)
                        best_f1 = max(best_f1, 2.0 * prec * rec / (prec + rec))
                f1_scores.append(best_f1)

            diag = torch.diagonal(out.adj_new)
            mean_diag = float(diag.mean().item())

            # Contrast ratio: mean diagonal clique strength vs mean off-diagonal super-edge weight
            off_diag_mask = ~torch.eye(num_cliques, dtype=torch.bool, device=out.adj_new.device)
            off_diag_mean = float(out.adj_new[off_diag_mask].mean().item()) if off_diag_mask.any() else 1e-6
            contrast_ratio = float(mean_diag / max(off_diag_mean, 1e-6))

            entropy_val, _ = compute_deluca_termini_and_bezdek(out.s)

        records.append(
            CliqueMotifBenchmarkRecord(
                key=key,
                code=spec.code,
                title=spec.title,
                internal_clique_density=float(np.mean(densities)),
                clique_recovery_f1=float(np.mean(f1_scores)),
                diagonal_reflexivity=mean_diag,
                motif_contrast_ratio=contrast_ratio,
                deluca_termini_entropy=entropy_val,
                ms_per_step=elapsed_ms,
            )
        )

    return records
