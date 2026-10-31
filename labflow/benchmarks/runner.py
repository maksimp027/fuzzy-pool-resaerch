"""Experimental benchmarks evaluating all 13 graph reduction methods on synthetic tasks:
1. SBM Unsupervised Community & Relational Autoencoding Benchmark (NMI, Modularity, Gradient Sparsity)
2. Planted Clique Isolation Benchmark (Universal Quantifier Bandler-Kohout vs Existential & Classical)
3. End-to-End Hierarchical Graph Classification Benchmark (GNN -> Pooling -> Post-GNN -> Readout)
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
from torch import Tensor

from labflow.composite import CANONICAL_METHOD_SPECS, METHOD_SPEC_MAP, GraphReductionLayer
from labflow.datasets.synthetic import (
    generate_graph_classification_dataset,
    generate_planted_cliques_graph,
    generate_sbm_graph,
)
from labflow.gnn import GraphSAGEOperator
from labflow.types import GraphData, PoolingOutput


@dataclass
class SBMBenchmarkRecord:
    key: str
    code: str
    title: str
    nmi: float
    modularity_q: float
    conductance: float
    adj_new_max: float
    exact_sparsity: float
    topo_grad_active_ratio: float
    ms_per_step: float

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


@dataclass
class CliqueBenchmarkRecord:
    key: str
    code: str
    title: str
    internal_clique_density: float
    clique_recovery_f1: float
    mean_diag_a_new: float

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


@dataclass
class ClassificationBenchmarkRecord:
    key: str
    code: str
    title: str
    train_acc: float
    test_acc: float
    final_loss: float
    learned_params_summary: str

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


def compute_nmi(pred_labels: np.ndarray, true_labels: np.ndarray) -> float:
    """Compute Normalized Mutual Information (NMI) without requiring scikit-learn."""
    n = len(true_labels)
    if n == 0:
        return 0.0
    u_true, t_idx = np.unique(true_labels, return_inverse=True)
    u_pred, p_idx = np.unique(pred_labels, return_inverse=True)
    contingency = np.zeros((len(u_true), len(u_pred)), dtype=np.float64)
    for ti, pi in zip(t_idx, p_idx):
        contingency[ti, pi] += 1.0
    contingency /= float(n)

    p_t = contingency.sum(axis=1, keepdims=True)
    p_p = contingency.sum(axis=0, keepdims=True)

    nz = contingency > 0.0
    denom_mi = (p_t @ p_p)[nz]
    mi = float(np.sum(contingency[nz] * np.log(contingency[nz] / np.maximum(denom_mi, 1e-15))))

    h_t = -float(np.sum(p_t[p_t > 0] * np.log(p_t[p_t > 0])))
    h_p = -float(np.sum(p_p[p_p > 0] * np.log(p_p[p_p > 0])))
    if h_t + h_p < 1e-15:
        return 1.0
    return float(2.0 * mi / (h_t + h_p))


def compute_hard_modularity_and_conductance(adj: Tensor, pred_labels: np.ndarray, num_clusters: int) -> tuple[float, float]:
    """Compute discrete Newman-Girvan modularity Q and average cluster conductance."""
    a_np = adj.detach().cpu().numpy().astype(np.float64)
    deg = a_np.sum(axis=1)
    two_m = float(deg.sum())
    if two_m <= 1e-12:
        return 0.0, 1.0

    q = 0.0
    conductances: List[float] = []
    for c in range(num_clusters):
        mask = pred_labels == c
        if not np.any(mask):
            continue
        vol = float(deg[mask].sum())
        internal = float(a_np[np.ix_(mask, mask)].sum())
        cut = vol - internal
        q += (internal / two_m) - ((vol / two_m) ** 2)
        denom = min(vol, two_m - vol)
        if denom > 1e-12:
            conductances.append(cut / denom)
    avg_cond = float(np.mean(conductances)) if conductances else 1.0
    return float(q), avg_cond


def _measure_topo_grad_active_ratio(layer: GraphReductionLayer, graph: GraphData) -> float:
    """Measure fraction of active non-zero entries in (dA_new / dS) matching the Jacobian summary table."""
    x = graph.x
    edge_index, edge_weight = graph.sparse_edges()
    with torch.no_grad():
        h, z = layer.encoder.forward_sparse(edge_index, edge_weight, x)
        s0 = layer.membership(z)

    s_probe = s0.detach().clone().requires_grad_(True)
    if layer.is_classical:
        assert layer.classical_reducer is not None
        out = layer.classical_reducer(None, h.detach(), s_probe, edge_index=edge_index, edge_weight=edge_weight)
        out.adj_new.sum().backward()
    else:
        assert layer.topology_reducer is not None
        a_new = layer.topology_reducer(s_probe, edge_index=edge_index, edge_weight=edge_weight)
        a_new.sum().backward()

    assert s_probe.grad is not None
    active = (s_probe.grad.abs() > 1e-12).to(torch.float32).mean().item()
    return float(active)


def run_sbm_community_benchmark(
    method_keys: Optional[Sequence[str]] = None,
    epochs: int = 80,
    lr: float = 0.025,
    seed: int = 42,
) -> List[SBMBenchmarkRecord]:
    """Train all reduction layers unsupervised on an SBM graph and measure community & Jacobian metrics."""
    keys = list(method_keys) if method_keys is not None else [s.key for s in CANONICAL_METHOD_SPECS]
    graph = generate_sbm_graph(block_sizes=(15, 15, 15), p_in=0.68, p_out=0.04, feature_dim=16, seed=seed)
    adj = graph.dense_adj()
    x = graph.x
    assert graph.node_labels is not None
    true_labels = graph.node_labels.cpu().numpy()
    n = graph.num_nodes
    k = 3

    records: List[SBMBenchmarkRecord] = []
    for key in keys:
        spec = METHOD_SPEC_MAP[key]
        torch.manual_seed(seed)
        layer = GraphReductionLayer.from_spec(key, in_features=16, out_features=16, num_clusters=k)
        dec_x = nn.Linear(16, 16, bias=False)
        optimizer = torch.optim.Adam(list(layer.parameters()) + list(dec_x.parameters()), lr=lr)

        t0 = time.perf_counter()
        for _ in range(epochs):
            optimizer.zero_grad()
            out = layer(graph)
            s = out.s
            s_row_norm = s / (s.sum(dim=1, keepdim=True) + 1e-8)

            # Feature + relational topology reconstruction so both G_X and G_A drive learning
            x_hat = s_row_norm @ dec_x(out.x_new)
            loss_feat = F.mse_loss(x_hat, x)

            a_hat = (s_row_norm @ out.adj_new @ s_row_norm.t()).clamp(0.0, 1.0)
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
            nmi = compute_nmi(pred, true_labels)
            mod_q, cond = compute_hard_modularity_and_conductance(adj, pred, num_clusters=k)
            adj_max = float(final_out.adj_new.max().item())
            sparsity = float((final_out.adj_new == 0.0).to(torch.float32).mean().item())

        active_ratio = _measure_topo_grad_active_ratio(layer, graph)
        _ = n
        records.append(
            SBMBenchmarkRecord(
                key=key,
                code=spec.code,
                title=spec.title,
                nmi=nmi,
                modularity_q=mod_q,
                conductance=cond,
                adj_new_max=adj_max,
                exact_sparsity=sparsity,
                topo_grad_active_ratio=active_ratio,
                ms_per_step=elapsed_ms,
            )
        )

    return records


def run_planted_clique_benchmark(
    method_keys: Optional[Sequence[str]] = None,
    epochs: int = 120,
    lr: float = 0.05,
    seed: int = 7,
) -> List[CliqueBenchmarkRecord]:
    """Compare Bandler-Kohout universal-quantifier clique pooling against existential & classical methods."""
    default_keys = [
        "bk_reichenbach",
        "bk_lukasiewicz",
        "bk_kleene_dienes",
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
    clique_size = 6
    num_cliques = 3
    graph_raw = generate_planted_cliques_graph(
        num_cliques=num_cliques,
        clique_size=clique_size,
        num_noise_nodes=12,
        noise_edge_prob=0.10,
        feature_dim=12,
        seed=seed,
    )
    n_clq = graph_raw.num_nodes
    # Use identity node features (no ground-truth cluster feature leakage) to test pure topological clique discovery
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

    records: List[CliqueBenchmarkRecord] = []
    for key in compare_keys:
        spec = METHOD_SPEC_MAP[key]
        torch.manual_seed(seed)
        layer = GraphReductionLayer.from_spec(key, in_features=n_clq, out_features=12, num_clusters=num_cliques)
        optimizer = torch.optim.Adam(layer.parameters(), lr=lr)

        for _ in range(epochs):
            optimizer.zero_grad()
            out = layer(graph)
            s = out.s

            # Maximize super-node self-clique strength Tr(A_new) on normal, crisp-tending fuzzy sets
            if layer.is_classical:
                loss = out.aux_loss
            else:
                diag_clique_score = torch.diagonal(out.adj_new).mean()
                cluster_sizes = s.sum(dim=0)
                size_penalty = ((cluster_sizes - float(clique_size)) ** 2).mean() / float(clique_size ** 2)
                # Normality (height -> 1) + De Luca-Termini fuzziness penalty S(1-S) prevents diffuse S_{iI}=0.2 triviality
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
                densities.append(float(sub.sum()) / float(max_edges))

                pred_set = set(top_nodes.tolist())
                best_f1 = 0.0
                for p_set in planted_sets:
                    inter = len(pred_set & p_set)
                    if inter > 0:
                        prec = inter / len(pred_set)
                        rec = inter / len(p_set)
                        best_f1 = max(best_f1, 2.0 * prec * rec / (prec + rec))
                f1_scores.append(best_f1)

            mean_diag = float(torch.diagonal(out.adj_new).mean().item())

        records.append(
            CliqueBenchmarkRecord(
                key=key,
                code=spec.code,
                title=spec.title,
                internal_clique_density=float(np.mean(densities)),
                clique_recovery_f1=float(np.mean(f1_scores)),
                mean_diag_a_new=mean_diag,
            )
        )

    return records


class _PoolingGraphClassifier(nn.Module):
    """2-stage hierarchical GNN classifier: ReductionLayer -> Post-Pooling GraphSAGE -> Readout -> MLP."""

    def __init__(self, method_key: str, in_features: int, hidden_dim: int, num_clusters: int, num_classes: int) -> None:
        super().__init__()
        self.pool = GraphReductionLayer.from_spec(
            method_key,
            in_features=in_features,
            out_features=hidden_dim,
            num_clusters=num_clusters,
        )
        self.post_gnn = GraphSAGEOperator(hidden_dim, hidden_dim, activation="elu")
        self.head = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, graph: GraphData) -> tuple[Tensor, PoolingOutput]:
        out = self.pool(graph)
        # Normalize A_new safely if DiffPool blows up scale, while keeping fuzzy [0,1] topologies intact
        a_next = out.adj_new
        if self.pool.topology_name == "diffpool":
            a_next = a_next / (a_next.detach().abs().max().clamp(min=1.0))
        h2 = self.post_gnn(a_next, out.x_new)
        readout = torch.cat([h2.mean(dim=0), h2.amax(dim=0)], dim=-1)
        logits = self.head(readout)
        return logits, out


def _summarize_learned_params(layer: GraphReductionLayer) -> str:
    parts: List[str] = []
    if hasattr(layer.membership, "tau"):
        parts.append(f"tau={float(layer.membership.tau.detach().item()):.3f}")  # type: ignore[union-attr]
    if hasattr(layer.membership, "gamma"):
        g = layer.membership.gamma.detach()  # type: ignore[union-attr]
        parts.append(f"gamma_mean={float(g.mean().item()):.3f}")
    if hasattr(layer.membership, "sigma"):
        sg = layer.membership.sigma.detach()  # type: ignore[union-attr]
        parts.append(f"sigma_mean={float(sg.mean().item()):.3f}")

    tr = layer.topology_reducer
    if tr is not None:
        if hasattr(tr, "eta"):
            parts.append(f"eta_mean={float(tr.eta.detach().mean().item()):.3f}")  # type: ignore[union-attr]
        if hasattr(tr, "gamma"):
            parts.append(f"H_gamma={float(tr.gamma.detach().mean().item()):.3f}")  # type: ignore[union-attr]
        if hasattr(tr, "q"):
            parts.append(f"SS_q={float(tr.q.detach().item()):.3f}")  # type: ignore[union-attr]
        if hasattr(tr, "lam"):
            parts.append(f"Dombi_lam={float(tr.lam.detach().item()):.3f}")  # type: ignore[union-attr]
        if hasattr(tr, "tau_c"):
            parts.append(f"tau_c={float(tr.tau_c.detach().item()):.3f}")  # type: ignore[union-attr]

    fr = layer.feature_reducer
    if fr is not None:
        if hasattr(fr, "m"):
            parts.append(f"COG_m={float(fr.m.detach().item()):.3f}")  # type: ignore[union-attr]
        if hasattr(fr, "alpha"):
            parts.append(f"Choquet_alpha={float(fr.alpha.detach().mean().item()):.3f}")  # type: ignore[union-attr]

    return ", ".join(parts) if parts else "default"


def run_graph_classification_benchmark(
    method_keys: Optional[Sequence[str]] = None,
    epochs: int = 22,
    lr: float = 0.015,
    seed: int = 123,
) -> List[ClassificationBenchmarkRecord]:
    """Train end-to-end hierarchical GNN classifiers across all canonical reduction methods."""
    keys = list(method_keys) if method_keys is not None else [s.key for s in CANONICAL_METHOD_SPECS]
    dataset = generate_graph_classification_dataset(num_graphs_per_class=16, feature_dim=8, seed=seed)

    # Stratified 75/25 train/test split
    by_class: Dict[int, List[GraphData]] = {0: [], 1: [], 2: [], 3: []}
    for g in dataset:
        assert g.y is not None
        by_class[int(g.y.item())].append(g)
    train_set: List[GraphData] = []
    test_set: List[GraphData] = []
    for cid in range(4):
        items = by_class[cid]
        split = int(0.75 * len(items))
        train_set.extend(items[:split])
        test_set.extend(items[split:])

    records: List[ClassificationBenchmarkRecord] = []
    for key in keys:
        spec = METHOD_SPEC_MAP[key]
        torch.manual_seed(seed)
        model = _PoolingGraphClassifier(
            method_key=key,
            in_features=8,
            hidden_dim=16,
            num_clusters=4,
            num_classes=4,
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        final_loss = 0.0
        for _ in range(epochs):
            model.train()
            optimizer.zero_grad()
            batch_loss = torch.tensor(0.0)
            for g in train_set:
                assert g.y is not None
                logits, out = model(g)
                ce = F.cross_entropy(logits.unsqueeze(0), g.y.unsqueeze(0))
                batch_loss = batch_loss + ce + 0.15 * out.aux_loss
            batch_loss = batch_loss / float(len(train_set))
            batch_loss.backward()
            optimizer.step()
            final_loss = float(batch_loss.detach().item())

        model.eval()
        with torch.no_grad():
            train_correct = sum(
                int(model(g)[0].argmax().item() == int(g.y.item()))  # type: ignore[union-attr]
                for g in train_set
            )
            test_correct = sum(
                int(model(g)[0].argmax().item() == int(g.y.item()))  # type: ignore[union-attr]
                for g in test_set
            )

        records.append(
            ClassificationBenchmarkRecord(
                key=key,
                code=spec.code,
                title=spec.title,
                train_acc=train_correct / float(len(train_set)),
                test_acc=test_correct / float(len(test_set)),
                final_loss=final_loss,
                learned_params_summary=_summarize_learned_params(model.pool),
            )
        )

    return records
