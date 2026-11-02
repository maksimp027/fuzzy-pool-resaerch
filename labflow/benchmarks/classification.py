"""Hierarchical graph classification benchmark test for graph reduction methods.

Evaluates end-to-end downstream representation learning in a 2-stage hierarchical GNN:
    GNN_1 -> GraphReductionLayer -> Post-GNN -> Global Readout -> MLP Classifier.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from labflow.analytics.metrics import compute_feature_reconstruction_r2
from labflow.composite import CANONICAL_METHOD_SPECS, METHOD_SPEC_MAP, GraphReductionLayer
from labflow.datasets.synthetic import generate_graph_classification_dataset
from labflow.gnn import GraphSAGEOperator
from labflow.types import GraphData, PoolingOutput


@dataclass
class ClassificationBenchmarkRecord:
    """Scientific record for Hierarchical Graph Classification Test."""

    key: str
    code: str
    title: str
    train_acc: float
    test_acc: float
    generalization_gap: float
    final_loss: float
    feature_recon_r2: float
    learned_params_summary: str
    ms_per_epoch: float

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


class _PoolingGraphClassifier(nn.Module):
    """2-stage hierarchical GNN classifier: ReductionLayer -> Post-Pooling GraphSAGE -> Readout -> MLP."""

    def __init__(
        self,
        method_key: str,
        in_features: int,
        hidden_dim: int,
        num_clusters: int,
        num_classes: int,
    ) -> None:
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
    num_graphs_per_class: int = 16,
) -> List[ClassificationBenchmarkRecord]:
    """Execute dedicated Hierarchical Graph Classification Benchmark Test."""
    keys = list(method_keys) if method_keys is not None else [s.key for s in CANONICAL_METHOD_SPECS]
    dataset = generate_graph_classification_dataset(
        num_graphs_per_class=num_graphs_per_class, feature_dim=8, seed=seed
    )

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

        t0 = time.perf_counter()
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

        elapsed_ms_per_epoch = (time.perf_counter() - t0) * 1000.0 / max(epochs, 1)

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

            # Feature preservation R^2 across test set
            r2_scores: List[float] = []
            for g in test_set:
                _, out = model(g)
                r2_val = compute_feature_reconstruction_r2(target_feat=out.h, s=out.s, x_new=out.x_new)
                r2_scores.append(r2_val)
            mean_r2 = float(sum(r2_scores) / max(len(r2_scores), 1))

        tr_acc = train_correct / float(len(train_set))
        te_acc = test_correct / float(len(test_set))
        gap = abs(tr_acc - te_acc)

        records.append(
            ClassificationBenchmarkRecord(
                key=key,
                code=spec.code,
                title=spec.title,
                train_acc=tr_acc,
                test_acc=te_acc,
                generalization_gap=gap,
                final_loss=final_loss,
                feature_recon_r2=mean_r2,
                learned_params_summary=_summarize_learned_params(model.pool),
                ms_per_epoch=elapsed_ms_per_epoch,
            )
        )

    return records
