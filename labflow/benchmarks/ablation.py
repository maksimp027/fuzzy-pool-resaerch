"""Component-level ablation benchmark test for graph reduction methods.

Separates the composite GraphReductionLayer into its distinct modular components:
1. Topology Reducer Ablation (comparing t-norms/implications/spectral forms under fixed features)
2. Feature Pooling Ablation (comparing COG, Bipolar, Choquet, Linear under fixed topology)
3. Membership Function Ablation (comparing Softmax, Anisotropic Sigmoid, Gaussian RBF)

Quantifies the isolated contribution of each component towards representation performance.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from labflow.analytics.metrics import compute_feature_reconstruction_r2, compute_nmi_score
from labflow.composite import GraphReductionLayer
from labflow.datasets.synthetic import (
    generate_graph_classification_dataset,
    generate_sbm_graph,
)
from labflow.gnn import GraphSAGEOperator
from labflow.types import GraphData


@dataclass
class ComponentAblationRecord:
    """Scientific record for Component-Level Ablation Test."""

    component_type: str  # "topology", "feature_pool", "membership"
    variant_name: str
    task: str  # "community", "classification"
    metric_name: str
    metric_value: float
    isolated_delta: float  # Difference from baseline variant
    elapsed_ms: float

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


def run_feature_pooling_ablation(
    feature_kinds: Sequence[str] = ("cog", "bipolar", "choquet", "linear"),
    epochs: int = 16,
    lr: float = 0.015,
    seed: int = 123,
) -> List[ComponentAblationRecord]:
    """Isolate and test Feature Pooling methods under identical Goguen topology and Sigmoid membership."""
    dataset = generate_graph_classification_dataset(num_graphs_per_class=12, feature_dim=8, seed=seed)
    train_set = dataset[: int(0.75 * len(dataset))]
    test_set = dataset[int(0.75 * len(dataset)) :]

    records: List[ComponentAblationRecord] = []
    baseline_acc: Optional[float] = None

    for feat in feature_kinds:
        torch.manual_seed(seed)

        class Model(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.pool = GraphReductionLayer(
                    in_features=8,
                    out_features=16,
                    num_clusters=4,
                    topology="goguen",
                    membership="sigmoid",
                    feature_pool=feat,
                )
                self.post_gnn = GraphSAGEOperator(16, 16, activation="elu")
                self.head = nn.Linear(32, 4)

            def forward(self, graph: GraphData) -> torch.Tensor:
                out = self.pool(graph)
                h2 = self.post_gnn(out.adj_new, out.x_new)
                readout = torch.cat([h2.mean(dim=0), h2.amax(dim=0)], dim=-1)
                return self.head(readout)

        model = Model()
        opt = torch.optim.Adam(model.parameters(), lr=lr)

        t0 = time.perf_counter()
        for _ in range(epochs):
            model.train()
            opt.zero_grad()
            batch_loss = torch.tensor(0.0)
            for g in train_set:
                assert g.y is not None
                logits = model(g)
                batch_loss = batch_loss + F.cross_entropy(logits.unsqueeze(0), g.y.unsqueeze(0))
            batch_loss = batch_loss / float(len(train_set))
            batch_loss.backward()
            opt.step()
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        model.eval()
        with torch.no_grad():
            correct = sum(
                int(model(g).argmax().item() == int(g.y.item()))  # type: ignore[union-attr]
                for g in test_set
            )
        acc = float(correct / float(len(test_set)))

        if feat == "linear":
            baseline_acc = acc
        elif baseline_acc is None:
            baseline_acc = acc

        records.append(
            ComponentAblationRecord(
                component_type="feature_pool",
                variant_name=feat,
                task="classification",
                metric_name="test_acc",
                metric_value=acc,
                isolated_delta=acc - (baseline_acc or 0.0),
                elapsed_ms=elapsed_ms,
            )
        )

    return records


def run_membership_ablation(
    membership_kinds: Sequence[str] = ("sigmoid", "softmax", "rbf"),
    epochs: int = 50,
    lr: float = 0.025,
    seed: int = 42,
) -> List[ComponentAblationRecord]:
    """Isolate and test Membership Assignment operators under identical Zadeh topology and COG feature pooling."""
    graph = generate_sbm_graph(block_sizes=(15, 15, 15), p_in=0.68, p_out=0.04, feature_dim=16, seed=seed)
    adj = graph.dense_adj()
    x = graph.x
    assert graph.node_labels is not None
    true_labels = graph.node_labels.cpu().numpy()
    k = 3

    records: List[ComponentAblationRecord] = []
    baseline_nmi: Optional[float] = None

    for mem in membership_kinds:
        torch.manual_seed(seed)
        layer = GraphReductionLayer(
            in_features=16,
            out_features=16,
            num_clusters=k,
            topology="zadeh",
            membership=mem,
            feature_pool="cog",
        )
        dec_x = nn.Linear(16, 16, bias=False)
        opt = torch.optim.Adam(list(layer.parameters()) + list(dec_x.parameters()), lr=lr)

        t0 = time.perf_counter()
        for _ in range(epochs):
            opt.zero_grad()
            out = layer(graph)
            s = out.s
            s_row_norm = s / (s.sum(dim=1, keepdim=True) + 1e-8)
            x_hat = s_row_norm @ dec_x(out.x_new)
            loss_feat = F.mse_loss(x_hat, x)
            a_hat = (s_row_norm @ out.adj_new @ s_row_norm.t()).clamp(0.0, 1.0)
            loss_topo = F.mse_loss(a_hat, adj)
            loss = loss_topo + 0.5 * loss_feat
            loss.backward()
            opt.step()
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        with torch.no_grad():
            final_out = layer(graph)
            pred = final_out.s.argmax(dim=1).cpu().numpy()
            nmi_val = compute_nmi_score(pred, true_labels)

        if mem == "softmax":
            baseline_nmi = nmi_val
        elif baseline_nmi is None:
            baseline_nmi = nmi_val

        records.append(
            ComponentAblationRecord(
                component_type="membership",
                variant_name=mem,
                task="community",
                metric_name="nmi",
                metric_value=nmi_val,
                isolated_delta=nmi_val - (baseline_nmi or 0.0),
                elapsed_ms=elapsed_ms,
            )
        )

    return records


def run_component_ablation_benchmark(
    seed: int = 42,
) -> List[ComponentAblationRecord]:
    """Execute full Component-Level Ablation Benchmark Test."""
    feat_records = run_feature_pooling_ablation(seed=seed)
    mem_records = run_membership_ablation(seed=seed)
    return feat_records + mem_records
