"""Unified end-to-end GraphReductionLayer and canonical method registry for Labflow."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from labflow.classical import DiffPoolReducer, DMoNReducer, MinCutPoolReducer
from labflow.fuzzy_features import (
    BipolarSupremalFeaturePooling,
    ChoquetFeaturePooling,
    COGFeaturePooling,
    LinearFeaturePooling,
)
from labflow.fuzzy_relational import (
    DombiTopologyReducer,
    GoguenTopologyReducer,
    HamacherTopologyReducer,
    LukasiewiczTopologyReducer,
    SchweizerSklarTopologyReducer,
    ZadehTopologyReducer,
)
from labflow.fuzzy_smooth import (
    BandlerKohoutTopologyReducer,
    BoltzmannTopologyReducer,
    ProbSumTopologyReducer,
)
from labflow.gnn import BaseGraphEncoder
from labflow.memberships import (
    AnisotropicSigmoidMembership,
    GaussianRBFMembership,
    TemperatureSoftmaxMembership,
)
from labflow.types import GraphData, PoolingOutput


@dataclass(frozen=True)
class MethodSpec:
    """Metadata and canonical configuration for each reduction method in the research suite."""

    key: str
    code: str
    title: str
    section: str
    topology: str
    membership: str
    feature_pool: str
    complexity: str


CANONICAL_METHOD_SPECS: List[MethodSpec] = [
    MethodSpec("diffpool", "M1", "DiffPool (Bilinear Probabilistic Reduction)", "I.1", "diffpool", "softmax", "builtin", "O(M K + N K^2)"),
    MethodSpec("mincut", "M2", "MinCutPool (Spectral Minimum Cut)", "I.2", "mincut", "softmax", "builtin", "O(M K + N K^2)"),
    MethodSpec("dmon", "M3", "DMoN (Chung-Lu Modularity Reduction)", "I.3", "dmon", "softmax_unit", "builtin", "O(M K + N K)"),
    MethodSpec("zadeh", "M4", "Zadeh-Godel FuzzyPool (max-min)", "II.4", "zadeh", "sigmoid", "cog", "O(M K + N K^2)"),
    MethodSpec("zadeh_rbf", "M4-RBF", "Zadeh-Godel FuzzyPool with Gaussian RBF", "II.4", "zadeh", "rbf", "cog", "O(M K + N K^2)"),
    MethodSpec("goguen", "M5", "Goguen Algebraic FuzzyPool (max-product)", "II.5", "goguen", "sigmoid", "cog", "O(M K + N K^2)"),
    MethodSpec("lukasiewicz", "M6", "Bounded Lukasiewicz FuzzyPool ([0,1]-clipping)", "II.6", "lukasiewicz", "sigmoid", "cog", "O(M K + N K^2)"),
    MethodSpec("lukasiewicz_smooth", "M6-Sm", "Smooth Lukasiewicz FuzzyPool (Softplus window)", "II.6", "lukasiewicz_smooth", "sigmoid", "cog", "O(M K + N K^2)"),
    MethodSpec("hamacher", "M7", "Hamacher Rational FuzzyPool", "II.7", "hamacher", "sigmoid", "cog", "O(M K + N K^2)"),
    MethodSpec("einstein", "M7-Ein", "Einstein Relativistic FuzzyPool (gamma=2)", "II.7", "einstein", "sigmoid", "cog", "O(M K + N K^2)"),
    MethodSpec("schweizer_sklar", "M8A", "Schweizer-Sklar Parameterized FuzzyPool", "II.8A", "schweizer_sklar", "sigmoid", "cog", "O(M K + N K^2)"),
    MethodSpec("dombi", "M8B", "Dombi Parameterized FuzzyPool", "II.8B", "dombi", "sigmoid", "cog", "O(M K + N K^2)"),
    MethodSpec("boltzmann", "M9A", "Boltzmann Energy FuzzyPool (Global Edge)", "III.9.1", "boltzmann", "sigmoid", "cog", "O(M K^2)"),
    MethodSpec("boltzmann_adaptive", "M9-Ad", "Scale-Adaptive Boltzmann FuzzyPool (ln M/K^2)", "III.9.1", "boltzmann_adaptive", "sigmoid", "cog", "O(M K^2)"),
    MethodSpec("boltzmann_two_step", "M9B", "Two-Step Boltzmann FuzzyPool", "III.9.2", "boltzmann_two_step", "sigmoid", "cog", "O(M K + N K^2)"),
    MethodSpec("probsum", "M10", "Probabilistic Sum FuzzyPool (ProbSum)", "III.10", "probsum", "sigmoid", "cog", "O(M K^2)"),
    MethodSpec("probsum_norm", "M10-Nm", "Scale-Normalized ProbSum FuzzyPool (Mean-Field)", "III.10", "probsum_norm", "sigmoid", "cog", "O(M K^2)"),
    MethodSpec("bk_lukasiewicz", "M11A", "Bandler-Kohout Implicative FuzzyPool (Lukasiewicz)", "III.11.1", "bk_lukasiewicz", "sigmoid", "cog", "O((M+N)K log MK + NK^2)"),
    MethodSpec("bk_kleene_dienes", "M11B", "Bandler-Kohout Implicative FuzzyPool (Kleene-Dienes)", "III.11.2", "bk_kleene_dienes", "sigmoid", "cog", "O((M+N)K log MK + NK^2)"),
    MethodSpec("bk_reichenbach", "M11C", "Bandler-Kohout Implicative FuzzyPool (Reichenbach)", "III.11.3", "bk_reichenbach", "sigmoid", "cog", "O((M+N)K log MK + NK^2)"),
    MethodSpec("goguen_bipolar", "M13A", "Goguen FuzzyPool + Bipolar Supremal Feature Pooling", "IV.13A", "goguen", "sigmoid", "bipolar", "O(M K + N K^2)"),
    MethodSpec("goguen_choquet", "M13B", "Goguen FuzzyPool + Choquet Integral Feature Pooling", "IV.13B", "goguen", "sigmoid", "choquet", "O(M K + N K F')"),
]

METHOD_SPEC_MAP: Dict[str, MethodSpec] = {spec.key: spec for spec in CANONICAL_METHOD_SPECS}


def build_membership(
    kind: str,
    num_clusters: int,
    rbf_dim: Optional[int] = None,
) -> nn.Module:
    kind = kind.lower()
    if kind == "softmax":
        return TemperatureSoftmaxMembership(init_tau=1.0, learnable_tau=True)
    if kind == "softmax_unit":
        return TemperatureSoftmaxMembership(init_tau=1.0, learnable_tau=False)
    if kind == "sigmoid":
        return AnisotropicSigmoidMembership(num_clusters=num_clusters, init_gamma=1.0, init_beta=0.0)
    if kind == "rbf":
        return GaussianRBFMembership(num_clusters=num_clusters, latent_dim=rbf_dim or num_clusters)
    raise ValueError(f"Unknown membership parameterization: {kind}")


def build_feature_pooling(kind: str, num_clusters: int) -> nn.Module:
    kind = kind.lower()
    if kind == "cog":
        return COGFeaturePooling(init_m=2.0, learnable_m=True)
    if kind == "bipolar":
        return BipolarSupremalFeaturePooling()
    if kind == "choquet":
        return ChoquetFeaturePooling(num_clusters=num_clusters, init_alpha=1.0, learnable_alpha=True)
    if kind == "linear":
        return LinearFeaturePooling()
    raise ValueError(f"Unknown feature pooling kind: {kind}")


def build_fuzzy_topology_reducer(kind: str, num_clusters: int) -> nn.Module:
    kind = kind.lower()
    if kind == "zadeh":
        return ZadehTopologyReducer()
    if kind == "goguen":
        return GoguenTopologyReducer()
    if kind == "lukasiewicz":
        return LukasiewiczTopologyReducer(
            num_clusters=num_clusters, learnable_eta=True, init_eta=1.5, clamp_unit=True, smooth_tau=0.0
        )
    if kind == "lukasiewicz_smooth":
        return LukasiewiczTopologyReducer(
            num_clusters=num_clusters, learnable_eta=True, init_eta=1.5, clamp_unit=True, smooth_tau=0.12
        )
    if kind == "lukasiewicz_classical":
        return LukasiewiczTopologyReducer(
            num_clusters=num_clusters, learnable_eta=False, init_eta=2.0, clamp_unit=True, smooth_tau=0.0
        )
    if kind == "hamacher":
        return HamacherTopologyReducer(init_gamma=1.2, learnable_gamma=True, matrix_gamma=False)
    if kind == "einstein":
        return HamacherTopologyReducer(init_gamma=2.0, learnable_gamma=False, matrix_gamma=False)
    if kind == "hamacher_matrix":
        return HamacherTopologyReducer(
            init_gamma=1.5, learnable_gamma=True, matrix_gamma=True, num_clusters=num_clusters
        )
    if kind == "schweizer_sklar":
        return SchweizerSklarTopologyReducer(init_q=1.0, learnable_q=True)
    if kind == "dombi":
        return DombiTopologyReducer(init_lambda=1.0, learnable_lambda=True)
    if kind == "boltzmann":
        return BoltzmannTopologyReducer(init_tau_c=0.25, learnable_tau_c=True, mode="global_edge")
    if kind == "boltzmann_adaptive":
        return BoltzmannTopologyReducer(
            init_tau_c=0.25, learnable_tau_c=True, mode="global_edge", scale_by_log_m=True
        )
    if kind == "boltzmann_two_step":
        return BoltzmannTopologyReducer(init_tau_c=0.25, learnable_tau_c=True, mode="two_step")
    if kind == "probsum":
        return ProbSumTopologyReducer(normalization="none")
    if kind == "probsum_norm":
        return ProbSumTopologyReducer(normalization="mean_field")
    if kind == "bk_lukasiewicz":
        return BandlerKohoutTopologyReducer(implication="lukasiewicz", ensure_self_loops=True)
    if kind == "bk_kleene_dienes":
        return BandlerKohoutTopologyReducer(implication="kleene_dienes", ensure_self_loops=True)
    if kind == "bk_reichenbach":
        return BandlerKohoutTopologyReducer(implication="reichenbach", ensure_self_loops=True)
    raise ValueError(f"Unknown fuzzy topology reducer: {kind}")


class GraphReductionLayer(nn.Module):
    """Complete Graph Reduction Layer: (A, X) -> (H, Z) -> S -> (X_new, A_new, aux_loss)."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        num_clusters: int,
        topology: str = "goguen",
        membership: Optional[str] = None,
        feature_pool: str = "cog",
        rbf_dim: Optional[int] = None,
        embed_activation: str = "elu",
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_clusters = num_clusters
        self.topology_name = topology.lower()

        is_classical = self.topology_name in ("diffpool", "mincut", "dmon")
        if membership is None:
            membership = (
                "softmax_unit"
                if self.topology_name == "dmon"
                else ("softmax" if is_classical else "sigmoid")
            )
        self.membership_name = membership.lower()
        self.is_classical = is_classical

        pool_dim = (rbf_dim or num_clusters) if self.membership_name == "rbf" else num_clusters
        self.encoder = BaseGraphEncoder(
            in_features=in_features,
            embed_dim=out_features,
            pool_dim=pool_dim,
            embed_activation=embed_activation,
        )
        self.membership = build_membership(self.membership_name, num_clusters=num_clusters, rbf_dim=rbf_dim)

        if self.topology_name == "diffpool":
            self.classical_reducer: Optional[nn.Module] = DiffPoolReducer()
            self.topology_reducer: Optional[nn.Module] = None
            self.feature_reducer: Optional[nn.Module] = None
        elif self.topology_name == "mincut":
            self.classical_reducer = MinCutPoolReducer()
            self.topology_reducer = None
            self.feature_reducer = None
        elif self.topology_name == "dmon":
            self.classical_reducer = DMoNReducer()
            self.topology_reducer = None
            self.feature_reducer = None
        else:
            self.classical_reducer = None
            self.topology_reducer = build_fuzzy_topology_reducer(self.topology_name, num_clusters=num_clusters)
            self.feature_reducer = build_feature_pooling(feature_pool, num_clusters=num_clusters)

    @classmethod
    def from_spec(
        cls,
        key: str,
        in_features: int,
        out_features: int,
        num_clusters: int,
        rbf_dim: Optional[int] = None,
    ) -> "GraphReductionLayer":
        if key not in METHOD_SPEC_MAP:
            raise KeyError(f"Unknown method key '{key}'. Available: {list(METHOD_SPEC_MAP)}")
        spec = METHOD_SPEC_MAP[key]
        feat = "cog" if spec.feature_pool == "builtin" else spec.feature_pool
        return cls(
            in_features=in_features,
            out_features=out_features,
            num_clusters=num_clusters,
            topology=spec.topology,
            membership=spec.membership,
            feature_pool=feat,
            rbf_dim=rbf_dim,
        )

    def forward(self, graph: GraphData | Tuple[Tensor, Tensor]) -> PoolingOutput:
        if isinstance(graph, tuple):
            adj, x = graph
            edge_index, edge_weight = None, None
            use_sparse = False
        else:
            x = graph.x
            if graph.adj is not None and graph.adj.requires_grad:
                adj = graph.adj
                edge_index, edge_weight = None, None
                use_sparse = False
            else:
                edge_index, edge_weight = graph.sparse_edges()
                adj = None
                use_sparse = True

        if use_sparse:
            assert edge_index is not None and edge_weight is not None
            h, z = self.encoder.forward_sparse(edge_index, edge_weight, x)
        else:
            assert adj is not None
            h, z = self.encoder(adj, x)

        s = self.membership(z)

        if self.is_classical:
            assert self.classical_reducer is not None
            if use_sparse:
                out = self.classical_reducer(None, h, s, edge_index=edge_index, edge_weight=edge_weight)
            else:
                out = self.classical_reducer(adj, h, s)
            out.z = z
            return out

        assert self.topology_reducer is not None and self.feature_reducer is not None
        if use_sparse:
            adj_new = self.topology_reducer(s, edge_index=edge_index, edge_weight=edge_weight)
        elif self.topology_name.startswith("bk_"):
            adj_new = self.topology_reducer(s, adj=adj)
        elif edge_index is not None and edge_weight is not None:
            adj_new = self.topology_reducer(s, edge_index=edge_index, edge_weight=edge_weight)
        else:
            adj_new = self.topology_reducer(s, adj=adj)

        x_new = self.feature_reducer(h, s)
        sparsity = float((adj_new.detach() == 0.0).to(torch.float32).mean().item())
        return PoolingOutput(
            x_new=x_new,
            adj_new=adj_new,
            s=s,
            h=h,
            z=z,
            aux_loss=torch.zeros((), dtype=x.dtype, device=x.device),
            diagnostics={
                "adj_new_min": float(adj_new.detach().min().item()),
                "adj_new_max": float(adj_new.detach().max().item()),
                "exact_zero_fraction": sparsity,
            },
        )

    def forward_reference(self, adj: Tensor, x: Tensor) -> PoolingOutput:
        """Pure PyTorch autograd reference forward pass across the entire layer."""
        h, z = self.encoder.forward_reference(adj, x)
        s = self.membership.forward_reference(z)  # type: ignore[operator]

        if self.is_classical:
            assert self.classical_reducer is not None
            out = self.classical_reducer.forward_reference(adj, h, s)  # type: ignore[operator]
            out.z = z
            return out

        assert self.topology_reducer is not None and self.feature_reducer is not None
        from labflow.types import dense_to_sparse

        edge_index, edge_weight = dense_to_sparse(adj, threshold=0.0)
        if hasattr(self.topology_reducer, "forward_dense_reference"):
            adj_new = self.topology_reducer.forward_dense_reference(adj, s)  # type: ignore[operator]
        elif hasattr(self.topology_reducer, "forward_direct_implication_reference"):
            adj_new = self.topology_reducer.forward_direct_implication_reference(adj, s)  # type: ignore[operator]
        elif hasattr(self.topology_reducer, "forward_closed_form_reference"):
            adj_new = self.topology_reducer.forward_closed_form_reference(s, edge_index, edge_weight)  # type: ignore[operator]
        elif hasattr(self.topology_reducer, "forward_edge_reference"):
            adj_new = self.topology_reducer.forward_edge_reference(s, edge_index, edge_weight)  # type: ignore[operator]
        else:
            adj_new = self.topology_reducer.forward_reference(s, edge_index, edge_weight)  # type: ignore[operator]

        x_new = self.feature_reducer.forward_reference(h, s)  # type: ignore[operator]
        return PoolingOutput(
            x_new=x_new,
            adj_new=adj_new,
            s=s,
            h=h,
            z=z,
            aux_loss=torch.zeros((), dtype=x.dtype, device=x.device),
        )
