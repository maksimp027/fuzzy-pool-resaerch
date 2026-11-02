"""Multi-benchmark comparative evaluation, Pareto frontier analysis, and leaderboard engine for all reducers."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from labflow.analytics.metrics import ReducerMetricProfile, evaluate_reducer_metrics
from labflow.analytics.profiler import (
    ReducerScalingSummary,
    ScalingPointRecord,
    run_scaling_profiler,
)
from labflow.analytics.stability_audit import AuditStressRecord, run_audit_stress_suite
from labflow.benchmarks.runner import (
    ClassificationBenchmarkRecord,
    CliqueBenchmarkRecord,
    run_graph_classification_benchmark,
    run_planted_clique_benchmark,
)
from labflow.composite import CANONICAL_METHOD_SPECS, METHOD_SPEC_MAP, GraphReductionLayer
from labflow.datasets import (
    generate_multiscale_noisy_sbm_graph,
    generate_sbm_graph,
    load_classic_network,
)
from labflow.types import GraphData


@dataclass
class ReducerLeaderboardEntry:
    """Multi-criteria comparative ranking record for a single graph reduction method."""

    rank: int
    method_key: str
    method_code: str
    title: str
    section: str
    complexity: str
    # Normalized [0, 1] category scores
    community_score: float
    topology_spectral_score: float
    clique_isolation_score: float
    classification_score: float
    gradient_stability_score: float
    efficiency_score: float
    # Aggregate metrics & Pareto status
    quality_index: float
    composite_score: float
    mean_nmi: float
    mean_ari: float
    mean_modularity_q: float
    mean_link_auc: float
    dead_superedge_ratio: float
    ms_large_graph: float
    empirical_scaling_exp: float
    is_pareto_optimal: bool
    recommended_use_case: str
    # Domain-specific test ranks (separated scientific evaluation)
    community_rank: int = 0
    clique_rank: int = 0
    topology_rank: int = 0
    classification_rank: int = 0
    gradient_rank: int = 0
    efficiency_rank: int = 0

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


@dataclass
class AnalyticsPackageResult:
    """Complete output bundle of the Labflow Analytical Software Package."""

    leaderboard: List[ReducerLeaderboardEntry]
    dataset_profiles: Dict[str, List[ReducerMetricProfile]]
    clique_records: List[CliqueBenchmarkRecord]
    classification_records: List[ClassificationBenchmarkRecord]
    audit_stress_records: List[AuditStressRecord]
    scaling_points: List[ScalingPointRecord]
    scaling_summaries: List[ReducerScalingSummary]
    # Domain-specific test leaderboards (separated from monolithic composite)
    community_leaderboard: Optional[List[ReducerLeaderboardEntry]] = None
    clique_leaderboard: Optional[List[ReducerLeaderboardEntry]] = None
    topology_leaderboard: Optional[List[ReducerLeaderboardEntry]] = None
    classification_leaderboard: Optional[List[ReducerLeaderboardEntry]] = None
    gradient_leaderboard: Optional[List[ReducerLeaderboardEntry]] = None
    efficiency_leaderboard: Optional[List[ReducerLeaderboardEntry]] = None
    domain_winners: Optional[Dict[str, ReducerLeaderboardEntry]] = None

    def to_dict(self) -> Dict[str, object]:
        base = {
            "leaderboard": [e.to_dict() for e in self.leaderboard],
            "dataset_profiles": {
                ds: [p.to_dict() for p in profs] for ds, profs in self.dataset_profiles.items()
            },
            "clique_benchmark": [r.to_dict() for r in self.clique_records],
            "classification_benchmark": [r.to_dict() for r in self.classification_records],
            "audit_stress_tests": [r.to_dict() for r in self.audit_stress_records],
            "scaling_points": [p.to_dict() for p in self.scaling_points],
            "scaling_summaries": [s.to_dict() for s in self.scaling_summaries],
        }
        if self.community_leaderboard is not None:
            base["community_leaderboard"] = [e.to_dict() for e in self.community_leaderboard]
        if self.clique_leaderboard is not None:
            base["clique_leaderboard"] = [e.to_dict() for e in self.clique_leaderboard]
        if self.topology_leaderboard is not None:
            base["topology_leaderboard"] = [e.to_dict() for e in self.topology_leaderboard]
        if self.classification_leaderboard is not None:
            base["classification_leaderboard"] = [e.to_dict() for e in self.classification_leaderboard]
        if self.gradient_leaderboard is not None:
            base["gradient_leaderboard"] = [e.to_dict() for e in self.gradient_leaderboard]
        if self.efficiency_leaderboard is not None:
            base["efficiency_leaderboard"] = [e.to_dict() for e in self.efficiency_leaderboard]
        if self.domain_winners is not None:
            base["domain_winners"] = {k: v.to_dict() for k, v in self.domain_winners.items()}
        return base


def train_and_profile_unsupervised_dataset(
    graph: GraphData,
    dataset_name: str,
    num_clusters: int,
    method_keys: Sequence[str],
    epochs: int = 70,
    lr: float = 0.025,
    seed: int = 42,
) -> List[ReducerMetricProfile]:
    """Train each canonical reducer unsupervised on `graph` and extract its 24-metric profile."""
    adj = graph.dense_adj()
    x = graph.x
    f_dim = graph.num_features

    profiles: List[ReducerMetricProfile] = []
    for key in method_keys:
        spec = METHOD_SPEC_MAP[key]
        torch.manual_seed(seed)
        layer = GraphReductionLayer.from_spec(
            key,
            in_features=f_dim,
            out_features=f_dim,
            num_clusters=num_clusters,
        )
        dec_x = nn.Linear(f_dim, f_dim, bias=False)
        optimizer = torch.optim.Adam(list(layer.parameters()) + list(dec_x.parameters()), lr=lr)

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
            eye_k = torch.eye(num_clusters, dtype=s.dtype, device=s.device) / math.sqrt(float(num_clusters))
            loss_orth = torch.linalg.norm(q_mat / q_norm - eye_k, ord="fro")

            if layer.is_classical:
                loss = out.aux_loss + 0.4 * loss_feat + 0.2 * loss_topo
            else:
                loss = loss_topo + 0.5 * loss_feat + 0.35 * loss_orth

            loss.backward()
            optimizer.step()

        profile = evaluate_reducer_metrics(
            layer=layer,
            graph=graph,
            method_key=key,
            method_code=spec.code,
            dataset_name=dataset_name,
        )
        profiles.append(profile)

    return profiles


def _recommend_use_case(key: str) -> str:
    if key.startswith("bk_"):
        return "Dense clique, motif, and fully-connected core extraction (universal quantifier)"
    if key == "lukasiewicz_smooth":
        return "Sparse [0, 1] reduction without Dead Super-Edges for deep hierarchical GNNs"
    if key == "lukasiewicz":
        return "Exact topological COO/CSR sparsity with strict threshold pruning"
    if key in ("dmon", "mincut"):
        return "Spectral and modular community clustering without overlap"
    if key == "diffpool":
        return "Dense bilinear aggregation of small/medium graphs (conserves edge mass)"
    if key in ("probsum_norm", "boltzmann_adaptive"):
        return "Scale-invariant smooth parallel-path accumulation on large graphs"
    if key in ("probsum", "boltzmann", "boltzmann_two_step"):
        return "Smooth (C^inf) probabilistic/energy amplification of multi-edge bridges"
    if key in ("schweizer_sklar", "dombi"):
        return "Adaptive parametric continuum of t-norms (log-domain stable)"
    if key in ("goguen_bipolar", "goguen_choquet"):
        return "Preservation of signed extrema and nonlinear feature interactions (Bipolar / Choquet)"
    if key in ("hamacher", "einstein", "goguen"):
        return "Multiplicative rational fuzzy reduction with mutual Hebbian reinforcement"
    return "Idempotent boundary reduction (max-min / RBF prototypes)"


def run_comprehensive_analytics(
    method_keys: Optional[Sequence[str]] = None,
    unsupervised_epochs: int = 65,
    clique_epochs: int = 90,
    cls_epochs: int = 20,
    profiler_sizes: Sequence[int] = (150, 450, 1200, 2400),
    profiler_repeats: int = 2,
    seed: int = 42,
) -> AnalyticsPackageResult:
    """Execute the full multi-benchmark, audit-stress, and scaling analytics suite across all reducers."""
    keys = list(method_keys) if method_keys is not None else [s.key for s in CANONICAL_METHOD_SPECS]

    # 1. Three structural community & topology datasets (Clean SBM, Multi-Scale Noisy SBM, Real Karate Club)
    g_sbm = generate_sbm_graph(block_sizes=(15, 15, 15), p_in=0.68, p_out=0.04, feature_dim=16, seed=seed)
    g_multi = generate_multiscale_noisy_sbm_graph(
        block_sizes=(8, 16, 24, 12), p_in_range=(0.48, 0.78), p_out=0.06, feature_dim=16, seed=seed + 17
    )
    g_karate = load_classic_network("karate", feature_dim=16, seed=seed)

    dataset_profiles: Dict[str, List[ReducerMetricProfile]] = {
        "sbm_clean": train_and_profile_unsupervised_dataset(
            g_sbm, "sbm_clean", num_clusters=3, method_keys=keys, epochs=unsupervised_epochs, seed=seed
        ),
        "sbm_multiscale_noisy": train_and_profile_unsupervised_dataset(
            g_multi,
            "sbm_multiscale_noisy",
            num_clusters=4,
            method_keys=keys,
            epochs=unsupervised_epochs,
            seed=seed + 1,
        ),
        "karate_real": train_and_profile_unsupervised_dataset(
            g_karate, "karate_real", num_clusters=2, method_keys=keys, epochs=unsupervised_epochs, seed=seed + 2
        ),
    }

    # 2. Planted Clique Isolation across all tested reducers
    clique_records = run_planted_clique_benchmark(method_keys=keys, epochs=clique_epochs, seed=7)
    clique_map = {r.key: r for r in clique_records}

    # 3. End-to-End 4-Class Hierarchical Graph Classification
    cls_records = run_graph_classification_benchmark(method_keys=keys, epochs=cls_epochs, seed=123)
    cls_map = {r.key: r for r in cls_records}

    # 4. Mathematical & Numerical Audit Stress Suite (ST-01 .. ST-08)
    audit_records = run_audit_stress_suite()

    # 5. Computational Scaling & Memory Profiler
    scaling_points, scaling_summaries = run_scaling_profiler(
        method_keys=keys,
        node_sizes=profiler_sizes,
        repeats=profiler_repeats,
        seed=2026,
    )
    scale_map = {s.method_key: s for s in scaling_summaries}

    # Index profiles by method_key
    prof_by_key: Dict[str, List[ReducerMetricProfile]] = {k: [] for k in keys}
    for ds_list in dataset_profiles.values():
        for p in ds_list:
            prof_by_key[p.method_key].append(p)

    median_ms = float(np.median([s.ms_at_largest_n for s in scaling_summaries])) if scaling_summaries else 1.0
    median_ms = max(median_ms, 1e-3)

    raw_entries: List[ReducerLeaderboardEntry] = []
    for key in keys:
        spec = METHOD_SPEC_MAP[key]
        profs = prof_by_key[key]
        clq = clique_map[key]
        cls_r = cls_map[key]
        sc = scale_map[key]

        mean_nmi = float(np.mean([p.nmi for p in profs]))
        mean_ari = float(np.mean([p.ari for p in profs]))
        mean_f1 = float(np.mean([p.macro_f1 for p in profs]))
        mean_mod_q = float(np.mean([p.hard_modularity_q for p in profs]))
        # Normalize modularity Q (typically up to ~0.65 on 2-4 blocks) into [0, 1]
        mod_norm = float(np.clip(mean_mod_q / 0.60, 0.0, 1.0))
        ari_norm = float(np.clip(mean_ari, 0.0, 1.0))
        community_score = 0.30 * mean_nmi + 0.25 * ari_norm + 0.25 * mean_f1 + 0.20 * mod_norm

        mean_auc = float(np.mean([p.link_recon_auc for p in profs]))
        mean_spec_sim = float(np.mean([1.0 / (1.0 + p.spectral_laplacian_dist) for p in profs]))
        mean_unit = float(np.mean([p.unit_interval_compliance for p in profs]))
        mean_cut_qual = float(np.mean([1.0 - np.clip(p.conductance, 0.0, 1.0) for p in profs]))
        topology_spectral_score = (
            0.35 * mean_auc + 0.25 * mean_spec_sim + 0.20 * mean_unit + 0.20 * mean_cut_qual
        )

        clique_isolation_score = 0.50 * clq.internal_clique_density + 0.50 * clq.clique_recovery_f1
        classification_score = float(cls_r.test_acc)

        mean_dead = float(np.mean([p.dead_superedge_ratio for p in profs]))
        mean_finite = float(np.mean([p.param_grad_finite_ratio for p in profs]))
        mean_active_c = float(np.mean([p.active_cluster_ratio for p in profs]))
        mean_r2 = float(np.mean([max(0.0, p.feature_recon_r2) for p in profs]))
        gradient_stability_score = (
            0.30 * mean_finite
            + 0.30 * (1.0 - 0.65 * mean_dead)
            + 0.20 * mean_active_c
            + 0.20 * mean_r2
        )

        speed_factor = 1.0 / (1.0 + 0.5 * (sc.ms_at_largest_n / median_ms))
        linear_scaling_bonus = float(np.clip(2.0 - sc.empirical_time_exponent, 0.0, 1.0))
        efficiency_score = 0.65 * speed_factor + 0.35 * linear_scaling_bonus

        quality_index = (
            0.28 * community_score
            + 0.20 * topology_spectral_score
            + 0.16 * clique_isolation_score
            + 0.20 * classification_score
            + 0.16 * gradient_stability_score
        )
        composite_score = 0.88 * quality_index + 0.12 * efficiency_score

        raw_entries.append(
            ReducerLeaderboardEntry(
                rank=0,
                method_key=key,
                method_code=spec.code,
                title=spec.title,
                section=spec.section,
                complexity=spec.complexity,
                community_score=community_score,
                topology_spectral_score=topology_spectral_score,
                clique_isolation_score=clique_isolation_score,
                classification_score=classification_score,
                gradient_stability_score=gradient_stability_score,
                efficiency_score=efficiency_score,
                quality_index=quality_index,
                composite_score=composite_score,
                mean_nmi=mean_nmi,
                mean_ari=mean_ari,
                mean_modularity_q=mean_mod_q,
                mean_link_auc=mean_auc,
                dead_superedge_ratio=mean_dead,
                ms_large_graph=sc.ms_at_largest_n,
                empirical_scaling_exp=sc.empirical_time_exponent,
                is_pareto_optimal=False,
                recommended_use_case=_recommend_use_case(key),
            )
        )

    # Compute 2D Pareto frontier: maximize `quality_index`, minimize `ms_large_graph`
    for i, cand in enumerate(raw_entries):
        dominated = False
        for j, other in enumerate(raw_entries):
            if i == j:
                continue
            better_or_eq_qual = other.quality_index >= cand.quality_index - 1e-12
            faster_or_eq = other.ms_large_graph <= cand.ms_large_graph + 1e-12
            strictly_better = (other.quality_index > cand.quality_index + 1e-9) or (
                other.ms_large_graph < cand.ms_large_graph - 1e-6
            )
            if better_or_eq_qual and faster_or_eq and strictly_better:
                dominated = True
                break
        cand.is_pareto_optimal = not dominated

    # Sort descending by composite_score and assign 1-based ranks
    raw_entries.sort(key=lambda e: e.composite_score, reverse=True)
    for idx, entry in enumerate(raw_entries, start=1):
        entry.rank = idx

    # Assign separate domain test ranks (decoupling from composite score)
    domain_sort_fields = [
        ("community_score", "community_rank"),
        ("clique_isolation_score", "clique_rank"),
        ("topology_spectral_score", "topology_rank"),
        ("classification_score", "classification_rank"),
        ("gradient_stability_score", "gradient_rank"),
        ("efficiency_score", "efficiency_rank"),
    ]
    for score_attr, rank_attr in domain_sort_fields:
        sorted_by_attr = sorted(raw_entries, key=lambda e: getattr(e, score_attr), reverse=True)
        for d_rank, e in enumerate(sorted_by_attr, start=1):
            setattr(e, rank_attr, d_rank)

    community_lead = sorted(raw_entries, key=lambda e: (e.community_rank, -e.community_score))
    clique_lead = sorted(raw_entries, key=lambda e: (e.clique_rank, -e.clique_isolation_score))
    topology_lead = sorted(raw_entries, key=lambda e: (e.topology_rank, -e.topology_spectral_score))
    classification_lead = sorted(raw_entries, key=lambda e: (e.classification_rank, -e.classification_score))
    gradient_lead = sorted(raw_entries, key=lambda e: (e.gradient_rank, -e.gradient_stability_score))
    efficiency_lead = sorted(raw_entries, key=lambda e: (e.efficiency_rank, -e.efficiency_score))

    domain_winners = {
        "community": community_lead[0],
        "clique": clique_lead[0],
        "topology": topology_lead[0],
        "classification": classification_lead[0],
        "gradient": gradient_lead[0],
        "efficiency": efficiency_lead[0],
    }

    return AnalyticsPackageResult(
        leaderboard=raw_entries,
        dataset_profiles=dataset_profiles,
        clique_records=clique_records,
        classification_records=cls_records,
        audit_stress_records=audit_records,
        scaling_points=scaling_points,
        scaling_summaries=scaling_summaries,
        community_leaderboard=community_lead,
        clique_leaderboard=clique_lead,
        topology_leaderboard=topology_lead,
        classification_leaderboard=classification_lead,
        gradient_leaderboard=gradient_lead,
        efficiency_leaderboard=efficiency_lead,
        domain_winners=domain_winners,
    )
