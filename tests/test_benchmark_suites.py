"""Comprehensive test suite verifying modular benchmark tests, separated metrics, and domain leaderboards.

Verifies:
1. Community Detection & Partition Alignment Benchmark (NMI, ARI, Modularity Q, Conductance, Purity)
2. Structural Motif & Planted Clique Isolation Benchmark (Density, F1, Reflexivity, Contrast)
3. Topological & Spectral Fidelity Benchmark (Link AUC, SpecDist, Sparsity, Unit Interval)
4. Hierarchical Graph Classification Benchmark (Accuracy, Generalization Gap, Loss, Feat R^2)
5. Gradient Dynamics & Optimization Health Benchmark (Jacobian density, Dead edges, Finiteness)
6. Computational Efficiency & Asymptotic Scaling Benchmark (Scaling alpha, ms, memory)
7. Component-Level Modular Ablation Benchmark (Topology vs Feature vs Membership)
8. Domain-specific leaderboards and scientific champions (decoupling from composite score)
9. CLI benchmark command with `--test` dispatcher
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from labflow.analytics import run_comprehensive_analytics
from labflow.benchmarks import (
    run_clique_motif_benchmark,
    run_community_detection_benchmark,
    run_component_ablation_benchmark,
    run_efficiency_benchmark,
    run_feature_pooling_ablation,
    run_gradient_stability_benchmark,
    run_graph_classification_benchmark,
    run_membership_ablation,
    run_topology_fidelity_benchmark,
)
from labflow.cli import main as cli_main


def test_community_detection_benchmark_metrics() -> None:
    subset = ["dmon", "goguen", "lukasiewicz_smooth"]
    records = run_community_detection_benchmark(method_keys=subset, epochs=6, seed=42)

    assert len(records) == len(subset)
    for r in records:
        assert 0.0 <= r.nmi <= 1.0
        assert -1.0 <= r.ari <= 1.0
        assert -1.0 <= r.modularity_q <= 1.0
        assert -1.0 <= r.fuzzy_modularity_q <= 1.0
        assert 0.0 <= r.conductance <= 1.0
        assert 0.0 <= r.cluster_purity <= 1.0
        assert 0.0 <= r.macro_f1 <= 1.0
        assert r.adj_new_max >= 0.0
        assert r.ms_per_step >= 0.0


def test_clique_motif_benchmark_metrics() -> None:
    subset = ["bk_kleene_dienes", "goguen", "mincut"]
    records = run_clique_motif_benchmark(
        method_keys=subset,
        epochs=8,
        seed=7,
        clique_size=5,
        num_cliques=2,
        num_noise_nodes=6,
    )

    assert len(records) == len(subset)
    rec_by_key = {r.key: r for r in records}

    # Bandler-Kohout must compute valid reflexivity and motif metrics
    bk = rec_by_key["bk_kleene_dienes"]
    assert 0.0 <= bk.internal_clique_density <= 1.0
    assert 0.0 <= bk.clique_recovery_f1 <= 1.0
    assert bk.diagonal_reflexivity >= 0.0
    assert bk.motif_contrast_ratio > 0.0
    assert bk.deluca_termini_entropy >= 0.0
    # Backward compatibility alias
    assert bk.mean_diag_a_new == bk.diagonal_reflexivity


def test_topology_fidelity_benchmark_metrics() -> None:
    subset = ["lukasiewicz_smooth", "lukasiewicz", "probsum_norm"]
    records = run_topology_fidelity_benchmark(method_keys=subset, epochs=6, seed=42)

    assert len(records) == len(subset)
    for r in records:
        assert 0.0 <= r.link_recon_auc <= 1.0
        assert r.link_recon_rmse >= 0.0
        assert r.spectral_laplacian_dist >= 0.0
        assert 0.0 <= r.exact_sparsity <= 1.0
        assert 0.0 <= r.effective_sparsity <= 1.0
        assert 0.0 <= r.unit_interval_compliance <= 1.0
        assert r.adj_new_max >= 0.0

    # Bounded Lukasiewicz should exhibit higher exact sparsity than smooth
    rec_by_key = {r.key: r for r in records}
    assert rec_by_key["lukasiewicz"].exact_sparsity >= rec_by_key["lukasiewicz_smooth"].exact_sparsity


def test_classification_benchmark_metrics() -> None:
    subset = ["goguen_bipolar", "goguen_choquet", "goguen"]
    records = run_graph_classification_benchmark(
        method_keys=subset,
        epochs=4,
        seed=123,
        num_graphs_per_class=8,
    )

    assert len(records) == len(subset)
    for r in records:
        assert 0.0 <= r.train_acc <= 1.0
        assert 0.0 <= r.test_acc <= 1.0
        assert 0.0 <= r.generalization_gap <= 1.0
        assert r.final_loss >= 0.0
        assert -10.0 <= r.feature_recon_r2 <= 1.0
        assert len(r.learned_params_summary) > 0


def test_gradient_stability_benchmark_metrics() -> None:
    subset = ["diffpool", "lukasiewicz", "lukasiewicz_smooth"]
    records = run_gradient_stability_benchmark(method_keys=subset, seed=42)

    assert len(records) == len(subset)
    rec_by_key = {r.key: r for r in records}

    # DiffPool has 100% dense Jacobian (O(N) active nodes)
    assert rec_by_key["diffpool"].topo_jacobian_density >= 0.80
    assert rec_by_key["diffpool"].dead_superedge_ratio == 0.0
    assert rec_by_key["diffpool"].param_grad_finite_ratio == 1.0

    # Smooth Lukasiewicz avoids dead superedges
    assert rec_by_key["lukasiewicz_smooth"].param_grad_finite_ratio == 1.0
    assert rec_by_key["lukasiewicz_smooth"].dead_superedge_ratio <= rec_by_key["lukasiewicz"].dead_superedge_ratio


def test_efficiency_benchmark_metrics() -> None:
    subset = ["mincut", "goguen"]
    records = run_efficiency_benchmark(
        method_keys=subset,
        node_sizes=(60, 120),
        repeats=1,
        seed=2026,
    )

    assert len(records) == len(subset)
    for r in records:
        assert torch.isfinite(torch.tensor(r.empirical_scaling_exp))
        assert r.ms_at_smallest_n > 0.0
        assert r.ms_at_largest_n > 0.0
        assert r.peak_memory_mb_largest_n > 0.0
        assert r.throughput_nodes_per_sec > 0.0


def test_component_ablation_benchmarks() -> None:
    feat_records = run_feature_pooling_ablation(
        feature_kinds=("cog", "bipolar", "linear"),
        epochs=3,
        seed=123,
    )
    assert len(feat_records) == 3
    assert all(r.component_type == "feature_pool" for r in feat_records)
    assert all(0.0 <= r.metric_value <= 1.0 for r in feat_records)

    mem_records = run_membership_ablation(
        membership_kinds=("sigmoid", "softmax"),
        epochs=4,
        seed=42,
    )
    assert len(mem_records) == 2
    assert all(r.component_type == "membership" for r in mem_records)
    assert all(0.0 <= r.metric_value <= 1.0 for r in mem_records)

    full_abl = run_component_ablation_benchmark(seed=42)
    assert len(full_abl) >= 5


def test_domain_specific_leaderboards() -> None:
    subset = ["dmon", "goguen", "lukasiewicz_smooth", "bk_kleene_dienes", "goguen_bipolar"]
    res = run_comprehensive_analytics(
        method_keys=subset,
        unsupervised_epochs=6,
        clique_epochs=8,
        cls_epochs=3,
        profiler_sizes=(80, 160),
        profiler_repeats=1,
    )

    # Check that domain leaderboards exist and are correctly populated
    assert res.community_leaderboard is not None
    assert res.clique_leaderboard is not None
    assert res.topology_leaderboard is not None
    assert res.classification_leaderboard is not None
    assert res.gradient_leaderboard is not None
    assert res.efficiency_leaderboard is not None
    assert res.domain_winners is not None

    assert len(res.community_leaderboard) == len(subset)
    assert len(res.clique_leaderboard) == len(subset)

    # Each entry must have domain rank attributes
    for e in res.leaderboard:
        assert 1 <= e.community_rank <= len(subset)
        assert 1 <= e.clique_rank <= len(subset)
        assert 1 <= e.topology_rank <= len(subset)
        assert 1 <= e.classification_rank <= len(subset)
        assert 1 <= e.gradient_rank <= len(subset)
        assert 1 <= e.efficiency_rank <= len(subset)

    # Verify domain winners are top-ranked in their respective categories
    assert res.domain_winners["community"].community_rank == 1
    assert res.domain_winners["clique"].clique_rank == 1
    assert res.domain_winners["topology"].topology_rank == 1
    assert res.domain_winners["classification"].classification_rank == 1

    # Serialization test
    d = res.to_dict()
    assert "community_leaderboard" in d
    assert "domain_winners" in d


def test_cli_benchmark_and_separate_tests(tmp_path: Path) -> None:
    # Test specific benchmark dispatches
    assert cli_main(["benchmark", "--test", "community", "--sbm-epochs", "2", "--output-dir", str(tmp_path)]) == 0
    assert cli_main(["benchmark", "--test", "clique", "--clique-epochs", "2", "--output-dir", str(tmp_path)]) == 0
    assert cli_main(["benchmark", "--test", "topology", "--sbm-epochs", "2", "--output-dir", str(tmp_path)]) == 0
    assert cli_main(["benchmark", "--test", "classification", "--cls-epochs", "1", "--output-dir", str(tmp_path)]) == 0
    assert cli_main(["benchmark", "--test", "gradient", "--output-dir", str(tmp_path)]) == 0
    assert cli_main(["benchmark", "--test", "ablation", "--output-dir", str(tmp_path)]) == 0
