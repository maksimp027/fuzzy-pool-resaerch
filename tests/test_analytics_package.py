"""Unit & integration tests for the Labflow Analytical Software Package (`labflow.analytics`)."""

from __future__ import annotations

from pathlib import Path

import torch

from labflow.analytics import (
    evaluate_reducer_metrics,
    run_comprehensive_analytics,
    run_scaling_profiler,
)
from labflow.cli import main as cli_main
from labflow.composite import GraphReductionLayer
from labflow.datasets import (
    generate_large_sparse_sbm_graph,
    generate_multiscale_noisy_sbm_graph,
    load_classic_network,
)
from labflow.reporting import write_analytics_reports


def test_classic_and_large_sparse_datasets() -> None:
    for net_name in ("karate", "les_miserables", "florentine", "davis"):
        g = load_classic_network(net_name, feature_dim=12, seed=42)
        assert g.num_nodes >= 15
        assert g.num_features == 12
        assert g.node_labels is not None
        assert g.adj is not None
        assert float(g.adj.max().item()) <= 1.0 + 1e-6

    g_multi = generate_multiscale_noisy_sbm_graph(block_sizes=(6, 12, 18), feature_dim=10, seed=17)
    assert g_multi.num_nodes == 36
    assert g_multi.node_labels is not None

    g_large = generate_large_sparse_sbm_graph(num_nodes=1500, num_blocks=5, avg_degree=10.0, feature_dim=8)
    assert g_large.adj is None  # Strictly sparse!
    assert g_large.num_nodes == 1500
    assert g_large.num_edges > 2000


def test_24_metric_profile_evaluation() -> None:
    g = load_classic_network("karate", feature_dim=12, seed=7)
    for key, code in (
        ("dmon", "M3"),
        ("goguen", "M5"),
        ("lukasiewicz_smooth", "M6-Sm"),
        ("probsum_norm", "M10-Nm"),
        ("bk_reichenbach", "M11C"),
    ):
        layer = GraphReductionLayer.from_spec(key, in_features=12, out_features=10, num_clusters=2)
        prof = evaluate_reducer_metrics(layer, g, method_key=key, method_code=code, dataset_name="karate")
        assert 0.0 <= prof.nmi <= 1.0
        assert -1.0 <= prof.ari <= 1.0
        assert 0.0 <= prof.cluster_purity <= 1.0
        assert 0.0 <= prof.macro_f1 <= 1.0
        assert -1.0 <= prof.hard_modularity_q <= 1.0
        assert -1.0 <= prof.fuzzy_modularity_q <= 1.0
        assert 0.0 <= prof.conductance <= 1.0
        assert 0.0 <= prof.link_recon_auc <= 1.0
        assert 0.0 < prof.unit_interval_compliance <= 1.0
        assert 0.0 <= prof.dead_superedge_ratio <= 1.0
        assert prof.param_grad_finite_ratio == 1.0


def test_profiler_comparator_and_cli_reporting(tmp_path: Path) -> None:
    subset_keys = ["dmon", "goguen", "lukasiewicz_smooth", "probsum_norm", "bk_reichenbach"]
    pts, summaries = run_scaling_profiler(
        method_keys=subset_keys,
        node_sizes=(100, 300, 700),
        repeats=1,
    )
    assert len(pts) == len(subset_keys) * 3
    assert len(summaries) == len(subset_keys)
    assert all(s.ms_at_largest_n > 0.0 and s.peak_memory_mb_largest_n > 0.0 for s in summaries)

    res = run_comprehensive_analytics(
        method_keys=subset_keys,
        unsupervised_epochs=8,
        clique_epochs=10,
        cls_epochs=4,
        profiler_sizes=(100, 300),
        profiler_repeats=1,
    )
    assert len(res.leaderboard) == len(subset_keys)
    assert [e.rank for e in res.leaderboard] == [1, 2, 3, 4, 5]
    assert any(e.is_pareto_optimal for e in res.leaderboard)
    assert all(r.passed for r in res.audit_stress_records)

    paths = write_analytics_reports(tmp_path, res)
    assert paths["analytics_json"].exists()
    assert paths["analytics_markdown"].exists()
    assert paths["leaderboard_csv"].exists()
    assert (tmp_path / "figures" / "multi_criteria_heatmap.png").exists()
    assert (tmp_path / "figures" / "scaling_and_pareto.png").exists()

    # CLI subcommands
    assert cli_main(["audit-check"]) == 0
    assert cli_main(["profile", "--sizes", "100,250", "--repeats", "1"]) == 0
