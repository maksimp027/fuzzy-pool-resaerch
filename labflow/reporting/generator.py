"""Markdown, JSON, CSV leaderboard, and Matplotlib diagnostic chart generator for Labflow experiments."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from labflow.benchmarks.runner import (
    ClassificationBenchmarkRecord,
    CliqueBenchmarkRecord,
    SBMBenchmarkRecord,
)
from labflow.validation.gradcheck_suite import GradCheckRecord
from labflow.validation.invariants import InvariantRecord
from labflow.validation.sparse_equivalence import SparseEquivalenceRecord

if TYPE_CHECKING:
    from labflow.analytics.comparator import AnalyticsPackageResult


def generate_diagnostic_plots(
    output_dir: Path,
    sbm_records: Optional[List[SBMBenchmarkRecord]] = None,
    clique_records: Optional[List[CliqueBenchmarkRecord]] = None,
    cls_records: Optional[List[ClassificationBenchmarkRecord]] = None,
) -> List[Path]:
    """Generate comparative Matplotlib figures for benchmarks and Jacobian sparsity."""
    fig_dir = output_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    saved_paths: List[Path] = []

    if sbm_records:
        codes = [r.code for r in sbm_records]
        nmis = [r.nmi for r in sbm_records]
        mods = [max(0.0, r.modularity_q) for r in sbm_records]
        grad_ratios = [r.topo_grad_active_ratio * 100.0 for r in sbm_records]

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 8.5), sharex=True)
        x_pos = np.arange(len(codes))
        width = 0.38

        ax1.bar(x_pos - width / 2, nmis, width, label="NMI (SBM Communities)", color="#2563eb")
        ax1.bar(x_pos + width / 2, mods, width, label="Modularity Q", color="#10b981")
        ax1.set_ylabel("Clustering Quality")
        ax1.set_ylim(0.0, 1.08)
        ax1.set_title("Experimental Comparison of Graph Reduction Methods on SBM Benchmark (N=45, K=3)")
        ax1.grid(axis="y", linestyle="--", alpha=0.4)
        ax1.legend(loc="upper right")

        colors = [
            "#ef4444" if ratio > 65.0 else "#f59e0b" if ratio > 30.0 else "#6366f1"
            for ratio in grad_ratios
        ]
        ax2.bar(x_pos, grad_ratios, width=0.55, color=colors)
        ax2.set_ylabel("Active Nodes in dA_new/dS (%)")
        ax2.set_xlabel("Graph Reduction Method Code")
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(codes, rotation=35, ha="right")
        ax2.set_ylim(0.0, 108.0)
        ax2.set_title("Empirical Topological Jacobian Density dA_new / dS (O(N) vs Critical Edges)")
        ax2.grid(axis="y", linestyle="--", alpha=0.4)

        fig.tight_layout()
        p1 = fig_dir / "sbm_and_jacobian_comparison.png"
        fig.savefig(p1, dpi=160)
        plt.close(fig)
        saved_paths.append(p1)

    if clique_records or cls_records:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

        if clique_records:
            ax = axes[0]
            c_codes = [r.code for r in clique_records]
            dens = [r.internal_clique_density for r in clique_records]
            f1s = [r.clique_recovery_f1 for r in clique_records]
            y_pos = np.arange(len(c_codes))
            h_bar = 0.38
            ax.barh(y_pos - h_bar / 2, dens, h_bar, label="Clique Subgraph Density", color="#8b5cf6")
            ax.barh(y_pos + h_bar / 2, f1s, h_bar, label="Clique Recovery F1", color="#ec4899")
            ax.set_yticks(y_pos)
            ax.set_yticklabels(c_codes)
            ax.invert_yaxis()
            ax.set_xlim(0.0, 1.08)
            ax.set_xlabel("Metric Score [0, 1]")
            ax.set_title("Planted Clique Isolation: Bandler-Kohout vs Existential Methods")
            ax.grid(axis="x", linestyle="--", alpha=0.4)
            ax.legend(loc="lower right")

        if cls_records:
            ax = axes[1]
            k_codes = [r.code for r in cls_records]
            test_accs = [r.test_acc * 100.0 for r in cls_records]
            x_p = np.arange(len(k_codes))
            ax.bar(x_p, test_accs, color="#0ea5e9", width=0.6)
            ax.set_xticks(x_p)
            ax.set_xticklabels(k_codes, rotation=45, ha="right")
            ax.set_ylim(0.0, 108.0)
            ax.set_ylabel("Test Accuracy (%)")
            ax.set_title("Hierarchical Graph Classification (GNN -> Pool -> GNN -> Readout)")
            ax.grid(axis="y", linestyle="--", alpha=0.4)
            ax.grid(axis="y", linestyle="--", alpha=0.4)

        fig.tight_layout()
        p2 = fig_dir / "clique_and_classification.png"
        fig.savefig(p2, dpi=160)
        plt.close(fig)
        saved_paths.append(p2)

    return saved_paths


def write_labflow_reports(
    output_dir: Path,
    gradchecks: Optional[List[GradCheckRecord]] = None,
    sparse_eqs: Optional[List[SparseEquivalenceRecord]] = None,
    invariants: Optional[List[InvariantRecord]] = None,
    sbm_records: Optional[List[SBMBenchmarkRecord]] = None,
    clique_records: Optional[List[CliqueBenchmarkRecord]] = None,
    cls_records: Optional[List[ClassificationBenchmarkRecord]] = None,
) -> Dict[str, Path]:
    """Write JSON data, Markdown summary report, and PNG figures to `output_dir`."""
    output_dir.mkdir(parents=True, exist_ok=True)
    fig_paths = generate_diagnostic_plots(
        output_dir=output_dir,
        sbm_records=sbm_records,
        clique_records=clique_records,
        cls_records=cls_records,
    )

    payload = {
        "gradcheck": [r.to_dict() for r in (gradchecks or [])],
        "sparse_equivalence": [r.to_dict() for r in (sparse_eqs or [])],
        "invariants": [r.to_dict() for r in (invariants or [])],
        "benchmarks": {
            "sbm_community": [r.to_dict() for r in (sbm_records or [])],
            "planted_clique": [r.to_dict() for r in (clique_records or [])],
            "graph_classification": [r.to_dict() for r in (cls_records or [])],
        },
        "figures": [str(p) for p in fig_paths],
    }

    json_path = output_dir / "labflow_results.json"
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    lines: List[str] = [
        "# Labflow: Mathematical Validation and Graph Reduction Benchmarking Report",
        "",
    ]

    if gradchecks is not None:
        n_ok = sum(1 for r in gradchecks if r.passed)
        lines.extend(
            [
                f"## 1. Triple Verification of Analytical Jacobians (GradCheck): {n_ok}/{len(gradchecks)} passed",
                "",
                "| Operator / Method | Status | Max |Fwd - Ref| | Max |Grad - Autograd| | Numerical Finite-Diff |",
                "|---|---|---|---|---|",
            ]
        )
        for r in gradchecks:
            status = "PASS" if r.passed else "FAIL"
            num_s = "PASS" if r.numerical_gradcheck else "FAIL"
            lines.append(
                f"| {r.name} | **{status}** | `{r.forward_max_err:.2e}` | `{r.autograd_grad_max_err:.2e}` | {num_s} |"
            )
        lines.append("")

    if sparse_eqs is not None:
        n_ok = sum(1 for r in sparse_eqs if r.passed)
        lines.extend(
            [
                f"## 2. Equivalence of Two-Step Sparse Decomposition O(MK + NK^2): {n_ok}/{len(sparse_eqs)} passed",
                "",
                "| Verification | Status | Max |Sparse - Dense| | Tolerance |",
                "|---|---|---|---|",
            ]
        )
        for r in sparse_eqs:
            status = "PASS" if r.passed else "FAIL"
            lines.append(f"| {r.name} | **{status}** | `{r.max_abs_diff:.2e}` | `{r.tolerance:.1e}` |")
        lines.append("")

    if invariants is not None:
        n_ok = sum(1 for r in invariants if r.passed)
        lines.extend(
            [
                f"## 3. Spectral, Algebraic, and Topological Invariants: {n_ok}/{len(invariants)} passed",
                "",
                "| Code | Property / Theorem | Status | Measurement Details |",
                "|---|---|---|---|",
            ]
        )
        for r in invariants:
            status = "PASS" if r.passed else "FAIL"
            lines.append(f"| `{r.code}` | {r.title} | **{status}** | {r.details} |")
        lines.append("")

    if sbm_records is not None:
        lines.extend(
            [
                "## 4. Benchmark I: SBM Community Detection & Topological Gradient Sparsity",
                "",
                "| Code | Method | NMI | Modularity Q | Conductance | max(A_new) | Exact Zeros A_new | Active dA_new/dS | Time (ms/step) |",
                "|---|---|---|---|---|---|---|---|---|",
            ]
        )
        for r in sbm_records:
            lines.append(
                f"| `{r.code}` | {r.title} | `{r.nmi:.3f}` | `{r.modularity_q:.3f}` | `{r.conductance:.3f}` | "
                f"`{r.adj_new_max:.3f}` | `{r.exact_sparsity:.1%}` | `{r.topo_grad_active_ratio:.1%}` | `{r.ms_per_step:.2f}` |"
            )
        lines.append("")

    if clique_records is not None:
        lines.extend(
            [
                "## 5. Benchmark II: Planted Clique Isolation",
                "",
                "| Code | Method | Internal Clique Density | Clique Recovery F1 | Mean Diagonal A_new(I,I) |",
                "|---|---|---|---|---|",
            ]
        )
        for r in clique_records:
            lines.append(
                f"| `{r.code}` | {r.title} | `{r.internal_clique_density:.3f}` | `{r.clique_recovery_f1:.3f}` | `{r.mean_diag_a_new:.3f}` |"
            )
        lines.append("")

    if cls_records is not None:
        lines.extend(
            [
                "## 6. Benchmark III: Hierarchical Graph Classification (End-to-End GNN)",
                "",
                "| Code | Method | Train Acc | Test Acc | Final Loss | Learned Layer Parameters |",
                "|---|---|---|---|---|---|",
            ]
        )
        for r in cls_records:
            lines.append(
                f"| `{r.code}` | {r.title} | `{r.train_acc:.1%}` | `{r.test_acc:.1%}` | `{r.final_loss:.4f}` | `{r.learned_params_summary}` |"
            )
        lines.append("")

    md_path = output_dir / "labflow_report.md"
    md_path.write_text("\n".join(lines), encoding="utf-8")

    result: Dict[str, Path] = {"json": json_path, "markdown": md_path}
    for idx, fp in enumerate(fig_paths):
        result[f"figure_{idx}"] = fp
    return result


def generate_analytics_plots(
    output_dir: Path,
    analytics: "AnalyticsPackageResult",
) -> List[Path]:
    """Generate multi-criteria heatmap and scaling/Pareto frontier diagnostic figures."""
    fig_dir = output_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    saved_paths: List[Path] = []

    lb = analytics.leaderboard
    if lb:
        criteria_labels = [
            "Communities\n(NMI/ARI/Q)",
            "Topology &\nSpectrum",
            "Clique\nIsolation",
            "GNN\nClassification",
            "Gradient\nHealth",
            "Efficiency\nO(N)",
            "Composite\nIndex",
        ]
        row_labels = [
            f"#{e.rank} [{e.method_code}] {e.method_key}{' ★' if e.is_pareto_optimal else ''}"
            for e in lb
        ]
        mat = np.array(
            [
                [
                    e.community_score,
                    e.topology_spectral_score,
                    e.clique_isolation_score,
                    e.classification_score,
                    e.gradient_stability_score,
                    e.efficiency_score,
                    e.composite_score,
                ]
                for e in lb
            ],
            dtype=np.float64,
        )

        fig_h = max(7.0, 0.38 * len(lb) + 1.8)
        fig, ax = plt.subplots(figsize=(11.5, fig_h))
        im = ax.imshow(mat, cmap="YlGnBu", aspect="auto", vmin=0.0, vmax=1.0)
        ax.set_xticks(np.arange(len(criteria_labels)))
        ax.set_xticklabels(criteria_labels, fontsize=9.5)
        ax.set_yticks(np.arange(len(row_labels)))
        ax.set_yticklabels(row_labels, fontsize=9)
        ax.set_title("Multi-Criteria Evaluation Matrix of Labflow Reducers (★ = Pareto-optimal)")

        for r_i in range(mat.shape[0]):
            for c_i in range(mat.shape[1]):
                val = mat[r_i, c_i]
                txt_color = "white" if val > 0.62 else "#111827"
                ax.text(c_i, r_i, f"{val:.2f}", ha="center", va="center", color=txt_color, fontsize=8.5)

        fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02, label="Normalized Score [0, 1]")
        fig.tight_layout()
        p_heat = fig_dir / "multi_criteria_heatmap.png"
        fig.savefig(p_heat, dpi=165)
        plt.close(fig)
        saved_paths.append(p_heat)

    if analytics.scaling_points and lb:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14.5, 5.8))

        # Left: Log-Log runtime scaling across graph sizes N
        highlight_keys = [
            "diffpool",
            "dmon",
            "goguen",
            "lukasiewicz_smooth",
            "schweizer_sklar",
            "probsum_norm",
            "bk_reichenbach",
        ]
        present_keys = [k for k in highlight_keys if any(p.method_key == k for p in analytics.scaling_points)]
        if not present_keys:
            present_keys = list({p.method_key for p in analytics.scaling_points})[:7]

        for m_key in present_keys:
            pts = sorted(
                [p for p in analytics.scaling_points if p.method_key == m_key],
                key=lambda x: x.num_nodes,
            )
            ns = [p.num_nodes for p in pts]
            ts = [p.total_ms for p in pts]
            code = pts[0].method_code
            ax1.plot(ns, ts, marker="o", linewidth=1.9, label=f"{code}: {m_key}")

        ax1.set_xscale("log")
        ax1.set_yscale("log")
        ax1.set_xlabel("Number of Sparse Graph Nodes N (log)")
        ax1.set_ylabel("Forward + Backward Runtime, ms (log)")
        ax1.set_title("Asymptotic Scaling of True Sparse Pipeline")
        ax1.grid(True, which="both", linestyle="--", alpha=0.4)
        ax1.legend(fontsize=8.5, loc="upper left")

        # Right: 2D Pareto Frontier (Quality Index vs Runtime at Largest N)
        pareto_pts = sorted([e for e in lb if e.is_pareto_optimal], key=lambda e: e.ms_large_graph)
        non_pareto_pts = [e for e in lb if not e.is_pareto_optimal]

        if non_pareto_pts:
            ax2.scatter(
                [e.ms_large_graph for e in non_pareto_pts],
                [e.quality_index for e in non_pareto_pts],
                s=58,
                color="#64748b",
                alpha=0.75,
                label="Other Methods",
            )
        if pareto_pts:
            px = [e.ms_large_graph for e in pareto_pts]
            py = [e.quality_index for e in pareto_pts]
            ax2.plot(px, py, linestyle="--", color="#e11d48", linewidth=1.5, alpha=0.8)
            ax2.scatter(px, py, s=115, marker="*", color="#e11d48", label="Pareto Frontier (Quality vs Time)")

        for e in lb:
            ax2.annotate(
                e.method_code,
                (e.ms_large_graph, e.quality_index),
                textcoords="offset points",
                xytext=(5, 4),
                fontsize=8,
            )

        ax2.set_xlabel("Processing Time on Large Sparse Graph (ms)")
        ax2.set_ylabel("Composite Quality Index [0, 1]")
        ax2.set_title("Pareto-Optimal Frontier of Graph Reduction Methods")
        ax2.grid(True, linestyle="--", alpha=0.4)
        ax2.legend(loc="lower right")

        fig.tight_layout()
        p_par = fig_dir / "scaling_and_pareto.png"
        fig.savefig(p_par, dpi=165)
        plt.close(fig)
        saved_paths.append(p_par)

    return saved_paths


def write_analytics_reports(
    output_dir: Path,
    analytics: "AnalyticsPackageResult",
) -> Dict[str, Path]:
    """Export comprehensive Markdown report, JSON bundle, CSV leaderboard, and Matplotlib figures."""
    output_dir.mkdir(parents=True, exist_ok=True)
    fig_paths = generate_analytics_plots(output_dir=output_dir, analytics=analytics)

    json_path = output_dir / "analytics_results.json"
    payload = analytics.to_dict()
    payload["figures"] = [str(p) for p in fig_paths]
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    # CSV Leaderboard
    csv_path = output_dir / "reducer_leaderboard.csv"
    if analytics.leaderboard:
        fieldnames = list(analytics.leaderboard[0].to_dict().keys())
        with csv_path.open("w", newline="", encoding="utf-8") as f_csv:
            writer = csv.DictWriter(f_csv, fieldnames=fieldnames)
            writer.writeheader()
            for entry in analytics.leaderboard:
                writer.writerow(entry.to_dict())
    else:
        csv_path.write_text("", encoding="utf-8")

    # Markdown Analytical Report
    lines: List[str] = [
        "# Labflow Analytics: Comprehensive Analytical Report and Reducer Comparison",
        "",
        "This report was generated by the `labflow.analytics` package following mathematical and algorithmic verification of all reduction methods.",
        "",
    ]

    # 1. Audit Stress Suite
    st_records = analytics.audit_stress_records
    n_st_ok = sum(1 for r in st_records if r.passed)
    lines.extend(
        [
            f"## 1. Stress Verification of Stability Audit Fixes: {n_st_ok}/{len(st_records)} PASS",
            "",
            "| Code | Audit Section | Stress Test | Status | Measurement Details | Time (ms) |",
            "|---|---|---|---|---|---|",
        ]
    )
    for r in st_records:
        st_mark = "PASS" if r.passed else "FAIL"
        lines.append(
            f"| `{r.code}` | {r.audit_section} | {r.title} | **{st_mark}** | {r.details} | `{r.elapsed_ms:.1f}` |"
        )
    lines.append("")

    # 2. Multi-Criteria Leaderboard
    lines.extend(
        [
            "## 2. Consolidated Multi-Criteria Ranking and Pareto Frontier",
            "",
            "| Rank | Code | Method | Pareto | Composite | Communities | Topology | Cliques | GNN Acc | Gradient | Speed | Mean NMI | Dead Edges | Max-N Time (ms) | alpha_emp |",
            "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
        ]
    )
    for e in analytics.leaderboard:
        p_star = "★ YES" if e.is_pareto_optimal else "—"
        lines.append(
            f"| **{e.rank}** | `{e.method_code}` | `{e.method_key}` | {p_star} | **{e.composite_score:.3f}** | "
            f"`{e.community_score:.3f}` | `{e.topology_spectral_score:.3f}` | `{e.clique_isolation_score:.3f}` | "
            f"`{e.classification_score:.1%}` | `{e.gradient_stability_score:.3f}` | `{e.efficiency_score:.3f}` | "
            f"`{e.mean_nmi:.3f}` | `{e.dead_superedge_ratio:.0%}` | `{e.ms_large_graph:.2f}` | `{e.empirical_scaling_exp:.2f}` |"
        )
    lines.append("")

    # 2b. Domain Recommendations
    lines.extend(
        [
            "### Recommended Application Domains (Top Methods & Specialized Operators)",
            "",
            "| Code | Key | Theoretical Complexity | Recommended Use Scenario |",
            "|---|---|---|---|",
        ]
    )
    for e in analytics.leaderboard:
        lines.append(
            f"| `{e.method_code}` | `{e.method_key}` | `{e.complexity}` | {e.recommended_use_case} |"
        )
    lines.append("")

    # 3. Per-Dataset 24-Metric Summary Tables
    ds_titles = {
        "sbm_clean": "Balanced SBM Community Graph (N=45, K=3)",
        "sbm_multiscale_noisy": "Unbalanced Multi-Scale Noisy SBM (N=60, K=4)",
        "karate_real": "Zachary's Karate Club Real Social Network (N=34, K=2)",
    }
    lines.append("## 3. Detailed Reduction Quality Profiles across Datasets (24 Metrics)")
    lines.append("")
    for ds_key, profs in analytics.dataset_profiles.items():
        title_ds = ds_titles.get(ds_key, ds_key)
        lines.extend(
            [
                f"### Dataset: {title_ds}",
                "",
                "| Code | Method | NMI | ARI | Macro-F1 | Hard Q | Fuzzy Q | Cond. | SpecDist | Link AUC | [0,1] | Sparsity | Bezdek PC | Feat R^2 | Jac.Dens | DeadSuper |",
                "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
            ]
        )
        for p in profs:
            lines.append(
                f"| `{p.method_code}` | `{p.method_key}` | `{p.nmi:.3f}` | `{p.ari:.3f}` | `{p.macro_f1:.3f}` | "
                f"`{p.hard_modularity_q:.3f}` | `{p.fuzzy_modularity_q:.3f}` | `{p.conductance:.3f}` | "
                f"`{p.spectral_laplacian_dist:.3f}` | `{p.link_recon_auc:.3f}` | `{p.unit_interval_compliance:.2f}` | "
                f"`{p.exact_sparsity:.0%}` | `{p.bezdek_pc:.3f}` | `{p.feature_recon_r2:.3f}` | "
                f"`{p.topo_jacobian_density:.1%}` | `{p.dead_superedge_ratio:.0%}` |"
            )
        lines.append("")

    # 4. Scaling & Memory Summary
    if analytics.scaling_summaries:
        lines.extend(
            [
                "## 4. Asymptotic Profiling of Computational Complexity & Memory (True Sparse COO)",
                "",
                "| Code | Method | Theoretical Complexity | Empirical Exponent alpha (T ~ N^alpha) | Min-N Time (ms) | Max-N Time (ms) | Peak Memory (MiB) |",
                "|---|---|---|---|---|---|---|",
            ]
        )
        for s in analytics.scaling_summaries:
            lines.append(
                f"| `{s.method_code}` | `{s.method_key}` | `{s.theoretical_complexity}` | `{s.empirical_time_exponent:.2f}` | "
                f"`{s.ms_at_smallest_n:.2f}` | `{s.ms_at_largest_n:.2f}` | `{s.peak_memory_mb_largest_n:.2f}` |"
            )
        lines.append("")

    md_path = output_dir / "analytics_report.md"
    md_path.write_text("\n".join(lines), encoding="utf-8")

    result: Dict[str, Path] = {
        "analytics_json": json_path,
        "analytics_markdown": md_path,
        "leaderboard_csv": csv_path,
    }
    for idx, fp in enumerate(fig_paths):
        result[f"analytics_figure_{idx}"] = fp
    return result
