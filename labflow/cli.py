"""Command-line interface for Labflow: mathematical validation, audit stress testing, profiling, and comparative analytics."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional, Sequence

from labflow.analytics import (
    run_audit_stress_suite,
    run_comprehensive_analytics,
    run_scaling_profiler,
)
from labflow.benchmarks import (
    run_clique_motif_benchmark,
    run_community_detection_benchmark,
    run_component_ablation_benchmark,
    run_efficiency_benchmark,
    run_gradient_stability_benchmark,
    run_graph_classification_benchmark,
    run_planted_clique_benchmark,
    run_real_world_benchmark_suite,
    run_sbm_community_benchmark,
    run_topology_fidelity_benchmark,
)
from labflow.composite import CANONICAL_METHOD_SPECS
from labflow.reporting import (
    write_analytics_reports,
    write_labflow_reports,
    write_real_world_reports,
)
from labflow.validation import (
    run_full_gradcheck_suite,
    run_invariants_suite,
    run_sparse_equivalence_suite,
)


def _cmd_list() -> int:
    print(f"{'Code':<8} {'Key':<21} {'Section':<8} {'Membership':<13} {'Features':<9} {'Complexity':<24} Title")
    print("-" * 126)
    for s in CANONICAL_METHOD_SPECS:
        print(
            f"{s.code:<8} {s.key:<21} {s.section:<8} {s.membership:<13} {s.feature_pool:<9} {s.complexity:<24} {s.title}"
        )
    return 0


def _cmd_validate(output_dir: Optional[Path], only: Optional[str] = None) -> int:
    gc = run_full_gradcheck_suite() if only in (None, "gradcheck") else None
    sp = run_sparse_equivalence_suite() if only in (None, "sparse") else None
    inv = run_invariants_suite() if only in (None, "invariants") else None

    failed = False
    if gc is not None:
        ok = sum(r.passed for r in gc)
        print(f"[GradCheck] Analytical Jacobians vs Autograd & Finite-Diff: {ok}/{len(gc)} PASS")
        for r in gc:
            if not r.passed:
                failed = True
                print(
                    f"  FAIL: {r.name} (fwd={r.forward_max_err:.2e}, grad={r.autograd_grad_max_err:.2e}, num={r.numerical_gradcheck})"
                )

    if sp is not None:
        ok = sum(r.passed for r in sp)
        print(f"[SparseEq]  Two-step sparse decomposition O(MK+NK^2):     {ok}/{len(sp)} PASS")
        for r in sp:
            if not r.passed:
                failed = True
                print(f"  FAIL: {r.name} (diff={r.max_abs_diff:.2e})")

    if inv is not None:
        ok = sum(r.passed for r in inv)
        print(f"[Invariants] Spectral & topological invariants:              {ok}/{len(inv)} PASS")
        for r in inv:
            mark = "PASS" if r.passed else "FAIL"
            if not r.passed:
                failed = True
            print(f"  [{mark}] {r.code}: {r.title} -> {r.details}")

    if output_dir is not None:
        paths = write_labflow_reports(output_dir, gradchecks=gc, sparse_eqs=sp, invariants=inv)
        print(f"Reports saved to: {paths['markdown']} and {paths['json']}")

    return 1 if failed else 0


def _cmd_audit_check() -> int:
    records = run_audit_stress_suite()
    ok = sum(1 for r in records if r.passed)
    print(f"[AuditStress] Verification of stability audit fixes: {ok}/{len(records)} PASS")
    for r in records:
        mark = "PASS" if r.passed else "FAIL"
        print(f"  [{mark}] {r.code} ({r.audit_section}): {r.title}")
        print(f"         -> {r.details} [{r.elapsed_ms:.1f} ms]")
    return 0 if ok == len(records) else 1


def _cmd_profile(sizes_csv: str, repeats: int, device: str) -> int:
    sizes = [int(tok.strip()) for tok in sizes_csv.split(",") if tok.strip()]
    print(f"[Profiler] Asymptotic profiling across node sizes N={sizes} (device={device})...")
    _, summaries = run_scaling_profiler(node_sizes=sizes, repeats=repeats, device=device)
    print(f"{'Code':<8} {'Key':<21} {'Complexity':<24} {'alpha_emp':<10} {'min-N(ms)':<11} {'max-N(ms)':<11} {'Mem(MiB)'}")
    print("-" * 105)
    for s in summaries:
        print(
            f"{s.method_code:<8} {s.method_key:<21} {s.theoretical_complexity:<24} "
            f"{s.empirical_time_exponent:<10.2f} {s.ms_at_smallest_n:<11.2f} "
            f"{s.ms_at_largest_n:<11.2f} {s.peak_memory_mb_largest_n:.2f}"
        )
    return 0


def _cmd_compare(
    output_dir: Path,
    unsup_epochs: int,
    clique_epochs: int,
    cls_epochs: int,
    sizes_csv: str,
) -> int:
    sizes = [int(tok.strip()) for tok in sizes_csv.split(",") if tok.strip()]
    print("=== LABFLOW ANALYTICS: COMPREHENSIVE BENCHMARKING & MULTI-CRITERIA REDUCER COMPARISON ===")
    res = run_comprehensive_analytics(
        unsupervised_epochs=unsup_epochs,
        clique_epochs=clique_epochs,
        cls_epochs=cls_epochs,
        profiler_sizes=sizes,
    )

    ok_st = sum(1 for r in res.audit_stress_records if r.passed)
    print(f"\n[1] Stability Audit Stress Tests: {ok_st}/{len(res.audit_stress_records)} PASS")

    print("\n[2] Reducer Leaderboard (Top -> Bottom, ★ = Pareto-optimal):")
    print(
        f"{'Rank':<5} {'Code':<8} {'Key':<21} {'Pareto':<7} {'Comp.':<7} {'Comm.':<7} "
        f"{'Topo.':<7} {'Cliq.':<7} {'ClsAcc':<7} {'Grad':<7} {'NMI':<7} {'Dead%':<7} {'MaxN(ms)'}"
    )
    print("-" * 116)
    for e in res.leaderboard:
        p_flag = "★" if e.is_pareto_optimal else "-"
        print(
            f"#{e.rank:<4} {e.method_code:<8} {e.method_key:<21} {p_flag:<7} "
            f"{e.composite_score:<7.3f} {e.community_score:<7.3f} {e.topology_spectral_score:<7.3f} "
            f"{e.clique_isolation_score:<7.3f} {e.classification_score:<7.1%} "
            f"{e.gradient_stability_score:<7.3f} {e.mean_nmi:<7.3f} "
            f"{e.dead_superedge_ratio:<7.0%} {e.ms_large_graph:.2f}"
        )

    if res.domain_winners:
        print("\n[3] Domain-Specific Benchmark Champions (Decoupled Scientific Tests):")
        print(f"  - Community Detection:      {res.domain_winners['community'].method_code:<8} ({res.domain_winners['community'].method_key:<20}) score={res.domain_winners['community'].community_score:.3f}")
        print(f"  - Planted Clique Isolation: {res.domain_winners['clique'].method_code:<8} ({res.domain_winners['clique'].method_key:<20}) score={res.domain_winners['clique'].clique_isolation_score:.3f}")
        print(f"  - Topological Fidelity:     {res.domain_winners['topology'].method_code:<8} ({res.domain_winners['topology'].method_key:<20}) score={res.domain_winners['topology'].topology_spectral_score:.3f}")
        print(f"  - GNN Graph Classification: {res.domain_winners['classification'].method_code:<8} ({res.domain_winners['classification'].method_key:<20}) test_acc={res.domain_winners['classification'].classification_score:.1%}")
        print(f"  - Gradient Stability:       {res.domain_winners['gradient'].method_code:<8} ({res.domain_winners['gradient'].method_key:<20}) score={res.domain_winners['gradient'].gradient_stability_score:.3f}")
        print(f"  - Asymptotic Efficiency:    {res.domain_winners['efficiency'].method_code:<8} ({res.domain_winners['efficiency'].method_key:<20}) time={res.domain_winners['efficiency'].ms_large_graph:.2f} ms")

    paths = write_analytics_reports(output_dir=output_dir, analytics=res)
    print("\n[4] Generated analytical reports, CSV leaderboard, and diagnostic figures:")
    for k_name, p in paths.items():
        print(f"  - {k_name}: {p}")

    return 0 if ok_st == len(res.audit_stress_records) else 1


def _cmd_benchmark(
    output_dir: Path,
    sbm_epochs: int,
    clique_epochs: int,
    cls_epochs: int,
    test: str = "all",
) -> int:
    if test in ("all", "community"):
        print("[Test 1/6] Running Dedicated Community Detection & Modularity Benchmark...")
        comm = run_community_detection_benchmark(epochs=sbm_epochs)
        for r in comm:
            print(
                f"  {r.code:<7} {r.key:<21} NMI={r.nmi:.3f}  Q={r.modularity_q:.3f}  "
                f"Cond={r.conductance:.3f}  Purity={r.cluster_purity:.3f}  F1={r.macro_f1:.3f}"
            )

    if test in ("all", "clique"):
        print("[Test 2/6] Running Dedicated Planted Clique & Motif Isolation Benchmark...")
        clq = run_clique_motif_benchmark(epochs=clique_epochs)
        for r in clq:
            print(
                f"  {r.code:<7} {r.key:<21} density={r.internal_clique_density:.3f}  "
                f"F1={r.clique_recovery_f1:.3f}  reflexivity={r.diagonal_reflexivity:.3f}  contrast={r.motif_contrast_ratio:.2f}"
            )

    if test in ("all", "topology"):
        print("[Test 3/6] Running Dedicated Topological & Spectral Fidelity Benchmark...")
        topo = run_topology_fidelity_benchmark(epochs=max(sbm_epochs // 2, 20))
        for r in topo:
            print(
                f"  {r.code:<7} {r.key:<21} AUC={r.link_recon_auc:.3f}  RMSE={r.link_recon_rmse:.3f}  "
                f"SpecDist={r.spectral_laplacian_dist:.3f}  exact_zeros={r.exact_sparsity:.0%}  unit={r.unit_interval_compliance:.0%}"
            )

    if test in ("all", "classification"):
        print("[Test 4/6] Running Dedicated Hierarchical Graph Classification Benchmark...")
        cls_res = run_graph_classification_benchmark(epochs=cls_epochs)
        for r in cls_res:
            print(
                f"  {r.code:<7} {r.key:<21} train_acc={r.train_acc:.1%}  test_acc={r.test_acc:.1%}  "
                f"gap={r.generalization_gap:.1%}  loss={r.final_loss:.4f}  [{r.learned_params_summary}]"
            )

    if test in ("all", "gradient"):
        print("[Test 5/6] Running Dedicated Gradient Dynamics & Dead Super-Edge Benchmark...")
        grad = run_gradient_stability_benchmark()
        for r in grad:
            print(
                f"  {r.code:<7} {r.key:<21} jac_density={r.topo_jacobian_density:.1%}  "
                f"dead_edges={r.dead_superedge_ratio:.0%}  finite_grad={r.param_grad_finite_ratio:.0%}  grad_norm={r.mean_gradient_norm:.3e}"
            )

    if test in ("all", "efficiency"):
        print("[Test 6/6] Running Dedicated Computational Efficiency & Scaling Benchmark...")
        eff = run_efficiency_benchmark(node_sizes=(150, 450, 1200))
        for r in eff:
            print(
                f"  {r.code:<7} {r.key:<21} alpha={r.empirical_scaling_exp:.2f}  "
                f"ms_min={r.ms_at_smallest_n:.2f}  ms_max={r.ms_at_largest_n:.2f}  mem_mb={r.peak_memory_mb_largest_n:.2f}"
            )

    if test == "ablation":
        print("[Ablation] Running Component-Level Reduction Ablation Benchmark...")
        abl = run_component_ablation_benchmark()
        for r in abl:
            print(
                f"  [{r.component_type:<12}] {r.variant_name:<14} -> {r.metric_name}={r.metric_value:.3f} (delta={r.isolated_delta:+.3f})"
            )

    if test == "real-world":
        return _cmd_real_world(output_dir=output_dir, epochs=cls_epochs)

    if test == "all":
        sbm_legacy = run_sbm_community_benchmark(epochs=min(sbm_epochs, 25))
        clq_legacy = run_planted_clique_benchmark(epochs=min(clique_epochs, 35))
        cls_legacy = run_graph_classification_benchmark(epochs=min(cls_epochs, 10))
        paths = write_labflow_reports(
            output_dir=output_dir,
            sbm_records=sbm_legacy,
            clique_records=clq_legacy,
            cls_records=cls_legacy,
        )
        print(f"\nBenchmark reports and figures saved to: {paths['markdown']}")
    return 0


def _cmd_real_world(
    output_dir: Path,
    epochs: int = 16,
    datasets_csv: Optional[str] = None,
    all_methods: bool = False,
) -> int:
    print("=== LABFLOW REAL-WORLD BENCHMARK: 10 DATASETS (TUDataset & OGB ogbg-molhiv) ===")
    ds_list = (
        [tok.strip() for tok in datasets_csv.split(",") if tok.strip()]
        if datasets_csv
        else None
    )
    m_keys = "all" if all_methods else None
    suite = run_real_world_benchmark_suite(
        dataset_names=ds_list,
        method_keys=m_keys,
        epochs=epochs,
        verbose=True,
    )
    print("\n[Cross-Dataset Real-World Leaderboard]")
    print(
        f"{'Rank':<5} {'Code':<8} {'Key':<21} {'MeanAcc':<9} {'MacroF1':<9} "
        f"{'MeanAUC':<9} {'AvgRank':<8} {'Wins':<6} {'BioChem':<9} {'Social':<9} {'MolHIV-AUC'}"
    )
    print("-" * 110)
    for m in suite.method_leaderboard:
        print(
            f"#{m.rank:<4} {m.method_code:<8} {m.method_key:<21} {m.mean_test_acc:<9.1%} "
            f"{m.mean_macro_f1:<9.3f} {m.mean_roc_auc:<9.3f} {m.mean_rank:<8.2f} "
            f"{m.datasets_won_or_tied:<6d} {m.bio_chem_mean_acc:<9.1%} "
            f"{m.social_mean_acc:<9.1%} {m.ogb_molhiv_auc:.3f}"
        )
    paths = write_real_world_reports(output_dir=output_dir, suite=suite)
    print("\n[Generated Real-World Research Artifacts]")
    for k_name, p in paths.items():
        print(f"  - {k_name}: {p}")
    return 0


def _cmd_run_all(output_dir: Path, sbm_epochs: int, clique_epochs: int, cls_epochs: int) -> int:
    print("=== STEP 1: MATHEMATICAL VALIDATION (JACOBIANS, SPARSE DECOMPOSITION, INVARIANTS) ===")
    gc = run_full_gradcheck_suite()
    sp = run_sparse_equivalence_suite()
    inv = run_invariants_suite()

    gc_ok = sum(r.passed for r in gc)
    sp_ok = sum(r.passed for r in sp)
    inv_ok = sum(r.passed for r in inv)
    print(f"  GradCheck:          {gc_ok}/{len(gc)} PASS")
    print(f"  Sparse Equivalence: {sp_ok}/{len(sp)} PASS")
    print(f"  Invariants:         {inv_ok}/{len(inv)} PASS")

    if gc_ok != len(gc) or sp_ok != len(sp) or inv_ok != len(inv):
        print("Mathematical validation failed!")
        return 1

    print("\n=== STEP 2: EXPERIMENTAL BENCHMARKS ON SYNTHETIC GRAPHS ===")
    sbm = run_sbm_community_benchmark(epochs=sbm_epochs)
    clq = run_planted_clique_benchmark(epochs=clique_epochs)
    cls_res = run_graph_classification_benchmark(epochs=cls_epochs)

    paths = write_labflow_reports(
        output_dir=output_dir,
        gradchecks=gc,
        sparse_eqs=sp,
        invariants=inv,
        sbm_records=sbm,
        clique_records=clq,
        cls_records=cls_res,
    )
    print("\n=== COMPLETE! Generated Labflow Artifacts ===")
    for k_name, p in paths.items():
        print(f"  - {k_name}: {p}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="labflow",
        description="Labflow: Mathematical validation, stress audit, and analytical comparison of graph reduction methods",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("list", help="List all implemented reduction methods (1..13B + stabilized variants)")

    p_gc = sub.add_parser("gradcheck", help="Triple verification of analytical gradients (Jacobians)")
    p_gc.add_argument("--output-dir", type=Path, default=None)

    p_sp = sub.add_parser("sparse-eq", help="Verify equivalence of two-step and matrix-free sparse decomposition")
    p_sp.add_argument("--output-dir", type=Path, default=None)

    p_inv = sub.add_parser("invariants", help="Verify all 16 spectral and topological invariants")
    p_inv.add_argument("--output-dir", type=Path, default=None)

    p_val = sub.add_parser("validate", help="Run full mathematical validation (gradcheck + sparse + invariants)")
    p_val.add_argument("--output-dir", type=Path, default=Path("reports"))

    sub.add_parser("audit-check", help="Run 8 stability audit stress tests (ST-01 .. ST-08)")

    p_prof = sub.add_parser("profile", help="Profile asymptotic complexity and memory on large sparse graphs")
    p_prof.add_argument("--sizes", type=str, default="150,450,1200,2400")
    p_prof.add_argument("--repeats", type=int, default=3)
    p_prof.add_argument("--device", type=str, default="cpu")

    p_cmp = sub.add_parser(
        "compare",
        help="Comprehensive analytical comparison of all reducers (24 metrics, 5 benchmarks, Pareto front, CSV/MD/PNG)",
    )
    p_cmp.add_argument("--output-dir", type=Path, default=Path("reports"))
    p_cmp.add_argument("--unsup-epochs", type=int, default=65)
    p_cmp.add_argument("--clique-epochs", type=int, default=90)
    p_cmp.add_argument("--cls-epochs", type=int, default=20)
    p_cmp.add_argument("--sizes", type=str, default="150,450,1200,2400")

    p_bm = sub.add_parser("benchmark", help="Run experimental benchmarks (SBM, Planted Cliques, Classification)")
    p_bm.add_argument("--output-dir", type=Path, default=Path("reports"))
    p_bm.add_argument("--sbm-epochs", type=int, default=80)
    p_bm.add_argument("--clique-epochs", type=int, default=120)
    p_bm.add_argument("--cls-epochs", type=int, default=22)
    p_bm.add_argument(
        "--test",
        type=str,
        default="all",
        choices=["all", "community", "clique", "topology", "classification", "gradient", "efficiency", "ablation", "real-world"],
        help="Select specific benchmark test to run (splits composite into independent tests)",
    )

    p_rw = sub.add_parser(
        "real-world",
        help="Run real-world multi-domain benchmark on 10 datasets (PROTEINS, DD, NCI1, NCI109, ENZYMES, IMDB-B, IMDB-M, COLLAB, REDDIT-B, ogbg-molhiv)",
    )
    p_rw.add_argument("--output-dir", type=Path, default=Path("reports"))
    p_rw.add_argument("--epochs", type=int, default=16)
    p_rw.add_argument("--datasets", type=str, default=None, help="Comma-separated dataset names (default: all 10)")
    p_rw.add_argument("--all-methods", action="store_true", help="Evaluate all 22 canonical methods instead of 13 representative methods")

    p_all = sub.add_parser("run-all", help="Complete Labflow cycle: mathematical validation + benchmarks + reports & plots")
    p_all.add_argument("--output-dir", type=Path, default=Path("reports"))
    p_all.add_argument("--sbm-epochs", type=int, default=80)
    p_all.add_argument("--clique-epochs", type=int, default=120)
    p_all.add_argument("--cls-epochs", type=int, default=22)

    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "list":
        return _cmd_list()
    if args.command == "gradcheck":
        return _cmd_validate(args.output_dir, only="gradcheck")
    if args.command == "sparse-eq":
        return _cmd_validate(args.output_dir, only="sparse")
    if args.command == "invariants":
        return _cmd_validate(args.output_dir, only="invariants")
    if args.command == "validate":
        return _cmd_validate(args.output_dir, only=None)
    if args.command == "audit-check":
        return _cmd_audit_check()
    if args.command == "profile":
        return _cmd_profile(args.sizes, args.repeats, args.device)
    if args.command == "compare":
        return _cmd_compare(
            args.output_dir,
            args.unsup_epochs,
            args.clique_epochs,
            args.cls_epochs,
            args.sizes,
        )
    if args.command == "benchmark":
        return _cmd_benchmark(
            args.output_dir,
            args.sbm_epochs,
            args.clique_epochs,
            args.cls_epochs,
            test=args.test,
        )
    if args.command == "real-world":
        return _cmd_real_world(
            output_dir=args.output_dir,
            epochs=args.epochs,
            datasets_csv=args.datasets,
            all_methods=args.all_methods,
        )
    if args.command == "run-all":
        return _cmd_run_all(args.output_dir, args.sbm_epochs, args.clique_epochs, args.cls_epochs)
    return 1


if __name__ == "__main__":
    sys.exit(main())
