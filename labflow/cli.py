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
    run_graph_classification_benchmark,
    run_planted_clique_benchmark,
    run_sbm_community_benchmark,
)
from labflow.composite import CANONICAL_METHOD_SPECS
from labflow.reporting import write_analytics_reports, write_labflow_reports
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

    paths = write_analytics_reports(output_dir=output_dir, analytics=res)
    print("\n[3] Generated analytical reports, CSV leaderboard, and diagnostic figures:")
    for k_name, p in paths.items():
        print(f"  - {k_name}: {p}")

    return 0 if ok_st == len(res.audit_stress_records) else 1


def _cmd_benchmark(output_dir: Path, sbm_epochs: int, clique_epochs: int, cls_epochs: int) -> int:
    print("[1/3] Running SBM Community & Jacobian Sparsity benchmark...")
    sbm = run_sbm_community_benchmark(epochs=sbm_epochs)
    for r in sbm:
        print(
            f"  {r.code:<7} {r.key:<21} NMI={r.nmi:.3f}  Q={r.modularity_q:.3f}  "
            f"max(A_new)={r.adj_new_max:.3f}  zeros={r.exact_sparsity:.0%}  active_grad={r.topo_grad_active_ratio:.1%}"
        )

    print("[2/3] Running Planted Clique Isolation benchmark...")
    clq = run_planted_clique_benchmark(epochs=clique_epochs)
    for r in clq:
        print(
            f"  {r.code:<7} {r.key:<21} clique_density={r.internal_clique_density:.3f}  "
            f"F1={r.clique_recovery_f1:.3f}  diag(A_new)={r.mean_diag_a_new:.3f}"
        )

    print("[3/3] Running End-to-End Graph Classification benchmark...")
    cls_res = run_graph_classification_benchmark(epochs=cls_epochs)
    for r in cls_res:
        print(
            f"  {r.code:<7} {r.key:<21} train_acc={r.train_acc:.1%}  test_acc={r.test_acc:.1%}  [{r.learned_params_summary}]"
        )

    paths = write_labflow_reports(
        output_dir=output_dir,
        sbm_records=sbm,
        clique_records=clq,
        cls_records=cls_res,
    )
    print(f"Benchmark reports and figures saved to: {paths['markdown']}")
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
        return _cmd_benchmark(args.output_dir, args.sbm_epochs, args.clique_epochs, args.cls_epochs)
    if args.command == "run-all":
        return _cmd_run_all(args.output_dir, args.sbm_epochs, args.clique_epochs, args.cls_epochs)
    return 1


if __name__ == "__main__":
    sys.exit(main())
