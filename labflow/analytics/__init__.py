"""Labflow Analytical Software Package: multi-domain metrics, audit stress testing, scaling profiler, and comparator."""

from labflow.analytics.comparator import (
    AnalyticsPackageResult,
    ReducerLeaderboardEntry,
    run_comprehensive_analytics,
    train_and_profile_unsupervised_dataset,
)
from labflow.analytics.metrics import (
    ReducerMetricProfile,
    compute_ari_score,
    compute_feature_reconstruction_r2,
    compute_jacobian_and_dead_superedges,
    compute_link_reconstruction_metrics,
    compute_nmi_score,
    compute_purity_and_macro_f1,
    compute_sparse_cut_and_modularity,
    compute_spectral_laplacian_distance,
    evaluate_reducer_metrics,
)
from labflow.analytics.profiler import (
    ReducerScalingSummary,
    ScalingPointRecord,
    run_scaling_profiler,
)
from labflow.analytics.stability_audit import (
    AuditStressRecord,
    run_audit_stress_suite,
)

__all__ = [
    "ReducerMetricProfile",
    "evaluate_reducer_metrics",
    "compute_nmi_score",
    "compute_ari_score",
    "compute_purity_and_macro_f1",
    "compute_sparse_cut_and_modularity",
    "compute_spectral_laplacian_distance",
    "compute_link_reconstruction_metrics",
    "compute_feature_reconstruction_r2",
    "compute_jacobian_and_dead_superedges",
    "AuditStressRecord",
    "run_audit_stress_suite",
    "ScalingPointRecord",
    "ReducerScalingSummary",
    "run_scaling_profiler",
    "ReducerLeaderboardEntry",
    "AnalyticsPackageResult",
    "train_and_profile_unsupervised_dataset",
    "run_comprehensive_analytics",
]
