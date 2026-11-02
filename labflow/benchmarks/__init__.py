"""Experimental benchmark suite for Labflow graph reduction methods.

Separates composite evaluation into dedicated, scientifically grounded benchmark tests:
- Community Detection & Partition Alignment Benchmark
- Structural Motif & Planted Clique Isolation Benchmark
- Topological & Spectral Fidelity Benchmark
- Hierarchical Graph Classification Benchmark
- Gradient Dynamics & Optimization Health Benchmark
- Computational Efficiency & Asymptotic Scaling Benchmark
- Component-Level Modular Ablation Benchmark
"""

from labflow.benchmarks.ablation import (
    ComponentAblationRecord,
    run_component_ablation_benchmark,
    run_feature_pooling_ablation,
    run_membership_ablation,
)
from labflow.benchmarks.classification import (
    ClassificationBenchmarkRecord,
    run_graph_classification_benchmark,
)
from labflow.benchmarks.clique_motif import (
    CliqueMotifBenchmarkRecord,
    run_clique_motif_benchmark,
)
from labflow.benchmarks.community import (
    CommunityBenchmarkRecord,
    run_community_detection_benchmark,
)
from labflow.benchmarks.efficiency import (
    EfficiencyBenchmarkRecord,
    run_efficiency_benchmark,
)
from labflow.benchmarks.gradient_stability import (
    GradientStabilityBenchmarkRecord,
    run_gradient_stability_benchmark,
)
from labflow.benchmarks.runner import (
    CliqueBenchmarkRecord,
    SBMBenchmarkRecord,
    SeparateBenchmarkSuiteResult,
    run_all_separate_benchmarks,
    run_planted_clique_benchmark,
    run_sbm_community_benchmark,
)
from labflow.benchmarks.topology_fidelity import (
    TopologyFidelityBenchmarkRecord,
    run_topology_fidelity_benchmark,
)

__all__ = [
    # Records
    "SBMBenchmarkRecord",
    "CliqueBenchmarkRecord",
    "CliqueMotifBenchmarkRecord",
    "ClassificationBenchmarkRecord",
    "CommunityBenchmarkRecord",
    "TopologyFidelityBenchmarkRecord",
    "GradientStabilityBenchmarkRecord",
    "EfficiencyBenchmarkRecord",
    "ComponentAblationRecord",
    "SeparateBenchmarkSuiteResult",
    # Modular Benchmark Test Runners
    "run_community_detection_benchmark",
    "run_clique_motif_benchmark",
    "run_topology_fidelity_benchmark",
    "run_graph_classification_benchmark",
    "run_gradient_stability_benchmark",
    "run_efficiency_benchmark",
    "run_component_ablation_benchmark",
    "run_feature_pooling_ablation",
    "run_membership_ablation",
    "run_all_separate_benchmarks",
    # Legacy runners
    "run_sbm_community_benchmark",
    "run_planted_clique_benchmark",
]
