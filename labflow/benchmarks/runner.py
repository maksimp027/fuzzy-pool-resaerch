"""Unified benchmark runner and canonical test registry for Labflow.

Exports both modular domain-specific benchmark tests with distinct scientific metrics,
and legacy unified benchmark functions for backward compatibility:
1. Community Detection & Partition Alignment Benchmark (`run_community_detection_benchmark`)
2. Structural Motif & Planted Clique Isolation Benchmark (`run_clique_motif_benchmark`)
3. Topological & Spectral Fidelity Benchmark (`run_topology_fidelity_benchmark`)
4. Hierarchical Graph Classification Benchmark (`run_graph_classification_benchmark`)
5. Gradient Dynamics & Optimization Health Benchmark (`run_gradient_stability_benchmark`)
6. Computational Efficiency & Asymptotic Scaling Benchmark (`run_efficiency_benchmark`)
7. Component-Level Modular Ablation Benchmark (`run_component_ablation_benchmark`)
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Sequence

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
from labflow.benchmarks.topology_fidelity import (
    TopologyFidelityBenchmarkRecord,
    run_topology_fidelity_benchmark,
)
from labflow.composite import CANONICAL_METHOD_SPECS, METHOD_SPEC_MAP, GraphReductionLayer
from labflow.datasets.synthetic import generate_sbm_graph
from labflow.types import GraphData


# Backward compatibility aliases
CliqueBenchmarkRecord = CliqueMotifBenchmarkRecord


@dataclass
class SBMBenchmarkRecord:
    """Legacy SBM benchmark record for backward compatibility."""

    key: str
    code: str
    title: str
    nmi: float
    modularity_q: float
    conductance: float
    adj_new_max: float
    exact_sparsity: float
    topo_grad_active_ratio: float
    ms_per_step: float

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


def run_sbm_community_benchmark(
    method_keys: Optional[Sequence[str]] = None,
    epochs: int = 80,
    lr: float = 0.025,
    seed: int = 42,
) -> List[SBMBenchmarkRecord]:
    """Backward-compatible runner for SBM community detection benchmark."""
    comm_records = run_community_detection_benchmark(
        method_keys=method_keys,
        epochs=epochs,
        lr=lr,
        seed=seed,
    )
    topo_records = run_topology_fidelity_benchmark(
        method_keys=method_keys,
        epochs=max(epochs // 2, 10),
        lr=lr,
        seed=seed,
    )
    topo_map = {r.key: r for r in topo_records}

    grad_records = run_gradient_stability_benchmark(
        method_keys=method_keys,
        seed=seed,
    )
    grad_map = {r.key: r for r in grad_records}

    records: List[SBMBenchmarkRecord] = []
    for c_rec in comm_records:
        t_rec = topo_map.get(c_rec.key)
        g_rec = grad_map.get(c_rec.key)
        records.append(
            SBMBenchmarkRecord(
                key=c_rec.key,
                code=c_rec.code,
                title=c_rec.title,
                nmi=c_rec.nmi,
                modularity_q=c_rec.modularity_q,
                conductance=c_rec.conductance,
                adj_new_max=c_rec.adj_new_max,
                exact_sparsity=t_rec.exact_sparsity if t_rec else 0.0,
                topo_grad_active_ratio=g_rec.topo_jacobian_density if g_rec else 0.0,
                ms_per_step=c_rec.ms_per_step,
            )
        )
    return records


def run_planted_clique_benchmark(
    method_keys: Optional[Sequence[str]] = None,
    epochs: int = 120,
    lr: float = 0.05,
    seed: int = 7,
) -> List[CliqueBenchmarkRecord]:
    """Backward-compatible runner for planted clique benchmark."""
    return run_clique_motif_benchmark(
        method_keys=method_keys,
        epochs=epochs,
        lr=lr,
        seed=seed,
    )


@dataclass
class SeparateBenchmarkSuiteResult:
    """Consolidated bundle of all modular benchmark test results."""

    community_records: List[CommunityBenchmarkRecord]
    clique_motif_records: List[CliqueMotifBenchmarkRecord]
    topology_fidelity_records: List[TopologyFidelityBenchmarkRecord]
    classification_records: List[ClassificationBenchmarkRecord]
    gradient_stability_records: List[GradientStabilityBenchmarkRecord]
    efficiency_records: List[EfficiencyBenchmarkRecord]
    ablation_records: List[ComponentAblationRecord]

    def to_dict(self) -> Dict[str, object]:
        return {
            "community_benchmark": [r.to_dict() for r in self.community_records],
            "clique_motif_benchmark": [r.to_dict() for r in self.clique_motif_records],
            "topology_fidelity_benchmark": [r.to_dict() for r in self.topology_fidelity_records],
            "classification_benchmark": [r.to_dict() for r in self.classification_records],
            "gradient_stability_benchmark": [r.to_dict() for r in self.gradient_stability_records],
            "efficiency_benchmark": [r.to_dict() for r in self.efficiency_records],
            "ablation_benchmark": [r.to_dict() for r in self.ablation_records],
        }


def run_all_separate_benchmarks(
    method_keys: Optional[Sequence[str]] = None,
    community_epochs: int = 80,
    clique_epochs: int = 120,
    cls_epochs: int = 22,
    topo_epochs: int = 70,
    profiler_sizes: Sequence[int] = (150, 450, 1200, 2400),
    seed: int = 42,
) -> SeparateBenchmarkSuiteResult:
    """Execute all separate scientific benchmark tests with independent metrics."""
    comm = run_community_detection_benchmark(method_keys=method_keys, epochs=community_epochs, seed=seed)
    clq = run_clique_motif_benchmark(method_keys=method_keys, epochs=clique_epochs, seed=seed)
    topo = run_topology_fidelity_benchmark(method_keys=method_keys, epochs=topo_epochs, seed=seed)
    cls = run_graph_classification_benchmark(method_keys=method_keys, epochs=cls_epochs, seed=seed)
    grad = run_gradient_stability_benchmark(method_keys=method_keys, seed=seed)
    eff = run_efficiency_benchmark(method_keys=method_keys, node_sizes=profiler_sizes, seed=seed)
    abl = run_component_ablation_benchmark(seed=seed)

    return SeparateBenchmarkSuiteResult(
        community_records=comm,
        clique_motif_records=clq,
        topology_fidelity_records=topo,
        classification_records=cls,
        gradient_stability_records=grad,
        efficiency_records=eff,
        ablation_records=abl,
    )
