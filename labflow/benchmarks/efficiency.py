"""Computational efficiency and asymptotic scaling benchmark test for graph reduction methods.

Profiles forward + backward execution runtime and peak memory across growing node sizes N
on large sparse graphs with zero N x N matrix allocation.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Sequence

from labflow.analytics.profiler import run_scaling_profiler
from labflow.composite import CANONICAL_METHOD_SPECS


@dataclass
class EfficiencyBenchmarkRecord:
    """Scientific record for Computational Efficiency & Asymptotic Scaling Test."""

    key: str
    code: str
    title: str
    theoretical_complexity: str
    empirical_scaling_exp: float
    ms_at_smallest_n: float
    ms_at_largest_n: float
    peak_memory_mb_largest_n: float
    throughput_nodes_per_sec: float

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


def run_efficiency_benchmark(
    method_keys: Optional[Sequence[str]] = None,
    node_sizes: Sequence[int] = (150, 450, 1200, 2400),
    repeats: int = 2,
    device: str = "cpu",
    seed: int = 2026,
) -> List[EfficiencyBenchmarkRecord]:
    """Execute dedicated Computational Efficiency & Asymptotic Scaling Benchmark Test."""
    keys = list(method_keys) if method_keys is not None else [s.key for s in CANONICAL_METHOD_SPECS]

    _, summaries = run_scaling_profiler(
        method_keys=keys,
        node_sizes=node_sizes,
        repeats=repeats,
        device=device,
        seed=seed,
    )

    records: List[EfficiencyBenchmarkRecord] = []
    largest_n = max(node_sizes) if node_sizes else 1
    for s in summaries:
        ms = max(s.ms_at_largest_n, 1e-4)
        throughput = float(largest_n / (ms / 1000.0))
        records.append(
            EfficiencyBenchmarkRecord(
                key=s.method_key,
                code=s.method_code,
                title=s.title,
                theoretical_complexity=s.theoretical_complexity,
                empirical_scaling_exp=s.empirical_time_exponent,
                ms_at_smallest_n=s.ms_at_smallest_n,
                ms_at_largest_n=s.ms_at_largest_n,
                peak_memory_mb_largest_n=s.peak_memory_mb_largest_n,
                throughput_nodes_per_sec=throughput,
            )
        )

    return records
