"""Computational latency, memory footprint, and empirical asymptotic scaling profiler for all reducers."""

from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch

from labflow.composite import CANONICAL_METHOD_SPECS, METHOD_SPEC_MAP, GraphReductionLayer
from labflow.datasets.classic import generate_large_sparse_sbm_graph


@dataclass
class ScalingPointRecord:
    """Single latency & memory measurement for a reducer at graph size N."""

    method_key: str
    method_code: str
    num_nodes: int
    num_edges: int
    num_clusters: int
    forward_ms: float
    backward_ms: float
    total_ms: float
    working_memory_mb: float

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


@dataclass
class ReducerScalingSummary:
    """Aggregated scaling summary across multiple graph sizes N for a single reducer."""

    method_key: str
    method_code: str
    title: str
    theoretical_complexity: str
    empirical_time_exponent: float
    ms_at_smallest_n: float
    ms_at_largest_n: float
    peak_memory_mb_largest_n: float

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


def _estimate_working_memory_mb(
    num_nodes: int,
    num_edges: int,
    num_clusters: int,
    feat_dim: int,
    topology_name: str,
    elem_bytes: int = 4,
) -> float:
    """Deterministic active tensor working-set estimate (in MiB) during sparse forward+backward."""
    # Base sparse COO graph + node embeddings H, Z, S + gradients
    bytes_used = (
        2 * (num_edges + num_nodes) * 8  # COO int64 indices (with self-loops)
        + 2 * (num_edges + num_nodes) * elem_bytes  # edge weights + normalized weights
        + 4 * num_nodes * feat_dim * elem_bytes  # X, H, grad_X, grad_H
        + 4 * num_nodes * num_clusters * elem_bytes  # Z, S, grad_Z, grad_S
        + 2 * num_clusters * num_clusters * elem_bytes  # A_new, grad_A_new
    )
    if topology_name in ("boltzmann", "boltzmann_adaptive", "probsum", "probsum_norm"):
        # [M, K, K] omega & pi / m_eij tensors
        bytes_used += 2 * num_edges * num_clusters * num_clusters * elem_bytes
    elif topology_name.startswith("bk_"):
        # Sorted S permutations [N, K] + MEX flattened rank keys [M_u * K]
        m_u = num_edges + num_nodes
        bytes_used += (2 * num_nodes * num_clusters + 3 * m_u * num_clusters) * 8
    else:
        # 2-step sparse intermediate [M, K] and [N, K, K] tensors
        bytes_used += (2 * num_edges * num_clusters + num_nodes * num_clusters * num_clusters) * elem_bytes

    return float(bytes_used) / (1024.0 * 1024.0)


def run_scaling_profiler(
    method_keys: Optional[Sequence[str]] = None,
    node_sizes: Sequence[int] = (150, 450, 1200, 2400),
    num_clusters: int = 6,
    avg_degree: float = 12.0,
    feature_dim: int = 16,
    repeats: int = 3,
    device: str = "cpu",
    seed: int = 2026,
) -> tuple[List[ScalingPointRecord], List[ReducerScalingSummary]]:
    """Profile forward/backward runtime and empirical scaling exponent across sparse graph sizes N."""
    keys = list(method_keys) if method_keys is not None else [s.key for s in CANONICAL_METHOD_SPECS]
    dev = torch.device(device)

    graphs = {
        n_sz: generate_large_sparse_sbm_graph(
            num_nodes=n_sz,
            num_blocks=num_clusters,
            avg_degree=avg_degree,
            feature_dim=feature_dim,
            seed=seed + n_sz,
        ).to(device=dev)
        for n_sz in node_sizes
    }

    points: List[ScalingPointRecord] = []
    summaries: List[ReducerScalingSummary] = []

    for key in keys:
        spec = METHOD_SPEC_MAP[key]
        torch.manual_seed(seed)
        layer = GraphReductionLayer.from_spec(
            key,
            in_features=feature_dim,
            out_features=feature_dim,
            num_clusters=num_clusters,
        ).to(device=dev)

        method_points: List[ScalingPointRecord] = []
        for n_sz in node_sizes:
            g = graphs[n_sz]
            # Warmup run
            layer.zero_grad()
            out_w = layer(g)
            (out_w.x_new.sum() + out_w.adj_new.sum() + out_w.aux_loss).backward()

            fwd_times: List[float] = []
            bwd_times: List[float] = []
            cuda_mem_mb = 0.0

            for _ in range(max(repeats, 1)):
                layer.zero_grad()
                if dev.type == "cuda":
                    torch.cuda.reset_peak_memory_stats(dev)
                    torch.cuda.synchronize(dev)

                t0 = time.perf_counter()
                out = layer(g)
                if dev.type == "cuda":
                    torch.cuda.synchronize(dev)
                t1 = time.perf_counter()

                loss = out.x_new.sum() + out.adj_new.sum() + out.aux_loss
                loss.backward()
                if dev.type == "cuda":
                    torch.cuda.synchronize(dev)
                    cuda_mem_mb = max(cuda_mem_mb, float(torch.cuda.max_memory_allocated(dev)) / (1024.0 * 1024.0))
                t2 = time.perf_counter()

                fwd_times.append((t1 - t0) * 1000.0)
                bwd_times.append((t2 - t1) * 1000.0)

            fwd_med = float(np.median(fwd_times))
            bwd_med = float(np.median(bwd_times))
            mem_mb = (
                cuda_mem_mb
                if dev.type == "cuda" and cuda_mem_mb > 0.0
                else _estimate_working_memory_mb(
                    num_nodes=n_sz,
                    num_edges=g.num_edges,
                    num_clusters=num_clusters,
                    feat_dim=feature_dim,
                    topology_name=layer.topology_name,
                )
            )
            pt = ScalingPointRecord(
                method_key=key,
                method_code=spec.code,
                num_nodes=n_sz,
                num_edges=g.num_edges,
                num_clusters=num_clusters,
                forward_ms=fwd_med,
                backward_ms=bwd_med,
                total_ms=fwd_med + bwd_med,
                working_memory_mb=mem_mb,
            )
            points.append(pt)
            method_points.append(pt)

        # Fit log(total_ms) ~ alpha * log(N) + beta across largest sizes where asymptotic regime dominates
        fit_pts = method_points[-min(len(method_points), 3) :]
        if len(fit_pts) >= 2:
            log_n = np.array([math.log(max(p.num_nodes, 2)) for p in fit_pts], dtype=np.float64)
            log_t = np.array([math.log(max(p.total_ms, 1e-4)) for p in fit_pts], dtype=np.float64)
            slope = float(np.polyfit(log_n, log_t, deg=1)[0])
            exponent = max(0.0, slope)
        else:
            exponent = 1.0

        summaries.append(
            ReducerScalingSummary(
                method_key=key,
                method_code=spec.code,
                title=spec.title,
                theoretical_complexity=spec.complexity,
                empirical_time_exponent=exponent,
                ms_at_smallest_n=method_points[0].total_ms,
                ms_at_largest_n=method_points[-1].total_ms,
                peak_memory_mb_largest_n=method_points[-1].working_memory_mb,
            )
        )

    return points, summaries
