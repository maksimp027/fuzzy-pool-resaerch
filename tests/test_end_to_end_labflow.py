"""End-to-end integration tests across all 19 canonical specs, multi-layer hierarchical pooling, CUDA, and CLI."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from labflow.cli import main as cli_main
from labflow.composite import CANONICAL_METHOD_SPECS, GraphReductionLayer
from labflow.datasets import generate_sbm_graph
from labflow.validation import run_invariants_suite, run_sparse_equivalence_suite


@pytest.mark.parametrize("spec_key", [s.key for s in CANONICAL_METHOD_SPECS])
def test_all_canonical_layers_forward_backward(spec_key: str) -> None:
    graph = generate_sbm_graph(block_sizes=(6, 6, 6), feature_dim=8, seed=11, dtype=torch.float64)
    layer = GraphReductionLayer.from_spec(spec_key, in_features=8, out_features=6, num_clusters=3).double()

    out = layer(graph)
    assert out.x_new.shape == (3, 6)
    assert out.adj_new.shape == (3, 3)
    assert out.s.shape == (18, 3)

    loss = out.x_new.sum() + out.adj_new.sum() + out.aux_loss
    loss.backward()
    grads = [p.grad for p in layer.parameters() if p.requires_grad]
    assert all(g is not None and torch.isfinite(g).all() for g in grads)


def test_two_layer_stacked_hierarchical_fuzzy_pooling() -> None:
    """Verify gradients propagate through A_new^{(1)} and X_new^{(1)} across two stacked fuzzy pooling layers."""
    graph = generate_sbm_graph(block_sizes=(8, 8, 8, 8), feature_dim=10, seed=19, dtype=torch.float64)
    layer1 = GraphReductionLayer(in_features=10, out_features=8, num_clusters=8, topology="goguen", feature_pool="cog").double()
    layer2 = GraphReductionLayer(in_features=8, out_features=6, num_clusters=3, topology="hamacher", feature_pool="choquet").double()

    out1 = layer1(graph)
    out2 = layer2((out1.adj_new, out1.x_new))
    assert out2.x_new.shape == (3, 6)
    assert out2.adj_new.shape == (3, 3)

    (out2.x_new.sum() + out2.adj_new.sum()).backward()
    for name, p in list(layer1.named_parameters()) + list(layer2.named_parameters()):
        assert p.grad is not None and torch.isfinite(p.grad).all(), f"Missing/NaN grad on {name}"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_cuda_cpu_parity() -> None:
    graph_cpu = generate_sbm_graph(block_sizes=(6, 6, 6), feature_dim=8, seed=29, dtype=torch.float64)
    graph_gpu = graph_cpu.to(device="cuda")

    for key in ("mincut", "zadeh", "lukasiewicz", "schweizer_sklar", "boltzmann_two_step", "bk_reichenbach", "goguen_choquet"):
        torch.manual_seed(55)
        layer_cpu = GraphReductionLayer.from_spec(key, in_features=8, out_features=6, num_clusters=3).double()
        torch.manual_seed(55)
        layer_gpu = GraphReductionLayer.from_spec(key, in_features=8, out_features=6, num_clusters=3).double().cuda()

        out_c = layer_cpu(graph_cpu)
        out_g = layer_gpu(graph_gpu)
        assert torch.allclose(out_c.x_new, out_g.x_new.cpu(), atol=1e-10)
        assert torch.allclose(out_c.adj_new, out_g.adj_new.cpu(), atol=1e-10)


def test_sparse_equivalence_and_invariants_suites() -> None:
    sp = run_sparse_equivalence_suite()
    assert all(r.passed for r in sp)

    inv = run_invariants_suite()
    assert all(r.passed for r in inv)


def test_cli_list_and_invariants(tmp_path: Path) -> None:
    assert cli_main(["list"]) == 0
    assert cli_main(["invariants", "--output-dir", str(tmp_path)]) == 0
    assert (tmp_path / "labflow_results.json").exists()
    assert (tmp_path / "labflow_report.md").exists()
