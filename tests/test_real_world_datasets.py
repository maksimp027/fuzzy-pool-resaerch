"""Unit and integration tests for real-world datasets (TUDataset & OGB ogbg-molhiv) and benchmark runner."""

from __future__ import annotations

import pytest
import torch

from labflow.benchmarks.real_world import (
    REPRESENTATIVE_REALWORLD_METHOD_KEYS,
    evaluate_methods_on_real_dataset,
    run_real_world_benchmark_suite,
)
from labflow.datasets.real_world import (
    REAL_WORLD_DATASET_SPECS,
    load_real_world_dataset,
    resolve_dataset_spec,
)


def test_real_world_dataset_aliases() -> None:
    """Verify all 10 datasets and their shorthand aliases resolve to canonical specs."""
    assert len(REAL_WORLD_DATASET_SPECS) == 10
    assert resolve_dataset_spec("PROTEINS").name == "PROTEINS"
    assert resolve_dataset_spec("D&D").name == "DD"
    assert resolve_dataset_spec("NCT1").name == "NCI1"
    assert resolve_dataset_spec("NCI109").name == "NCI109"
    assert resolve_dataset_spec("ENZYMES").name == "ENZYMES"
    assert resolve_dataset_spec("IMDB-B").name == "IMDB-BINARY"
    assert resolve_dataset_spec("IMDB-M").name == "IMDB-MULTI"
    assert resolve_dataset_spec("COLLAB").name == "COLLAB"
    assert resolve_dataset_spec("REDDIT-B").name == "REDDIT-BINARY"
    assert resolve_dataset_spec("ogbg-molhiv").name == "ogbg-molhiv"


@pytest.mark.parametrize(
    "ds_name",
    ["PROTEINS", "NCI1", "ENZYMES", "IMDB-BINARY", "ogbg-molhiv"],
)
def test_load_real_world_dataset_integrity(ds_name: str) -> None:
    """Verify sparse GraphData structure, features, and stratified splits on real datasets."""
    bundle = load_real_world_dataset(ds_name)
    assert len(bundle.graphs) > 20
    assert len(bundle.train_graphs) + len(bundle.test_graphs) == len(bundle.graphs)
    assert bundle.num_features >= 18
    assert bundle.num_classes == bundle.spec.num_classes
    assert len(bundle.class_counts) == bundle.num_classes

    g0 = bundle.graphs[0]
    assert g0.adj is None  # Strictly sparse O(N + M)
    assert g0.edge_index is not None and g0.edge_weight is not None
    assert g0.x.shape[1] == bundle.num_features
    assert torch.isfinite(g0.x).all()
    assert g0.y is not None and 0 <= int(g0.y.item()) < bundle.num_classes


def test_real_world_benchmark_evaluation_smoke() -> None:
    """Verify end-to-end training and metric computation on a real dataset across classical and fuzzy methods."""
    bundle = load_real_world_dataset("IMDB-BINARY")
    records = evaluate_methods_on_real_dataset(
        bundle=bundle,
        method_keys=["diffpool", "dmon", "goguen", "lukasiewicz_smooth", "goguen_bipolar"],
        epochs=2,
        hidden_dim=16,
        seed=42,
    )
    assert len(records) == 5
    for rec in records:
        assert 0.0 <= rec.train_acc <= 1.0
        assert 0.0 <= rec.test_acc <= 1.0
        assert 0.0 <= rec.test_macro_f1 <= 1.0
        assert 0.0 <= rec.test_roc_auc <= 1.0
        assert rec.ms_per_epoch > 0.0


def test_representative_methods_coverage() -> None:
    """Ensure representative real-world method list covers all 4 divisions."""
    assert len(REPRESENTATIVE_REALWORLD_METHOD_KEYS) == 13
    for required in ("diffpool", "mincut", "dmon", "zadeh", "goguen", "bk_kleene_dienes", "goguen_bipolar"):
        assert required in REPRESENTATIVE_REALWORLD_METHOD_KEYS
