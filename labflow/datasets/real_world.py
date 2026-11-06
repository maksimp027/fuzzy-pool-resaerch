"""Real-world graph classification datasets (TUDataset & OGB ogbg-molhiv) for Labflow.

Supports all 10 benchmark datasets across 6 scientific domains:
1. PROTEINS, DD (D&D)           — Bioinformatics: Protein enzyme vs non-enzyme classification
2. NCI1, NCI109                 — Cheminformatics: Anti-cancer molecular activity screening
3. ENZYMES                      — Bioinformatics: 6 EC top-level enzyme classes
4. IMDB-BINARY, IMDB-MULTI      — Social Networks: Ego-networks of movie actors
5. COLLAB, REDDIT-BINARY        — Large Social Networks: Scientific collaborations & Reddit threads
6. ogbg-molhiv (OGB)            — Molecular Property Prediction: HIV replication inhibition
"""

from __future__ import annotations

import gzip
import io
import os
import urllib.request
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from labflow.types import GraphData


@dataclass(frozen=True)
class RealWorldDatasetSpec:
    """Canonical specification for a real-world graph benchmark dataset."""

    name: str
    aliases: Tuple[str, ...]
    source: str  # "tudataset" or "ogb"
    category: str
    task_ua: str
    task_en: str
    nominal_graphs: int
    num_classes: int
    default_max_graphs: int
    default_num_clusters: int
    url: str


REAL_WORLD_DATASET_SPECS: List[RealWorldDatasetSpec] = [
    RealWorldDatasetSpec(
        name="PROTEINS",
        aliases=("PROTEINS", "proteins"),
        source="tudataset",
        category="Bioinformatics (Proteins)",
        task_ua="Білки: фермент чи ні",
        task_en="Proteins: enzyme vs non-enzyme",
        nominal_graphs=1113,
        num_classes=2,
        default_max_graphs=200,
        default_num_clusters=6,
        url="https://www.chrsmrrs.com/graphkerneldatasets/PROTEINS.zip",
    ),
    RealWorldDatasetSpec(
        name="DD",
        aliases=("DD", "D&D", "dd", "d&d"),
        source="tudataset",
        category="Bioinformatics (Proteins)",
        task_ua="Білки: фермент чи ні (великі структури D&D)",
        task_en="Proteins: enzyme vs non-enzyme (large D&D structures)",
        nominal_graphs=1178,
        num_classes=2,
        default_max_graphs=140,
        default_num_clusters=8,
        url="https://www.chrsmrrs.com/graphkerneldatasets/DD.zip",
    ),
    RealWorldDatasetSpec(
        name="NCI1",
        aliases=("NCI1", "NCT1", "nci1", "nct1"),
        source="tudataset",
        category="Cheminformatics (Molecules)",
        task_ua="Молекули: протираковий ефект (NSCLC)",
        task_en="Molecules: anti-cancer activity (NSCLC)",
        nominal_graphs=4110,
        num_classes=2,
        default_max_graphs=220,
        default_num_clusters=6,
        url="https://www.chrsmrrs.com/graphkerneldatasets/NCI1.zip",
    ),
    RealWorldDatasetSpec(
        name="NCI109",
        aliases=("NCI109", "nci109"),
        source="tudataset",
        category="Cheminformatics (Molecules)",
        task_ua="Молекули: протираковий ефект (Ovarian)",
        task_en="Molecules: anti-cancer activity (Ovarian)",
        nominal_graphs=4127,
        num_classes=2,
        default_max_graphs=220,
        default_num_clusters=6,
        url="https://www.chrsmrrs.com/graphkerneldatasets/NCI109.zip",
    ),
    RealWorldDatasetSpec(
        name="ENZYMES",
        aliases=("ENZYMES", "enzymes"),
        source="tudataset",
        category="Bioinformatics (Enzymes)",
        task_ua="6 класів ферментів (EC 1..6)",
        task_en="6 enzyme EC top-level classes",
        nominal_graphs=600,
        num_classes=6,
        default_max_graphs=240,
        default_num_clusters=6,
        url="https://www.chrsmrrs.com/graphkerneldatasets/ENZYMES.zip",
    ),
    RealWorldDatasetSpec(
        name="IMDB-BINARY",
        aliases=("IMDB-BINARY", "IMDB-B", "imdb-binary", "imdb-b"),
        source="tudataset",
        category="Social Networks (Actors)",
        task_ua="Соцмережа акторів (2 жанри)",
        task_en="Actor ego-networks (Action vs Romance)",
        nominal_graphs=1000,
        num_classes=2,
        default_max_graphs=200,
        default_num_clusters=4,
        url="https://www.chrsmrrs.com/graphkerneldatasets/IMDB-BINARY.zip",
    ),
    RealWorldDatasetSpec(
        name="IMDB-MULTI",
        aliases=("IMDB-MULTI", "IMDB-M", "imdb-multi", "imdb-m"),
        source="tudataset",
        category="Social Networks (Actors)",
        task_ua="Соцмережа акторів (3 жанри)",
        task_en="Actor ego-networks (Comedy, Romance, Sci-Fi)",
        nominal_graphs=1500,
        num_classes=3,
        default_max_graphs=210,
        default_num_clusters=4,
        url="https://www.chrsmrrs.com/graphkerneldatasets/IMDB-MULTI.zip",
    ),
    RealWorldDatasetSpec(
        name="COLLAB",
        aliases=("COLLAB", "collab"),
        source="tudataset",
        category="Large Social Networks",
        task_ua="Великі соціальні графи (наукові колаборації)",
        task_en="Large scientific collaboration ego-networks",
        nominal_graphs=5000,
        num_classes=3,
        default_max_graphs=150,
        default_num_clusters=6,
        url="https://www.chrsmrrs.com/graphkerneldatasets/COLLAB.zip",
    ),
    RealWorldDatasetSpec(
        name="REDDIT-BINARY",
        aliases=("REDDIT-BINARY", "REDDIT-B", "reddit-binary", "reddit-b"),
        source="tudataset",
        category="Large Social Networks",
        task_ua="Великі соціальні графи (дискусійні гілки Reddit)",
        task_en="Large online Reddit discussion graphs",
        nominal_graphs=2000,
        num_classes=2,
        default_max_graphs=140,
        default_num_clusters=8,
        url="https://www.chrsmrrs.com/graphkerneldatasets/REDDIT-BINARY.zip",
    ),
    RealWorldDatasetSpec(
        name="ogbg-molhiv",
        aliases=("ogbg-molhiv", "molhiv", "hiv", "OGBG-MOLHIV"),
        source="ogb",
        category="Molecular OGB",
        task_ua="Молекули: активність проти ВІЛ",
        task_en="Molecules: HIV replication inhibition (OGB)",
        nominal_graphs=41127,
        num_classes=2,
        default_max_graphs=240,
        default_num_clusters=6,
        url="https://snap.stanford.edu/ogb/data/graphproppred/csv_mol_download/hiv.zip",
    ),
]

DATASET_SPEC_MAP: Dict[str, RealWorldDatasetSpec] = {}
for _spec in REAL_WORLD_DATASET_SPECS:
    DATASET_SPEC_MAP[_spec.name] = _spec
    for _alias in _spec.aliases:
        DATASET_SPEC_MAP[_alias] = _spec
        DATASET_SPEC_MAP[_alias.lower()] = _spec


def resolve_dataset_spec(name: str) -> RealWorldDatasetSpec:
    """Resolve dataset name or alias (e.g., 'D&D' -> 'DD', 'IMDB-B' -> 'IMDB-BINARY')."""
    key = name.strip()
    if key in DATASET_SPEC_MAP:
        return DATASET_SPEC_MAP[key]
    if key.lower() in DATASET_SPEC_MAP:
        return DATASET_SPEC_MAP[key.lower()]
    raise ValueError(
        f"Unknown real-world dataset '{name}'. Available: {[s.name for s in REAL_WORLD_DATASET_SPECS]}"
    )


@dataclass
class RealWorldDatasetBundle:
    """Loaded and preprocessed real-world graph classification dataset with train/test split."""

    spec: RealWorldDatasetSpec
    graphs: List[GraphData]
    train_graphs: List[GraphData]
    test_graphs: List[GraphData]
    num_features: int
    num_classes: int
    total_available_graphs: int
    avg_nodes: float
    avg_edges: float
    max_nodes: int
    class_counts: Dict[int, int]

    def summary_dict(self) -> Dict[str, object]:
        return {
            "name": self.spec.name,
            "category": self.spec.category,
            "task_ua": self.spec.task_ua,
            "task_en": self.spec.task_en,
            "nominal_graphs": self.spec.nominal_graphs,
            "total_available_graphs": self.total_available_graphs,
            "sampled_graphs": len(self.graphs),
            "train_graphs": len(self.train_graphs),
            "test_graphs": len(self.test_graphs),
            "num_classes": self.num_classes,
            "num_features": self.num_features,
            "avg_nodes": round(self.avg_nodes, 2),
            "avg_edges": round(self.avg_edges, 2),
            "max_nodes": self.max_nodes,
            "class_counts": self.class_counts,
        }


def _compute_sparse_structural_features(
    num_nodes: int,
    src: np.ndarray,
    dst: np.ndarray,
) -> np.ndarray:
    """Compute O(N + M) structural & ego-network node features without N x N matrices.

    Produces 18 topological features per node:
    - [0]: normalized degree d_i / max(d_max, 1)
    - [1]: log degree ln(1 + d_i) / ln(1 + max(d_max, 1))
    - [2]: relative degree d_i / (mean_d + 1)
    - [3]: mean neighbor degree (normalized)
    - [4]: max neighbor degree (normalized)
    - [5]: min neighbor degree (normalized)
    - [6]: std neighbor degree (normalized)
    - [7]: 2-hop walk volume proxy (normalized)
    - [8..17]: 10 soft/one-hot degree quantile bins
    """
    if num_nodes == 0:
        return np.zeros((0, 18), dtype=np.float32)

    deg = np.bincount(src, minlength=num_nodes).astype(np.float32)
    d_max = float(max(deg.max(), 1.0))
    d_mean = float(deg.mean())

    deg_norm = deg / d_max
    deg_log = np.log1p(deg) / max(float(np.log1p(d_max)), 1.0)
    deg_rel = np.tanh(deg / (d_mean + 1.0))

    if len(src) > 0:
        dst_deg = deg[dst]
        sum_nb = np.bincount(src, weights=dst_deg, minlength=num_nodes).astype(np.float32)
        mean_nb = (sum_nb / np.maximum(deg, 1.0)) / d_max

        sum_sq_nb = np.bincount(src, weights=dst_deg * dst_deg, minlength=num_nodes).astype(np.float32)
        var_nb = np.maximum(sum_sq_nb / np.maximum(deg, 1.0) - (sum_nb / np.maximum(deg, 1.0)) ** 2, 0.0)
        std_nb = np.sqrt(var_nb) / d_max

        max_nb = np.zeros((num_nodes,), dtype=np.float32)
        np.maximum.at(max_nb, src, dst_deg)
        max_nb = max_nb / d_max

        min_nb = np.full((num_nodes,), d_max, dtype=np.float32)
        np.minimum.at(min_nb, src, dst_deg)
        min_nb = np.where(deg > 0, min_nb, 0.0) / d_max

        two_hop = np.log1p(sum_nb) / max(float(np.log1p(sum_nb.max())), 1.0)
    else:
        mean_nb = np.zeros((num_nodes,), dtype=np.float32)
        std_nb = np.zeros((num_nodes,), dtype=np.float32)
        max_nb = np.zeros((num_nodes,), dtype=np.float32)
        min_nb = np.zeros((num_nodes,), dtype=np.float32)
        two_hop = np.zeros((num_nodes,), dtype=np.float32)

    # 10 degree bins: 0, 1, 2, 3, 4, 5-6, 7-9, 10-14, 15-24, 25+
    bins = np.zeros((num_nodes, 10), dtype=np.float32)
    thresholds = [0, 1, 2, 3, 4, 6, 9, 14, 24]
    bucket = np.digitize(deg, thresholds, right=True)
    bucket = np.clip(bucket, 0, 9)
    bins[np.arange(num_nodes), bucket] = 1.0

    cont = np.stack([deg_norm, deg_log, deg_rel, mean_nb, max_nb, min_nb, std_nb, two_hop], axis=1)
    return np.concatenate([cont, bins], axis=1).astype(np.float32)


def _select_stratified_indices(
    labels: np.ndarray,
    valid_mask: np.ndarray,
    max_graphs: Optional[int],
    seed: int,
    min_minority_fraction: float = 0.0,
) -> np.ndarray:
    """Select stratified graph indices (0-based) respecting valid_mask and optional minority floor."""
    rng = np.random.default_rng(seed)
    valid_idx = np.flatnonzero(valid_mask)
    if max_graphs is None or max_graphs <= 0 or len(valid_idx) <= max_graphs:
        return np.sort(valid_idx)

    u_classes = np.unique(labels[valid_idx])
    k_cls = len(u_classes)
    by_cls = {c: valid_idx[labels[valid_idx] == c] for c in u_classes}

    # Compute target allocation per class
    total_valid = float(len(valid_idx))
    alloc: Dict[int, int] = {}
    if k_cls == 2 and min_minority_fraction > 0.0:
        c0, c1 = int(u_classes[0]), int(u_classes[1])
        n0, n1 = len(by_cls[c0]), len(by_cls[c1])
        if n1 < n0:
            min_c, maj_c = c1, c0
        else:
            min_c, maj_c = c0, c1
        target_min = min(len(by_cls[min_c]), max(int(round(max_graphs * min_minority_fraction)), 2))
        target_maj = min(len(by_cls[maj_c]), max_graphs - target_min)
        alloc[min_c] = target_min
        alloc[maj_c] = target_maj
    else:
        rem = max_graphs
        for i, c in enumerate(u_classes):
            c_int = int(c)
            if i == k_cls - 1:
                take = min(len(by_cls[c_int]), max(rem, 2))
            else:
                prop = len(by_cls[c_int]) / total_valid
                take = min(len(by_cls[c_int]), max(int(round(max_graphs * prop)), 2))
                rem -= take
            alloc[c_int] = take

    chosen: List[np.ndarray] = []
    for c in u_classes:
        c_int = int(c)
        pool = by_cls[c_int]
        take = min(len(pool), alloc[c_int])
        chosen.append(rng.choice(pool, size=take, replace=False))

    return np.sort(np.concatenate(chosen))


def _ensure_downloaded(spec: RealWorldDatasetSpec, cache_dir: Path) -> Path:
    """Ensure dataset archive is downloaded and extracted into cache_dir."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    if spec.source == "tudataset":
        ext_dir = cache_dir / spec.name
        if not ext_dir.exists():
            zpath = cache_dir / f"{spec.name}.zip"
            if not zpath.exists():
                urllib.request.urlretrieve(spec.url, zpath)
            with zipfile.ZipFile(zpath, "r") as zf:
                zf.extractall(cache_dir)
        return ext_dir
    else:
        ext_dir = cache_dir / "hiv"
        if not ext_dir.exists():
            zpath = cache_dir / "hiv.zip"
            if not zpath.exists():
                urllib.request.urlretrieve(spec.url, zpath)
            with zipfile.ZipFile(zpath, "r") as zf:
                zf.extractall(cache_dir)
        return ext_dir


def _parse_tudataset(
    spec: RealWorldDatasetSpec,
    ext_dir: Path,
    max_graphs: Optional[int],
    max_nodes_filter: int,
    seed: int,
) -> Tuple[List[GraphData], int]:
    """Parse a TUDataset directory into a list of sparse GraphData objects."""
    name = spec.name
    gi = np.loadtxt(ext_dir / f"{name}_graph_indicator.txt", dtype=np.int64)
    gl_raw = np.loadtxt(ext_dir / f"{name}_graph_labels.txt", dtype=np.int64)
    total_graphs = len(gl_raw)

    # Map graph labels to 0..C-1
    uniq_y, y_mapped = np.unique(gl_raw, return_inverse=True)

    # Node counts per graph (1-indexed in gi)
    node_counts = np.bincount(gi, minlength=total_graphs + 1)[1:]
    valid_mask = (node_counts >= 3) & (node_counts <= max_nodes_filter)
    if int(valid_mask.sum()) < 20:
        valid_mask = node_counts >= 2

    selected_gids_0 = _select_stratified_indices(
        labels=y_mapped,
        valid_mask=valid_mask,
        max_graphs=max_graphs,
        seed=seed,
    )
    selected_gids_1 = selected_gids_0 + 1  # 1-indexed graph IDs

    # Node ranges per graph (nodes in TUDataset are contiguous by graph_indicator, but handle general case)
    node_g0 = gi - 1  # 0-indexed graph ID per node
    in_selected_graph = np.isin(node_g0, selected_gids_0)

    # Load optional node_labels and node_attributes
    nl_path = ext_dir / f"{name}_node_labels.txt"
    na_path = ext_dir / f"{name}_node_attributes.txt"

    nl_onehot_all: Optional[np.ndarray] = None
    if nl_path.exists():
        with open(nl_path, "rb") as f:
            raw_nl = f.read().replace(b",", b" ")
        nl_vec = np.fromstring(raw_nl.decode("ascii"), dtype=np.int64, sep=" ")
        # If multiple columns exist, take first column
        if len(nl_vec) > len(gi):
            nl_vec = nl_vec.reshape(len(gi), -1)[:, 0]
        _, nl_idx = np.unique(nl_vec, return_inverse=True)
        n_nl_classes = min(int(nl_idx.max()) + 1, 64)
        nl_clipped = np.clip(nl_idx, 0, n_nl_classes - 1)
        nl_onehot_all = np.zeros((len(gi), n_nl_classes), dtype=np.float32)
        nl_onehot_all[np.arange(len(gi)), nl_clipped] = 1.0

    na_all: Optional[np.ndarray] = None
    if na_path.exists():
        na_mat = np.loadtxt(na_path, delimiter=",", dtype=np.float32)
        if na_mat.ndim == 1:
            na_mat = na_mat[:, None]
        mean_na = na_mat.mean(axis=0, keepdims=True)
        std_na = np.maximum(na_mat.std(axis=0, keepdims=True), 1e-6)
        na_all = np.clip((na_mat - mean_na) / std_na, -5.0, 5.0).astype(np.float32)

    # Fast load edges from DS_A.txt
    with open(ext_dir / f"{name}_A.txt", "rb") as f:
        raw_a = f.read().replace(b",", b" ")
    edges_1 = np.fromstring(raw_a.decode("ascii"), dtype=np.int64, sep=" ").reshape(-1, 2)
    u_0 = edges_1[:, 0] - 1
    v_0 = edges_1[:, 1] - 1

    # Filter edges belonging to selected graphs
    edge_mask = in_selected_graph[u_0] & (node_g0[u_0] == node_g0[v_0]) & (u_0 != v_0)
    u_sel = u_0[edge_mask]
    v_sel = v_0[edge_mask]
    g_edge = node_g0[u_sel]

    # Group nodes by graph ID
    node_order = np.argsort(node_g0, kind="stable")
    g_sorted_nodes = node_g0[node_order]
    n_splits = np.searchsorted(g_sorted_nodes, np.arange(total_graphs + 1))

    # Group selected edges by graph ID
    edge_order = np.argsort(g_edge, kind="stable")
    u_sel = u_sel[edge_order]
    v_sel = v_sel[edge_order]
    g_edge_sorted = g_edge[edge_order]
    e_splits = np.searchsorted(g_edge_sorted, np.arange(total_graphs + 1))

    graphs: List[GraphData] = []
    global_to_local = np.zeros((len(gi),), dtype=np.int64)

    for g0 in selected_gids_0:
        g_nodes = node_order[n_splits[g0] : n_splits[g0 + 1]]
        n_g = len(g_nodes)
        global_to_local[g_nodes] = np.arange(n_g, dtype=np.int64)

        e_lo, e_hi = e_splits[g0], e_splits[g0 + 1]
        if e_hi > e_lo:
            lu = global_to_local[u_sel[e_lo:e_hi]]
            lv = global_to_local[v_sel[e_lo:e_hi]]
            # Symmetrize and deduplicate
            su = np.concatenate([lu, lv])
            sv = np.concatenate([lv, lu])
            keys = su * n_g + sv
            uniq_keys = np.unique(keys)
            src_g = uniq_keys // n_g
            dst_g = uniq_keys - src_g * n_g
        else:
            src_g = np.empty((0,), dtype=np.int64)
            dst_g = np.empty((0,), dtype=np.int64)

        struct_x = _compute_sparse_structural_features(n_g, src_g, dst_g)
        feat_parts = [struct_x]
        if nl_onehot_all is not None:
            feat_parts.append(nl_onehot_all[g_nodes])
        if na_all is not None:
            feat_parts.append(na_all[g_nodes])
        x_np = np.concatenate(feat_parts, axis=1)

        edge_index = torch.tensor(np.stack([src_g, dst_g], axis=0), dtype=torch.long)
        edge_weight = torch.ones((len(src_g),), dtype=torch.float32)
        y_val = int(y_mapped[g0])

        graphs.append(
            GraphData(
                x=torch.tensor(x_np, dtype=torch.float32),
                adj=None,  # True sparse representation O(N + M)
                edge_index=edge_index,
                edge_weight=edge_weight,
                y=torch.tensor(y_val, dtype=torch.long),
                metadata={
                    "dataset": name,
                    "graph_id": int(g0),
                    "num_nodes": int(n_g),
                    "num_edges": int(len(src_g)),
                },
            )
        )

    return graphs, total_graphs


def _parse_ogbg_molhiv(
    spec: RealWorldDatasetSpec,
    ext_dir: Path,
    max_graphs: Optional[int],
    max_nodes_filter: int,
    seed: int,
) -> Tuple[List[GraphData], int]:
    """Parse OGB ogbg-molhiv raw CSV files into sparse GraphData objects."""
    raw_dir = ext_dir / "raw"
    with gzip.open(raw_dir / "num-node-list.csv.gz", "rt") as f:
        num_nodes_list = np.loadtxt(f, dtype=np.int64)
    with gzip.open(raw_dir / "num-edge-list.csv.gz", "rt") as f:
        num_edges_list = np.loadtxt(f, dtype=np.int64)
    with gzip.open(raw_dir / "graph-label.csv.gz", "rt") as f:
        labels = np.loadtxt(f, dtype=np.int64)

    total_graphs = len(labels)
    valid_mask = (num_nodes_list >= 4) & (num_nodes_list <= max_nodes_filter) & (num_edges_list >= 2)

    # Ensure sufficient positive HIV molecules (at least 30% positive in subsample) for low-variance evaluation
    selected_gids = _select_stratified_indices(
        labels=labels,
        valid_mask=valid_mask,
        max_graphs=max_graphs,
        seed=seed,
        min_minority_fraction=0.30,
    )

    node_offsets = np.concatenate([[0], np.cumsum(num_nodes_list)])
    edge_offsets = np.concatenate([[0], np.cumsum(num_edges_list)])

    with gzip.open(raw_dir / "node-feat.csv.gz", "rb") as f:
        raw_nf = f.read().replace(b",", b" ")
    node_feats_all = np.fromstring(raw_nf.decode("ascii"), dtype=np.float32, sep=" ").reshape(-1, 9)

    # Standardize the 9 chemical atom features across the dataset
    nf_mean = node_feats_all.mean(axis=0, keepdims=True)
    nf_std = np.maximum(node_feats_all.std(axis=0, keepdims=True), 1e-6)
    node_feats_norm = np.clip((node_feats_all - nf_mean) / nf_std, -5.0, 5.0).astype(np.float32)

    with gzip.open(raw_dir / "edge.csv.gz", "rb") as f:
        raw_ed = f.read().replace(b",", b" ")
    edges_all = np.fromstring(raw_ed.decode("ascii"), dtype=np.int64, sep=" ").reshape(-1, 2)

    graphs: List[GraphData] = []
    for gid in selected_gids:
        n_g = int(num_nodes_list[gid])
        n_lo, n_hi = int(node_offsets[gid]), int(node_offsets[gid + 1])
        e_lo, e_hi = int(edge_offsets[gid]), int(edge_offsets[gid + 1])

        if e_hi > e_lo:
            e_chunk = edges_all[e_lo:e_hi]
            lu = e_chunk[:, 0]
            lv = e_chunk[:, 1]
            mask = (lu != lv) & (lu >= 0) & (lu < n_g) & (lv >= 0) & (lv < n_g)
            lu, lv = lu[mask], lv[mask]
            su = np.concatenate([lu, lv])
            sv = np.concatenate([lv, lu])
            keys = su * n_g + sv
            uniq_keys = np.unique(keys)
            src_g = uniq_keys // n_g
            dst_g = uniq_keys - src_g * n_g
        else:
            src_g = np.empty((0,), dtype=np.int64)
            dst_g = np.empty((0,), dtype=np.int64)

        struct_x = _compute_sparse_structural_features(n_g, src_g, dst_g)
        atom_x = node_feats_norm[n_lo:n_hi]
        x_np = np.concatenate([atom_x, struct_x], axis=1)

        edge_index = torch.tensor(np.stack([src_g, dst_g], axis=0), dtype=torch.long)
        edge_weight = torch.ones((len(src_g),), dtype=torch.float32)

        graphs.append(
            GraphData(
                x=torch.tensor(x_np, dtype=torch.float32),
                adj=None,
                edge_index=edge_index,
                edge_weight=edge_weight,
                y=torch.tensor(int(labels[gid]), dtype=torch.long),
                metadata={
                    "dataset": spec.name,
                    "graph_id": int(gid),
                    "num_nodes": n_g,
                    "num_edges": int(len(src_g)),
                },
            )
        )

    return graphs, total_graphs


def split_stratified_train_test(
    graphs: Sequence[GraphData],
    train_ratio: float = 0.78,
    seed: int = 42,
) -> Tuple[List[GraphData], List[GraphData]]:
    """Perform a deterministic stratified train/test split over GraphData items."""
    rng = np.random.default_rng(seed)
    by_class: Dict[int, List[GraphData]] = {}
    for g in graphs:
        assert g.y is not None
        c = int(g.y.item())
        by_class.setdefault(c, []).append(g)

    train_set: List[GraphData] = []
    test_set: List[GraphData] = []
    for c in sorted(by_class.keys()):
        items = list(by_class[c])
        perm = rng.permutation(len(items))
        n_train = min(max(int(round(train_ratio * len(items))), 1), max(len(items) - 1, 1))
        for idx in perm[:n_train]:
            train_set.append(items[int(idx)])
        for idx in perm[n_train:]:
            test_set.append(items[int(idx)])

    # Shuffle train_set deterministically so mini-batches/epochs are well-mixed
    train_perm = rng.permutation(len(train_set))
    train_set = [train_set[int(i)] for i in train_perm]
    return train_set, test_set


def load_real_world_dataset(
    name: str,
    max_graphs: Optional[int] = None,
    max_nodes_filter: int = 500,
    train_ratio: float = 0.78,
    seed: int = 42,
    cache_dir: str | Path = "data/real_world",
    use_disk_cache: bool = True,
) -> RealWorldDatasetBundle:
    """Download (if needed), parse, and split a real-world TUDataset or OGB benchmark dataset."""
    spec = resolve_dataset_spec(name)
    effective_max = spec.default_max_graphs if max_graphs is None else max_graphs
    root = Path(cache_dir)
    proc_dir = root / "processed"
    proc_dir.mkdir(parents=True, exist_ok=True)

    cache_file = proc_dir / f"{spec.name}_max{effective_max}_mn{max_nodes_filter}_s{seed}.pt"
    if use_disk_cache and cache_file.exists():
        payload = torch.load(cache_file, weights_only=False)
        graphs = payload["graphs"]
        total_available = int(payload["total_available"])
    else:
        ext_dir = _ensure_downloaded(spec, root)
        if spec.source == "tudataset":
            graphs, total_available = _parse_tudataset(
                spec=spec,
                ext_dir=ext_dir,
                max_graphs=effective_max,
                max_nodes_filter=max_nodes_filter,
                seed=seed,
            )
        else:
            graphs, total_available = _parse_ogbg_molhiv(
                spec=spec,
                ext_dir=ext_dir,
                max_graphs=effective_max,
                max_nodes_filter=max_nodes_filter,
                seed=seed,
            )
        if use_disk_cache:
            torch.save({"graphs": graphs, "total_available": total_available}, cache_file)

    train_graphs, test_graphs = split_stratified_train_test(
        graphs, train_ratio=train_ratio, seed=seed
    )

    num_features = int(graphs[0].x.shape[1]) if graphs else 0
    node_list = [int(g.num_nodes) for g in graphs]
    edge_list = [int(g.edge_weight.numel()) if g.edge_weight is not None else 0 for g in graphs]

    class_counts: Dict[int, int] = {}
    for g in graphs:
        assert g.y is not None
        c = int(g.y.item())
        class_counts[c] = class_counts.get(c, 0) + 1

    return RealWorldDatasetBundle(
        spec=spec,
        graphs=graphs,
        train_graphs=train_graphs,
        test_graphs=test_graphs,
        num_features=num_features,
        num_classes=spec.num_classes,
        total_available_graphs=total_available,
        avg_nodes=float(np.mean(node_list)) if node_list else 0.0,
        avg_edges=float(np.mean(edge_list)) if edge_list else 0.0,
        max_nodes=int(max(node_list)) if node_list else 0,
        class_counts=class_counts,
    )
