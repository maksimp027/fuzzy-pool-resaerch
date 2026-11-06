"""Real-world TUDataset & OGB benchmark suite for Labflow graph reduction methods.

Evaluates classical spectral/bilinear and fuzzy relational/smooth/implicative/feature-space
graph reduction operators across 10 real-world benchmark datasets from 6 domains:
- Bioinformatics (Proteins): PROTEINS, DD (D&D)
- Cheminformatics (Anti-cancer molecules): NCI1, NCI109
- Multi-class Enzyme Classification: ENZYMES (6 classes)
- Social Actor Ego-Networks: IMDB-BINARY, IMDB-MULTI
- Large Social Graphs: COLLAB, REDDIT-BINARY
- Open Graph Benchmark (Molecular HIV): ogbg-molhiv
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from labflow.analytics.metrics import compute_feature_reconstruction_r2
from labflow.benchmarks.classification import _summarize_learned_params
from labflow.composite import CANONICAL_METHOD_SPECS, METHOD_SPEC_MAP, GraphReductionLayer
from labflow.datasets.real_world import (
    REAL_WORLD_DATASET_SPECS,
    RealWorldDatasetBundle,
    load_real_world_dataset,
)
from labflow.gnn import GraphSAGEOperator
from labflow.types import GraphData, PoolingOutput


# Representative methods covering every theoretical family in Labflow:
# Div I (Classical): M1 (DiffPool), M2 (MinCutPool), M3 (DMoN)
# Div II (Relational sup-T): M4 (Zadeh), M5 (Goguen), M6-Sm (Smooth Lukasiewicz), M7 (Hamacher), M8B (Dombi)
# Div III (Smooth & Implicative): M9B (Two-Step Boltzmann), M10-Nm (Normalized ProbSum), M11B (Bandler-Kohout KD)
# Div IV (Fuzzy Feature Space): M13A (Goguen + Bipolar), M13B (Goguen + Choquet)
REPRESENTATIVE_REALWORLD_METHOD_KEYS: Tuple[str, ...] = (
    "diffpool",
    "mincut",
    "dmon",
    "zadeh",
    "goguen",
    "lukasiewicz_smooth",
    "hamacher",
    "dombi",
    "boltzmann_two_step",
    "probsum_norm",
    "bk_kleene_dienes",
    "goguen_bipolar",
    "goguen_choquet",
)


METHOD_FAMILY_LABELS: Dict[str, str] = {
    "diffpool": "I. Класичні білінійні/спектральні",
    "mincut": "I. Класичні білінійні/спектральні",
    "dmon": "I. Класичні білінійні/спектральні",
    "zadeh": "II. Реляційні sup-T (Zadeh/Goguen)",
    "zadeh_rbf": "II. Реляційні sup-T (Zadeh/Goguen)",
    "goguen": "II. Реляційні sup-T (Zadeh/Goguen)",
    "lukasiewicz": "II. Реляційні порогові (Lukasiewicz)",
    "lukasiewicz_smooth": "II. Реляційні порогові (Lukasiewicz)",
    "hamacher": "II. Параметричні t-норми (Hamacher/Dombi)",
    "einstein": "II. Параметричні t-норми (Hamacher/Dombi)",
    "schweizer_sklar": "II. Параметричні t-норми (Hamacher/Dombi)",
    "dombi": "II. Параметричні t-норми (Hamacher/Dombi)",
    "boltzmann": "III. Гладкі енергетичні/ймовірнісні",
    "boltzmann_adaptive": "III. Гладкі енергетичні/ймовірнісні",
    "boltzmann_two_step": "III. Гладкі енергетичні/ймовірнісні",
    "probsum": "III. Гладкі енергетичні/ймовірнісні",
    "probsum_norm": "III. Гладкі енергетичні/ймовірнісні",
    "bk_lukasiewicz": "III. Імплікативні кванторні (Bandler-Kohout)",
    "bk_kleene_dienes": "III. Імплікативні кванторні (Bandler-Kohout)",
    "bk_reichenbach": "III. Імплікативні кванторні (Bandler-Kohout)",
    "goguen_bipolar": "IV. Нечітка агрегація ознак (Bipolar/Choquet)",
    "goguen_choquet": "IV. Нечітка агрегація ознак (Bipolar/Choquet)",
}


@dataclass
class RealWorldBenchmarkRecord:
    """Empirical evaluation record for one (dataset, reduction_method) pair."""

    dataset: str
    category: str
    task_ua: str
    num_graphs_total: int
    num_graphs_sampled: int
    num_classes: int
    avg_nodes: float
    avg_edges: float
    num_clusters: int
    method_key: str
    method_code: str
    method_title: str
    family: str
    train_acc: float
    test_acc: float
    test_macro_f1: float
    test_roc_auc: float
    generalization_gap: float
    final_loss: float
    feature_recon_r2: float
    supergraph_density: float
    supergraph_sparsity: float
    bezdek_pc: float
    learned_params_summary: str
    ms_per_epoch: float

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


@dataclass
class RealWorldMethodSummary:
    """Aggregated cross-dataset ranking summary for a single graph reduction method."""

    rank: int
    method_key: str
    method_code: str
    method_title: str
    family: str
    mean_test_acc: float
    mean_macro_f1: float
    mean_roc_auc: float
    mean_rank: float
    datasets_won_or_tied: int
    bio_chem_mean_acc: float
    social_mean_acc: float
    ogb_molhiv_auc: float
    mean_feature_r2: float
    mean_ms_per_epoch: float

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


@dataclass
class RealWorldBenchmarkSuiteResult:
    """Complete result bundle for the 10-dataset real-world evaluation."""

    dataset_summaries: List[Dict[str, object]]
    records: List[RealWorldBenchmarkRecord]
    method_leaderboard: List[RealWorldMethodSummary]
    dataset_champions: Dict[str, RealWorldBenchmarkRecord]

    def to_dict(self) -> Dict[str, object]:
        return {
            "dataset_summaries": self.dataset_summaries,
            "records": [r.to_dict() for r in self.records],
            "method_leaderboard": [m.to_dict() for m in self.method_leaderboard],
            "dataset_champions": {k: v.to_dict() for k, v in self.dataset_champions.items()},
        }


class RealWorldPoolingClassifier(nn.Module):
    """2-stage Hierarchical GNN Classifier for real-world graphs with relational energy readout."""

    def __init__(
        self,
        method_key: str,
        in_features: int,
        hidden_dim: int,
        num_clusters: int,
        num_classes: int,
    ) -> None:
        super().__init__()
        self.num_clusters = num_clusters
        self.pool = GraphReductionLayer.from_spec(
            method_key,
            in_features=in_features,
            out_features=hidden_dim,
            num_clusters=num_clusters,
        )
        self.post_gnn = GraphSAGEOperator(hidden_dim, hidden_dim, activation="elu")
        # Readout combines:
        # 1) h2.mean(dim=0) [hidden_dim]
        # 2) h2.amax(dim=0) [hidden_dim]
        # 3) relational bilinear energy: sum_{I,J} A_new(I,J) (X_new_I * X_new_J) / K^2 [hidden_dim]
        # 4) topological invariants of A_new (mean, std, diag_mean, offdiag_mean, frob_norm, s_entropy) [6]
        readout_dim = hidden_dim * 3 + 6
        self.norm = nn.LayerNorm(readout_dim)
        self.head = nn.Sequential(
            nn.Linear(readout_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, graph: GraphData) -> Tuple[Tensor, PoolingOutput]:
        out = self.pool(graph)
        a_next = out.adj_new
        k = self.num_clusters
        if self.pool.topology_name == "diffpool":
            a_next = a_next / (a_next.detach().abs().max().clamp(min=1.0))

        # Preserve exact fuzzy super-edge weights in [0, 1] during post-pooling message passing
        a_msg = a_next / float(max(k, 1))
        h2 = self.post_gnn(a_msg, out.x_new, prenormalized=True)

        # Relational interaction energy weighted by reduced topology A_new
        ax_new = a_msg @ out.x_new  # [K, d]
        rel_energy = (out.x_new * ax_new).mean(dim=0)  # [d]

        # Super-graph topological descriptors from A_new and S
        diag_a = torch.diagonal(a_next)
        diag_mean = diag_a.mean()
        if k > 1:
            off_mask = 1.0 - torch.eye(k, dtype=a_next.dtype, device=a_next.device)
            off_mean = (a_next * off_mask).sum() / float(k * (k - 1))
        else:
            off_mean = diag_mean
        a_mean = a_next.mean()
        a_std = a_next.std(unbiased=False)
        a_frob = torch.linalg.norm(a_next, ord="fro") / float(max(k, 1))
        s_sharpness = (out.s * out.s).mean()
        topo_stats = torch.stack([a_mean, a_std, diag_mean, off_mean, a_frob, s_sharpness], dim=0)

        readout = torch.cat([h2.mean(dim=0), h2.amax(dim=0), rel_energy, topo_stats], dim=-1)
        readout = self.norm(readout)
        logits = self.head(readout)
        return logits, out


def _compute_multiclass_f1_and_auc(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    probs: np.ndarray,
    num_classes: int,
) -> Tuple[float, float]:
    """Compute unweighted Macro-F1 and binary/OvR ROC-AUC without external sklearn dependency."""
    f1s: List[float] = []
    for c in range(num_classes):
        tp = float(np.sum((y_true == c) & (y_pred == c)))
        fp = float(np.sum((y_true != c) & (y_pred == c)))
        fn = float(np.sum((y_true == c) & (y_pred != c)))
        denom = 2.0 * tp + fp + fn
        f1s.append((2.0 * tp / denom) if denom > 0 else 0.0)
    macro_f1 = float(np.mean(f1s)) if f1s else 0.0

    def _binary_auc(bin_true: np.ndarray, scores: np.ndarray) -> Optional[float]:
        pos = bin_true == 1
        neg = bin_true == 0
        n_pos = int(pos.sum())
        n_neg = int(neg.sum())
        if n_pos == 0 or n_neg == 0:
            return None
        order = np.argsort(scores, kind="mergesort")
        ranks = np.empty_like(order, dtype=np.float64)
        ranks[order] = np.arange(1, len(scores) + 1, dtype=np.float64)
        # Handle ties by averaging ranks
        sorted_scores = scores[order]
        i = 0
        n = len(scores)
        while i < n:
            j = i + 1
            while j < n and sorted_scores[j] == sorted_scores[i]:
                j += 1
            if j > i + 1:
                avg_r = 0.5 * (i + 1 + j)
                ranks[order[i:j]] = avg_r
            i = j
        sum_pos_ranks = float(ranks[pos].sum())
        auc = (sum_pos_ranks - n_pos * (n_pos + 1) * 0.5) / float(n_pos * n_neg)
        return float(np.clip(auc, 0.0, 1.0))

    if num_classes == 2:
        auc_val = _binary_auc((y_true == 1).astype(np.int64), probs[:, 1])
        roc_auc = auc_val if auc_val is not None else 0.5
    else:
        aucs: List[float] = []
        for c in range(num_classes):
            a = _binary_auc((y_true == c).astype(np.int64), probs[:, c])
            if a is not None:
                aucs.append(a)
        roc_auc = float(np.mean(aucs)) if aucs else 0.5

    return macro_f1, roc_auc


def evaluate_methods_on_real_dataset(
    bundle: RealWorldDatasetBundle,
    method_keys: Sequence[str] = REPRESENTATIVE_REALWORLD_METHOD_KEYS,
    epochs: int = 14,
    hidden_dim: int = 24,
    lr: float = 0.015,
    mini_batch_size: int = 16,
    seed: int = 42,
) -> List[RealWorldBenchmarkRecord]:
    """Train and evaluate each graph reduction method on a single real-world dataset bundle."""
    train_set = bundle.train_graphs
    test_set = bundle.test_graphs
    num_classes = bundle.num_classes
    num_clusters = bundle.spec.default_num_clusters

    # Use standard cross-entropy for Bayes-optimal accuracy; mild weighting only for extreme imbalance (<25% minority)
    train_y = np.array([int(g.y.item()) for g in train_set if g.y is not None], dtype=np.int64)
    counts = np.bincount(train_y, minlength=num_classes).astype(np.float32)
    counts = np.maximum(counts, 1.0)
    min_share = float(counts.min() / max(len(train_y), 1))
    if num_classes == 2 and min_share < 0.25:
        inv_w = np.sqrt(len(train_y) / (float(num_classes) * counts))
        inv_w = inv_w / inv_w.mean()
        class_weights: Optional[Tensor] = torch.tensor(inv_w, dtype=torch.float32)
    else:
        class_weights = None
    eye_k = torch.eye(num_clusters, dtype=torch.float32) / float(np.sqrt(max(num_clusters, 1)))

    records: List[RealWorldBenchmarkRecord] = []
    for key in method_keys:
        spec = METHOD_SPEC_MAP[key]
        torch.manual_seed(seed)
        np_rng = np.random.default_rng(seed)

        model = RealWorldPoolingClassifier(
            method_key=key,
            in_features=bundle.num_features,
            hidden_dim=hidden_dim,
            num_clusters=num_clusters,
            num_classes=num_classes,
        )
        # Compensate O(1/N) COG quotient gradient attenuation on the pooling assignment branch
        pool_branch_params: List[nn.Parameter] = []
        base_params: List[nn.Parameter] = []
        for pname, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if any(tok in pname for tok in ("gnn_pool", "membership", "topology_reducer", "feature_reducer")):
                pool_branch_params.append(param)
            else:
                base_params.append(param)

        optimizer = torch.optim.AdamW(
            [
                {"params": base_params, "lr": lr},
                {"params": pool_branch_params, "lr": lr * 3.0},
            ],
            weight_decay=1e-4,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1), eta_min=lr * 0.1)

        t0 = time.perf_counter()
        final_loss = 0.0
        n_train = len(train_set)
        log_k = float(np.log(max(num_clusters, 2)))

        for _ in range(epochs):
            model.train()
            perm = np_rng.permutation(n_train)
            epoch_loss_sum = 0.0
            n_steps = 0

            for start in range(0, n_train, mini_batch_size):
                batch_idx = perm[start : start + mini_batch_size]
                optimizer.zero_grad()
                logits_list: List[Tensor] = []
                targets_list: List[Tensor] = []
                aux_sum = torch.tensor(0.0)

                for idx in batch_idx:
                    g = train_set[int(idx)]
                    assert g.y is not None
                    logits, out = model(g)
                    logits_list.append(logits)
                    targets_list.append(g.y)
                    if model.pool.is_classical:
                        aux_sum = aux_sum + out.aux_loss
                    else:
                        # Row-normalized fuzzy partition diversity + entropy sharpening regularizer
                        s_row = out.s / (out.s.sum(dim=1, keepdim=True).clamp(min=1e-8))
                        sts = s_row.t() @ s_row
                        sts_norm = sts / (torch.linalg.norm(sts, ord="fro").clamp(min=1e-8))
                        l_div = torch.linalg.norm(sts_norm - eye_k, ord="fro")
                        l_ent = -(s_row * torch.log(s_row + 1e-8)).sum(dim=1).mean() / log_k
                        aux_sum = aux_sum + (l_div + 0.35 * l_ent)

                batch_logits = torch.stack(logits_list, dim=0)
                batch_targets = torch.stack(targets_list, dim=0)
                ce_loss = F.cross_entropy(batch_logits, batch_targets, weight=class_weights)
                loss = ce_loss + 0.15 * (aux_sum / float(len(batch_idx)))
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()

                epoch_loss_sum += float(loss.detach().item())
                n_steps += 1

            scheduler.step()
            final_loss = epoch_loss_sum / max(n_steps, 1)

        ms_per_epoch = (time.perf_counter() - t0) * 1000.0 / max(epochs, 1)

        model.eval()
        with torch.no_grad():
            tr_probs_list: List[np.ndarray] = []
            tr_true: List[int] = []
            for g in train_set:
                assert g.y is not None
                lg, _ = model(g)
                tr_probs_list.append(F.softmax(lg, dim=-1).cpu().numpy())
                tr_true.append(int(g.y.item()))
            tr_probs_np = np.stack(tr_probs_list, axis=0)
            tr_true_np = np.array(tr_true, dtype=np.int64)

            # Calibrate binary decision threshold strictly on training set to prevent majority-class collapse
            best_thr = 0.5
            if num_classes == 2:
                best_score = -1.0
                for cand_t in np.linspace(0.25, 0.75, 51):
                    cand_pred = (tr_probs_np[:, 1] >= cand_t).astype(np.int64)
                    cand_acc = float(np.mean(cand_pred == tr_true_np))
                    cand_f1, _ = _compute_multiclass_f1_and_auc(tr_true_np, cand_pred, tr_probs_np, 2)
                    score = 0.5 * cand_acc + 0.5 * cand_f1
                    if score > best_score:
                        best_score = score
                        best_thr = float(cand_t)
                tr_preds_np = (tr_probs_np[:, 1] >= best_thr).astype(np.int64)
            else:
                tr_preds_np = tr_probs_np.argmax(axis=1)
            train_acc = float(np.mean(tr_preds_np == tr_true_np))

            te_preds: List[int] = []
            te_true: List[int] = []
            te_probs: List[np.ndarray] = []
            r2_list: List[float] = []
            dens_list: List[float] = []
            spars_list: List[float] = []
            pc_list: List[float] = []

            for g in test_set:
                assert g.y is not None
                lg, out = model(g)
                prob = F.softmax(lg, dim=-1).cpu().numpy()
                te_probs.append(prob)
                if num_classes == 2:
                    te_preds.append(int(prob[1] >= best_thr))
                else:
                    te_preds.append(int(lg.argmax().item()))
                te_true.append(int(g.y.item()))

                r2_list.append(compute_feature_reconstruction_r2(target_feat=out.h, s=out.s, x_new=out.x_new))
                a_np = out.adj_new.detach().cpu().numpy()
                dens_list.append(float(np.mean(np.clip(a_np, 0.0, 1.0))))
                spars_list.append(float(np.mean(np.abs(a_np) < 1e-3)))

                s_np = out.s.detach().cpu().numpy()
                s_row_sum = np.maximum(s_np.sum(axis=1, keepdims=True), 1e-12)
                s_norm = s_np / s_row_sum
                pc_list.append(float(np.mean(np.sum(s_norm * s_norm, axis=1))))

            y_true_np = np.array(te_true, dtype=np.int64)
            y_pred_np = np.array(te_preds, dtype=np.int64)
            probs_np = np.stack(te_probs, axis=0)

            test_acc = float(np.mean(y_true_np == y_pred_np))
            macro_f1, roc_auc = _compute_multiclass_f1_and_auc(
                y_true=y_true_np,
                y_pred=y_pred_np,
                probs=probs_np,
                num_classes=num_classes,
            )

        records.append(
            RealWorldBenchmarkRecord(
                dataset=bundle.spec.name,
                category=bundle.spec.category,
                task_ua=bundle.spec.task_ua,
                num_graphs_total=bundle.total_available_graphs,
                num_graphs_sampled=len(bundle.graphs),
                num_classes=bundle.num_classes,
                avg_nodes=round(bundle.avg_nodes, 2),
                avg_edges=round(bundle.avg_edges, 2),
                num_clusters=num_clusters,
                method_key=key,
                method_code=spec.code,
                method_title=spec.title,
                family=METHOD_FAMILY_LABELS.get(key, spec.section),
                train_acc=train_acc,
                test_acc=test_acc,
                test_macro_f1=macro_f1,
                test_roc_auc=roc_auc,
                generalization_gap=abs(train_acc - test_acc),
                final_loss=final_loss,
                feature_recon_r2=float(np.mean(r2_list)) if r2_list else 0.0,
                supergraph_density=float(np.mean(dens_list)) if dens_list else 0.0,
                supergraph_sparsity=float(np.mean(spars_list)) if spars_list else 0.0,
                bezdek_pc=float(np.mean(pc_list)) if pc_list else 0.0,
                learned_params_summary=_summarize_learned_params(model.pool),
                ms_per_epoch=ms_per_epoch,
            )
        )

    return records


def run_real_world_benchmark_suite(
    dataset_names: Optional[Sequence[str]] = None,
    method_keys: Optional[Sequence[str]] = None,
    epochs: int = 16,
    hidden_dim: int = 24,
    lr: float = 0.012,
    max_graphs_override: Optional[int] = None,
    seed: int = 42,
    verbose: bool = True,
) -> RealWorldBenchmarkSuiteResult:
    """Execute the multi-domain real-world benchmark across all 10 datasets and aggregate rankings."""
    ds_names = (
        list(dataset_names)
        if dataset_names is not None
        else [s.name for s in REAL_WORLD_DATASET_SPECS]
    )
    m_keys = (
        [s.key for s in CANONICAL_METHOD_SPECS]
        if method_keys == "all"
        else (list(method_keys) if method_keys is not None else list(REPRESENTATIVE_REALWORLD_METHOD_KEYS))
    )

    dataset_summaries: List[Dict[str, object]] = []
    all_records: List[RealWorldBenchmarkRecord] = []
    dataset_champions: Dict[str, RealWorldBenchmarkRecord] = {}

    for ds_name in ds_names:
        bundle = load_real_world_dataset(
            ds_name,
            max_graphs=max_graphs_override,
            seed=seed,
        )
        dataset_summaries.append(bundle.summary_dict())
        if verbose:
            print(
                f"  -> Dataset [{bundle.spec.name}] ({bundle.spec.task_ua}): "
                f"sampled={len(bundle.graphs)}/{bundle.total_available_graphs}, "
                f"C={bundle.num_classes}, F={bundle.num_features}, "
                f"avg_N={bundle.avg_nodes:.1f}, avg_M={bundle.avg_edges:.1f}, K={bundle.spec.default_num_clusters}"
            )

        ds_records = evaluate_methods_on_real_dataset(
            bundle=bundle,
            method_keys=m_keys,
            epochs=epochs,
            hidden_dim=hidden_dim,
            lr=lr,
            seed=seed,
        )
        all_records.extend(ds_records)

        # Primary selection criterion: ROC-AUC for ogbg-molhiv, (test_acc + 0.5*macro_f1 + 0.25*roc_auc) for TUDataset
        if bundle.spec.name == "ogbg-molhiv":
            champ = max(ds_records, key=lambda r: (r.test_roc_auc, r.test_macro_f1, r.test_acc))
        else:
            champ = max(ds_records, key=lambda r: (r.test_acc, r.test_macro_f1, r.test_roc_auc))
        dataset_champions[bundle.spec.name] = champ

        if verbose:
            print(
                f"     Champion on {bundle.spec.name}: {champ.method_code} ({champ.method_key}) "
                f"Acc={champ.test_acc:.1%}, F1={champ.test_macro_f1:.3f}, AUC={champ.test_roc_auc:.3f}"
            )

    # Compute per-dataset ranks and cross-dataset method summaries
    by_dataset: Dict[str, List[RealWorldBenchmarkRecord]] = {}
    for rec in all_records:
        by_dataset.setdefault(rec.dataset, []).append(rec)

    method_ranks: Dict[str, List[float]] = {k: [] for k in m_keys}
    wins_count: Dict[str, int] = {k: 0 for k in m_keys}

    for ds_name, recs in by_dataset.items():
        if ds_name == "ogbg-molhiv":
            scores = [round(r.test_roc_auc, 4) for r in recs]
        else:
            scores = [round(r.test_acc + 0.15 * r.test_roc_auc, 4) for r in recs]
        max_sc = max(scores)
        unique_desc = sorted(set(scores), reverse=True)
        for r, sc in zip(recs, scores):
            rk = float(unique_desc.index(sc) + 1)
            method_ranks[r.method_key].append(rk)
            if abs(sc - max_sc) < 1e-6:
                wins_count[r.method_key] += 1

    bio_chem_ds = {"PROTEINS", "DD", "NCI1", "NCI109", "ENZYMES", "ogbg-molhiv"}
    social_ds = {"IMDB-BINARY", "IMDB-MULTI", "COLLAB", "REDDIT-BINARY"}

    summaries: List[RealWorldMethodSummary] = []
    for k in m_keys:
        spec = METHOD_SPEC_MAP[k]
        k_recs = [r for r in all_records if r.method_key == k]
        if not k_recs:
            continue
        bio_recs = [r for r in k_recs if r.dataset in bio_chem_ds]
        soc_recs = [r for r in k_recs if r.dataset in social_ds]
        hiv_recs = [r for r in k_recs if r.dataset == "ogbg-molhiv"]

        summaries.append(
            RealWorldMethodSummary(
                rank=0,
                method_key=k,
                method_code=spec.code,
                method_title=spec.title,
                family=METHOD_FAMILY_LABELS.get(k, spec.section),
                mean_test_acc=float(np.mean([r.test_acc for r in k_recs])),
                mean_macro_f1=float(np.mean([r.test_macro_f1 for r in k_recs])),
                mean_roc_auc=float(np.mean([r.test_roc_auc for r in k_recs])),
                mean_rank=float(np.mean(method_ranks[k])) if method_ranks[k] else 0.0,
                datasets_won_or_tied=wins_count[k],
                bio_chem_mean_acc=float(np.mean([r.test_acc for r in bio_recs])) if bio_recs else 0.0,
                social_mean_acc=float(np.mean([r.test_acc for r in soc_recs])) if soc_recs else 0.0,
                ogb_molhiv_auc=float(hiv_recs[0].test_roc_auc) if hiv_recs else 0.0,
                mean_feature_r2=float(np.mean([r.feature_recon_r2 for r in k_recs])),
                mean_ms_per_epoch=float(np.mean([r.ms_per_epoch for r in k_recs])),
            )
        )

    # Sort leaderboard by (mean_test_acc + 0.35 * mean_roc_auc - 0.02 * mean_rank) descending
    summaries.sort(
        key=lambda s: (s.mean_test_acc + 0.35 * s.mean_roc_auc - 0.02 * s.mean_rank),
        reverse=True,
    )
    for idx, s in enumerate(summaries, start=1):
        s.rank = idx

    return RealWorldBenchmarkSuiteResult(
        dataset_summaries=dataset_summaries,
        records=all_records,
        method_leaderboard=summaries,
        dataset_champions=dataset_champions,
    )
