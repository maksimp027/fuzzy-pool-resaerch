"""Verification of equivalence between 2-step sparse decompositions O(M K + N K^2) and dense/direct formulas."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, List

import torch

from labflow.classical import DiffPoolReducer, DMoNReducer, MinCutPoolReducer
from labflow.datasets.synthetic import generate_sbm_graph
from labflow.fuzzy_relational import (
    DombiTopologyReducer,
    GoguenTopologyReducer,
    HamacherTopologyReducer,
    LukasiewiczTopologyReducer,
    SchweizerSklarTopologyReducer,
    ZadehTopologyReducer,
)
from labflow.fuzzy_smooth import BandlerKohoutTopologyReducer
from labflow.gnn import BaseGraphEncoder


@dataclass
class SparseEquivalenceRecord:
    """Equivalence check between sparse 2-step decomposition and dense/closed-form formula."""

    name: str
    passed: bool
    max_abs_diff: float
    tolerance: float = 1e-12

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


def run_sparse_equivalence_suite() -> List[SparseEquivalenceRecord]:
    """Verify 2-step sparse algorithms match dense O(N^2 K^2) / closed-form ternary equations."""
    graph = generate_sbm_graph(block_sizes=(12, 10, 14), p_in=0.55, p_out=0.08, seed=101, dtype=torch.float64)
    adj = graph.dense_adj()
    edge_index, edge_weight = graph.sparse_edges()
    n = graph.num_nodes
    k = 4

    gen = torch.Generator().manual_seed(777)
    s = torch.rand((n, k), dtype=torch.float64, generator=gen) * 0.85 + 0.1
    s_simplex = torch.softmax(torch.randn((n, k), dtype=torch.float64, generator=gen), dim=-1)

    records: List[SparseEquivalenceRecord] = []

    def _check(name: str, a: torch.Tensor, b: torch.Tensor, tol: float = 1e-12) -> None:
        diff = float((a - b).abs().max().item())
        records.append(
            SparseEquivalenceRecord(
                name=name,
                passed=diff <= tol,
                max_abs_diff=diff,
                tolerance=tol,
            )
        )

    # Base GNN Encoder: True Sparse COO vs Dense normalized A_hat
    torch.manual_seed(707)
    enc = BaseGraphEncoder(in_features=graph.num_features, embed_dim=8, pool_dim=k).double()
    h_sp, z_sp = enc.forward_sparse(edge_index, edge_weight, graph.x)
    h_dn, z_dn = enc(adj, graph.x)
    _check("Base GNN (GraphSAGE): true sparse O(M F) vs dense O(N^2 F) H & Z", torch.cat([h_sp, z_sp], dim=1), torch.cat([h_dn, z_dn], dim=1))

    # Classical Methods 1..3: True Sparse COO vs Dense O(N^2)
    dp = DiffPoolReducer()
    dp_sp = dp(None, h_dn, s_simplex, edge_index=edge_index, edge_weight=edge_weight)
    dp_dn = dp(adj, h_dn, s_simplex)
    _check(
        "Method 1 (DiffPool): true sparse Frobenius trace identity vs dense ||A - S S^T||_F^2",
        torch.cat([dp_sp.adj_new.reshape(-1), dp_sp.aux_loss.reshape(1)]),
        torch.cat([dp_dn.adj_new.reshape(-1), dp_dn.aux_loss.reshape(1)]),
    )

    mc = MinCutPoolReducer()
    mc_sp = mc(None, h_dn, s_simplex, edge_index=edge_index, edge_weight=edge_weight)
    mc_dn = mc(adj, h_dn, s_simplex)
    _check(
        "Method 2 (MinCutPool): true sparse O(M K) vs dense spectral cut & A_new",
        torch.cat([mc_sp.adj_new.reshape(-1), mc_sp.aux_loss.reshape(1)]),
        torch.cat([mc_dn.adj_new.reshape(-1), mc_dn.aux_loss.reshape(1)]),
    )

    dm = DMoNReducer()
    dm_sp = dm(None, h_dn, s_simplex, edge_index=edge_index, edge_weight=edge_weight)
    dm_dn = dm(adj, h_dn, s_simplex)
    _check(
        "Method 3 (DMoN): true sparse Chung-Lu modularity vs dense DMoN",
        torch.cat([dm_sp.adj_new.reshape(-1), dm_sp.aux_loss.reshape(1)]),
        torch.cat([dm_dn.adj_new.reshape(-1), dm_dn.aux_loss.reshape(1)]),
    )

    # Method 4: Zadeh
    m4 = ZadehTopologyReducer()
    _check(
        "Method 4 (Zadeh): 2-step sparse vs dense O(N^2 K^2) max-min",
        m4(s, edge_index=edge_index, edge_weight=edge_weight),
        m4.forward_dense_reference(adj, s),
    )

    # Method 5: Goguen
    m5 = GoguenTopologyReducer()
    _check(
        "Method 5 (Goguen): 2-step sparse vs dense O(N^2 K^2) max-product",
        m5(s, edge_index=edge_index, edge_weight=edge_weight),
        m5.forward_dense_reference(adj, s),
    )

    # Method 6: Lukasiewicz (classical eta = 2 and generalized learnable eta_IJ)
    m6_cls = LukasiewiczTopologyReducer(num_clusters=k, learnable_eta=False, init_eta=2.0).double()
    _check(
        "Method 6A (Lukasiewicz eta=2): generalized 2-step vs classical 2-step formula",
        m6_cls(s, edge_index=edge_index, edge_weight=edge_weight),
        m6_cls.forward_classical_two_step(s, edge_index, edge_weight),
    )
    _check(
        "Method 6B (Lukasiewicz eta=2): classical 2-step vs direct edge max",
        m6_cls.forward_classical_two_step(s, edge_index, edge_weight),
        m6_cls.forward_edge_reference(s, edge_index, edge_weight),
    )

    m6_gen = LukasiewiczTopologyReducer(num_clusters=k, learnable_eta=True, init_eta=1.45, clamp_unit=True).double()
    _check(
        "Method 6C (Lukasiewicz [0,1]-clamped learnable eta_IJ): 2-step sparse vs direct edge O(M K^2)",
        m6_gen(s, edge_index=edge_index, edge_weight=edge_weight),
        m6_gen.forward_edge_reference(s, edge_index, edge_weight),
    )

    m6_smooth = LukasiewiczTopologyReducer(
        num_clusters=k, learnable_eta=True, init_eta=1.45, clamp_unit=True, smooth_tau=0.12
    ).double()
    _check(
        "Method 6D (Lukasiewicz smooth Softplus window): 2-step sparse vs direct edge O(M K^2)",
        m6_smooth(s, edge_index=edge_index, edge_weight=edge_weight),
        m6_smooth.forward_edge_reference(s, edge_index, edge_weight),
    )

    # Method 7: Hamacher-Einstein (2-step binary vs closed-form ternary formula from Section 7.1)
    m7_ein = HamacherTopologyReducer(init_gamma=2.0, learnable_gamma=False).double()
    _check(
        "Method 7A (Einstein gamma=2): 2-step sparse vs closed-form ternary equation",
        m7_ein(s, edge_index=edge_index, edge_weight=edge_weight),
        m7_ein.forward_closed_form_reference(s, edge_index, edge_weight),
    )
    m7_gen = HamacherTopologyReducer(init_gamma=0.65, learnable_gamma=True).double()
    _check(
        "Method 7B (Hamacher gamma=0.65): 2-step sparse vs closed-form ternary equation",
        m7_gen(s, edge_index=edge_index, edge_weight=edge_weight),
        m7_gen.forward_closed_form_reference(s, edge_index, edge_weight),
    )

    # Method 8A & 8B: Schweizer-Sklar & Dombi (2-step generator vs direct edge O(M K^2))
    m8_ss = SchweizerSklarTopologyReducer(init_q=1.8, learnable_q=True).double()
    _check(
        "Method 8A (Schweizer-Sklar): 2-step log-domain generator vs direct edge formula",
        m8_ss(s, edge_index=edge_index, edge_weight=edge_weight),
        m8_ss.forward_edge_reference(s, edge_index, edge_weight),
    )
    m8_dombi = DombiTopologyReducer(init_lambda=1.6, learnable_lambda=True).double()
    _check(
        "Method 8B (Dombi): 2-step log-domain generator vs direct edge formula",
        m8_dombi(s, edge_index=edge_index, edge_weight=edge_weight),
        m8_dombi.forward_edge_reference(s, edge_index, edge_weight),
    )

    # Method 11A/B/C: Bandler-Kohout dense 2-step AND Matrix-Free Sparse MEX vs direct min_{i,j} implication
    for impl in ("lukasiewicz", "kleene_dienes", "reichenbach"):
        bk = BandlerKohoutTopologyReducer(implication=impl, ensure_self_loops=True)
        ref_bk = bk.forward_direct_implication_reference(adj, s)
        _check(
            f"Method 11 (Bandler-Kohout {impl}): dense 2-step complement vs universal min_(i,j) implication",
            bk(s, adj=adj),
            ref_bk,
        )
        _check(
            f"Method 11-Sparse (Bandler-Kohout {impl}): Matrix-Free MEX sparse O((M+N)K log MK) vs dense reference",
            bk(s, edge_index=edge_index, edge_weight=edge_weight),
            ref_bk,
        )

    return records
