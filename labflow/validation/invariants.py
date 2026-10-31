"""Mathematical verification of all 16 spectral, algebraic, and topological invariants from the specification."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Dict, List

import torch

from labflow.classical import DiffPoolReducer, DMoNReducer, MinCutPoolReducer
from labflow.datasets.synthetic import (
    generate_parallel_bridges_graph,
    generate_planted_cliques_graph,
    generate_sbm_graph,
)
from labflow.fuzzy_features import (
    BipolarSupremalFeaturePooling,
    ChoquetFeaturePooling,
    COGFeaturePooling,
)
from labflow.fuzzy_relational import (
    GoguenTopologyReducer,
    HamacherTopologyReducer,
    LukasiewiczTopologyReducer,
    SchweizerSklarTopologyReducer,
    ZadehTopologyReducer,
)
from labflow.fuzzy_smooth import (
    BandlerKohoutTopologyReducer,
    BoltzmannTopologyReducer,
    ProbSumTopologyReducer,
)
from labflow.types import PoolingOutput, dense_to_sparse


@dataclass
class InvariantRecord:
    """Verification outcome for a theoretical spectral or topological invariant."""

    code: str
    title: str
    passed: bool
    measured_value: float
    expected_bound_or_target: str
    details: str

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


def run_invariants_suite() -> List[InvariantRecord]:
    """Run all 16 theoretical invariant checks across Methods 1..13B."""
    records: List[InvariantRecord] = []

    graph = generate_sbm_graph(block_sizes=(10, 10, 10), p_in=0.7, p_out=0.08, seed=55, dtype=torch.float64)
    adj = graph.dense_adj()
    x = graph.x
    edge_index, edge_weight = graph.sparse_edges()
    n = graph.num_nodes
    k = 3

    gen = torch.Generator().manual_seed(12345)
    z = torch.randn((n, k), dtype=torch.float64, generator=gen)
    s_simplex = torch.softmax(z, dim=-1)
    s_fuzzy = torch.sigmoid(z)

    # 1. DiffPool Edge Mass Conservation: sum_{I,J} A_new(I,J) == sum_{i,j} A_{ij}
    dp_out = DiffPoolReducer()(adj, x, s_simplex)
    mass_in = float(adj.sum().item())
    mass_out = float(dp_out.adj_new.sum().item())
    mass_err = abs(mass_in - mass_out)
    max_entry = float(dp_out.adj_new.max().item())
    records.append(
        InvariantRecord(
            code="INV-01",
            title="DiffPool: Edge Mass Conservation Law and [0, 1] Scale Breakdown",
            passed=(mass_err < 1e-10) and (max_entry > 1.0),
            measured_value=mass_err,
            expected_bound_or_target="|sum(A_new) - sum(A)| < 1e-10, max(A_new) > 1",
            details=f"sum(A)={mass_in:.4f}, sum(A_new)={mass_out:.4f}, max(A_new)={max_entry:.4f} (> 1.0)",
        )
    )

    # 2. MinCutPool Zero-Diagonal & Spectral Loss Bounds: L_cut in [-1, 0], L_orth in [0, sqrt(2)]
    mc_out = MinCutPoolReducer()(adj, x, s_simplex)
    diag_norm = float(torch.diagonal(mc_out.adj_new).abs().max().item())
    l_cut = float(mc_out.loss_components["l_cut"].item())
    l_orth = float(mc_out.loss_components["l_orth"].item())
    records.append(
        InvariantRecord(
            code="INV-02",
            title="MinCutPool: Self-loop Annihilation and Spectral Bounds L_cut in [-1, 0], L_orth in [0, sqrt(2)]",
            passed=(diag_norm == 0.0) and (-1.0 - 1e-9 <= l_cut <= 1e-9) and (0.0 <= l_orth <= math.sqrt(2.0) + 1e-9),
            measured_value=l_cut,
            expected_bound_or_target="diag==0, L_cut in [-1,0], L_orth in [0, 1.4142]",
            details=f"max|diag|={diag_norm:.2e}, L_cut={l_cut:.4f}, L_orth={l_orth:.4f}",
        )
    )

    # 3. DMoN Rank-1 Modularity Identity & Collapse Regularizer Bound
    dm = DMoNReducer()
    dm_out = dm(adj, x, s_simplex)
    dm_ref = dm.forward_reference(adj, x, s_simplex)
    mod_diff = float((dm_out.loss_components["l_mod"] - dm_ref.loss_components["l_mod"]).abs().item())
    l_col = float(dm_out.loss_components["l_collapse"].item())
    records.append(
        InvariantRecord(
            code="INV-03",
            title="DMoN: Equivalence of Rank-1 Chung-Lu Modularity Matrix Decomposition",
            passed=(mod_diff < 1e-12) and (-1e-9 <= l_col <= math.sqrt(k) - 1.0 + 1e-9),
            measured_value=mod_diff,
            expected_bound_or_target="|Q_rank1 - Q_dense_B| < 1e-12, L_collapse >= 0",
            details=f"|dQ|={mod_diff:.2e}, Q={-float(dm_out.loss_components['l_mod'].item()):.4f}, L_collapse={l_col:.4f}",
        )
    )

    # 4. Zadeh-Godel Idempotence & Height Identity: A_new(I, I) == max_i S_{iI} when A_{ii} == 1
    adj_refl = torch.maximum(adj, torch.eye(n, dtype=torch.float64))
    zadeh_refl = ZadehTopologyReducer()(s_fuzzy, adj=adj_refl)
    heights = s_fuzzy.amax(dim=0)
    height_err = float((torch.diagonal(zadeh_refl) - heights).abs().max().item())
    in_unit = bool((zadeh_refl >= 0.0).all() and (zadeh_refl <= 1.0).all())
    records.append(
        InvariantRecord(
            code="INV-04",
            title="Zadeh-Godel FuzzyPool: [0, 1] Closure and Height Identity A_new(I,I) = height(S_{.I})",
            passed=(height_err < 1e-12) and in_unit,
            measured_value=height_err,
            expected_bound_or_target="|A_new(I,I) - max_i S_{iI}| < 1e-12, A_new in [0, 1]",
            details=f"max|diag - height|={height_err:.2e}, range=[{zadeh_refl.min().item():.3f}, {zadeh_refl.max().item():.3f}]",
        )
    )

    # 5. Zadeh-Godel Edge Bottleneck Masking Theorem: dA_new(I,J)/dS == 0 when A_{i*j*} < min(S_{i*I}, S_{j*J})
    adj_bottle = torch.tensor([[0.0, 0.25], [0.25, 0.0]], dtype=torch.float64)
    s_bottle = torch.tensor([[0.85, 0.10], [0.12, 0.90]], dtype=torch.float64, requires_grad=True)
    a_bottle_new = ZadehTopologyReducer()(s_bottle, adj=adj_bottle)
    # Pair (0, 1): edge (0, 1) has A_{01}=0.25 < min(S_{00}=0.85, S_{11}=0.90) => exact bottleneck masking!
    a_bottle_new[0, 1].backward()
    assert s_bottle.grad is not None
    grad_norm_bottle = float(s_bottle.grad.abs().max().item())
    records.append(
        InvariantRecord(
            code="INV-05",
            title="Zadeh-Godel FuzzyPool: Edge Bottleneck Gradient Masking Effect",
            passed=(abs(float(a_bottle_new[0, 1].item()) - 0.25) < 1e-12) and (grad_norm_bottle == 0.0),
            measured_value=grad_norm_bottle,
            expected_bound_or_target="||dA_new(0,1)/dS|| == 0 when A_{i*j*} < min(S_{i*I}, S_{j*J})",
            details=f"A_new(0,1)={a_bottle_new[0,1].item():.3f} == A_01=0.25, max|dA_new/dS|={grad_norm_bottle:.2e}",
        )
    )

    # 6. Goguen Tropical Shortest-Path Isomorphism: -ln A_new(I,J) == min_{(i,j) in E} (-ln S_{iI} - ln A_{ij} - ln S_{jJ})
    a_goguen = GoguenTopologyReducer()(s_fuzzy, edge_index=edge_index, edge_weight=edge_weight)
    neg_log_a_new = -torch.log(a_goguen)
    src, dst = edge_index[0], edge_index[1]
    tropical_paths = (
        (-torch.log(s_fuzzy[src])).unsqueeze(2)
        + (-torch.log(edge_weight))[:, None, None]
        + (-torch.log(s_fuzzy[dst])).unsqueeze(1)
    ).amin(dim=0)
    trop_err = float((neg_log_a_new - tropical_paths).abs().max().item())
    records.append(
        InvariantRecord(
            code="INV-06",
            title="Goguen FuzzyPool: Tropical Logarithmic Isomorphism to Shortest Path Problem",
            passed=trop_err < 1e-12,
            measured_value=trop_err,
            expected_bound_or_target="|-ln A_new - min_e(-ln S_i - ln A_ij - ln S_j)| < 1e-12",
            details=f"max tropical shortest-path discrepancy = {trop_err:.2e}",
        )
    )

    # 7. Goguen Hebbian Mutual Reinforcement Scaling
    s_heb = torch.tensor([[0.80, 0.10], [0.10, 0.70]], dtype=torch.float64, requires_grad=True)
    a_heb = GoguenTopologyReducer()(s_heb, adj=adj_bottle)
    a_heb[0, 1].backward()
    assert s_heb.grad is not None
    expected_g0 = 0.25 * 0.70  # A_{01} * S_{1,1}
    expected_g1 = 0.80 * 0.25  # S_{0,0} * A_{01}
    heb_err = max(abs(float(s_heb.grad[0, 0].item()) - expected_g0), abs(float(s_heb.grad[1, 1].item()) - expected_g1))
    records.append(
        InvariantRecord(
            code="INV-07",
            title="Goguen FuzzyPool: Hebbian Scaling of Gradient on Both Incident Nodes of Winner Edge",
            passed=heb_err < 1e-12,
            measured_value=heb_err,
            expected_bound_or_target="dA_new/dS_{0,0} = A_{01} S_{1,1}, dA_new/dS_{1,1} = S_{0,0} A_{01}",
            details=f"grad_S[0,0]={s_heb.grad[0,0].item():.4f} ( != 0 unlike Zadeh! ), grad_S[1,1]={s_heb.grad[1,1].item():.4f}",
        )
    )

    # 8. Lukasiewicz Exact Mathematical Sparsity Theorem & Lossless COO/CSR Conversion
    luk = LukasiewiczTopologyReducer(num_clusters=k, learnable_eta=False, init_eta=1.95).double()
    s_luk_test = s_fuzzy.clone()
    s_luk_test[:, 0] *= 0.45  # Make cluster 0 pairs fall below threshold eta=1.95 -> exact zeros
    a_luk = luk(s_luk_test, edge_index=edge_index, edge_weight=edge_weight)
    zero_frac = float((a_luk == 0.0).to(torch.float64).mean().item())
    po_luk = PoolingOutput(x_new=torch.zeros((k, 2), dtype=torch.float64), adj_new=a_luk, s=s_luk_test)
    coo_roundtrip = po_luk.to_sparse_coo().to_dense()
    csr_roundtrip = po_luk.to_sparse_csr().to_dense()
    sparse_rt_err = max(
        float((a_luk - coo_roundtrip).abs().max().item()),
        float((a_luk - csr_roundtrip).abs().max().item()),
    )
    records.append(
        InvariantRecord(
            code="INV-08",
            title="Lukasiewicz FuzzyPool: Exact Mathematical Sparsity Theorem and COO/CSR Export",
            passed=(0.0 < zero_frac < 1.0) and (sparse_rt_err == 0.0),
            measured_value=zero_frac,
            expected_bound_or_target="exact IEEE-754 zeros > 0 and lossless COO/CSR conversion",
            details=f"exact_zero_fraction={zero_frac:.2%}, COO/CSR roundtrip err={sparse_rt_err:.2e}",
        )
    )

    # 9. Hamacher-Einstein Monotonicity in gamma & gamma=1 Goguen Identity
    from labflow.memberships import inverse_softplus

    ham_mod_1 = HamacherTopologyReducer(init_gamma=1.0, learnable_gamma=False).double()
    ham_mod_1.gamma_tilde.fill_(inverse_softplus(1.0))
    ham_g1 = ham_mod_1(s_fuzzy, edge_index=edge_index, edge_weight=edge_weight)
    ham_g2 = HamacherTopologyReducer(init_gamma=2.0, learnable_gamma=False).double()(
        s_fuzzy, edge_index=edge_index, edge_weight=edge_weight
    )
    ham_g5 = HamacherTopologyReducer(init_gamma=5.0, learnable_gamma=False).double()(
        s_fuzzy, edge_index=edge_index, edge_weight=edge_weight
    )
    g1_goguen_err = float((ham_g1 - a_goguen).abs().max().item())
    monotone_ok = bool((ham_g1 >= ham_g2 - 1e-12).all() and (ham_g2 >= ham_g5 - 1e-12).all())
    records.append(
        InvariantRecord(
            code="INV-09",
            title="Hamacher-Einstein FuzzyPool: gamma=1 Identity with Goguen and Monotonic Decrease in gamma",
            passed=(g1_goguen_err < 1e-12) and monotone_ok,
            measured_value=g1_goguen_err,
            expected_bound_or_target="T_{H,1} == Goguen and A_new(gamma=1) >= A_new(gamma=2) >= A_new(gamma=5)",
            details=f"|Hamacher(1) - Goguen|={g1_goguen_err:.2e}, monotone_decreasing={monotone_ok}",
        )
    )

    # 10. Schweizer-Sklar Continuum Asymptotics: q -> inf (Zadeh), q -> 0+ (Goguen), p = 1 (Lukasiewicz eta=2)
    a_zadeh = ZadehTopologyReducer()(s_fuzzy, edge_index=edge_index, edge_weight=edge_weight)
    a_ss_large_q = SchweizerSklarTopologyReducer(init_q=80.0, learnable_q=False).double()(
        s_fuzzy, edge_index=edge_index, edge_weight=edge_weight
    )
    a_ss_small_q = SchweizerSklarTopologyReducer(init_q=1e-4, learnable_q=False).double()(
        s_fuzzy, edge_index=edge_index, edge_weight=edge_weight
    )
    a_ss_p1 = SchweizerSklarTopologyReducer.forward_general_p(s_fuzzy, edge_index, edge_weight, p=1.0)
    a_luk_2 = LukasiewiczTopologyReducer(num_clusters=k, learnable_eta=False, init_eta=2.0).double()(
        s_fuzzy, edge_index=edge_index, edge_weight=edge_weight
    )
    err_q_inf = float((a_ss_large_q - a_zadeh).abs().max().item())
    err_q_zero = float((a_ss_small_q - a_goguen).abs().max().item())
    err_p1 = float((a_ss_p1 - a_luk_2).abs().max().item())
    records.append(
        InvariantRecord(
            code="INV-10",
            title="Schweizer-Sklar FuzzyPool: Continuous Continuum between Zadeh (q->inf), Goguen (q->0+), and Lukasiewicz (p=1)",
            passed=(err_q_inf < 0.02) and (err_q_zero < 1e-4) and (err_p1 < 1e-12),
            measured_value=max(err_q_zero, err_p1),
            expected_bound_or_target="q=80 -> Zadeh (<0.02), q=1e-4 -> Goguen (<1e-4), p=1 == Lukasiewicz",
            details=f"|SS(q=80)-Zadeh|={err_q_inf:.4f}, |SS(q=1e-4)-Goguen|={err_q_zero:.2e}, |SS(p=1)-Luk|={err_p1:.2e}",
        )
    )

    # 11. Boltzmann Temperature Limit (tau_c -> 0+ -> Goguen) & Non-Positive Variance Derivative dA/dtau_c <= 0
    boltz_cold = BoltzmannTopologyReducer(init_tau_c=1e-3, learnable_tau_c=False, mode="global_edge").double()
    a_boltz_cold = boltz_cold(s_fuzzy, edge_index=edge_index, edge_weight=edge_weight)
    boltz_lim_err = float((a_boltz_cold - a_goguen).abs().max().item())

    boltz_warm = BoltzmannTopologyReducer(init_tau_c=0.25, learnable_tau_c=True, mode="global_edge").double()
    a_bw = boltz_warm(s_fuzzy, edge_index=edge_index, edge_weight=edge_weight)
    a_bw.sum().backward()
    assert boltz_warm.tau_c_tilde.grad is not None
    grad_tau = float(boltz_warm.tau_c_tilde.grad.item())
    records.append(
        InvariantRecord(
            code="INV-11",
            title="Boltzmann FuzzyPool: Limit tau_c -> 0+ to Hard Maximum and dA_new/dtau_c = -Var/tau_c^2 <= 0",
            passed=(boltz_lim_err < 1e-3) and (grad_tau <= 0.0),
            measured_value=grad_tau,
            expected_bound_or_target="|Boltz(1e-3) - Goguen| < 1e-3 and d(sum A_new)/dtau_c <= 0",
            details=f"|Boltz(1e-3)-Goguen|={boltz_lim_err:.2e}, d(sum A_new)/d(tau_c_tilde)={grad_tau:.4f} <= 0",
        )
    )

    # 12. ProbSum Parallel-Bridge Subadditivity & Saturation: m parallel edges of weight w -> 1 - (1 - w)^m
    m_bridges = 6
    w_bridge = 0.35
    g_par, s_par = generate_parallel_bridges_graph(
        cluster_size=5, num_bridges=m_bridges, bridge_weight=w_bridge, dtype=torch.float64
    )
    ei_p, ew_p = g_par.sparse_edges()
    a_ps = ProbSumTopologyReducer()(s_par, edge_index=ei_p, edge_weight=ew_p)
    expected_ps = 1.0 - ((1.0 - w_bridge) ** m_bridges)
    ps_err = abs(float(a_ps[0, 1].item()) - expected_ps)
    a_dp_par = DiffPoolReducer()(g_par.dense_adj(), g_par.x, s_par).adj_new
    records.append(
        InvariantRecord(
            code="INV-12",
            title="ProbSum FuzzyPool: Exact Saturation of Parallel Paths 1 - (1 - w)^m within [0, 1]",
            passed=(ps_err < 1e-11) and (0.0 <= float(a_ps[0, 1].item()) <= 1.0) and (float(a_dp_par[0, 1].item()) > 1.0),
            measured_value=ps_err,
            expected_bound_or_target=f"A_new(0,1) == 1 - (1 - {w_bridge})^{m_bridges} = {expected_ps:.6f} in [0,1]",
            details=f"ProbSum={a_ps[0,1].item():.6f} (in [0,1]) vs DiffPool={a_dp_par[0,1].item():.4f} (>1 overflow)",
        )
    )

    # 13. Bandler-Kohout De Morgan Complement-Graph Duality Theorem
    a_refl = torch.maximum(adj, torch.eye(n, dtype=torch.float64))
    a_bar = 1.0 - a_refl
    ei_bar, ew_bar = dense_to_sparse(a_bar, threshold=0.0)
    bk_r = BandlerKohoutTopologyReducer(implication="reichenbach", ensure_self_loops=True)(s_fuzzy, adj=adj)
    goguen_bar = GoguenTopologyReducer()(s_fuzzy, edge_index=ei_bar, edge_weight=ew_bar)
    duality_err = float((bk_r - (1.0 - goguen_bar)).abs().max().item())
    records.append(
        InvariantRecord(
            code="INV-13",
            title="Bandler-Kohout (Reichenbach): De Morgan Duality Theorem 1 - Goguen(1 1^T - A_refl)",
            passed=duality_err < 1e-12,
            measured_value=duality_err,
            expected_bound_or_target="|BK_Reichenbach(A) - (1 - Goguen(A_bar))| < 1e-12",
            details=f"max De Morgan duality error = {duality_err:.2e}",
        )
    )

    # 14. Bandler-Kohout Exact Clique Pooling Theorem: A_new(I, I) == 1 iff support of S_{.I} is a clique
    g_clq = generate_planted_cliques_graph(
        num_cliques=2, clique_size=5, num_noise_nodes=5, noise_edge_prob=0.0, seed=9, dtype=torch.float64
    )
    adj_clq = g_clq.dense_adj()
    # Cluster 0: exact planted 5-clique (nodes 0..4). Cluster 1: 4 clique nodes + 1 disconnected node (not a clique!)
    s_clq = torch.zeros((15, 2), dtype=torch.float64)
    s_clq[0:5, 0] = 0.95
    s_clq[5:9, 1] = 0.95
    s_clq[10, 1] = 0.95  # noise node missing edges to 5..8
    a_bk_clq = BandlerKohoutTopologyReducer(implication="reichenbach", ensure_self_loops=True)(s_clq, adj=adj_clq)
    clique_diag = float(a_bk_clq[0, 0].item())
    non_clique_diag = float(a_bk_clq[1, 1].item())
    records.append(
        InvariantRecord(
            code="INV-14",
            title="Bandler-Kohout: Clique Extraction Theorem (A_new(I,I) == 1 iff cluster is a clique)",
            passed=(abs(clique_diag - 1.0) < 1e-12) and (non_clique_diag < 0.2),
            measured_value=clique_diag,
            expected_bound_or_target="A_new(clique, clique) == 1.0, A_new(non_clique, non_clique) << 1.0",
            details=f"A_new(0,0) [true clique] = {clique_diag:.4f}, A_new(1,1) [1 missing node] = {non_clique_diag:.4f}",
        )
    )

    # 15. Method 12 (COG): Limit m -> 1+ recovers D_S^{-1} S^T H and translation equivariance
    cog_m1 = COGFeaturePooling(init_m=1.0 + 1e-7, learnable_m=False).double()
    x_cog_1 = cog_m1(x, s_fuzzy)
    x_mean = (s_fuzzy.t() @ x) / s_fuzzy.sum(dim=0, keepdim=True).t()
    cog_lim_err = float((x_cog_1 - x_mean).abs().max().item())

    cog_m2 = COGFeaturePooling(init_m=2.0, learnable_m=False).double()
    shift = 3.75
    shift_err = float((cog_m2(x + shift, s_fuzzy) - (cog_m2(x, s_fuzzy) + shift)).abs().max().item())
    records.append(
        InvariantRecord(
            code="INV-15",
            title="COG Feature Pooling: Limit m -> 1+ to Weighted Mean and Translation Equivariance",
            passed=(cog_lim_err < 1e-6) and (shift_err < 1e-11),
            measured_value=max(cog_lim_err, shift_err),
            expected_bound_or_target="|COG(m->1) - Mean| < 1e-6 and |COG(H+c) - (COG(H)+c)| < 1e-11",
            details=f"|COG(1+1e-7)-Mean|={cog_lim_err:.2e}, shift_equivariance_err={shift_err:.2e}",
        )
    )

    # 16. Method 13A (Bipolar) Anti-Symmetry & Method 13B (Choquet) Min-Mean-Max Interpolation
    bip = BipolarSupremalFeaturePooling()
    antisym_err = float((bip(-x, s_fuzzy) + bip(x, s_fuzzy)).abs().max().item())

    choq_m1 = ChoquetFeaturePooling(num_clusters=k, init_alpha=1.0, learnable_alpha=False).double()
    choq_m1.alpha_tilde.fill_(inverse_softplus(1.0))
    choq_mean = choq_m1(x, s_fuzzy)
    choq_max = ChoquetFeaturePooling(num_clusters=k, init_alpha=1e-4, learnable_alpha=False).double()(x, s_fuzzy)
    choq_min = ChoquetFeaturePooling(num_clusters=k, init_alpha=400.0, learnable_alpha=False).double()(x, s_fuzzy)
    choq_mean_err = float((choq_mean - x_mean).abs().max().item())
    col_max = x.amax(dim=0).unsqueeze(0).expand_as(choq_max)
    col_min = x.amin(dim=0).unsqueeze(0).expand_as(choq_min)
    max_interp_err = float((choq_max - col_max).abs().max().item())
    min_interp_err = float((choq_min - col_min).abs().max().item())
    ordered_ok = bool((choq_min <= choq_mean + 1e-10).all() and (choq_mean <= choq_max + 1e-10).all())
    records.append(
        InvariantRecord(
            code="INV-16",
            title="Bipolar Pooling & Choquet Integral: Cone Anti-Symmetry and Continuous Min-Mean-Max Interpolation",
            passed=(antisym_err < 1e-12) and (choq_mean_err < 1e-10) and ordered_ok and (max_interp_err < 0.01) and (min_interp_err < 0.01),
            measured_value=choq_mean_err,
            expected_bound_or_target="Bipolar(-H) == -Bipolar(H), Choquet(alpha=1) == Mean, Choquet(400) <= Mean <= Choquet(1e-4)",
            details=(
                f"bipolar_antisym_err={antisym_err:.2e}, |Choquet(1)-Mean|={choq_mean_err:.2e}, "
                f"|Choquet(1e-4)-Max|={max_interp_err:.4f}, |Choquet(400)-Min|={min_interp_err:.4f}"
            ),
        )
    )

    return records
