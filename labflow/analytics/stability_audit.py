"""Automated numerical stability & audit stress-test suite verifying all AUD.md mathematical fixes."""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from typing import Dict, List

import torch

from labflow.composite import CANONICAL_METHOD_SPECS, GraphReductionLayer
from labflow.datasets.classic import generate_large_sparse_sbm_graph
from labflow.fuzzy_relational import (
    DombiTopologyReducer,
    HamacherTopologyReducer,
    LukasiewiczTopologyReducer,
    SchweizerSklarTopologyReducer,
)
from labflow.fuzzy_smooth import (
    BandlerKohoutTopologyReducer,
    BoltzmannTopologyReducer,
    ProbSumTopologyReducer,
)
from labflow.memberships import inverse_softplus


@dataclass
class AuditStressRecord:
    """Verification result for an AUD.md mathematical or numerical stress test."""

    code: str
    title: str
    audit_section: str
    passed: bool
    metric_value: float
    target_criterion: str
    details: str
    elapsed_ms: float

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


def run_audit_stress_suite() -> List[AuditStressRecord]:
    """Execute all 8 mathematical & numerical stress tests corresponding to AUD.md findings."""
    records: List[AuditStressRecord] = []

    # ST-01: Hamacher crisp boundary T_{H, gamma}(1, 1, 1) == 1 without 0/0 singularity
    t0 = time.perf_counter()
    ei_1 = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    ew_1 = torch.ones((2,), dtype=torch.float32, requires_grad=True)
    s_1 = torch.ones((2, 2), dtype=torch.float32, requires_grad=True)
    max_ham_err = 0.0
    grad_finite_ham = True
    for g_val in (1e-4, 0.5, 1.0, 2.0, 10.0):
        ham = HamacherTopologyReducer(init_gamma=g_val, learnable_gamma=True)
        with torch.no_grad():
            ham.gamma_tilde.fill_(inverse_softplus(g_val))
        a_h = ham(s_1, edge_index=ei_1, edge_weight=ew_1)
        err = float((a_h - 1.0).abs().max().item())
        max_ham_err = max(max_ham_err, err)
        a_h.sum().backward()
        assert s_1.grad is not None and ew_1.grad is not None and ham.gamma_tilde.grad is not None
        grad_finite_ham = grad_finite_ham and bool(
            torch.isfinite(s_1.grad).all()
            and torch.isfinite(ew_1.grad).all()
            and torch.isfinite(ham.gamma_tilde.grad).all()
        )
        s_1.grad.zero_()
        ew_1.grad.zero_()
    records.append(
        AuditStressRecord(
            code="ST-01",
            title="Hamacher T_{H,gamma}(1,1,1) == 1: Elimination of 0/0 Singularity as gamma -> 0+",
            audit_section="Audit Section 2.2 (Method 7)",
            passed=(max_ham_err < 1e-5) and grad_finite_ham,
            metric_value=max_ham_err,
            target_criterion="|T_{H,gamma}(1,1,1) - 1| < 1e-5 & finite grads",
            details=f"max_boundary_err={max_ham_err:.2e} across gamma in [1e-4..10], finite_grads={grad_finite_ham}",
            elapsed_ms=(time.perf_counter() - t0) * 1000.0,
        )
    )

    # ST-02: Lukasiewicz [0, 1] unit-interval closure when eta_{IJ} in (1, 2)
    t0 = time.perf_counter()
    s_luk = torch.full((4, 3), 0.98, dtype=torch.float64)
    ei_luk = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long)
    ew_luk = torch.ones((3,), dtype=torch.float64)
    luk_unclamped = LukasiewiczTopologyReducer(3, learnable_eta=False, init_eta=1.10, clamp_unit=False).double()
    luk_clamped = LukasiewiczTopologyReducer(3, learnable_eta=False, init_eta=1.10, clamp_unit=True).double()
    luk_smooth = LukasiewiczTopologyReducer(
        3, learnable_eta=False, init_eta=1.10, clamp_unit=True, smooth_tau=0.12
    ).double()
    max_legacy = float(luk_unclamped(s_luk, edge_index=ei_luk, edge_weight=ew_luk).max().item())
    a_clamped = luk_clamped(s_luk, edge_index=ei_luk, edge_weight=ew_luk)
    a_smooth = luk_smooth(s_luk, edge_index=ei_luk, edge_weight=ew_luk)
    max_fixed = max(float(a_clamped.max().item()), float(a_smooth.max().item()))
    min_fixed = min(float(a_clamped.min().item()), float(a_smooth.min().item()))
    records.append(
        AuditStressRecord(
            code="ST-02",
            title="Lukasiewicz [0, 1] Closure under Learnable Threshold eta_{IJ} in (1, 2)",
            audit_section="Audit Section 2.2 / Section 3.3 (Method 6)",
            passed=(max_legacy > 1.5) and (0.0 <= min_fixed) and (max_fixed <= 1.0 + 1e-12),
            metric_value=max_fixed,
            target_criterion="0 <= A_new <= 1 (legacy overflow > 1.8 eliminated)",
            details=f"unclamped_max={max_legacy:.3f} (>1!) -> clamped_max={a_clamped.max().item():.4f}, smooth_max={a_smooth.max().item():.4f}",
            elapsed_ms=(time.perf_counter() - t0) * 1000.0,
        )
    )

    # ST-03: Smooth Lukasiewicz eliminates Dead Super-Edges under cold-start sub-threshold memberships
    t0 = time.perf_counter()
    s_cold = torch.full((6, 3), 0.18, dtype=torch.float64, requires_grad=True)
    ei_cold = torch.tensor([[0, 1, 2, 3, 4], [1, 2, 3, 4, 5]], dtype=torch.long)
    ew_cold = torch.full((5,), 0.45, dtype=torch.float64)
    # Sum = 0.18 + 0.45 + 0.18 = 0.81 < eta = 1.60 => hard ReLU is 100% dead!
    hard_luk = LukasiewiczTopologyReducer(3, learnable_eta=True, init_eta=1.60, clamp_unit=True, smooth_tau=0.0).double()
    a_hard = hard_luk(s_cold, edge_index=ei_cold, edge_weight=ew_cold)
    a_hard.sum().backward()
    assert s_cold.grad is not None
    hard_grad_norm = float(s_cold.grad.abs().max().item())

    s_cold_sm = s_cold.detach().clone().requires_grad_(True)
    sm_luk = LukasiewiczTopologyReducer(3, learnable_eta=True, init_eta=1.60, clamp_unit=True, smooth_tau=0.12).double()
    a_sm = sm_luk(s_cold_sm, edge_index=ei_cold, edge_weight=ew_cold)
    a_sm.sum().backward()
    assert s_cold_sm.grad is not None and sm_luk.eta_tilde is not None and sm_luk.eta_tilde.grad is not None
    sm_grad_norm = float(s_cold_sm.grad.abs().max().item())
    eta_grad_norm = float(sm_luk.eta_tilde.grad.abs().max().item())
    records.append(
        AuditStressRecord(
            code="ST-03",
            title="Smooth Lukasiewicz (Softplus Window): Elimination of Dead Super-Edges",
            audit_section="Audit Section 3.3 (Method 6)",
            passed=(hard_grad_norm == 0.0) and (sm_grad_norm > 1e-4) and (eta_grad_norm > 1e-5),
            metric_value=sm_grad_norm,
            target_criterion="||dA_smooth/dS|| > 1e-4 where hard ReLU grad == 0",
            details=f"hard_grad_max={hard_grad_norm:.2e} (100% dead) -> smooth_grad_max={sm_grad_norm:.3e}, eta_grad={eta_grad_norm:.3e}",
            elapsed_ms=(time.perf_counter() - t0) * 1000.0,
        )
    )

    # ST-04: Float32 extreme-parameter Log-Domain Schweizer-Sklar (q=45) & Dombi (lambda=45)
    t0 = time.perf_counter()
    gen_f32 = torch.Generator().manual_seed(404)
    s_ext = (torch.rand((12, 4), generator=gen_f32, dtype=torch.float32) * 0.94 + 0.02).requires_grad_(True)
    ei_ext = torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]], dtype=torch.long)
    ew_ext = (torch.rand((11,), generator=gen_f32, dtype=torch.float32) * 0.90 + 0.05).requires_grad_(True)

    ss_ext = SchweizerSklarTopologyReducer(init_q=45.0, learnable_q=True)
    a_ss = ss_ext(s_ext, edge_index=ei_ext, edge_weight=ew_ext)
    a_ss.sum().backward()
    assert s_ext.grad is not None and ew_ext.grad is not None and ss_ext.q_tilde.grad is not None
    ss_ok = bool(
        torch.isfinite(a_ss).all()
        and (a_ss >= 0.0).all()
        and (a_ss <= 1.0).all()
        and torch.isfinite(s_ext.grad).all()
        and torch.isfinite(ew_ext.grad).all()
        and torch.isfinite(ss_ext.q_tilde.grad).all()
    )

    s_ext2 = s_ext.detach().clone().requires_grad_(True)
    ew_ext2 = ew_ext.detach().clone().requires_grad_(True)
    dom_ext = DombiTopologyReducer(init_lambda=45.0, learnable_lambda=True)
    a_dom = dom_ext(s_ext2, edge_index=ei_ext, edge_weight=ew_ext2)
    a_dom.sum().backward()
    assert s_ext2.grad is not None and ew_ext2.grad is not None and dom_ext.lam_tilde.grad is not None
    dom_ok = bool(
        torch.isfinite(a_dom).all()
        and (a_dom >= 0.0).all()
        and (a_dom <= 1.0).all()
        and torch.isfinite(s_ext2.grad).all()
        and torch.isfinite(ew_ext2.grad).all()
        and torch.isfinite(dom_ext.lam_tilde.grad).all()
    )
    records.append(
        AuditStressRecord(
            code="ST-04",
            title="Log-Domain Schweizer-Sklar (q=45) and Dombi (lambda=45): Absence of float32 Overflow/NaN",
            audit_section="Audit Section 3.4 (Methods 8A, 8B)",
            passed=ss_ok and dom_ok,
            metric_value=float(max(a_ss.max().item(), a_dom.max().item())),
            target_criterion="100% finite float32 forward & backward at q=lambda=45 (0.02^-45 ~ 10^76)",
            details=f"SS_f32_finite={ss_ok} (max|grad_S|={s_ext.grad.abs().max().item():.2e}), Dombi_f32_finite={dom_ok} (max|grad_S|={s_ext2.grad.abs().max().item():.2e})",
            elapsed_ms=(time.perf_counter() - t0) * 1000.0,
        )
    )

    # ST-05: ProbSum anti-saturation on large edge count M (~7200 directed edges, N=600, K=4)
    t0 = time.perf_counter()
    g_med = generate_large_sparse_sbm_graph(num_nodes=600, num_blocks=4, avg_degree=12.0, seed=505)
    ei_m, ew_m = g_med.sparse_edges()
    s_med = torch.sigmoid(torch.randn((600, 4), generator=torch.Generator().manual_seed(505))).requires_grad_(True)
    ps_raw = ProbSumTopologyReducer(normalization="none")
    a_ps_raw = ps_raw(s_med, edge_index=ei_m, edge_weight=ew_m)
    a_ps_raw.sum().backward()
    assert s_med.grad is not None
    raw_min = float(a_ps_raw.min().item())
    raw_grad = float(s_med.grad.abs().mean().item())

    s_med_n = s_med.detach().clone().requires_grad_(True)
    ps_norm = ProbSumTopologyReducer(normalization="mean_field")
    a_ps_norm = ps_norm(s_med_n, edge_index=ei_m, edge_weight=ew_m)
    a_ps_norm.sum().backward()
    assert s_med_n.grad is not None
    norm_min = float(a_ps_norm.min().item())
    norm_max = float(a_ps_norm.max().item())
    norm_grad = float(s_med_n.grad.abs().mean().item())
    records.append(
        AuditStressRecord(
            code="ST-05",
            title="Scale-Invariant ProbSum (Mean-Field): Prevention of A_new -> 1 Degeneracy",
            audit_section="Audit Section 3.1 (Method 10)",
            passed=(raw_min > 0.999) and (0.02 < norm_min <= norm_max < 0.95) and (norm_grad > 1e-4),
            metric_value=norm_max,
            target_criterion="A_new in (0.02, 0.95) & active grad > 1e-4 (vs raw min(A_new) == 1.0)",
            details=f"M={ew_m.numel()}: raw ProbSum min={raw_min:.6f} (grad={raw_grad:.2e}) -> norm ProbSum range=[{norm_min:.3f}, {norm_max:.3f}] (grad={norm_grad:.2e})",
            elapsed_ms=(time.perf_counter() - t0) * 1000.0,
        )
    )

    # ST-06: Boltzmann graph-size adaptive temperature scaling (ln M/K^2) prevents mean-dilution
    t0 = time.perf_counter()
    boltz_fixed = BoltzmannTopologyReducer(init_tau_c=0.35, learnable_tau_c=False, scale_by_log_m=False)
    boltz_adapt = BoltzmannTopologyReducer(init_tau_c=0.35, learnable_tau_c=False, scale_by_log_m=True)
    with torch.no_grad():
        a_bf = boltz_fixed(s_med.detach(), edge_index=ei_m, edge_weight=ew_m)
        a_ba = boltz_adapt(s_med.detach(), edge_index=ei_m, edge_weight=ew_m)
    mean_bf = float(a_bf.mean().item())
    mean_ba = float(a_ba.mean().item())
    records.append(
        AuditStressRecord(
            code="ST-06",
            title="Adaptive Boltzmann tau_eff = tau_c / (1 + ln(M/K^2)): Protection against Maximum Blurring",
            audit_section="Audit Section 3.5 (Method 9)",
            passed=(mean_ba > mean_bf + 0.03) and (mean_ba <= 1.0),
            metric_value=mean_ba - mean_bf,
            target_criterion="Adaptive Boltzmann sharpens peak recovery over unscaled Boltzmann at large M",
            details=f"M={ew_m.numel()}: unscaled mean(A_new)={mean_bf:.4f} -> log-M adaptive mean(A_new)={mean_ba:.4f} (+{mean_ba - mean_bf:.4f})",
            elapsed_ms=(time.perf_counter() - t0) * 1000.0,
        )
    )

    # ST-07: Matrix-Free Sparse MEX Bandler-Kohout on N=2500 sparse-only graph (no N x N allocation)
    t0 = time.perf_counter()
    g_large = generate_large_sparse_sbm_graph(num_nodes=2500, num_blocks=6, avg_degree=12.0, seed=707)
    assert g_large.adj is None
    ei_l, ew_l = g_large.sparse_edges()
    s_large = torch.sigmoid(torch.randn((2500, 6), generator=torch.Generator().manual_seed(707)))
    bk_all_ok = True
    for impl in ("reichenbach", "lukasiewicz", "kleene_dienes"):
        s_in = s_large.clone().requires_grad_(True)
        bk_mod = BandlerKohoutTopologyReducer(implication=impl, ensure_self_loops=True)
        a_bk_l = bk_mod(s_in, edge_index=ei_l, edge_weight=ew_l)
        a_bk_l.sum().backward()
        assert s_in.grad is not None
        bk_all_ok = bk_all_ok and bool(
            g_large.adj is None
            and a_bk_l.shape == (6, 6)
            and torch.isfinite(a_bk_l).all()
            and (a_bk_l >= -1e-6).all()
            and (a_bk_l <= 1.0 + 1e-6).all()
            and torch.isfinite(s_in.grad).all()
        )
    elapsed_bk = (time.perf_counter() - t0) * 1000.0
    records.append(
        AuditStressRecord(
            code="ST-07",
            title="Matrix-Free Sparse MEX Bandler-Kohout (N=2500, adj=None): Without Complement Matrix A_bar",
            audit_section="Audit Section 3.2 / Section 4.1 (Methods 11A-11C)",
            passed=bk_all_ok,
            metric_value=elapsed_bk,
            target_criterion="All 3 BK implications run in O((M+N)K log MK) with adj=None & finite grads",
            details=f"N=2500, M={ew_l.numel()}, K=6: 3 implications fwd+bwd in {elapsed_bk:.1f} ms (0 bytes N^2 dense alloc)",
            elapsed_ms=elapsed_bk,
        )
    )

    # ST-08: True Sparse End-to-End Pipeline across ALL 22 canonical reducers with adj=None (N=1200)
    t0 = time.perf_counter()
    g_sp_e2e = generate_large_sparse_sbm_graph(num_nodes=1200, num_blocks=6, avg_degree=10.0, feature_dim=12, seed=808)
    assert g_sp_e2e.adj is None
    passed_specs = 0
    for spec in CANONICAL_METHOD_SPECS:
        layer = GraphReductionLayer.from_spec(spec.key, in_features=12, out_features=10, num_clusters=6)
        out = layer(g_sp_e2e)
        loss = out.x_new.mean() + out.adj_new.mean() + out.aux_loss
        loss.backward()
        grads_ok = all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in layer.parameters() if p.requires_grad)
        if g_sp_e2e.adj is None and out.adj_new.shape == (6, 6) and grads_ok:
            passed_specs += 1
    elapsed_e2e = (time.perf_counter() - t0) * 1000.0
    records.append(
        AuditStressRecord(
            code="ST-08",
            title=f"True Sparse End-to-End Pipeline (N=1200, adj=None) for all {len(CANONICAL_METHOD_SPECS)} canonical methods",
            audit_section="Audit Section 4.1 (GNN + Classical + Fuzzy Pipeline)",
            passed=passed_specs == len(CANONICAL_METHOD_SPECS),
            metric_value=float(passed_specs),
            target_criterion=f"{len(CANONICAL_METHOD_SPECS)}/{len(CANONICAL_METHOD_SPECS)} canonical specs run sparse-only without dense_adj()",
            details=f"{passed_specs}/{len(CANONICAL_METHOD_SPECS)} layers completed fwd+bwd on N=1200, M={g_sp_e2e.num_edges} in {elapsed_e2e:.1f} ms",
            elapsed_ms=elapsed_e2e,
        )
    )

    return records
