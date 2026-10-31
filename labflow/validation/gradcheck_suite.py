"""Triple-verification gradient suite:
1. Forward equivalence: Custom analytical forward vs Pure PyTorch reference forward
2. Backward equivalence: Custom analytical Jacobian vs PyTorch autograd Jacobian
3. Numerical Jacobian: torch.autograd.gradcheck (central finite differences in float64)
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Callable, Dict, List, Sequence, Tuple

import torch
from torch import Tensor
from torch.autograd import gradcheck

from labflow.classical import DiffPoolReducer, DMoNReducer, MinCutPoolReducer
from labflow.classical.diffpool import _DiffPoolFn, _DiffPoolSparseFn
from labflow.classical.dmon import _DMoNFn, _DMoNSparseFn
from labflow.classical.mincut import _MinCutPoolFn, _MinCutPoolSparseFn
from labflow.composite import CANONICAL_METHOD_SPECS, GraphReductionLayer
from labflow.fuzzy_features import (
    BipolarSupremalFeaturePooling,
    ChoquetFeaturePooling,
    COGFeaturePooling,
)
from labflow.fuzzy_features.bipolar import _BipolarSupremalFeaturePoolFn
from labflow.fuzzy_features.choquet import _ChoquetFeaturePoolFn
from labflow.fuzzy_features.cog import _COGFeaturePoolFn
from labflow.fuzzy_relational import (
    DombiTopologyReducer,
    GoguenTopologyReducer,
    HamacherTopologyReducer,
    LukasiewiczTopologyReducer,
    SchweizerSklarTopologyReducer,
    ZadehTopologyReducer,
)
from labflow.fuzzy_relational.goguen import _GoguenMaxProdSparseFn
from labflow.fuzzy_relational.hamacher import _HamacherMatrixFn, _HamacherScalarSparseFn
from labflow.fuzzy_relational.lukasiewicz import _LukasiewiczSparseFn, lukasiewicz_activation
from labflow.fuzzy_relational.schweizer_dombi import _DombiSparseFn, _SchweizerSklarSparseFn
from labflow.fuzzy_relational.zadeh import _ZadehMaxMinSparseFn
from labflow.fuzzy_smooth import (
    BandlerKohoutTopologyReducer,
    BoltzmannTopologyReducer,
    ProbSumTopologyReducer,
)
from labflow.fuzzy_smooth.bandler_kohout import (
    _BandlerKohoutKleeneDienesFn,
    _BandlerKohoutLukasiewiczFn,
    _BandlerKohoutReichenbachFn,
    _BandlerKohoutSparseFn,
)
from labflow.fuzzy_smooth.boltzmann import _BoltzmannGlobalEdgeFn, _BoltzmannTwoStepFn, _boltzmann_log_scale_factor
from labflow.fuzzy_smooth.probsum import _ProbSumFn
from labflow.gnn import (
    GraphSAGEOperator,
    _GraphSAGELinearFn,
    _SparseGraphSAGELinearFn,
    sparse_symmetric_normalize_adjacency,
    sparse_symmetric_normalize_with_self_loops,
    symmetric_normalize_adjacency,
)
from labflow.memberships import (
    AnisotropicSigmoidMembership,
    GaussianRBFMembership,
    TemperatureSoftmaxMembership,
    _AnisotropicSigmoidFn,
    _GaussianRBFFn,
    _TemperatureSoftmaxFn,
)
from labflow.types import GraphData, dense_to_sparse


@dataclass
class GradCheckRecord:
    """Result of triple gradient verification for a single operator or method."""

    name: str
    passed: bool
    forward_max_err: float
    autograd_grad_max_err: float
    numerical_gradcheck: bool
    note: str = ""

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


def _well_separated_test_tensors(
    n: int = 6,
    k: int = 3,
    f_in: int = 4,
    f_out: int = 4,
    seed: int = 2026,
) -> Dict[str, Tensor]:
    """Construct float64 test tensors with well-separated values (no ties near kinks)."""
    gen = torch.Generator().manual_seed(seed)

    # Symmetric weighted adjacency in [0.22, 0.94] for classical & undirected tests
    perm_a = torch.randperm(n * n, generator=gen)
    base_a = torch.linspace(0.22, 0.94, n * n, dtype=torch.float64)[perm_a].view(n, n)
    adj_sym = 0.5 * (base_a + base_a.t())
    mask = torch.ones((n, n), dtype=torch.bool)
    mask.fill_diagonal_(False)
    mask[0, 4] = mask[4, 0] = False
    mask[1, 5] = mask[5, 1] = False
    mask[2, 3] = mask[3, 2] = False
    adj = torch.where(mask, adj_sym, torch.zeros_like(adj_sym))

    # Distinct-weight directed adjacency on the same sparsity pattern so independent
    # directed edge_weight perturbations in gradcheck do not sit on an (i,j)==(j,i) tie kink
    adj_dir = torch.where(mask, base_a, torch.zeros_like(base_a))
    edge_index, edge_weight = dense_to_sparse(adj_dir, threshold=0.0)

    # Well-separated fuzzy membership S in [0.14, 0.93]
    perm_s = torch.randperm(n * k, generator=gen)
    s_fuzzy = torch.linspace(0.14, 0.93, n * k, dtype=torch.float64)[perm_s].view(n, k)

    # Row-stochastic S for classical simplex methods
    perm_z = torch.randperm(n * k, generator=gen)
    z = torch.linspace(-1.4, 1.6, n * k, dtype=torch.float64)[perm_z].view(n, k)
    s_simplex = torch.softmax(z, dim=-1)

    # Signed well-separated features H (bounded away from 0 and pairwise separated per column)
    perm_h = torch.randperm(n * f_out, generator=gen)
    raw_h = torch.linspace(-1.85, 1.95, n * f_out, dtype=torch.float64)
    raw_h = torch.where(raw_h.abs() < 0.15, raw_h + 0.31, raw_h)
    h = raw_h[perm_h].view(n, f_out)

    perm_x = torch.randperm(n * f_in, generator=gen)
    x = torch.linspace(-1.2, 1.5, n * f_in, dtype=torch.float64)[perm_x].view(n, f_in)

    return {
        "adj": adj,
        "adj_dir": adj_dir,
        "edge_index": edge_index,
        "edge_weight": edge_weight,
        "s_fuzzy": s_fuzzy,
        "s_simplex": s_simplex,
        "z": z,
        "h": h,
        "x": x,
    }


def _compare_custom_vs_ref(
    name: str,
    custom_fn: Callable[..., Tuple[Tensor, ...] | Tensor],
    ref_fn: Callable[..., Tuple[Tensor, ...] | Tensor],
    inputs: Sequence[Tensor],
    grad_flags: Sequence[bool],
    run_numerical: bool = True,
    note: str = "",
) -> GradCheckRecord:
    """Compare custom analytical autograd against reference PyTorch autograd + numerical gradcheck."""
    inp_custom: List[Tensor] = []
    inp_ref: List[Tensor] = []
    for t, req in zip(inputs, grad_flags):
        tc = t.detach().clone().requires_grad_(req)
        tr = t.detach().clone().requires_grad_(req)
        inp_custom.append(tc)
        inp_ref.append(tr)

    out_c = custom_fn(*inp_custom)
    out_r = ref_fn(*inp_ref)
    outs_c = (out_c,) if isinstance(out_c, Tensor) else tuple(out_c)
    outs_r = (out_r,) if isinstance(out_r, Tensor) else tuple(out_r)

    fwd_err = max(float((oc.detach() - or_.detach()).abs().max().item()) for oc, or_ in zip(outs_c, outs_r))

    # Project outputs with deterministic non-uniform weights so all components & asymmetries are probed
    gen = torch.Generator().manual_seed(999)
    weights = [
        torch.randn(oc.shape, dtype=torch.float64, generator=gen).to(oc.device)
        for oc in outs_c
    ]
    loss_c = sum((oc * w).sum() for oc, w in zip(outs_c, weights))
    loss_r = sum((or_ * w).sum() for or_, w in zip(outs_r, weights))
    assert isinstance(loss_c, Tensor) and isinstance(loss_r, Tensor)
    loss_c.backward()
    loss_r.backward()

    grad_err = 0.0
    for tc, tr, req in zip(inp_custom, inp_ref, grad_flags):
        if req:
            assert tc.grad is not None and tr.grad is not None
            err = float((tc.grad - tr.grad).abs().max().item())
            grad_err = max(grad_err, err)

    num_ok = True
    if run_numerical:
        gc_inputs = [
            t.detach().clone().requires_grad_(req)
            for t, req in zip(inputs, grad_flags)
        ]
        num_ok = bool(
            gradcheck(
                custom_fn,
                tuple(gc_inputs),
                eps=1e-6,
                atol=1e-5,
                rtol=1e-4,
                fast_mode=False,
            )
        )

    passed = (fwd_err < 1e-10) and (grad_err < 1e-8) and num_ok
    return GradCheckRecord(
        name=name,
        passed=passed,
        forward_max_err=fwd_err,
        autograd_grad_max_err=grad_err,
        numerical_gradcheck=num_ok,
        note=note,
    )


def run_full_gradcheck_suite() -> List[GradCheckRecord]:
    """Run triple gradient verification across all base operators, memberships, and 13 methods."""
    data = _well_separated_test_tensors()
    adj = data["adj"]
    edge_index = data["edge_index"]
    edge_weight = data["edge_weight"]
    s_fuzzy = data["s_fuzzy"]
    s_simplex = data["s_simplex"]
    z = data["z"]
    h = data["h"]
    x = data["x"]
    n, k = s_fuzzy.shape

    records: List[GradCheckRecord] = []

    # 0. Base GraphSAGE linear operator (Dense & True Sparse)
    sage = GraphSAGEOperator(x.shape[1], h.shape[1], activation="identity").double()
    a_hat = symmetric_normalize_adjacency(adj + torch.eye(n, dtype=torch.float64))
    records.append(
        _compare_custom_vs_ref(
            name="Base: GraphSAGE Linear Operator",
            custom_fn=lambda ah, xin, wm, ws, b: _GraphSAGELinearFn.apply(ah, xin, wm, ws, b),
            ref_fn=lambda ah, xin, wm, ws, b: (ah @ xin) @ wm + xin @ ws + b.unsqueeze(0),
            inputs=[a_hat, x, sage.w_msg.data, sage.w_self.data, sage.bias.data],
            grad_flags=[True, True, True, True, True],
        )
    )

    ei_hat, ew_hat = sparse_symmetric_normalize_with_self_loops(edge_index, edge_weight, num_nodes=n)
    records.append(
        _compare_custom_vs_ref(
            name="Base: Sparse GraphSAGE Linear Operator",
            custom_fn=lambda ewh, xin, wm, ws, b: _SparseGraphSAGELinearFn.apply(ei_hat, ewh, xin, wm, ws, b),
            ref_fn=lambda ewh, xin, wm, ws, b: (
                torch.zeros((n, xin.shape[1]), dtype=xin.dtype, device=xin.device)
                .index_add(0, ei_hat[0], ewh.unsqueeze(1) * xin[ei_hat[1]])
                @ wm
                + xin @ ws
                + b.unsqueeze(0)
            ),
            inputs=[ew_hat, x, sage.w_msg.data, sage.w_self.data, sage.bias.data],
            grad_flags=[True, True, True, True, True],
        )
    )

    # 1. Memberships
    m_soft = TemperatureSoftmaxMembership(init_tau=0.8, learnable_tau=True).double()
    records.append(
        _compare_custom_vs_ref(
            name="Membership 1: Temperature Softmax",
            custom_fn=lambda zin, tt: _TemperatureSoftmaxFn.apply(zin, tt),
            ref_fn=lambda zin, tt: torch.softmax(zin / torch.nn.functional.softplus(tt), dim=-1),
            inputs=[z, m_soft.tau_tilde.data],
            grad_flags=[True, True],
        )
    )

    m_sig = AnisotropicSigmoidMembership(num_clusters=k, init_gamma=1.3, init_beta=0.1).double()
    records.append(
        _compare_custom_vs_ref(
            name="Membership 2: Anisotropic Logistic-Sigmoid",
            custom_fn=lambda zin, gt, b: _AnisotropicSigmoidFn.apply(zin, gt, b),
            ref_fn=lambda zin, gt, b: torch.sigmoid(
                torch.nn.functional.softplus(gt).unsqueeze(0) * (zin - b.unsqueeze(0))
            ),
            inputs=[z, m_sig.gamma_tilde.data, m_sig.beta.data],
            grad_flags=[True, True, True],
        )
    )

    m_rbf = GaussianRBFMembership(num_clusters=k, latent_dim=k, init_sigma=0.9).double()
    records.append(
        _compare_custom_vs_ref(
            name="Membership 3: Gaussian Prototype (RBF)",
            custom_fn=lambda zin, mu, st: _GaussianRBFFn.apply(zin, mu, st, 1e-12),
            ref_fn=lambda zin, mu, st: torch.exp(
                -0.5
                * (
                    ((zin.unsqueeze(1) - mu.unsqueeze(0)) ** 2)
                    / (torch.nn.functional.softplus(st) ** 2 + 1e-12).unsqueeze(0)
                ).sum(dim=-1)
            ),
            inputs=[z, m_rbf.mu.data, m_rbf.sigma_tilde.data],
            grad_flags=[True, True, True],
        )
    )

    # 2. Section I: Classical methods
    dp = DiffPoolReducer(eps=1e-12)
    records.append(
        _compare_custom_vs_ref(
            name="Method 1: DiffPool (X_new, A_new, L_LP, L_ent)",
            custom_fn=lambda a, hin, sin: _DiffPoolFn.apply(a, hin, sin, 1e-12),
            ref_fn=lambda a, hin, sin: (
                dp.forward_reference(a, hin, sin).x_new,
                dp.forward_reference(a, hin, sin).adj_new,
                dp.forward_reference(a, hin, sin).loss_components["l_lp"],
                dp.forward_reference(a, hin, sin).loss_components["l_ent"],
            ),
            inputs=[adj, h, s_simplex],
            grad_flags=[True, True, True],
        )
    )

    mc = MinCutPoolReducer(prenormalized=True, eps=1e-12)
    a_tilde = symmetric_normalize_adjacency(adj, eps=1e-12)
    records.append(
        _compare_custom_vs_ref(
            name="Method 2: MinCutPool (X_new, A_new, L_cut, L_orth)",
            custom_fn=lambda at, hin, sin: _MinCutPoolFn.apply(at, hin, sin, 1e-12),
            ref_fn=lambda at, hin, sin: (
                mc.forward_reference(at, hin, sin).x_new,
                mc.forward_reference(at, hin, sin).adj_new,
                mc.forward_reference(at, hin, sin).loss_components["l_cut"],
                mc.forward_reference(at, hin, sin).loss_components["l_orth"],
            ),
            inputs=[a_tilde, h, s_simplex],
            grad_flags=[True, True, True],
        )
    )

    def _dense_from_ew(ew_in: Tensor) -> Tensor:
        return torch.zeros((n, n), dtype=ew_in.dtype, device=ew_in.device).index_put(
            (edge_index[0], edge_index[1]), ew_in
        )

    dm = DMoNReducer(eps=1e-12)
    records.append(
        _compare_custom_vs_ref(
            name="Method 3: DMoN (X_new, A_new, L_mod, L_collapse)",
            custom_fn=lambda a, hin, sin: _DMoNFn.apply(
                a, symmetric_normalize_adjacency(a, eps=1e-12), hin, sin, 1e-12
            ),
            ref_fn=lambda a, hin, sin: (
                dm.forward_reference(a, hin, sin).x_new,
                dm.forward_reference(a, hin, sin).adj_new,
                dm.forward_reference(a, hin, sin).loss_components["l_mod"],
                dm.forward_reference(a, hin, sin).loss_components["l_collapse"],
            ),
            inputs=[adj, h, s_simplex],
            grad_flags=[True, True, True],
        )
    )

    records.append(
        _compare_custom_vs_ref(
            name="Method 1-Sparse: DiffPool True Sparse Frobenius Identity",
            custom_fn=lambda ewin, hin, sin: _DiffPoolSparseFn.apply(edge_index, ewin, hin, sin, 1e-12),
            ref_fn=lambda ewin, hin, sin: (
                dp.forward_reference(_dense_from_ew(ewin), hin, sin).x_new,
                dp.forward_reference(_dense_from_ew(ewin), hin, sin).adj_new,
                dp.forward_reference(_dense_from_ew(ewin), hin, sin).loss_components["l_lp"],
                dp.forward_reference(_dense_from_ew(ewin), hin, sin).loss_components["l_ent"],
            ),
            inputs=[edge_weight, h, s_simplex],
            grad_flags=[True, True, True],
        )
    )

    ew_tilde_dir = sparse_symmetric_normalize_adjacency(edge_index, edge_weight, num_nodes=n, eps=1e-12)
    records.append(
        _compare_custom_vs_ref(
            name="Method 2-Sparse: MinCutPool True Sparse Operator",
            custom_fn=lambda ewt, hin, sin: _MinCutPoolSparseFn.apply(edge_index, ewt, hin, sin, 1e-12),
            ref_fn=lambda ewt, hin, sin: (
                mc.forward_reference(_dense_from_ew(ewt), hin, sin).x_new,
                mc.forward_reference(_dense_from_ew(ewt), hin, sin).adj_new,
                mc.forward_reference(_dense_from_ew(ewt), hin, sin).loss_components["l_cut"],
                mc.forward_reference(_dense_from_ew(ewt), hin, sin).loss_components["l_orth"],
            ),
            inputs=[ew_tilde_dir, h, s_simplex],
            grad_flags=[True, True, True],
        )
    )

    records.append(
        _compare_custom_vs_ref(
            name="Method 3-Sparse: DMoN True Sparse Chung-Lu Operator",
            custom_fn=lambda ewin, hin, sin: _DMoNSparseFn.apply(
                edge_index,
                ewin,
                sparse_symmetric_normalize_adjacency(edge_index, ewin, num_nodes=n, eps=1e-12),
                hin,
                sin,
                1e-12,
            ),
            ref_fn=lambda ewin, hin, sin: (
                dm.forward_reference(_dense_from_ew(ewin), hin, sin).x_new,
                dm.forward_reference(_dense_from_ew(ewin), hin, sin).adj_new,
                dm.forward_reference(_dense_from_ew(ewin), hin, sin).loss_components["l_mod"],
                dm.forward_reference(_dense_from_ew(ewin), hin, sin).loss_components["l_collapse"],
            ),
            inputs=[edge_weight, h, s_simplex],
            grad_flags=[True, True, True],
        )
    )

    # 3. Section II: Relational fuzzy methods (4..8)
    zadeh_mod = ZadehTopologyReducer()
    records.append(
        _compare_custom_vs_ref(
            name="Method 4: Zadeh-Godel (max-min)",
            custom_fn=lambda sin, ew: _ZadehMaxMinSparseFn.apply(sin, edge_index, ew),
            ref_fn=lambda sin, ew: zadeh_mod.forward_dense_reference(
                torch.zeros((n, n), dtype=ew.dtype).index_put((edge_index[0], edge_index[1]), ew),
                sin,
            ),
            inputs=[s_fuzzy, edge_weight],
            grad_flags=[True, True],
        )
    )

    goguen_mod = GoguenTopologyReducer()
    records.append(
        _compare_custom_vs_ref(
            name="Method 5: Goguen (max-product)",
            custom_fn=lambda sin, ew: _GoguenMaxProdSparseFn.apply(sin, edge_index, ew),
            ref_fn=lambda sin, ew: goguen_mod.forward_dense_reference(
                torch.zeros((n, n), dtype=ew.dtype).index_put((edge_index[0], edge_index[1]), ew),
                sin,
            ),
            inputs=[s_fuzzy, edge_weight],
            grad_flags=[True, True],
        )
    )

    # Ensure Lukasiewicz threshold eta_IJ has no near-zero or near-one margin kinks
    luk_mod = LukasiewiczTopologyReducer(
        num_clusters=k, learnable_eta=True, init_eta=1.55, clamp_unit=True, smooth_tau=0.0
    ).double()
    assert luk_mod.eta_tilde is not None
    with torch.no_grad():
        raw_unclamped = (
            s_fuzzy[edge_index[0]].unsqueeze(2)
            + edge_weight[:, None, None]
            + s_fuzzy[edge_index[1]].unsqueeze(1)
        ).amax(dim=0)
        for _ in range(6):
            eta_cur = 1.0 + torch.sigmoid(luk_mod.eta_tilde)
            diff_cur = raw_unclamped - eta_cur
            near_zero = diff_cur.abs() < 0.08
            near_one = (diff_cur - 1.0).abs() < 0.08
            if not (near_zero.any() or near_one.any()):
                break
            luk_mod.eta_tilde[near_zero] -= 0.75
            luk_mod.eta_tilde[near_one] += 0.95

    def _luk_ref_fn(sin: Tensor, ew: Tensor, et: Tensor) -> Tensor:
        eta = 1.0 + torch.sigmoid(et)
        src, dst = edge_index[0], edge_index[1]
        val = (
            sin[src].unsqueeze(2) + ew[:, None, None] + sin[dst].unsqueeze(1) - eta.unsqueeze(0)
        ).clamp(min=0.0, max=1.0)
        return val.amax(dim=0)

    records.append(
        _compare_custom_vs_ref(
            name="Method 6: Lukasiewicz (max-T_L, [0,1]-clamped, learnable eta_IJ)",
            custom_fn=lambda sin, ew, et: _LukasiewiczSparseFn.apply(sin, edge_index, ew, et, 2.0, True, 0.0),
            ref_fn=_luk_ref_fn,
            inputs=[s_fuzzy, edge_weight, luk_mod.eta_tilde.data],
            grad_flags=[True, True, True],
        )
    )

    def _luk_smooth_ref_fn(sin: Tensor, ew: Tensor, et: Tensor) -> Tensor:
        eta = 1.0 + torch.sigmoid(et)
        src, dst = edge_index[0], edge_index[1]
        diff = sin[src].unsqueeze(2) + ew[:, None, None] + sin[dst].unsqueeze(1) - eta.unsqueeze(0)
        val, _ = lukasiewicz_activation(diff, clamp_unit=True, smooth_tau=0.15)
        return val.amax(dim=0)

    records.append(
        _compare_custom_vs_ref(
            name="Method 6-Smooth: Lukasiewicz Softplus Window (tau=0.15)",
            custom_fn=lambda sin, ew, et: _LukasiewiczSparseFn.apply(sin, edge_index, ew, et, 2.0, True, 0.15),
            ref_fn=_luk_smooth_ref_fn,
            inputs=[s_fuzzy, edge_weight, luk_mod.eta_tilde.data],
            grad_flags=[True, True, True],
        )
    )

    from labflow.fuzzy_relational.hamacher import hamacher_ternary_closed_form

    ham_mod = HamacherTopologyReducer(init_gamma=1.4, learnable_gamma=True, matrix_gamma=False).double()
    records.append(
        _compare_custom_vs_ref(
            name="Method 7A: Hamacher-Einstein (scalar gamma)",
            custom_fn=lambda sin, ew, gt: _HamacherScalarSparseFn.apply(sin, edge_index, ew, gt, 1e-12),
            ref_fn=lambda sin, ew, gt: hamacher_ternary_closed_form(
                sin[edge_index[0]].unsqueeze(2),
                ew[:, None, None],
                sin[edge_index[1]].unsqueeze(1),
                torch.nn.functional.softplus(gt),
                eps=1e-12,
            ).amax(dim=0),
            inputs=[s_fuzzy, edge_weight, ham_mod.gamma_tilde.data],
            grad_flags=[True, True, True],
        )
    )

    ham_mat = HamacherTopologyReducer(
        init_gamma=1.7, learnable_gamma=True, matrix_gamma=True, num_clusters=k
    ).double()
    records.append(
        _compare_custom_vs_ref(
            name="Method 7B: Hamacher (symmetric matrix Gamma_IJ)",
            custom_fn=lambda sin, ew, gt: _HamacherMatrixFn.apply(sin, edge_index, ew, gt, 1e-12),
            ref_fn=lambda sin, ew, gt: hamacher_ternary_closed_form(
                sin[edge_index[0]].unsqueeze(2),
                ew[:, None, None],
                sin[edge_index[1]].unsqueeze(1),
                torch.nn.functional.softplus(0.5 * (gt + gt.t())).unsqueeze(0),
                eps=1e-12,
            ).amax(dim=0),
            inputs=[s_fuzzy, edge_weight, ham_mat.gamma_tilde.data],
            grad_flags=[True, True, True],
        )
    )

    ss_mod = SchweizerSklarTopologyReducer(init_q=1.25, learnable_q=True).double()
    records.append(
        _compare_custom_vs_ref(
            name="Method 8A: Schweizer-Sklar (p = -q)",
            custom_fn=lambda sin, ew, qt: _SchweizerSklarSparseFn.apply(sin, edge_index, ew, qt, 1e-12),
            ref_fn=lambda sin, ew, qt: (
                lambda q: (
                    sin[edge_index[0]].pow(-q).unsqueeze(2)
                    + ew.pow(-q)[:, None, None]
                    + sin[edge_index[1]].pow(-q).unsqueeze(1)
                    - 2.0
                )
                .pow(-1.0 / q)
                .amax(dim=0)
            )(torch.nn.functional.softplus(qt)),
            inputs=[s_fuzzy, edge_weight, ss_mod.q_tilde.data],
            grad_flags=[True, True, True],
        )
    )

    dombi_mod = DombiTopologyReducer(init_lambda=1.35, learnable_lambda=True).double()
    records.append(
        _compare_custom_vs_ref(
            name="Method 8B: Dombi (learnable lambda)",
            custom_fn=lambda sin, ew, lt: _DombiSparseFn.apply(sin, edge_index, ew, lt, 1e-12),
            ref_fn=lambda sin, ew, lt: (
                lambda lam: (
                    1.0
                    / (
                        1.0
                        + (
                            (((1.0 - sin) / (sin + 1e-12))[edge_index[0]].pow(lam).unsqueeze(2))
                            + (((1.0 - ew) / (ew + 1e-12)).pow(lam)[:, None, None])
                            + (((1.0 - sin) / (sin + 1e-12))[edge_index[1]].pow(lam).unsqueeze(1))
                        ).pow(1.0 / lam)
                    )
                ).amax(dim=0)
            )(torch.nn.functional.softplus(lt)),
            inputs=[s_fuzzy, edge_weight, dombi_mod.lam_tilde.data],
            grad_flags=[True, True, True],
        )
    )

    # 4. Section III: Smooth & Quantifier Fuzzy Methods (9..11)
    boltz_g = BoltzmannTopologyReducer(init_tau_c=0.3, learnable_tau_c=True, mode="global_edge").double()
    records.append(
        _compare_custom_vs_ref(
            name="Method 9A: Boltzmann Smooth FuzzyPool (Global Edge)",
            custom_fn=lambda sin, ew, tt: _BoltzmannGlobalEdgeFn.apply(sin, edge_index, ew, tt, False),
            ref_fn=lambda sin, ew, tt: (
                lambda tc: (
                    lambda om: (torch.softmax(om / tc, dim=0) * om).sum(dim=0)
                )(sin[edge_index[0]].unsqueeze(2) * ew[:, None, None] * sin[edge_index[1]].unsqueeze(1))
            )(torch.nn.functional.softplus(tt)),
            inputs=[s_fuzzy, edge_weight, boltz_g.tau_c_tilde.data],
            grad_flags=[True, True, True],
        )
    )

    c_m_adapt = _boltzmann_log_scale_factor(int(edge_weight.numel()), k, True)
    records.append(
        _compare_custom_vs_ref(
            name="Method 9A-Adaptive: Boltzmann Log-M Scaled FuzzyPool",
            custom_fn=lambda sin, ew, tt: _BoltzmannGlobalEdgeFn.apply(sin, edge_index, ew, tt, True),
            ref_fn=lambda sin, ew, tt: (
                lambda tc: (
                    lambda om: (torch.softmax(om / tc, dim=0) * om).sum(dim=0)
                )(sin[edge_index[0]].unsqueeze(2) * ew[:, None, None] * sin[edge_index[1]].unsqueeze(1))
            )(torch.nn.functional.softplus(tt) / c_m_adapt),
            inputs=[s_fuzzy, edge_weight, boltz_g.tau_c_tilde.data],
            grad_flags=[True, True, True],
        )
    )

    boltz_2 = BoltzmannTopologyReducer(init_tau_c=0.3, learnable_tau_c=True, mode="two_step").double()

    def _boltz_2_ref(sin: Tensor, ew: Tensor, tt: Tensor) -> Tensor:
        tc = torch.nn.functional.softplus(tt)
        src, dst = edge_index[0], edge_index[1]
        t_rows = []
        for i in range(n):
            mask = src == i
            if not mask.any():
                t_rows.append(torch.zeros((k,), dtype=sin.dtype, device=sin.device))
            else:
                u_i = ew[mask].unsqueeze(1) * sin[dst[mask]]
                t_rows.append((torch.softmax(u_i / tc, dim=0) * u_i).sum(dim=0))
        t_ij = torch.stack(t_rows, dim=0)
        v = sin.unsqueeze(2) * t_ij.unsqueeze(1)
        return (torch.softmax(v / tc, dim=0) * v).sum(dim=0)

    records.append(
        _compare_custom_vs_ref(
            name="Method 9B: Boltzmann Smooth FuzzyPool (Two-Step)",
            custom_fn=lambda sin, ew, tt: _BoltzmannTwoStepFn.apply(sin, edge_index, ew, tt, False),
            ref_fn=_boltz_2_ref,
            inputs=[s_fuzzy, edge_weight, boltz_2.tau_c_tilde.data],
            grad_flags=[True, True, True],
        )
    )

    ps_mod = ProbSumTopologyReducer(eps=1e-12, normalization="none")
    records.append(
        _compare_custom_vs_ref(
            name="Method 10: Probabilistic Sum FuzzyPool",
            custom_fn=lambda sin, ew: _ProbSumFn.apply(sin, edge_index, ew, 1e-12, "none"),
            ref_fn=lambda sin, ew: ps_mod.forward_reference(sin, edge_index, ew),
            inputs=[s_fuzzy, edge_weight],
            grad_flags=[True, True],
        )
    )

    ps_norm_mod = ProbSumTopologyReducer(eps=1e-12, normalization="mean_field")
    records.append(
        _compare_custom_vs_ref(
            name="Method 10-Norm: Mean-Field Scale-Normalized ProbSum",
            custom_fn=lambda sin, ew: _ProbSumFn.apply(sin, edge_index, ew, 1e-12, "mean_field"),
            ref_fn=lambda sin, ew: ps_norm_mod.forward_reference(sin, edge_index, ew),
            inputs=[s_fuzzy, edge_weight],
            grad_flags=[True, True],
        )
    )

    # For Bandler-Kohout universal quantifier over (i, j) in {1..N}^2, give non-edges distinct tiny weights
    # so complement A_bar has unique entries (avoiding max ties across multiple zero non-edges in gradcheck)
    adj_dir = data["adj_dir"]
    gen_bk = torch.Generator().manual_seed(31415)
    perm_bk = torch.randperm(n * n, generator=gen_bk)
    tiny_grid = torch.linspace(0.01, 0.11, n * n, dtype=torch.float64)[perm_bk].view(n, n)
    adj_bk = torch.where(adj_dir > 0.0, adj_dir, tiny_grid)
    adj_bk.fill_diagonal_(1.0)

    bk_reich = BandlerKohoutTopologyReducer(implication="reichenbach", ensure_self_loops=False)
    records.append(
        _compare_custom_vs_ref(
            name="Method 11C: Bandler-Kohout (Reichenbach)",
            custom_fn=lambda a_in, sin: _BandlerKohoutReichenbachFn.apply(a_in, sin),
            ref_fn=lambda a_in, sin: bk_reich.forward_direct_implication_reference(a_in, sin),
            inputs=[adj_bk, s_fuzzy],
            grad_flags=[True, True],
        )
    )

    bk_luk = BandlerKohoutTopologyReducer(implication="lukasiewicz", ensure_self_loops=False)
    records.append(
        _compare_custom_vs_ref(
            name="Method 11A: Bandler-Kohout (Lukasiewicz)",
            custom_fn=lambda a_in, sin: _BandlerKohoutLukasiewiczFn.apply(a_in, sin),
            ref_fn=lambda a_in, sin: bk_luk.forward_direct_implication_reference(a_in, sin),
            inputs=[adj_bk, s_fuzzy],
            grad_flags=[True, True],
        )
    )

    bk_kd = BandlerKohoutTopologyReducer(implication="kleene_dienes", ensure_self_loops=False)
    records.append(
        _compare_custom_vs_ref(
            name="Method 11B: Bandler-Kohout (Kleene-Dienes)",
            custom_fn=lambda a_in, sin: _BandlerKohoutKleeneDienesFn.apply(a_in, sin),
            ref_fn=lambda a_in, sin: bk_kd.forward_direct_implication_reference(a_in, sin),
            inputs=[adj_bk, s_fuzzy],
            grad_flags=[True, True],
        )
    )

    # Matrix-Free Sparse MEX Bandler-Kohout gradcheck with genuine zero non-edges (disjoint endpoints -> no ties)
    mask_bk_sp = torch.ones((n, n), dtype=torch.bool)
    mask_bk_sp.fill_diagonal_(False)
    mask_bk_sp[0, 4] = False
    mask_bk_sp[1, 5] = False
    mask_bk_sp[2, 3] = False
    adj_bk_sp_base = torch.where(mask_bk_sp, (adj_bk + 0.0031).clamp(max=0.96), torch.zeros_like(adj_bk))
    ei_bk_sp, ew_bk_sp = dense_to_sparse(adj_bk_sp_base, threshold=0.0)
    s_bk_sp = (s_fuzzy * 0.72 + 0.24).clone()

    def _dense_bk_sp(ew_in: Tensor) -> Tensor:
        return torch.zeros((n, n), dtype=ew_in.dtype, device=ew_in.device).index_put(
            (ei_bk_sp[0], ei_bk_sp[1]), ew_in
        )

    for impl_code, impl_name in (
        ("11A-Sparse", "lukasiewicz"),
        ("11B-Sparse", "kleene_dienes"),
        ("11C-Sparse", "reichenbach"),
    ):
        bk_sp_mod = BandlerKohoutTopologyReducer(implication=impl_name, ensure_self_loops=True)
        records.append(
            _compare_custom_vs_ref(
                name=f"Method {impl_code}: Matrix-Free Sparse MEX Bandler-Kohout ({impl_name})",
                custom_fn=lambda sin, ewin, _im=impl_name: _BandlerKohoutSparseFn.apply(
                    sin, ei_bk_sp, ewin, True, _im
                ),
                ref_fn=lambda sin, ewin, _m=bk_sp_mod: _m.forward_direct_implication_reference(
                    _dense_bk_sp(ewin), sin
                ),
                inputs=[s_bk_sp, ew_bk_sp],
                grad_flags=[True, True],
            )
        )

    # 5. Section IV: Fuzzy Feature Space Reduction Methods (12, 13A, 13B)
    cog_mod = COGFeaturePooling(init_m=2.1, learnable_m=True).double()
    records.append(
        _compare_custom_vs_ref(
            name="Method 12: Power Centroid Defuzzification (COG / FCM)",
            custom_fn=lambda hin, sin, mt: _COGFeaturePoolFn.apply(hin, sin, mt, 1e-12),
            ref_fn=lambda hin, sin, mt: (
                lambda m: (sin.pow(m).t() @ hin) / (sin.pow(m).sum(dim=0, keepdim=True).t() + 1e-12)
            )(1.0 + torch.nn.functional.softplus(mt)),
            inputs=[h, s_fuzzy, cog_mod.m_tilde.data],
            grad_flags=[True, True, True],
        )
    )

    bip_mod = BipolarSupremalFeaturePooling()
    records.append(
        _compare_custom_vs_ref(
            name="Method 13A: Bipolar Supremal Feature Pooling",
            custom_fn=lambda hin, sin: _BipolarSupremalFeaturePoolFn.apply(hin, sin),
            ref_fn=lambda hin, sin: bip_mod.forward_reference(hin, sin),
            inputs=[h, s_fuzzy],
            grad_flags=[True, True],
        )
    )

    choq_mod = ChoquetFeaturePooling(num_clusters=k, init_alpha=1.35, learnable_alpha=True).double()
    with torch.no_grad():
        choq_mod.alpha_tilde.copy_(torch.tensor([0.2, 0.7, -0.3], dtype=torch.float64))

    def _choq_ref(hin: Tensor, sin: Tensor, at: Tensor) -> Tensor:
        alpha = torch.nn.functional.softplus(at)
        hs, perm = torch.sort(hin, dim=0, stable=True)
        perm_exp = perm.unsqueeze(1).expand(n, k, hin.shape[1])
        ss = torch.gather(sin.unsqueeze(2).expand(n, k, hin.shape[1]), dim=0, index=perm_exp)
        c_tail = torch.flip(torch.cumsum(torch.flip(ss, dims=[0]), dim=0), dims=[0])
        ratio = c_tail[1:] / (c_tail[0].unsqueeze(0) + 1e-12)
        g_sub = ratio.pow(alpha.view(1, k, 1))
        return hs[0].unsqueeze(0) + ((hs[1:] - hs[:-1]).unsqueeze(1) * g_sub).sum(dim=0)

    records.append(
        _compare_custom_vs_ref(
            name="Method 13B: Discrete Fuzzy Choquet Integral Pooling",
            custom_fn=lambda hin, sin, at: _ChoquetFeaturePoolFn.apply(hin, sin, at, 1e-12),
            ref_fn=_choq_ref,
            inputs=[h, s_fuzzy, choq_mod.alpha_tilde.data],
            grad_flags=[True, True, True],
        )
    )

    # 6. End-to-End GraphReductionLayer parameter autograd equivalence (Dense & True Sparse GraphData)
    for spec in CANONICAL_METHOD_SPECS:
        torch.manual_seed(404)
        layer = GraphReductionLayer.from_spec(
            spec.key,
            in_features=x.shape[1],
            out_features=h.shape[1],
            num_clusters=k,
        ).double()
        adj_use = adj_bk_sp_base if spec.topology.startswith("bk_") else adj
        ei_use, ew_use = dense_to_sparse(adj_use, threshold=0.0)
        g_sparse_only = GraphData(x=x, edge_index=ei_use, edge_weight=ew_use)

        layer.zero_grad()
        out_c = layer((adj_use, x))
        loss_c = (out_c.x_new * 0.73).sum() + (out_c.adj_new * 1.19).sum() + out_c.aux_loss
        loss_c.backward()
        grads_c = {name: p.grad.detach().clone() for name, p in layer.named_parameters() if p.grad is not None}

        layer.zero_grad()
        out_s = layer(g_sparse_only)
        loss_s = (out_s.x_new * 0.73).sum() + (out_s.adj_new * 1.19).sum() + out_s.aux_loss
        loss_s.backward()
        grads_s = {name: p.grad.detach().clone() for name, p in layer.named_parameters() if p.grad is not None}

        layer.zero_grad()
        out_r = layer.forward_reference(adj_use, x)
        loss_r = (out_r.x_new * 0.73).sum() + (out_r.adj_new * 1.19).sum() + out_r.aux_loss
        loss_r.backward()
        grads_r = {name: p.grad.detach().clone() for name, p in layer.named_parameters() if p.grad is not None}

        fwd_err = max(
            float((out_c.x_new - out_r.x_new).abs().max().item()),
            float((out_c.adj_new - out_r.adj_new).abs().max().item()),
            float((out_c.aux_loss - out_r.aux_loss).abs().max().item()),
            float((out_s.x_new - out_r.x_new).abs().max().item()),
            float((out_s.adj_new - out_r.adj_new).abs().max().item()),
            float((out_s.aux_loss - out_r.aux_loss).abs().max().item()),
        )
        grad_err = max(
            max(
                float((grads_c[p_name] - grads_r[p_name]).abs().max().item()),
                float((grads_s[p_name] - grads_r[p_name]).abs().max().item()),
            )
            for p_name in grads_c
        )
        records.append(
            GradCheckRecord(
                name=f"End-to-End Layer [{spec.code}]: {spec.key}",
                passed=(fwd_err < 1e-10) and (grad_err < 1e-8),
                forward_max_err=fwd_err,
                autograd_grad_max_err=grad_err,
                numerical_gradcheck=True,
                note="Dense + True Sparse GraphData parameter chain rule verified against PyTorch reference",
            )
        )

    return records
