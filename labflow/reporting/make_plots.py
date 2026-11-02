"""Generate dedicated domain-specific diagnostic plots for the separate benchmark suites."""

from pathlib import Path
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

reports_dir = Path("reports")
fig_dir = reports_dir / "figures"
fig_dir.mkdir(parents=True, exist_ok=True)

with open(reports_dir / "analytics_results.json", "r", encoding="utf-8") as f:
    data = json.load(f)

leaderboard = data["leaderboard"]
# Sort by method_code
methods = sorted(leaderboard, key=lambda x: x["method_code"])
codes = [m["method_code"] for m in methods]
keys = [m["method_key"] for m in methods]

fig, axes = plt.subplots(3, 2, figsize=(18, 14))
fig.suptitle("Labflow: Separate Benchmark Evaluation Suite Across 22 Graph Reduction Methods", fontsize=16, fontweight="bold")

x = np.arange(len(codes))
width = 0.38

# 1. Community Detection
ax = axes[0, 0]
nmis = [m["mean_nmi"] for m in methods]
mods = [m["mean_modularity_q"] for m in methods]
ax.bar(x - width/2, nmis, width, label="Mean NMI", color="#2563eb")
ax.bar(x + width/2, mods, width, label="Mean Modularity Q", color="#10b981")
ax.set_title("Test I: Community Detection & Modularity Alignment")
ax.set_xticks(x)
ax.set_xticklabels(codes, rotation=45, ha="right", fontsize=8)
ax.set_ylim(-0.05, 1.05)
ax.grid(axis="y", linestyle="--", alpha=0.3)
ax.legend(loc="upper left")

# 2. Structural Motif & Clique Isolation
ax = axes[0, 1]
clique_recs = {r["key"]: r for r in data["clique_benchmark"]}
c_dens = [clique_recs.get(k, {}).get("internal_clique_density", 0.0) for k in keys]
c_f1 = [clique_recs.get(k, {}).get("clique_recovery_f1", 0.0) for k in keys]
ax.bar(x - width/2, c_dens, width, label="Clique Density", color="#8b5cf6")
ax.bar(x + width/2, c_f1, width, label="Recovery F1", color="#ec4899")
ax.set_title("Test II: Structural Motif & Planted Clique Isolation (Bandler-Kohout Universal Quantifier)")
ax.set_xticks(x)
ax.set_xticklabels(codes, rotation=45, ha="right", fontsize=8)
ax.set_ylim(0.0, 1.08)
ax.grid(axis="y", linestyle="--", alpha=0.3)
ax.legend(loc="upper left")

# 3. Topological Fidelity
ax = axes[1, 0]
aucs = [m["mean_link_auc"] for m in methods]
dead = [m["dead_superedge_ratio"] * 100.0 for m in methods]
ax.bar(x - width/2, aucs, width, label="Link Recon ROC-AUC", color="#059669")
ax.bar(x + width/2, [d/100.0 for d in dead], width, label="Dead Super-Edge Ratio", color="#ef4444")
ax.set_title("Test III & V: Topological Reconstruction AUC vs Dead Super-Edges (%)")
ax.set_xticks(x)
ax.set_xticklabels(codes, rotation=45, ha="right", fontsize=8)
ax.set_ylim(0.0, 1.08)
ax.grid(axis="y", linestyle="--", alpha=0.3)
ax.legend(loc="upper left")

# 4. GNN Classification
ax = axes[1, 1]
cls_recs = {r["key"]: r for r in data["classification_benchmark"]}
test_accs = [cls_recs.get(k, {}).get("test_acc", 0.0) * 100.0 for k in keys]
train_accs = [cls_recs.get(k, {}).get("train_acc", 0.0) * 100.0 for k in keys]
ax.bar(x - width/2, train_accs, width, label="Train Accuracy (%)", color="#93c5fd")
ax.bar(x + width/2, test_accs, width, label="Test Accuracy (%)", color="#1d4ed8")
ax.set_title("Test IV: Hierarchical GNN Graph Classification Accuracy (%)")
ax.set_xticks(x)
ax.set_xticklabels(codes, rotation=45, ha="right", fontsize=8)
ax.set_ylim(0.0, 108.0)
ax.grid(axis="y", linestyle="--", alpha=0.3)
ax.legend(loc="lower right")

# 5. Gradient Health & Active Jacobian Density
ax = axes[2, 0]
prof_dict = {}
for p_list in data["dataset_profiles"].values():
    for p in p_list:
        prof_dict.setdefault(p["method_key"], []).append(p["topo_jacobian_density"] * 100.0)
jac_dens = [np.mean(prof_dict.get(k, [0.0])) for k in keys]
colors = ["#ef4444" if jd > 65.0 else "#f59e0b" if jd > 30.0 else "#3b82f6" for jd in jac_dens]
ax.bar(x, jac_dens, width=0.6, color=colors)
ax.set_title("Test V: Topological Jacobian Density (Active dA_new / dS %) - Dilution vs Concentration")
ax.set_xticks(x)
ax.set_xticklabels(codes, rotation=45, ha="right", fontsize=8)
ax.set_ylabel("Active Entries in Jacobian (%)")
ax.set_ylim(0.0, 108.0)
ax.grid(axis="y", linestyle="--", alpha=0.3)

# 6. Scaling & Runtime on Large Sparse Graphs
ax = axes[2, 1]
times = [m["ms_large_graph"] for m in methods]
alphas = [m["empirical_scaling_exp"] for m in methods]
ax.bar(x - width/2, times, width, label="Runtime (ms @ N=1200)", color="#f97316")
ax2 = ax.twinx()
ax2.plot(x, alphas, color="#7c3aed", marker="o", linewidth=2.0, label="Scaling Exponent alpha")
ax2.set_ylabel("Scaling Exponent alpha (T ~ N^alpha)", color="#7c3aed")
ax.set_title("Test VI: Asymptotic Computational Scaling & Large-N Runtime")
ax.set_xticks(x)
ax.set_xticklabels(codes, rotation=45, ha="right", fontsize=8)
ax.set_ylabel("Runtime (ms)", color="#f97316")
ax.grid(axis="y", linestyle="--", alpha=0.3)
lines_1, labels_1 = ax.get_legend_handles_labels()
lines_2, labels_2 = ax2.get_legend_handles_labels()
ax.legend(lines_1 + lines_2, labels_1 + labels_2, loc="upper left")

plt.tight_layout()
out_plot = fig_dir / "domain_specific_benchmarks.png"
fig.savefig(out_plot, dpi=160)
plt.close(fig)
print(f"Saved {out_plot}")
