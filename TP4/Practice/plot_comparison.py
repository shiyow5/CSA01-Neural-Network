"""
Cosine-similarity WTA vs Euclidean-distance WTA on Iris.
  Fig 4: Side-by-side scatter (petal length vs petal width)
          showing how each criterion assigns clusters
"""
import numpy as np
import matplotlib.pyplot as plt
import os

os.makedirs("figures", exist_ok=True)

ALPHA_COS = 0.5
ALPHA_EUC = 0.1
N_UPDATE_COS = 20
N_UPDATE_EUC = 50
M            = 3

# ── load iris ──────────────────────────────────────────────────────────────────
data = []
with open("../Part1/iris.data") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        parts = line.split(",")
        data.append((list(map(float, parts[:4])), parts[4]))

X_raw   = np.array([d[0] for d in data])   # (150, 4)  raw cm
species = np.array([d[1] for d in data])

norms = np.linalg.norm(X_raw, axis=1, keepdims=True)
X_cos = X_raw / norms                       # L2-normalised

# ── cosine WTA ─────────────────────────────────────────────────────────────────
rng  = np.random.default_rng(42)
idx  = rng.choice(150, M, replace=False)
W_c  = X_cos[idx].copy()

for _ in range(N_UPDATE_COS):
    for p in range(150):
        m_star = np.argmax(W_c @ X_cos[p])
        W_c[m_star] += ALPHA_COS * (X_cos[p] - W_c[m_star])
        W_c[m_star] /= np.linalg.norm(W_c[m_star])

labels_cos = np.argmax(X_cos @ W_c.T, axis=1)

# ── euclidean WTA ─────────────────────────────────────────────────────────────
rng2 = np.random.default_rng(42)
idx2 = rng2.choice(150, M, replace=False)
W_e  = X_raw[idx2].copy().astype(float)

for _ in range(N_UPDATE_EUC):
    for p in range(150):
        dists  = np.sum((W_e - X_raw[p]) ** 2, axis=1)
        m_star = np.argmin(dists)
        W_e[m_star] += ALPHA_EUC * (X_raw[p] - W_e[m_star])

labels_euc = np.array([np.argmin(np.sum((W_e - X_raw[p]) ** 2, axis=1))
                        for p in range(150)])

# ── remap cluster indices to best-match true species for readability ───────────
# setosa=0-49, versicolor=50-99, virginica=100-149
def best_remap(labels):
    """Permute cluster IDs so each species' majority group gets a fixed colour."""
    true = np.array([0]*50 + [1]*50 + [2]*50)
    from itertools import permutations
    best_acc, best_perm = -1, None
    for perm in permutations(range(M)):
        mapped = np.array([perm[l] for l in labels])
        acc = np.mean(mapped == true)
        if acc > best_acc:
            best_acc, best_perm = acc, perm
    return np.array([best_perm[l] for l in labels]), best_acc

labels_cos_r, acc_cos = best_remap(labels_cos)
labels_euc_r, acc_euc = best_remap(labels_euc)

# prototype positions for each method (mean of assigned raw samples)
def raw_prototypes(labels, X_raw):
    return np.array([X_raw[labels == m].mean(axis=0) if (labels == m).any()
                     else np.zeros(4) for m in range(M)])

proto_cos = raw_prototypes(labels_cos_r, X_raw)
proto_euc = raw_prototypes(labels_euc_r, X_raw)

# ── Fig 4: side-by-side comparison ────────────────────────────────────────────
CLUSTER_COLORS  = ["#e06c6c", "#6c9ae0", "#6cbf6c"]
SPECIES_MARKERS = [("Iris-setosa", "o"), ("Iris-versicolor", "s"),
                   ("Iris-virginica", "^")]
SPECIES_LABELS  = {"Iris-setosa": "setosa", "Iris-versicolor": "versicolor",
                   "Iris-virginica": "virginica"}
CLUSTER_LABELS  = ["setosa cluster", "versicolor cluster", "virginica cluster"]

petal_len = X_raw[:, 2]
petal_wid = X_raw[:, 3]

fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)

for ax, labels_r, proto, title, acc in [
    (axes[0], labels_cos_r, proto_cos,
     f"Cosine-similarity WTA\n(accuracy {acc_cos*100:.1f}%)", acc_cos),
    (axes[1], labels_euc_r, proto_euc,
     f"Euclidean-distance WTA\n(accuracy {acc_euc*100:.1f}%)", acc_euc),
]:
    for sp, mk in SPECIES_MARKERS:
        mask = species == sp
        ax.scatter(
            petal_len[mask], petal_wid[mask],
            c=[CLUSTER_COLORS[labels_r[i]] for i in np.where(mask)[0]],
            marker=mk, s=55, edgecolors="k", linewidths=0.5, zorder=3
        )

    # prototype centroids
    for m in range(M):
        if (labels_r == m).any():
            ax.scatter(proto[m, 2], proto[m, 3],
                       s=250, marker="*", color=CLUSTER_COLORS[m],
                       edgecolors="k", linewidths=1.2, zorder=5)

    ax.set_xlabel("Petal length (cm)")
    ax.set_title(title, fontsize=11)
    ax.grid(True, alpha=0.3)

axes[0].set_ylabel("Petal width (cm)")

# shared legend
import matplotlib.lines as mlines
sp_handles = [mlines.Line2D([], [], color="gray", marker=mk, linestyle="None",
                             markersize=8, label=SPECIES_LABELS[sp])
              for sp, mk in SPECIES_MARKERS]
cl_handles = [plt.scatter([], [], s=70, color=CLUSTER_COLORS[m],
                          edgecolors="k", label=CLUSTER_LABELS[m])
              for m in range(M)]
proto_h    = [plt.scatter([], [], s=200, marker="*", color="gray",
                          edgecolors="k", label="Cluster centroid")]
fig.legend(handles=sp_handles + cl_handles + proto_h,
           loc="lower center", ncol=4, fontsize=9,
           bbox_to_anchor=(0.5, -0.08))

fig.suptitle("Cosine vs Euclidean WTA: Iris clustering\n"
             "(color = cluster assignment, marker = true species)",
             fontsize=12)
fig.tight_layout()
fig.savefig("figures/fig4_comparison.png", dpi=150, bbox_inches="tight")
print("Saved figures/fig4_comparison.png")
plt.close(fig)

# ── confusion summary ──────────────────────────────────────────────────────────
true_names = ["setosa"]*50 + ["versicolor"]*50 + ["virginica"]*50
cluster_names = ["setosa cluster", "versicolor cluster", "virginica cluster"]
true_ids = np.array([0]*50 + [1]*50 + [2]*50)

print("\nCosine WTA confusion (rows=true, cols=cluster):")
for ti, tn in enumerate(["setosa", "versicolor", "virginica"]):
    row = [np.sum((true_ids == ti) & (labels_cos_r == ci)) for ci in range(M)]
    print(f"  {tn:12s}: {row}  → {max(row)}/50 correct")

print(f"\nEuclidean WTA confusion (rows=true, cols=cluster):")
for ti, tn in enumerate(["setosa", "versicolor", "virginica"]):
    row = [np.sum((true_ids == ti) & (labels_euc_r == ci)) for ci in range(M)]
    print(f"  {tn:12s}: {row}  → {max(row)}/50 correct")
