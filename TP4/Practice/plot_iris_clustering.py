"""
Iris WTA clustering visualization.
  Fig 3: Petal length vs petal width colored by cosine-WTA cluster assignment
          with prototype positions and true species as marker shapes
"""
import numpy as np
import matplotlib.pyplot as plt
import os

os.makedirs("figures", exist_ok=True)

ALPHA    = 0.5
N_UPDATE = 20
M        = 3

# ── load iris data ─────────────────────────────────────────────────────────────
data = []
with open("../Part1/iris.data") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        parts = line.split(",")
        data.append((list(map(float, parts[:4])), parts[4]))

X_raw     = np.array([d[0] for d in data])
species   = [d[1] for d in data]

# L2-normalise for cosine WTA
norms = np.linalg.norm(X_raw, axis=1, keepdims=True)
X_cos = X_raw / norms

# ── cosine WTA ─────────────────────────────────────────────────────────────────
rng = np.random.default_rng(42)
idx = rng.choice(150, M, replace=False)
W   = X_cos[idx].copy()

for _ in range(N_UPDATE):
    for p in range(150):
        m_star = np.argmax(W @ X_cos[p])
        W[m_star] += ALPHA * (X_cos[p] - W[m_star])
        W[m_star] /= np.linalg.norm(W[m_star])

labels = np.argmax(X_cos @ W.T, axis=1)

# ── Fig 3: petal length vs petal width ────────────────────────────────────────
CLUSTER_COLORS  = ["#e06c6c", "#6c9ae0", "#6cbf6c"]   # red, blue, green
SPECIES_MARKERS = {"Iris-setosa": "o", "Iris-versicolor": "s", "Iris-virginica": "^"}
SPECIES_LABELS  = {"Iris-setosa": "setosa", "Iris-versicolor": "versicolor",
                   "Iris-virginica": "virginica"}

# raw petal length/width (features 2,3) for axis; cosine-cluster colours
petal_len = X_raw[:, 2]
petal_wid = X_raw[:, 3]

fig, ax = plt.subplots(figsize=(7, 5))

for sp, mk in SPECIES_MARKERS.items():
    mask = np.array([s == sp for s in species])
    sc = ax.scatter(
        petal_len[mask], petal_wid[mask],
        c=[CLUSTER_COLORS[labels[i]] for i in np.where(mask)[0]],
        marker=mk, s=60, edgecolors="k", linewidths=0.5,
        label=SPECIES_LABELS[sp], zorder=3
    )

# prototype positions in raw feature space (de-normalise by multiplying back)
# prototypes W are unit-norm direction vectors; project to raw space:
# use the mean norm of each assigned cluster for scaling
for m in range(M):
    mask_m = labels == m
    if mask_m.sum() == 0:
        continue
    mean_raw = X_raw[mask_m].mean(axis=0)
    ax.scatter(mean_raw[2], mean_raw[3],
               s=220, marker="*", color=CLUSTER_COLORS[m],
               edgecolors="k", linewidths=1.2, zorder=5,
               label=f"Prototype {m}" if m == 0 else f"Prototype {m}")

ax.set_xlabel("Petal length (cm)")
ax.set_ylabel("Petal width (cm)")
ax.set_title("Cosine-similarity WTA on Iris\n(color = cluster, marker = true species)")

# combine legend
import matplotlib.lines as mlines
sp_handles = [mlines.Line2D([], [], color="gray", marker=mk, linestyle="None",
                             markersize=8, label=SPECIES_LABELS[sp])
              for sp, mk in SPECIES_MARKERS.items()]
cl_handles = [plt.scatter([], [], s=80, color=CLUSTER_COLORS[m],
                          edgecolors="k", label=f"Cluster {m}")
              for m in range(M)]
proto_h    = [plt.scatter([], [], s=200, marker="*", color="gray",
                          edgecolors="k", label="Cluster centroid (raw)")]
ax.legend(handles=sp_handles + cl_handles + proto_h,
          loc="upper left", fontsize=9, ncol=2)
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig("figures/fig3_iris_cosine.png", dpi=150)
print("Saved figures/fig3_iris_cosine.png")
plt.close(fig)
