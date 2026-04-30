"""
Base WTA experiment visualizations.
  Fig 1: Unit circle – initial vs final weight vectors and cluster assignments
  Fig 2: Weight component convergence over epochs
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os

os.makedirs("figures", exist_ok=True)

# ── data ──────────────────────────────────────────────────────────────────────
X = np.array([
    [0.8,    0.6   ],
    [0.1736,-0.9848],
    [0.707,  0.707 ],
    [0.342, -0.9397],
    [0.6,    0.8   ],
])

ALPHA      = 0.5
N_UPDATE   = 20
N_PATTERNS = len(X)

# ── WTA with history ──────────────────────────────────────────────────────────
rng = np.random.default_rng(42)
raw = rng.uniform(-0.5, 0.5, (2, 2))
W0  = raw / np.linalg.norm(raw, axis=1, keepdims=True)   # initial weights

W       = W0.copy()
history = [W0.copy()]       # shape: (epoch+1, M, I)

for _ in range(N_UPDATE):
    for p in range(N_PATTERNS):
        scores = W @ X[p]
        m_star = np.argmax(scores)
        W[m_star] += ALPHA * (X[p] - W[m_star])
        W[m_star] /= np.linalg.norm(W[m_star])
    history.append(W.copy())

history = np.array(history)   # (21, 2, 2)

# final cluster assignment
def assign(W, X):
    return np.argmax(X @ W.T, axis=1)

labels_init  = assign(history[0],  X)
labels_final = assign(history[-1], X)

# ── Fig 1: unit circle ────────────────────────────────────────────────────────
CLUSTER_COLORS = ["#e06c6c", "#6c9ae0"]   # red, blue
MARKER_LABELS  = [f"P{i}" for i in range(N_PATTERNS)]

fig, ax = plt.subplots(figsize=(6, 6))

theta = np.linspace(0, 2 * np.pi, 300)
ax.plot(np.cos(theta), np.sin(theta), "lightgray", lw=1, zorder=0)
ax.axhline(0, color="lightgray", lw=0.8, zorder=0)
ax.axvline(0, color="lightgray", lw=0.8, zorder=0)

# data points (colored by FINAL cluster)
for idx, (x, y) in enumerate(X):
    c = CLUSTER_COLORS[labels_final[idx]]
    ax.scatter(x, y, s=100, color=c, zorder=3, edgecolors="k", linewidths=0.8)
    ax.annotate(MARKER_LABELS[idx], (x, y),
                textcoords="offset points", xytext=(8, 4), fontsize=10)

# initial weight vectors (dashed)
for m, (wx, wy) in enumerate(history[0]):
    ax.annotate("", xy=(wx, wy), xytext=(0, 0),
                arrowprops=dict(arrowstyle="->", color=CLUSTER_COLORS[m],
                                lw=1.8, linestyle="dashed"))
    ax.scatter(wx, wy, s=80, color=CLUSTER_COLORS[m], marker="D",
               edgecolors="k", linewidths=0.8, zorder=4, alpha=0.5)

# final weight vectors (solid)
for m, (wx, wy) in enumerate(history[-1]):
    ax.annotate("", xy=(wx, wy), xytext=(0, 0),
                arrowprops=dict(arrowstyle="->", color=CLUSTER_COLORS[m], lw=2.5))
    ax.scatter(wx, wy, s=120, color=CLUSTER_COLORS[m], marker="*",
               edgecolors="k", linewidths=0.8, zorder=5)

# legend
legend_elems = [
    mpatches.Patch(color=CLUSTER_COLORS[0], label="Class 0"),
    mpatches.Patch(color=CLUSTER_COLORS[1], label="Class 1"),
    plt.Line2D([0], [0], color="gray", lw=1.8, linestyle="--", label="Initial weight"),
    plt.Line2D([0], [0], color="gray", lw=2.5,                  label="Final weight"),
    plt.scatter([], [], s=100, color="gray", edgecolors="k", label="Data point"),
    plt.scatter([], [], s=120, marker="*", color="gray", edgecolors="k", label="Final prototype"),
]
ax.legend(handles=legend_elems, loc="lower right", fontsize=9)

ax.set_xlim(-1.25, 1.25)
ax.set_ylim(-1.25, 1.25)
ax.set_aspect("equal")
ax.set_xlabel("$w_1$")
ax.set_ylabel("$w_2$")
ax.set_title("Base WTA: unit circle – initial vs final prototypes")
fig.tight_layout()
fig.savefig("figures/fig1_base_circle.png", dpi=150)
print("Saved figures/fig1_base_circle.png")
plt.close(fig)

# ── Fig 2: convergence of weight components ───────────────────────────────────
epochs = np.arange(len(history))   # 0..20

fig, ax = plt.subplots(figsize=(7, 4))

styles = [
    ("C0", "-",  r"$w_0^{(1)}$"),
    ("C0", "--", r"$w_0^{(2)}$"),
    ("C1", "-",  r"$w_1^{(1)}$"),
    ("C1", "--", r"$w_1^{(2)}$"),
]
for m in range(2):
    for dim in range(2):
        label = styles[m * 2 + dim][2]
        color = styles[m * 2 + dim][0]
        ls    = styles[m * 2 + dim][1]
        ax.plot(epochs, history[:, m, dim], color=color, linestyle=ls,
                linewidth=1.8, marker="o", markersize=3, label=label)

# mark first epoch where max per-component change falls below 1e-3
diffs = np.max(np.abs(np.diff(history, axis=0)), axis=(1, 2))
stable_epoch = int(np.argmax(diffs < 1e-3))
ax.axvline(x=stable_epoch, color="gray", linestyle=":", lw=1.2,
           label=f"Stabilisation (epoch {stable_epoch})")

ax.set_xlabel("Epoch")
ax.set_ylabel("Weight value")
ax.set_title("Base WTA: convergence of weight components")
ax.set_xticks(epochs[::2])
ax.legend(fontsize=9, ncol=2)
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig("figures/fig2_base_convergence.png", dpi=150)
print("Saved figures/fig2_base_convergence.png")
plt.close(fig)
