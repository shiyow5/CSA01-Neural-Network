import matplotlib.pyplot as plt
import csv

p_vals, acc_vals = [], []
with open("capacity_results.txt") as f:
    reader = csv.DictReader(f)
    for row in reader:
        p_vals.append(int(row["P"]))
        acc_vals.append(float(row["avg_accuracy"]) * 100)

fig, ax = plt.subplots(figsize=(7, 4))

ax.plot(p_vals, acc_vals, "o-", color="steelblue", linewidth=2, markersize=7)
ax.axvline(x=16.56, color="tomato", linestyle="--", linewidth=1.5,
           label=r"Theoretical limit $P_{max} \approx 0.138N = 16.6$")
ax.axhline(y=50, color="gray", linestyle=":", linewidth=1, label="Chance level (50%)")

ax.set_xlabel("Number of stored patterns $P$")
ax.set_ylabel("Average recall accuracy (%)")
ax.set_title("Hopfield Network Capacity (N=120, noise=10%)")
ax.set_xticks(p_vals)
ax.set_ylim(40, 102)
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("capacity_curve.png", dpi=150)
print("Saved capacity_curve.png")
