#!/usr/bin/env python3
"""Plot close centers experiment results."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import csv

def read_csv(filename):
    epochs, errors = [], []
    with open(filename) as f:
        reader = csv.DictReader(f)
        for row in reader:
            epochs.append(int(row['epoch']))
            errors.append(float(row['error']))
    return epochs, errors

dataset_names = ['Original (far)', 'Medium', 'Close', 'Very close']
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# --- Plots 1-2: Error curves ---
colors = ['#2ecc71', '#3498db', '#e67e22', '#e74c3c']

ax = axes[0][0]
for i in range(4):
    try:
        ep, err = read_csv(f'close_perceptron_{i}.csv')
        ax.plot(ep, err, color=colors[i], linewidth=1.2, label=dataset_names[i])
    except FileNotFoundError:
        pass
ax.set_title('Perceptron: Error by Class Distance')
ax.set_xlabel('Epoch')
ax.set_ylabel('Total Error')
ax.legend()
ax.grid(True, alpha=0.3)

ax = axes[0][1]
for i in range(4):
    try:
        ep, err = read_csv(f'close_delta_{i}.csv')
        ax.plot(ep, err, color=colors[i], linewidth=1.2, label=dataset_names[i])
    except FileNotFoundError:
        pass
ax.set_title('Delta: Error by Class Distance')
ax.set_xlabel('Epoch')
ax.set_ylabel('Total Error')
ax.legend()
ax.grid(True, alpha=0.3)

# --- Plot 3: Bar chart from summary ---
ax = axes[1][0]
summary = {}
with open('close_centers_summary.csv') as f:
    reader = csv.DictReader(f)
    for row in reader:
        ds = row['dataset']
        method = row['method']
        avg = float(row['avg_epochs'])
        if ds not in summary:
            summary[ds] = {}
        summary[ds][method] = avg if avg > 0 else 0

ds_keys = list(summary.keys())
p_avgs = [summary[k].get('perceptron', 0) for k in ds_keys]
d_avgs = [summary[k].get('delta', 0) for k in ds_keys]

x_pos = range(len(ds_keys))
width = 0.35
bars1 = ax.bar([p - width/2 for p in x_pos], p_avgs, width, label='Perceptron', color='#e74c3c', alpha=0.8)
bars2 = ax.bar([p + width/2 for p in x_pos], d_avgs, width, label='Delta', color='#3498db', alpha=0.8)
ax.set_xticks(list(x_pos))
ax.set_xticklabels(dataset_names, fontsize=9)
ax.set_xlabel('Dataset (Class Distance)')
ax.set_ylabel('Average Epochs to Converge')
ax.set_title('Convergence Speed vs Class Distance')
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

for bar in bars1:
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
            f'{bar.get_height():.1f}', ha='center', va='bottom', fontsize=8)
for bar in bars2:
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
            f'{bar.get_height():.1f}', ha='center', va='bottom', fontsize=8)

# --- Plot 4: Scatter of input patterns ---
ax = axes[1][1]
datasets = [
    {'name': 'Original (far)', 'points': [(10, 2), (2, -5), (-5, 5)]},
    {'name': 'Medium', 'points': [(3, 1), (1, -2), (-2, 2)]},
    {'name': 'Close', 'points': [(1.5, 0.5), (0.5, -1.0), (-1.0, 1.0)]},
    {'name': 'Very close', 'points': [(1.0, 0.3), (0.3, -0.5), (-0.5, 0.5)]},
]
markers = ['o', 's', '^', 'D']
for i, ds in enumerate(datasets):
    xs = [p[0] for p in ds['points']]
    ys = [p[1] for p in ds['points']]
    ax.scatter(xs, ys, marker=markers[i], s=100, color=colors[i], label=ds['name'], zorder=5)
    for j, (px, py) in enumerate(ds['points']):
        ax.annotate(f'C{j+1}', (px, py), textcoords="offset points",
                    xytext=(5, 5), fontsize=8, color=colors[i])

ax.set_title('Input Pattern Distributions')
ax.set_xlabel('x1')
ax.set_ylabel('x2')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
ax.axhline(y=0, color='k', linewidth=0.5)
ax.axvline(x=0, color='k', linewidth=0.5)

plt.suptitle('Effect of Class Center Distance on Convergence', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('close_centers.png', dpi=150, bbox_inches='tight')
print('Saved close_centers.png')
