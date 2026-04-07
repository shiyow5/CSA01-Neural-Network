#!/usr/bin/env python3
"""Plot weight initialization influence results."""
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

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# --- Plot 1: Delta error curves for different init ranges ---
ax = axes[0][0]
detail_ranges = [0.01, 0.1, 0.5, 2.0, 5.0]
colors = ['#2ecc71', '#3498db', '#e67e22', '#e74c3c', '#9b59b6']
for i, r in enumerate(detail_ranges):
    try:
        ep, err = read_csv(f'winit_delta_{r:.2f}.csv')
        ax.plot(ep, err, color=colors[i], linewidth=1.2, label=f'range=[-{r}, {r}]')
    except FileNotFoundError:
        pass
ax.set_title('Delta: Error Curves by Init Range')
ax.set_xlabel('Epoch')
ax.set_ylabel('Total Error')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# --- Plot 2: Same but zoomed to first 50 epochs ---
ax = axes[0][1]
for i, r in enumerate(detail_ranges):
    try:
        ep, err = read_csv(f'winit_delta_{r:.2f}.csv')
        ep_z = [e for e in ep if e <= 50]
        err_z = err[:len(ep_z)]
        ax.plot(ep_z, err_z, color=colors[i], linewidth=1.2, label=f'range=[-{r}, {r}]')
    except FileNotFoundError:
        pass
ax.set_title('Delta: Error Curves (First 50 Epochs)')
ax.set_xlabel('Epoch')
ax.set_ylabel('Total Error')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# --- Plot 3: Bar chart of avg epochs + error bars ---
ax = axes[1][0]
summary = {}
with open('weight_init_summary.csv') as f:
    reader = csv.DictReader(f)
    for row in reader:
        r = float(row['init_range'])
        method = row['method']
        avg = float(row['avg_epochs'])
        std = float(row['std_epochs'])
        conv = float(row['convergence_rate'].replace('%', ''))
        if r not in summary:
            summary[r] = {}
        summary[r][method] = {'avg': avg if avg > 0 else 0, 'std': std, 'conv': conv}

ranges_all = sorted(summary.keys())
d_avgs = [summary[r]['delta']['avg'] for r in ranges_all]
d_stds = [summary[r]['delta']['std'] for r in ranges_all]
p_avgs = [summary[r]['perceptron']['avg'] for r in ranges_all]
p_stds = [summary[r]['perceptron']['std'] for r in ranges_all]

x_pos = range(len(ranges_all))
width = 0.35
ax.bar([p - width/2 for p in x_pos], p_avgs, width, yerr=p_stds, capsize=3,
       label='Perceptron', color='#e74c3c', alpha=0.8)
ax.bar([p + width/2 for p in x_pos], d_avgs, width, yerr=d_stds, capsize=3,
       label='Delta', color='#3498db', alpha=0.8)
ax.set_xticks(list(x_pos))
ax.set_xticklabels([f'{r}' for r in ranges_all])
ax.set_xlabel('Init Range')
ax.set_ylabel('Average Epochs (log scale)')
ax.set_title('Convergence Epochs vs Init Range')
ax.legend()
ax.grid(True, alpha=0.3, axis='y')
ax.set_yscale('log')

# --- Plot 4: Convergence rate ---
ax = axes[1][1]
d_conv = [summary[r]['delta']['conv'] for r in ranges_all]
p_conv = [summary[r]['perceptron']['conv'] for r in ranges_all]
ax.plot(range(len(ranges_all)), p_conv, 'o-', color='#e74c3c', linewidth=2, markersize=8, label='Perceptron')
ax.plot(range(len(ranges_all)), d_conv, 's-', color='#3498db', linewidth=2, markersize=8, label='Delta')
ax.set_xticks(list(range(len(ranges_all))))
ax.set_xticklabels([f'{r}' for r in ranges_all])
ax.set_xlabel('Init Range')
ax.set_ylabel('Convergence Rate (%)')
ax.set_title('Convergence Rate vs Init Range')
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_ylim(-5, 105)

plt.suptitle('Effect of Weight Initialization on Learning', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('weight_init.png', dpi=150, bbox_inches='tight')
print('Saved weight_init.png')
