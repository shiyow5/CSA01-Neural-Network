#!/usr/bin/env python3
"""Plot learning rate comparison for AND gate."""
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

etas = [0.1, 0.5, 1.0, 2.0]
colors_p = ['#e74c3c', '#e67e22', '#f1c40f', '#2ecc71']
colors_d = ['#3498db', '#9b59b6', '#1abc9c', '#34495e']

# --- Plot 1: Perceptron error curves ---
ax = axes[0][0]
for i, eta in enumerate(etas):
    try:
        ep, err = read_csv(f'eta_perceptron_{eta:.2f}.csv')
        ax.plot(ep, err, color=colors_p[i], linewidth=1.2, label=f'eta={eta}')
    except FileNotFoundError:
        pass
ax.set_title('Perceptron: Error Curves by Learning Rate')
ax.set_xlabel('Epoch')
ax.set_ylabel('Total Error')
ax.legend()
ax.grid(True, alpha=0.3)

# --- Plot 2: Delta error curves ---
ax = axes[0][1]
for i, eta in enumerate(etas):
    try:
        ep, err = read_csv(f'eta_delta_{eta:.2f}.csv')
        ax.plot(ep, err, color=colors_d[i], linewidth=1.2, label=f'eta={eta}')
    except FileNotFoundError:
        pass
ax.set_title('Delta: Error Curves by Learning Rate')
ax.set_xlabel('Epoch')
ax.set_ylabel('Total Error')
ax.legend()
ax.grid(True, alpha=0.3)

# --- Plot 3: Bar chart of average epochs ---
ax = axes[1][0]
summary = {}
with open('eta_summary.csv') as f:
    reader = csv.DictReader(f)
    for row in reader:
        eta = float(row['eta'])
        method = row['method']
        avg = float(row['avg_epochs'])
        if eta not in summary:
            summary[eta] = {}
        summary[eta][method] = avg if avg > 0 else None

etas_all = sorted(summary.keys())
p_avgs = [summary[e].get('perceptron', 0) or 0 for e in etas_all]
d_avgs = [summary[e].get('delta', 0) or 0 for e in etas_all]

x_pos = range(len(etas_all))
width = 0.35
bars1 = ax.bar([p - width/2 for p in x_pos], p_avgs, width, label='Perceptron', color='#e74c3c', alpha=0.8)
bars2 = ax.bar([p + width/2 for p in x_pos], d_avgs, width, label='Delta', color='#3498db', alpha=0.8)
ax.set_xticks(list(x_pos))
ax.set_xticklabels([f'{e}' for e in etas_all])
ax.set_xlabel('Learning Rate (eta)')
ax.set_ylabel('Average Epochs to Converge')
ax.set_title('Average Convergence Epochs by Learning Rate')
ax.legend()
ax.grid(True, alpha=0.3, axis='y')
ax.set_yscale('log')

# Add value labels
for bar in bars1:
    if bar.get_height() > 0:
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                f'{bar.get_height():.0f}', ha='center', va='bottom', fontsize=8)
for bar in bars2:
    if bar.get_height() > 0:
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                f'{bar.get_height():.0f}', ha='center', va='bottom', fontsize=8)

# --- Plot 4: Convergence rate ---
ax = axes[1][1]
p_rates = []
d_rates = []
with open('eta_summary.csv') as f:
    reader = csv.DictReader(f)
    for row in reader:
        rate = float(row['convergence_rate'].replace('%', ''))
        if row['method'] == 'perceptron':
            p_rates.append(rate)
        else:
            d_rates.append(rate)

ax.plot(etas_all, p_rates, 'o-', color='#e74c3c', linewidth=2, markersize=8, label='Perceptron')
ax.plot(etas_all, d_rates, 's-', color='#3498db', linewidth=2, markersize=8, label='Delta')
ax.set_xlabel('Learning Rate (eta)')
ax.set_ylabel('Convergence Rate (%)')
ax.set_title('Convergence Rate by Learning Rate')
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_ylim(-5, 105)

plt.suptitle('Learning Rate Comparison for AND Gate', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('eta_comparison.png', dpi=150, bbox_inches='tight')
print('Saved eta_comparison.png')
