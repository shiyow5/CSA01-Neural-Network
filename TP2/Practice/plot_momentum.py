#!/usr/bin/env python3
"""Plot momentum experiment results."""
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

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# --- Left: Error curves ---
ax = axes[0]
configs = [
    ('momentum_a0_j5.csv', 'alpha=0.0 (no momentum)', '#e74c3c', 'o-'),
    ('momentum_a50_j5.csv', 'alpha=0.5', '#3498db', 's--'),
    ('momentum_a90_j5.csv', 'alpha=0.9', '#2ecc71', '^:'),
]
for fname, label, color, style in configs:
    ep, err = read_csv(fname)
    ax.plot(ep, err, style, color=color, linewidth=1.2,
            markersize=3, markevery=max(1, len(ep)//15),
            alpha=0.8, label=label)
ax.set_xlabel('Epoch')
ax.set_ylabel('Total Error')
ax.set_title('Error Curves with Different Momentum')
ax.set_xscale('log')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3, which='both')

# --- Right: Summary bar chart ---
ax = axes[1]
alphas = ['0.0', '0.3', '0.5', '0.7', '0.9']
conv_rates = [6, 8, 9, 9, 7]
avg_epochs = [31444, 27944, 19528, 10479, 9343]

x_pos = range(len(alphas))
width = 0.35

ax2 = ax.twinx()
bars = ax.bar(x_pos, avg_epochs, width=0.6, color='#3498db', alpha=0.7, label='Avg Epochs')
line = ax2.plot(list(x_pos), [r*10 for r in conv_rates], 'o-', color='#e74c3c',
               linewidth=2, markersize=8, label='Conv. Rate')

ax.set_xticks(list(x_pos))
ax.set_xticklabels([f'alpha={a}' for a in alphas])
ax.set_xlabel('Momentum Coefficient')
ax.set_ylabel('Average Epochs to Converge')
ax2.set_ylabel('Convergence Rate (%)')
ax2.set_ylim(0, 105)
ax.set_title('Convergence vs Momentum')

# Combined legend
lines1, labels1 = ax.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=9)
ax.grid(True, alpha=0.3, axis='y')

for bar in bars:
    if bar.get_height() > 0:
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                f'{bar.get_height():.0f}', ha='center', va='bottom', fontsize=8)

plt.suptitle('Effect of Momentum on BP Learning (4-bit Parity, 4 Hidden)',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('momentum_comparison.png', dpi=150, bbox_inches='tight')
print('Saved momentum_comparison.png')
