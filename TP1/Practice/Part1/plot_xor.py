#!/usr/bin/env python3
"""Plot XOR error curves showing non-convergence."""
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

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Perceptron XOR
ep, err = read_csv('xor_perceptron_error.csv')
ax1.plot(ep, err, 'r-', linewidth=0.8)
ax1.set_title('Perceptron Learning Rule on XOR')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Total Error')
ax1.axhline(y=0.01, color='g', linestyle='--', label='Desired error = 0.01')
ax1.legend()
ax1.set_ylim(bottom=0)
ax1.grid(True, alpha=0.3)

# Delta XOR
ep, err = read_csv('xor_delta_error.csv')
ax2.plot(ep, err, 'b-', linewidth=0.8)
ax2.set_title('Delta Learning Rule on XOR')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Total Error')
ax2.axhline(y=0.01, color='g', linestyle='--', label='Desired error = 0.01')
ax2.legend()
ax2.set_ylim(bottom=0)
ax2.grid(True, alpha=0.3)

plt.suptitle('XOR Problem: Single Neuron Cannot Converge', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('xor_error.png', dpi=150, bbox_inches='tight')
print('Saved xor_error.png')
