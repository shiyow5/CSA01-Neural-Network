import re
import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams['font.family'] = 'Helvetica'
matplotlib.rcParams['axes.unicode_minus'] = False

def parse_log(filepath):
    cycles = []
    errors = []
    finished_cycle = None
    finished_error = None
    with open(filepath, 'r') as f:
        for line in f:
            m = re.match(r'Error in the (\d+)-th learning cycle = ([\d.]+)', line)
            if m:
                cycles.append(int(m.group(1)))
                errors.append(float(m.group(2)))
            m2 = re.match(r'Finished at the (\d+)-th learning cycle with Error = ([\d.]+)', line)
            if m2:
                finished_cycle = int(m2.group(1))
                finished_error = float(m2.group(2))
    return cycles, errors, finished_cycle, finished_error

hidden_sizes = [4, 6, 8, 10]
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

# --- 4-bit: individual learning curves ---
fig, axes = plt.subplots(2, 2, figsize=(12, 9))
fig.suptitle('4-bit Parity Check: Learning Curves by Number of Hidden Neurons', fontsize=14)
converge_info_4bit = {}

for idx, J in enumerate(hidden_sizes):
    ax = axes[idx // 2][idx % 2]
    cycles, errors, fc, fe = parse_log(f'j_{J}.txt')
    converge_info_4bit[J] = (fc, fe)
    ax.plot(cycles, errors, color=colors[idx], linewidth=1.2)
    if fc:
        ax.axvline(x=fc, color='gray', linestyle='--', alpha=0.7, linewidth=0.8)
        ax.annotate(f'Converged: {fc} cycles', xy=(fc, fe), fontsize=8,
                    xytext=(0.55, 0.5), textcoords='axes fraction',
                    arrowprops=dict(arrowstyle='->', color='gray'),
                    bbox=dict(boxstyle='round,pad=0.3', fc='lightyellow'))
    ax.set_title(f'Hidden Neurons = {J}', fontsize=11)
    ax.set_xlabel('Learning Cycles')
    ax.set_ylabel('Error')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('4bit_learning_curves_eng.png', dpi=150, bbox_inches='tight')
plt.close()

# --- 4-bit: combined comparison ---
fig, ax = plt.subplots(figsize=(10, 6))
for idx, J in enumerate(hidden_sizes):
    cycles, errors, fc, fe = parse_log(f'j_{J}.txt')
    label = f'J = {J} (converged: {fc} cycles)' if fc else f'J = {J}'
    ax.plot(cycles, errors, color=colors[idx], linewidth=1.5, label=label)
ax.set_title('4-bit Parity Check: Learning Curve Comparison by Hidden Neurons', fontsize=13)
ax.set_xlabel('Learning Cycles', fontsize=11)
ax.set_ylabel('Error', fontsize=11)
ax.set_yscale('log')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('4bit_comparison_eng.png', dpi=150, bbox_inches='tight')
plt.close()

# --- 8-bit: individual learning curves ---
fig, axes = plt.subplots(2, 2, figsize=(12, 9))
fig.suptitle('8-bit Parity Check: Learning Curves by Number of Hidden Neurons', fontsize=14)
converge_info_8bit = {}

for idx, J in enumerate(hidden_sizes):
    ax = axes[idx // 2][idx % 2]
    cycles, errors, fc, fe = parse_log(f'j8_{J}.txt')
    converge_info_8bit[J] = (fc, fe, errors[-1] if errors else None)
    ax.plot(cycles, errors, color=colors[idx], linewidth=1.2)
    ax.set_title(f'Hidden Neurons = {J}', fontsize=11)
    ax.set_xlabel('Learning Cycles')
    ax.set_ylabel('Error')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    if errors:
        ax.annotate(f'Final Error: {errors[-1]:.3f}', xy=(0.5, 0.95),
                    xycoords='axes fraction', fontsize=9, ha='center',
                    bbox=dict(boxstyle='round,pad=0.3', fc='lightyellow'))

plt.tight_layout()
plt.savefig('8bit_learning_curves_eng.png', dpi=150, bbox_inches='tight')
plt.close()

# --- 8-bit: combined comparison ---
fig, ax = plt.subplots(figsize=(10, 6))
for idx, J in enumerate(hidden_sizes):
    cycles, errors, fc, fe = parse_log(f'j8_{J}.txt')
    final_err = errors[-1] if errors else 0
    ax.plot(cycles, errors, color=colors[idx], linewidth=1.5,
            label=f'J = {J} (final error: {final_err:.2f})')
ax.set_title('8-bit Parity Check: Learning Curve Comparison by Hidden Neurons', fontsize=13)
ax.set_xlabel('Learning Cycles', fontsize=11)
ax.set_ylabel('Error', fontsize=11)
ax.set_yscale('log')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('8bit_comparison_eng.png', dpi=150, bbox_inches='tight')
plt.close()

# --- 4-bit: convergence bar chart ---
fig, ax = plt.subplots(figsize=(8, 5))
conv_cycles = [converge_info_4bit[J][0] for J in hidden_sizes]
bars = ax.bar([f'J = {J}' for J in hidden_sizes], conv_cycles, color=colors, edgecolor='black', linewidth=0.5)
for bar, val in zip(bars, conv_cycles):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 500,
            f'{val}', ha='center', va='bottom', fontsize=11)
ax.set_title('4-bit Parity Check: Learning Cycles to Convergence', fontsize=13)
ax.set_xlabel('Number of Hidden Neurons', fontsize=11)
ax.set_ylabel('Convergence Cycles', fontsize=11)
ax.grid(True, axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('4bit_convergence_bar_eng.png', dpi=150, bbox_inches='tight')
plt.close()

# --- 8-bit: final error bar chart ---
fig, ax = plt.subplots(figsize=(8, 5))
final_errors_8bit = []
for J in hidden_sizes:
    _, errors, fc, fe = parse_log(f'j8_{J}.txt')
    if fc:
        final_errors_8bit.append(fe)
    else:
        final_errors_8bit.append(errors[-1] if errors else 0)

bars = ax.bar([f'J = {J}' for J in hidden_sizes], final_errors_8bit, color=colors, edgecolor='black', linewidth=0.5)
for bar, val in zip(bars, final_errors_8bit):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
            f'{val:.2f}', ha='center', va='bottom', fontsize=11)
ax.set_title('8-bit Parity Check: Final Error after 2 Million Cycles', fontsize=13)
ax.set_xlabel('Number of Hidden Neurons', fontsize=11)
ax.set_ylabel('Final Error', fontsize=11)
ax.grid(True, axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('8bit_final_error_bar_eng.png', dpi=150, bbox_inches='tight')
plt.close()

print("All English graphs generated.")
