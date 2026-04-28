import re
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

matplotlib.rcParams['font.family'] = 'Hiragino Sans'
matplotlib.rcParams['axes.unicode_minus'] = False

def parse_log(filepath):
    cycles, errors = [], []
    finished_cycle, finished_error = None, None
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

# --- Comparison bar chart: original vs momentum (8-bit, J=4,6,8,10) ---
hidden_sizes = [4, 6, 8, 10]
original_errors = []
momentum_errors = []
for n in hidden_sizes:
    _, errs, _, fe = parse_log(f'j8_{n}.txt')
    original_errors.append(fe if fe else (errs[-1] if errs else 0))
    _, errs2, _, fe2 = parse_log(f'j8m_{n}.txt')
    momentum_errors.append(fe2 if fe2 else (errs2[-1] if errs2 else 0))

x_pos = np.arange(len(hidden_sizes))
width = 0.35

fig, ax = plt.subplots(figsize=(10, 6))
bars1 = ax.bar(x_pos - width/2, original_errors, width, label='モメンタムなし', color='#1f77b4', edgecolor='black', linewidth=0.5)
bars2 = ax.bar(x_pos + width/2, momentum_errors, width, label='モメンタムあり (α=0.9)', color='#ff7f0e', edgecolor='black', linewidth=0.5)

for bar, val in zip(bars1, original_errors):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, f'{val:.2f}', ha='center', va='bottom', fontsize=9)
for bar, val in zip(bars2, momentum_errors):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, f'{val:.2f}', ha='center', va='bottom', fontsize=9)

ax.set_xlabel('隠れニューロン数', fontsize=11)
ax.set_ylabel('最終誤差（200万サイクル後）', fontsize=11)
ax.set_title('8-bit パリティチェック：モメンタムの有無による最終誤差の比較', fontsize=13)
ax.set_xticks(x_pos)
ax.set_xticklabels([f'{n}' for n in hidden_sizes])
ax.legend(fontsize=10)
ax.grid(True, axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('8bit_momentum_comparison.png', dpi=150, bbox_inches='tight')
plt.close()

# --- Learning curve comparison for J=8 (best original) ---
fig, ax = plt.subplots(figsize=(10, 6))
cycles_orig, errors_orig, _, _ = parse_log('j8_8.txt')
cycles_mom, errors_mom, _, _ = parse_log('j8m_8.txt')
ax.plot(cycles_orig, errors_orig, linewidth=1.5, label='モメンタムなし (最終: {:.2f})'.format(errors_orig[-1]), color='#1f77b4')
ax.plot(cycles_mom, errors_mom, linewidth=1.5, label='モメンタムあり α=0.9 (最終: {:.2f})'.format(errors_mom[-1]), color='#ff7f0e')
ax.set_title('8-bit パリティチェック（隠れニューロン数8）：モメンタムの効果', fontsize=13)
ax.set_xlabel('学習サイクル数', fontsize=11)
ax.set_ylabel('誤差 (Error)', fontsize=11)
ax.set_yscale('log')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('8bit_momentum_j8_curve.png', dpi=150, bbox_inches='tight')
plt.close()

# --- Extended hidden neurons comparison (J=4..20, momentum) ---
fig, ax = plt.subplots(figsize=(10, 6))
all_neurons = [4, 6, 8, 10, 16, 20]
all_mom_errors = []
for n in all_neurons:
    _, errs, fc, fe = parse_log(f'j8m_{n}.txt')
    all_mom_errors.append(fe if fe else errs[-1])

colors_ext = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
bars = ax.bar([str(n) for n in all_neurons], all_mom_errors, color=colors_ext, edgecolor='black', linewidth=0.5)
for bar, val in zip(bars, all_mom_errors):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, f'{val:.2f}', ha='center', va='bottom', fontsize=10)
ax.set_title('8-bit パリティチェック（モメンタムあり）：隠れニューロン数の拡張実験', fontsize=13)
ax.set_xlabel('隠れニューロン数', fontsize=11)
ax.set_ylabel('最終誤差（200万サイクル後）', fontsize=11)
ax.grid(True, axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('8bit_momentum_extended.png', dpi=150, bbox_inches='tight')
plt.close()

print("Momentum comparison graphs generated.")
