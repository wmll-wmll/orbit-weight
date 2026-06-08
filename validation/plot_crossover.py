"""Visualize gather-vs-dense crossover from real domestic GPU measurements."""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os

OUT = os.path.join(os.path.dirname(__file__), '..', 'figures')
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 11,
    'figure.dpi': 150,
    'savefig.dpi': 200,
    'savefig.bbox': 'tight',
})

# Real measurements from domestic GPU (gfx936, HIP/DTK 25.04)
# Format: (N, B, D, gather_ms, dense_ms)
DATA = [
    (27,  64,  128, 0.043, 0.052),
    (27,  256, 512, 0.039, 0.046),
    (64,  64,  128, 0.015, 0.046),
    (64,  256, 512, 0.074, 0.049),   # CROSSOVER
    (125, 64,  128, 0.013, 0.046),
    (125, 256, 512, 0.138, 0.130),   # CROSSOVER
    (216, 64,  128, 0.020, 0.046),
    (216, 256, 512, 0.237, 0.201),   # CROSSOVER
]

C_GATHER = '#E24A33'
C_DENSE = '#4C72B0'
C_CROSSOVER = '#FFD700'


def plot_crossover_main():
    """Main crossover chart: grouped bars by (N, B, D) config."""
    fig, ax = plt.subplots(figsize=(10, 5.5))

    labels = [f'N={n}\nB={b} D={d}' for n, b, d, *_ in DATA]
    gather_times = [d[3] for d in DATA]
    dense_times = [d[4] for d in DATA]

    x = np.arange(len(labels))
    width = 0.35

    bars_g = ax.bar(x - width/2, gather_times, width, color=C_GATHER, edgecolor='white',
                    label='gather (HIP)')
    bars_d = ax.bar(x + width/2, dense_times, width, color=C_DENSE, edgecolor='white',
                    label='dense matmul (rocBLAS)')

    # Annotate winner
    for i, (n, b, d, g, dn) in enumerate(DATA):
        if dn < g:  # dense wins
            winner = 'dense'
            ratio = g / dn
            color = C_DENSE
            star = '★'
        else:
            winner = 'gather'
            ratio = dn / g
            color = C_GATHER
            star = '☆'

        y_max = max(g, dn)
        ax.text(i, y_max + 0.008, f'{star} {winner}\n{ratio:.2f}x',
                ha='center', fontsize=9, fontweight='bold', color=color)

    # Crossover zone shading
    for i in range(len(labels)):
        if DATA[i][4] < DATA[i][3]:  # dense wins
            ax.axvspan(i - 0.5, i + 0.5, alpha=0.08, color=C_CROSSOVER)

    ax.set_ylabel('Latency (ms)')
    ax.set_title('Gather vs Dense Matmul on Domestic GPU (Biren gfx936, fp16)')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.legend(fontsize=10, loc='upper left')
    ax.grid(axis='y', alpha=0.3)

    # Cross-over annotation
    ax.annotate('CROSSOVER ZONE\nB×D ≥ 131K\nDense matmul wins',
                xy=(5.5, 0.13), fontsize=11, ha='center', color=C_DENSE, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='#E8F0FE', alpha=0.9, edgecolor=C_DENSE))

    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'fig_crossover_bars.png'))
    fig.savefig(os.path.join(OUT, 'fig_crossover_bars.svg'))
    plt.close()
    print('Saved fig_crossover_bars')


def plot_crossover_bd():
    """Show crossover as function of B×D product."""
    fig, ax = plt.subplots(figsize=(8, 5))

    bed = [b * d for _, b, d, *_ in DATA]
    gather_times = [d[3] for d in DATA]
    dense_times = [d[4] for d in DATA]
    ratios = [dense / gather for _, _, _, gather, dense in DATA]

    # Color-code by winner
    colors = [C_DENSE if r < 1 else C_GATHER for r in ratios]
    sizes = [abs(1 - r) * 800 + 80 for r in ratios]

    scatter = ax.scatter(bed, ratios, c=colors, s=sizes, edgecolors='#333333',
                         linewidths=1.2, zorder=5, alpha=0.9)

    ax.axhline(y=1.0, color='#333333', linestyle='--', linewidth=1.5, alpha=0.5)
    ax.text(max(bed) * 1.02, 1.0, 'break-even', fontsize=9, color='#666', va='center')

    # Annotate points
    for i, (n, b, d, g, dn) in enumerate(DATA):
        label = f'N={n}'
        offset = 0.015 if ratios[i] > 1 else -0.025
        ax.annotate(label, (bed[i], ratios[i]),
                    textcoords="offset points", xytext=(0, 12 if ratios[i] > 1 else -14),
                    fontsize=8, ha='center', color=colors[i])

    # Shade regions
    ax.axhspan(0, 1.0, alpha=0.06, color=C_DENSE)
    ax.axhspan(1.0, max(ratios) * 1.1, alpha=0.06, color=C_GATHER)
    ax.text(max(bed) * 0.5, 0.5, 'Dense Wins →', fontsize=12, ha='center',
            color=C_DENSE, alpha=0.4, fontweight='bold')
    ax.text(max(bed) * 0.5, 2.8, '← Gather Wins', fontsize=12, ha='center',
            color=C_GATHER, alpha=0.4, fontweight='bold')

    ax.set_xlabel('B × D (batch × features)')
    ax.set_ylabel('Dense / Gather Latency Ratio')
    ax.set_title('Permutation Strategy Crossover on Domestic GPU\n(ratio < 1.0 → dense matmul is faster)')
    ax.set_xscale('log')
    ax.grid(alpha=0.3)

    # Decision boundary annotation
    ax.axvline(x=100000, color=C_CROSSOVER, linestyle=':', linewidth=1.8, alpha=0.7)
    ax.annotate('Decision boundary\nB×D ≈ 100K', xy=(100000, 2.5),
                fontsize=9, color='#B8860B', ha='center', fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='#FFFDE7', alpha=0.8, edgecolor=C_CROSSOVER))

    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'fig_crossover_ratio.png'))
    fig.savefig(os.path.join(OUT, 'fig_crossover_ratio.svg'))
    plt.close()
    print('Saved fig_crossover_ratio')


def plot_speedup_heatmap():
    """Heatmap: dense/gather ratio across (N, B×D) space."""
    fig, ax = plt.subplots(figsize=(8, 5))

    # Construct grid from measured points + interpolation
    N_vals = np.array([27, 64, 125, 216])
    BD_vals = np.array([8192, 131072])  # 64×128, 256×512

    # Measured ratios at grid points
    ratio_grid = np.array([
        [1/1.19, 1/1.19],    # N=27:  gather wins both
        [1/3.08, 1.52],       # N=64:  gather@BD=8K, dense@BD=131K
        [1/3.45, 1.06],       # N=125: gather@BD=8K, dense@BD=131K
        [1/2.35, 1.18],       # N=216: gather@BD=8K, dense@BD=131K
    ])  # Values > 1 = dense wins

    im = ax.pcolormesh(ratio_grid.T, cmap='RdBu_r', vmin=0.2, vmax=2.0,
                        edgecolors='white', linewidth=2)

    # Annotate cells
    for i in range(len(N_vals)):
        for j in range(len(BD_vals)):
            val = ratio_grid[i, j]
            winner = 'dense' if val > 1 else 'gather'
            color = 'white' if abs(val - 1) > 0.5 else '#333333'
            ax.text(i + 0.5, j + 0.5, f'{val:.2f}x\n{winner}',
                    ha='center', va='center', fontsize=10, fontweight='bold', color=color)

    ax.set_xticks(np.arange(len(N_vals)) + 0.5)
    ax.set_xticklabels([f'N={n}' for n in N_vals])
    ax.set_yticks(np.arange(len(BD_vals)) + 0.5)
    ax.set_yticklabels([f'B×D={bd/1000:.0f}K' for bd in BD_vals])
    ax.set_title('Permutation Strategy: Dense/Gather Ratio\n(>1 = dense wins, <1 = gather wins)')

    cbar = fig.colorbar(im, ax=ax, label='Dense/Gather Speed Ratio')
    cbar.ax.axhline(y=1.0, color='black', linestyle='--', linewidth=1.5)

    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'fig_crossover_heatmap.png'))
    fig.savefig(os.path.join(OUT, 'fig_crossover_heatmap.svg'))
    plt.close()
    print('Saved fig_crossover_heatmap')


def plot_matmul_tflops():
    """Matmul TFLOPS scaling on domestic GPU."""
    fig, ax = plt.subplots(figsize=(7, 4.5))

    matmul_sizes = [1024, 2048, 4096, 8192]
    tflops = [0.01, 37.7, 86.0, 131.9]  # near-0 for 1024 (JIT overhead)

    ax.plot([1024, 8192], [132, 132], '--', color='#888888', linewidth=1, alpha=0.6)
    ax.text(4500, 134, 'Peak: 132 TFLOPS (fp16)', fontsize=9, color='#888888')

    ax.plot(matmul_sizes[1:], tflops[1:], 'o-', color=C_DENSE, linewidth=2.5, markersize=10,
            label='gfx936 (domestic GPU)')

    # A100 reference
    ax.axhline(y=312, color='#55A868', linestyle=':', linewidth=1.2, alpha=0.7)
    ax.text(7000, 314, 'A100 peak: 312 TFLOPS', fontsize=9, color='#55A868')

    # Efficiency annotation
    for sz, tf in zip(matmul_sizes[2:], tflops[2:]):
        eff = tf / 132 * 100
        ax.annotate(f'{eff:.0f}% eff.', (sz, tf), textcoords="offset points",
                    xytext=(0, 12), fontsize=8, ha='center', color=C_DENSE)

    ax.set_xlabel('Matrix Size (N=K=M)')
    ax.set_ylabel('TFLOPS (fp16)')
    ax.set_title('Matmul Throughput on Domestic GPU (Biren gfx936)')
    ax.set_xscale('log')
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    ax.set_ylim(0, 350)

    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'fig_matmul_tflops.png'))
    fig.savefig(os.path.join(OUT, 'fig_matmul_tflops.svg'))
    plt.close()
    print('Saved fig_matmul_tflops')


def plot_roofline_vs_measured():
    """Scatter plot: measured vs roofline-predicted gather/dense ratios.

    Imports from backends.domestic: CROSSOVER_DATA and roofline_estimate.
    For each of the measured points:
      - Compute roofline prediction
      - Plot measured_ratio vs predicted_ratio as scatter
      - Add y=x reference line
      - Annotate each point with (N,B,D)

    Saves fig_theory_crossover_scatter.png and .svg
    """
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    from backends.domestic import CROSSOVER_DATA, roofline_estimate

    fig, ax = plt.subplots(figsize=(8, 7))

    measured_ratios = []
    predicted_ratios = []
    winners = []
    labels = []

    for N, B, D, g_ms, d_ms, ratio in CROSSOVER_DATA:
        # Measured speedup: always >= 1, determined directly from raw times
        if g_ms < d_ms:
            measured_ratio = d_ms / g_ms
            measured_winner = 'gather'
        else:
            measured_ratio = g_ms / d_ms
            measured_winner = 'dense'

        # Roofline prediction
        pred_g, pred_d, pred_winner = roofline_estimate(N, B, D)
        if pred_winner == 'gather':
            predicted_ratio = pred_d / pred_g
        else:
            predicted_ratio = pred_g / pred_d

        measured_ratios.append(measured_ratio)
        predicted_ratios.append(predicted_ratio)
        winners.append(measured_winner)
        labels.append((N, B, D))

    # Color by measured winner
    point_colors = ['#E24A33' if w == 'gather' else '#4C72B0' for w in winners]

    ax.scatter(measured_ratios, predicted_ratios, c=point_colors, s=120,
               edgecolors='#333333', linewidths=1.2, zorder=5)

    # y = x reference line
    max_val = max(max(measured_ratios), max(predicted_ratios)) * 1.1
    ax.plot([0, max_val], [0, max_val], 'k--', alpha=0.3, linewidth=1.0,
            label='y = x (perfect prediction)')

    # Annotate each point with (N,B,D)
    for i, (N, B, D) in enumerate(labels):
        offset_y = 8 if i % 2 == 0 else -15
        ax.annotate(f'({N},{B},{D})',
                    (measured_ratios[i], predicted_ratios[i]),
                    textcoords="offset points", xytext=(5, offset_y),
                    fontsize=7, ha='left', color=point_colors[i])

    ax.set_xlabel('Measured Speedup Ratio (faster / slower)')
    ax.set_ylabel('Roofline-Predicted Speedup Ratio')
    ax.set_title('Roofline Model Validation: Predicted vs Measured\n'
                 'Gather/Dense Crossover Speedup Ratios on gfx936')
    ax.grid(alpha=0.3)
    ax.set_xlim(0, max_val)
    ax.set_ylim(0, max_val)

    # Compute R² and winner accuracy
    ss_res = np.sum((np.array(measured_ratios) - np.array(predicted_ratios)) ** 2)
    ss_tot = np.sum((np.array(measured_ratios) - np.mean(measured_ratios)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0

    n_correct = 0
    for i, (N, B, D) in enumerate(labels):
        _, _, pred_w = roofline_estimate(N, B, D)
        if pred_w == winners[i]:
            n_correct += 1

    # Custom legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#E24A33', label='Gather wins (measured)'),
        Patch(facecolor='#4C72B0', label='Dense wins (measured)'),
        plt.Line2D([0], [0], color='black', linestyle='--', alpha=0.3,
                   label='y = x'),
    ]
    ax.legend(handles=legend_elements, fontsize=8, loc='lower right')

    # Stats annotation
    ax.text(0.98, 0.05,
            f'R² = {r2:.3f}\n'
            f'Winner accuracy: {n_correct}/{len(CROSSOVER_DATA)}\n'
            f'(gfx936, BW=180 GB/s, FLOPS=110 TFLOPS)',
            transform=ax.transAxes, fontsize=9, ha='right', va='bottom',
            bbox=dict(boxstyle='round', facecolor='#F0F0F0', alpha=0.8))

    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'fig_theory_crossover_scatter.png'))
    fig.savefig(os.path.join(OUT, 'fig_theory_crossover_scatter.svg'))
    plt.close()
    print('Saved fig_theory_crossover_scatter')


if __name__ == '__main__':
    plot_crossover_main()
    plot_crossover_bd()
    plot_speedup_heatmap()
    plot_matmul_tflops()
    plot_roofline_vs_measured()
    print(f'\nAll crossover figures saved to: {OUT}')
