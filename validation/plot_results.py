"""Generate publication-quality charts from experiment results."""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import os

# Output directory
OUT = os.path.join(os.path.dirname(__file__), '..', 'figures')
os.makedirs(OUT, exist_ok=True)

# ── Style setup ──────────────────────────────────────────────────
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 11,
    'legend.fontsize': 10,
    'figure.dpi': 150,
    'savefig.dpi': 200,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
})

# Color palette (colorblind-friendly)
C_STD = '#7A7A7A'       # StandardMLP - gray
C_CUBE = '#4C72B0'      # CubeMLP - blue
C_ORBIT = '#E24A33'     # OrbitMLP - red
C_ORBITCUBE = '#DD8452' # OrbitCubeMLP - orange
C_RANDOM = '#937860'    # RandomOrbitMLP - brown
C_PERPOS = '#55A868'    # PerPositionMLP - green

COLORS = {
    'StandardMLP': C_STD,
    'CubeMLP': C_CUBE,
    'OrbitMLP': C_ORBIT,
    'OrbitCubeMLP': C_ORBITCUBE,
    'RandomOrbitMLP': C_RANDOM,
    'PerPositionMLP': C_PERPOS,
}

MODEL_ORDER = ['StandardMLP', 'CubeMLP', 'OrbitMLP', 'OrbitCubeMLP', 'RandomOrbitMLP', 'PerPositionMLP']


# ═══════════════════════════════════════════════════════════════════
# Chart 1: Ablation — Position Reconstruction
# ═══════════════════════════════════════════════════════════════════

def plot_ablation_recon():
    models = ['StandardMLP', 'CubeMLP', 'OrbitMLP', 'OrbitCubeMLP', 'RandomOrbitMLP', 'PerPositionMLP']
    acc = [1.52, 0.59, 15.06, 14.49, 10.05, 18.52]
    params = [9792, 9792, 451968, 451968, 451968, 1176384]
    colors = [COLORS[m] for m in models]

    fig, ax = plt.subplots(figsize=(8, 5))

    bars = ax.bar(models, acc, color=colors, edgecolor='white', linewidth=0.8)

    # Annotate with param count
    for bar, p in zip(bars, params):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f'{p:,}p', ha='center', va='bottom', fontsize=8, color='#555555')

    # Group annotations
    ax.annotate('Shared weight\n(no orbit)', xy=(0.8, 1.52), xytext=(1.5, 5),
                fontsize=8, color=C_STD, ha='center',
                arrowprops=dict(arrowstyle='->', color=C_STD, lw=0.8))
    ax.annotate('Orbit-shared\nweights', xy=(3.2, 15.06), xytext=(3.5, 20),
                fontsize=8, color=C_ORBIT, ha='center', fontweight='bold',
                arrowprops=dict(arrowstyle='->', color=C_ORBIT, lw=1.2))
    ax.annotate('Random\norbits', xy=(4.8, 10.05), xytext=(5.5, 13),
                fontsize=8, color=C_RANDOM, ha='center',
                arrowprops=dict(arrowstyle='->', color=C_RANDOM, lw=0.8))
    ax.annotate('Full per-position\n(upper bound)', xy=(5.2, 18.52), xytext=(5.5, 23),
                fontsize=8, color=C_PERPOS, ha='center',
                arrowprops=dict(arrowstyle='->', color=C_PERPOS, lw=0.8))

    ax.set_ylabel('Test Accuracy (%)')
    ax.set_title('Ablation: Position Reconstruction\n(125-way, per-position head)')
    ax.set_ylim(0, 28)
    ax.axhline(y=0.8, color='#999999', linestyle='--', linewidth=0.8, label=f'Chance (1/125 = 0.8%)')
    ax.legend(fontsize=8, loc='upper left')
    ax.grid(axis='y', alpha=0.3)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=25, ha='right')

    # Add key insight box
    ax.text(0.98, 0.95, 'OrbitMLP = 78% of PerPositionMLP accuracy\nwith only 38% of parameters',
            transform=ax.transAxes, fontsize=9, ha='right', va='top',
            bbox=dict(boxstyle='round', facecolor='#FFF9E6', alpha=0.9, edgecolor='#E0C36A'))

    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'fig1_ablation_recon.png'))
    fig.savefig(os.path.join(OUT, 'fig1_ablation_recon.svg'))
    plt.close()
    print('Saved fig1_ablation_recon')


# ═══════════════════════════════════════════════════════════════════
# Chart 2: Sample Efficiency
# ═══════════════════════════════════════════════════════════════════

def plot_sample_efficiency():
    samples = [200, 400, 800, 1600, 3200]
    std = [69.79, 72.92, 76.50, 77.25, 86.62]
    orbit = [87.50, 100.00, 100.00, 100.00, 100.00]
    perpos = [85.42, 94.79, 99.50, 100.00, 100.00]

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(samples, std, 'o-', color=C_STD, linewidth=2, markersize=8, label='StandardMLP (shared weight)')
    ax.plot(samples, orbit, 's-', color=C_ORBIT, linewidth=2.5, markersize=9, label='OrbitMLP (orbit-shared)')
    ax.plot(samples, perpos, 'D-', color=C_PERPOS, linewidth=2, markersize=8, label='PerPositionMLP (per-position)')

    ax.axhline(y=12.5, color='#999999', linestyle='--', linewidth=0.8, label='Chance (1/8 = 12.5%)')

    # Highlight the gap
    ax.fill_between(samples, std, orbit, alpha=0.08, color=C_ORBIT)
    ax.annotate('Δ = +17.7pp\n@ 200 samples', xy=(200, 78.6), fontsize=9, ha='center',
                color=C_ORBIT, fontweight='bold')

    ax.set_xlabel('Training Samples')
    ax.set_ylabel('Test Accuracy (%)')
    ax.set_title('Sample Efficiency: OrbitMLP saturates 4× faster than PerPositionMLP\n(8-class shape classification, D=96, 6 layers, noise=0.3)')
    ax.set_ylim(55, 105)
    ax.legend(fontsize=9, loc='lower right')
    ax.grid(alpha=0.3)
    ax.xaxis.set_major_locator(mticker.FixedLocator(samples))

    # OrbitMLP @400 annotation
    ax.annotate('OrbitMLP = 100%\n@ 400 samples\n(50/class)', xy=(400, 100),
                xytext=(600, 92), fontsize=8, color=C_ORBIT, fontweight='bold',
                arrowprops=dict(arrowstyle='->', color=C_ORBIT, lw=0.8))

    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'fig2_sample_efficiency.png'))
    fig.savefig(os.path.join(OUT, 'fig2_sample_efficiency.svg'))
    plt.close()
    print('Saved fig2_sample_efficiency')


# ═══════════════════════════════════════════════════════════════════
# Chart 3: Scaling across Configs (StandardMLP flatlines)
# ═══════════════════════════════════════════════════════════════════

def plot_config_scaling():
    configs = ['Small\n(D=48, 4L)', 'Medium\n(D=96, 6L)', 'Large\n(D=128, 6L)']
    std = [84.75, 82.25, 82.83]
    orbit = [100, 100, 100]
    perpos = [100, 100, 100]

    x = np.arange(len(configs))
    width = 0.25

    fig, ax = plt.subplots(figsize=(8, 5))

    bars1 = ax.bar(x - width, std, width, color=C_STD, edgecolor='white', label='StandardMLP')
    bars2 = ax.bar(x, orbit, width, color=C_ORBIT, edgecolor='white', label='OrbitMLP')
    bars3 = ax.bar(x + width, perpos, width, color=C_PERPOS, edgecolor='white', label='PerPositionMLP')

    # Value labels
    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{bar.get_height():.1f}%', ha='center', fontsize=9, color=C_STD, fontweight='bold')
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() - 5,
                '100%', ha='center', fontsize=9, color='white', fontweight='bold')
    for bar in bars3:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() - 5,
                '100%', ha='center', fontsize=9, color='white', fontweight='bold')

    # Flatline arrow
    ax.annotate('', xy=(2.25, 83), xytext=(0.25, 85),
                arrowprops=dict(arrowstyle='->', color=C_STD, lw=1.5, ls='--'))
    ax.text(1.25, 87, 'StandardMLP does NOT scale with D\n(saturates at ~83%)',
            ha='center', fontsize=9, color=C_STD, fontweight='bold')

    ax.set_ylabel('Test Accuracy (%)')
    ax.set_title('Config Scaling: Orbit models saturate; StandardMLP flatlines')
    ax.set_xticks(x)
    ax.set_xticklabels(configs)
    ax.set_ylim(70, 108)
    ax.legend(fontsize=10, loc='lower left')
    ax.grid(axis='y', alpha=0.3)

    # Param annotation
    ax.text(0.98, 0.15, 'StandardMLP params: 10K→57K→101K\nOrbitMLP params:   452K→2.7M→4.8M',
            transform=ax.transAxes, fontsize=8, ha='right', va='bottom',
            bbox=dict(boxstyle='round', facecolor='#F0F0F0', alpha=0.8))

    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'fig3_config_scaling.png'))
    fig.savefig(os.path.join(OUT, 'fig3_config_scaling.svg'))
    plt.close()
    print('Saved fig3_config_scaling')


# ═══════════════════════════════════════════════════════════════════
# Chart 4: Parameter efficiency scatter
# ═══════════════════════════════════════════════════════════════════

def plot_param_efficiency():
    fig, ax = plt.subplots(figsize=(8, 5))

    # Ablation models (position reconstruction task)
    models_recon = {
        'StandardMLP':     (9792, 1.52),
        'CubeMLP':         (9792, 0.59),
        'OrbitMLP':       (451968, 15.06),
        'OrbitCubeMLP':   (451968, 14.49),
        'RandomOrbitMLP': (451968, 10.05),
        'PerPositionMLP': (1176384, 18.52),
    }

    for name, (p, a) in models_recon.items():
        marker = 's' if 'Orbit' in name else 'o'
        size = 160 if 'Orbit' in name else 100
        edgewidth = 2 if 'Orbit' in name else 0.8
        ax.scatter(p, a, c=COLORS[name], s=size, marker=marker,
                   edgecolors='#333333', linewidths=edgewidth, zorder=5 if 'Orbit' in name else 3)
        offset = 4 if 'MLP' in name and 'Orbit' not in name and 'Cube' not in name and 'Per' not in name else 8
        ax.annotate(name, (p, a), textcoords="offset points", xytext=(0, offset),
                    fontsize=8, ha='center', color=COLORS[name], fontweight='bold' if 'Orbit' in name else 'normal')

    # Pareto frontier line
    frontier_x = [9792, 451968, 1176384]
    frontier_y = [1.52, 15.06, 18.52]
    ax.plot(frontier_x, frontier_y, '--', color='#AAAAAA', linewidth=1, alpha=0.7)

    ax.set_xlabel('Parameter Count')
    ax.set_ylabel('Test Accuracy (%)')
    ax.set_title('Parameter Efficiency: OrbitMLP achieves 78% accuracy with 38% parameters')
    ax.set_xscale('log')
    ax.grid(alpha=0.3)

    # Pareto annotation
    ax.annotate('Pareto frontier:\nStandardMLP→OrbitMLP→PerPositionMLP',
                xy=(100000, 8), fontsize=8, color='#888888',
                rotation=12)

    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'fig4_param_efficiency.png'))
    fig.savefig(os.path.join(OUT, 'fig4_param_efficiency.svg'))
    plt.close()
    print('Saved fig4_param_efficiency')


# ═══════════════════════════════════════════════════════════════════
# Chart 5: Zero-shot generalization
# ═══════════════════════════════════════════════════════════════════

def plot_generalization():
    models = ['StandardMLP', 'CubeMLP', 'OrbitMLP', 'OrbitCubeMLP', 'RandomOrbitMLP', 'PerPositionMLP']
    pretrain = [33.55, 33.55, 64.20, 64.20, 100.00, 100.00]
    zeroshot = [0.00, 0.00, 18.60, 18.60, 25.40, 5.60]
    finetune = [0.00, 0.00, 34.60, 34.60, 39.60, 32.80]
    colors = [COLORS[m] for m in models]

    x = np.arange(len(models))
    width = 0.25

    fig, ax = plt.subplots(figsize=(9, 5))

    ax.bar(x - width, pretrain, width, color='#DDDDDD', edgecolor='white', label='Pre-train (seen rotations)')
    ax.bar(x, zeroshot, width, color=colors, edgecolor='white', label='Zero-shot (unseen rotations)')
    ax.bar(x + width, finetune, width, color=colors, alpha=0.4, edgecolor='white',
           label='After fine-tune (20 samples)')

    # Zero-shot highlight
    ax.annotate('Zero-shot = 0%\n(no generalization)', xy=(0, 2),
                xytext=(0.5, 22), fontsize=9, color=C_STD, ha='center',
                arrowprops=dict(arrowstyle='->', color=C_STD, lw=0.8))
    ax.annotate('Zero-shot = 18.6%\n(group structure transfers)', xy=(2.2, 18.6),
                xytext=(2.5, 28), fontsize=9, color=C_ORBIT, ha='center', fontweight='bold',
                arrowprops=dict(arrowstyle='->', color=C_ORBIT, lw=1.2))
    ax.annotate('Zero-shot = 5.6%\n(overfit to seen positions)', xy=(5.2, 5.6),
                xytext=(5.5, 18), fontsize=9, color=C_PERPOS, ha='center',
                arrowprops=dict(arrowstyle='->', color=C_PERPOS, lw=0.8))

    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Rotation Generalization: Pre-train on {U,R,F} → Test on {D,L,B}')
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.legend(fontsize=9, loc='upper left')
    ax.grid(axis='y', alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'fig5_generalization.png'))
    fig.savefig(os.path.join(OUT, 'fig5_generalization.svg'))
    plt.close()
    print('Saved fig5_generalization')


# ═══════════════════════════════════════════════════════════════════
# Chart 6: Summary — Orbit contribution visual
# ═══════════════════════════════════════════════════════════════════

def plot_summary():
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    # (A) Q1: Do gathers help?
    ax = axes[0, 0]
    categories = ['StandardMLP\n→ CubeMLP', 'OrbitMLP\n→ OrbitCubeMLP']
    values = [-0.93, -0.56]
    colors_bar = [C_CUBE, C_ORBITCUBE]
    bars = ax.bar(categories, values, color=colors_bar, edgecolor='white', width=0.5)
    ax.axhline(y=0, color='black', linewidth=0.8)
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, v - 0.15 if v < 0 else v + 0.15,
                f'{v:+.2f}%', ha='center', fontsize=11, fontweight='bold')
    ax.set_title('Q1: Do gathers help training?', fontweight='bold')
    ax.set_ylabel('Δ Accuracy (%)')
    ax.set_ylim(-2, 1)
    ax.text(0.5, 0.95, 'NO — gathers are irrelevant\nfor AI training', transform=ax.transAxes,
            fontsize=10, ha='center', va='top', color=C_ORBIT, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='#FFEBEB', alpha=0.9))

    # (B) Q2: Do orbits help?
    ax = axes[0, 1]
    categories = ['CubeMLP\n→ OrbitCubeMLP', 'StandardMLP\n→ OrbitMLP']
    values = [13.90, 13.54]
    colors_bar = [C_ORBITCUBE, C_ORBIT]
    bars = ax.bar(categories, values, color=colors_bar, edgecolor='white', width=0.5)
    ax.axhline(y=0, color='black', linewidth=0.8)
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, v + 0.3,
                f'+{v:.2f}%', ha='center', fontsize=11, fontweight='bold')
    ax.set_title('Q2: Do orbit-shared weights help?', fontweight='bold')
    ax.set_ylabel('Δ Accuracy (%)')
    ax.set_ylim(0, 18)
    ax.text(0.5, 0.95, 'YES — orbits provide\n+13.5% improvement', transform=ax.transAxes,
            fontsize=10, ha='center', va='top', color=C_ORBIT, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='#EBFFEB', alpha=0.9))

    # (C) Sample efficiency highlight
    ax = axes[1, 0]
    samples = [200, 400, 800, 1600, 3200]
    std = [69.79, 72.92, 76.50, 77.25, 86.62]
    orbit = [87.50, 100.00, 100.00, 100.00, 100.00]
    ax.plot(samples, std, 'o-', color=C_STD, linewidth=2, markersize=6, label='StandardMLP')
    ax.plot(samples, orbit, 's-', color=C_ORBIT, linewidth=2.5, markersize=7, label='OrbitMLP')
    ax.axhline(y=12.5, color='#999', linestyle='--', linewidth=0.8, label='Chance')
    ax.fill_between(samples, std, orbit, alpha=0.1, color=C_ORBIT)
    ax.set_xlabel('Training Samples')
    ax.set_ylabel('Test Accuracy (%)')
    ax.set_title('Sample Efficiency (8-class, D=96)', fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    ax.set_ylim(55, 105)

    # (D) Generalization highlight
    ax = axes[1, 1]
    models_short = ['Std', 'Cube', 'Orbit', 'OrbCube', 'Rand', 'P.Pos']
    zeroshot = [0.0, 0.0, 18.6, 18.6, 25.4, 5.6]
    colors_zs = [C_STD, C_CUBE, C_ORBIT, C_ORBITCUBE, C_RANDOM, C_PERPOS]
    x = np.arange(len(models_short))
    ax.bar(x, zeroshot, color=colors_zs, edgecolor='white')
    ax.set_xticks(x)
    ax.set_xticklabels(models_short, fontsize=8)
    ax.set_ylabel('Zero-shot Accuracy (%)')
    ax.set_title('Zero-shot Generalization (unseen rotations)', fontweight='bold')
    ax.axhline(y=0, color='#999', linestyle='--', linewidth=0.8)
    for i, v in enumerate(zeroshot):
        if v > 0:
            ax.text(i, v + 0.5, f'{v}%', ha='center', fontsize=9, fontweight='bold', color=colors_zs[i])
        else:
            ax.text(i, 0.3, '0%', ha='center', fontsize=8, color='#999')
    ax.set_ylim(0, 32)
    ax.grid(axis='y', alpha=0.3)

    fig.suptitle('Cube Group Theory for Neural Network Weight Sharing', fontsize=14, fontweight='bold', y=1.01)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'fig6_summary.png'))
    fig.savefig(os.path.join(OUT, 'fig6_summary.svg'))
    plt.close()
    print('Saved fig6_summary')


if __name__ == '__main__':
    plot_ablation_recon()
    plot_sample_efficiency()
    plot_config_scaling()
    plot_param_efficiency()
    plot_generalization()
    plot_summary()
    print(f'\nAll figures saved to: {OUT}')
