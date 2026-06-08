"""Group comparison plot module: OrbitMLP vs UniformMLP vs RandomMLP across group types."""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
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

# Color palette for groups
C_n = '#4C72B0'     # Cyclic - blue
D_n = '#E24A33'     # Dihedral - red
S_n = '#55A868'     # Symmetric - green
O_h = '#C44E52'     # Octahedral - brick red

GROUP_COLORS = {
    'C_n': C_n,
    'D_n': D_n,
    'S_n': S_n,
    'O_h': O_h,
}

# Model variant colors
C_ORBIT = '#4C72B0'      # OrbitMLP - blue
C_UNIFORM = '#7A7A7A'    # UniformMLP - gray
C_RANDOM = '#E24A33'     # RandomMLP - red

MODEL_ORDER = ['OrbitMLP', 'UniformMLP', 'RandomMLP']


# ═══════════════════════════════════════════════════════════════════
# Chart 10: Group Comparison Bar Chart
# ═══════════════════════════════════════════════════════════════════

def plot_group_comparison(results):
    """Bar chart: accuracy of OrbitMLP vs UniformMLP vs RandomMLP for each group.

    Args:
        results: dict mapping group_name -> dict of model_name -> accuracy
            e.g. {'C_n': {'OrbitMLP': 85, 'UniformMLP': 70, 'RandomMLP': 45}, ...}

    Saves fig10_group_comparison.png and .svg
    """
    groups = list(results.keys())
    models = MODEL_ORDER
    model_colors = [C_ORBIT, C_UNIFORM, C_RANDOM]

    x = np.arange(len(groups))
    width = 0.25

    fig, ax = plt.subplots(figsize=(9, 5.5))

    for i, (model, color) in enumerate(zip(models, model_colors)):
        values = [results[g].get(model, 0) for g in groups]
        offset = (i - 1) * width
        bars = ax.bar(x + offset, values, width, color=color,
                      edgecolor='white', linewidth=0.8, label=model)

        # Annotate bars
        for bar, val in zip(bars, values):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.5,
                        f'{val:.1f}', ha='center', fontsize=8,
                        fontweight='bold', color=color)

    # Highlight OrbitMLP advantage
    for i, g in enumerate(groups):
        orbit_val = results[g].get('OrbitMLP', 0)
        uniform_val = results[g].get('UniformMLP', 0)
        if orbit_val > 0 and uniform_val > 0:
            delta = orbit_val - uniform_val
            ax.annotate(f'+{delta:.1f}%', (i, max(orbit_val, uniform_val) + 3),
                        ha='center', fontsize=8, color=C_ORBIT, fontweight='bold')

    ax.set_ylabel('Test Accuracy (%)')
    ax.set_title('Group Comparison: OrbitMLP vs UniformMLP vs RandomMLP')
    ax.set_xticks(x)
    ax.set_xticklabels(groups, fontsize=12)
    ax.legend(fontsize=10, loc='upper left')
    ax.grid(axis='y', alpha=0.3)

    y_max = max(max(results[g].values()) for g in groups)
    ax.set_ylim(0, y_max * 1.3)

    # Insight box
    ax.text(0.98, 0.95,
            'OrbitMLP = group-informed weight sharing\n'
            'UniformMLP = standard shared weights\n'
            'RandomMLP = random orbit assignment',
            transform=ax.transAxes, fontsize=8, ha='right', va='top',
            bbox=dict(boxstyle='round', facecolor='#F0F0F0', alpha=0.8))

    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'fig10_group_comparison.png'))
    fig.savefig(os.path.join(OUT, 'fig10_group_comparison.svg'))
    plt.close()
    print('Saved fig10_group_comparison')


# ═══════════════════════════════════════════════════════════════════
# Chart 11: Orbit Structure Visualization
# ═══════════════════════════════════════════════════════════════════

def plot_orbit_visualization(group_name, orbit_ids, N, n_cube=None):
    """Visualize orbit structure. For 2D groups show colored grid, for 3D show slices.

    Args:
        group_name: str, one of 'C_n', 'D_n', 'S_n', 'O_h'
        orbit_ids: [N] tensor/array of orbit indices (0..K-1)
        N: total number of positions
        n_cube: int, cube side length (required for O_h to interpret 3D layout)

    Saves fig11_orbit_visualization.png and .svg
    """
    orbit_ids = np.asarray(orbit_ids)
    K = int(orbit_ids.max()) + 1

    # Colormap: tab20 for up to 20 orbits, gist_rainbow for more
    cmap = plt.cm.tab20 if K <= 20 else plt.cm.gist_rainbow

    if group_name == 'O_h' and n_cube is not None:
        # ── 3D visualization: XY and XZ slices through the cube ─────
        n = n_cube
        vol = orbit_ids.reshape(n, n, n)  # z-major ordering

        fig, axes = plt.subplots(2, 3, figsize=(13, 8))
        fig.subplots_adjust(right=0.90, wspace=0.35, hspace=0.40)

        slice_z = [0, n // 2, n - 1]
        titles_top = [f'z=0 (bottom)', f'z={n // 2} (middle)', f'z={n - 1} (top)']

        for col, (z, title) in enumerate(zip(slice_z, titles_top)):
            # XY slice at fixed z
            ax = axes[0, col]
            im = ax.imshow(vol[z, :, :], cmap=cmap, origin='lower',
                           aspect='equal', vmin=0, vmax=K - 1)
            ax.set_title(title, fontsize=10)
            ax.set_xlabel('x')
            ax.set_ylabel('y')

        slice_y = [0, n // 2, n - 1]
        titles_bot = [f'y=0', f'y={n // 2} (middle)', f'y={n - 1}']

        for col, (y_val, title) in enumerate(zip(slice_y, titles_bot)):
            # XZ slice at fixed y
            ax = axes[1, col]
            ax.imshow(vol[:, y_val, :], cmap=cmap, origin='lower',
                      aspect='equal', vmin=0, vmax=K - 1)
            ax.set_title(title, fontsize=10)
            ax.set_xlabel('x')
            ax.set_ylabel('z')

        fig.suptitle(r'$O_h({}^3)$ Orbit Structure — {} orbits, {} positions'.format(n, K, N),
                     fontsize=12, fontweight='bold')

        # Shared colorbar
        cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
        fig.colorbar(im, cax=cbar_ax, label='Orbit ID')

    else:
        # ── 1D / 2D visualization ───────────────────────────────────
        fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

        # Panel 1: Spatial layout colored by orbit
        ax = axes[0]

        if group_name == 'D_n':
            # Polygon layout for dihedral
            angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
            radius = 1.0
            x_pos = radius * np.cos(angles)
            y_pos = radius * np.sin(angles)

            scatter = ax.scatter(x_pos, y_pos, c=orbit_ids, cmap=cmap, s=220,
                                 edgecolors='black', linewidths=0.8, zorder=5,
                                 vmin=0, vmax=K - 1)
            # Label vertices
            for i, (xp, yp) in enumerate(zip(x_pos, y_pos)):
                ax.annotate(str(i), (xp, yp), textcoords="offset points",
                            xytext=(0, 8), fontsize=7, ha='center', color='#555555')
            ax.set_aspect('equal')
            ax.set_xlim(-1.35, 1.35)
            ax.set_ylim(-1.35, 1.35)
            ax.set_title(f'{group_name} — {N} vertices, {K} orbit(s)\n'
                         f'(circular layout)')
            ax.axis('off')
            cbar = fig.colorbar(scatter, ax=ax, label='Orbit ID',
                                fraction=0.046, pad=0.04)

        elif group_name == 'S_n':
            # Ordered positions with orbit coloring
            for i in range(N):
                ax.axvspan(i - 0.45, i + 0.45, 0, 1,
                           facecolor=cmap(orbit_ids[i] / max(1, K - 1)),
                           edgecolor='black', linewidth=0.5)
                ax.text(i, 0.5, f'{orbit_ids[i]}', ha='center', va='center',
                        fontsize=9, fontweight='bold',
                        color='white' if orbit_ids[i] < K / 2 else '#333333')
            ax.set_xlim(-0.5, N - 0.5)
            ax.set_ylim(0, 1)
            ax.set_yticks([])
            ax.set_xlabel('Position Index')
            ax.set_title(f'{group_name} — {N} positions, {K} orbit(s)')

        else:
            # C_n and other 1D groups: bar-style per position
            norm = plt.Normalize(0, max(1, K - 1))
            colors = cmap(norm(orbit_ids))
            ax.bar(range(N), [1] * N, color=colors, edgecolor='white',
                   linewidth=0.5, width=0.85)
            # Label orbit IDs
            for i, oid in enumerate(orbit_ids):
                ax.text(i, 0.5, str(oid), ha='center', va='center',
                        fontsize=8, fontweight='bold',
                        color='white' if oid < K / 2 else '#333333')
            ax.set_xlabel('Position Index')
            ax.set_yticks([])
            ax.set_title(f'{group_name} — {N} positions, {K} orbits')

        # Panel 2: Orbit size distribution
        ax = axes[1]
        unique_ids, counts = np.unique(orbit_ids, return_counts=True)
        colors_bar = cmap(unique_ids / max(1, K - 1)) if K > 1 else [cmap(0)] * len(unique_ids)
        ax.bar(unique_ids, counts, color=colors_bar, edgecolor='black',
               linewidth=0.5, width=0.7)
        ax.set_xlabel('Orbit ID')
        ax.set_ylabel('Number of Positions')
        ax.set_title(f'Orbit Size Distribution (K={K})')
        ax.grid(axis='y', alpha=0.3)

        # Mean size annotation
        mean_size = counts.mean()
        ax.axhline(y=mean_size, color='#333333', linestyle='--',
                   linewidth=0.8, alpha=0.5)
        ax.text(len(unique_ids) - 0.5, mean_size + 0.5,
                f'mean={mean_size:.1f}', fontsize=8, ha='right',
                color='#666666')

        # Uniform-sharing reference (single orbit)
        if K == 1:
            ax.text(0.5, 0.95,
                    'Single orbit →\nuniform weight sharing\nis group-optimal',
                    transform=ax.transAxes, fontsize=9, ha='center', va='top',
                    color=C_UNIFORM, fontweight='bold',
                    bbox=dict(boxstyle='round', facecolor='#EBFFEB', alpha=0.9))

    if group_name == 'O_h' and n_cube is not None:
        # Already handled with subplots_adjust above
        pass
    else:
        fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'fig11_orbit_visualization.png'))
    fig.savefig(os.path.join(OUT, 'fig11_orbit_visualization.svg'))
    plt.close()
    print('Saved fig11_orbit_visualization')


# ═══════════════════════════════════════════════════════════════════
# Test runner with placeholder data
# ═══════════════════════════════════════════════════════════════════

def run_all_plots():
    """Run all group plots with placeholder data (for testing)."""
    print("=" * 60)
    print("Generating group comparison plots...")
    print("=" * 60)

    # ── Placeholder results for group comparison ──────────────────
    placeholder_results = {
        'C_n':  {'OrbitMLP': 78.0, 'UniformMLP': 52.0, 'RandomMLP': 35.0},
        'D_n':  {'OrbitMLP': 82.0, 'UniformMLP': 55.0, 'RandomMLP': 38.0},
        'S_n':  {'OrbitMLP': 65.0, 'UniformMLP': 60.0, 'RandomMLP': 40.0},
        'O_h':  {'OrbitMLP': 88.0, 'UniformMLP': 58.0, 'RandomMLP': 42.0},
    }
    plot_group_comparison(placeholder_results)

    # ── Placeholder orbit visualizations ──────────────────────────

    # C_8: step=2 → 2 orbits of size 4 (even/odd positions)
    orbit_ids_c8 = np.array([0, 1, 0, 1, 0, 1, 0, 1], dtype=int)
    plot_orbit_visualization('C_n', orbit_ids_c8, 8)

    # D_6: all 6 vertices in 1 orbit under dihedral action
    orbit_ids_d6 = np.zeros(6, dtype=int)
    plot_orbit_visualization('D_n', orbit_ids_d6, 6)

    # S_5: all 5 positions in 1 orbit (fully transitive)
    orbit_ids_s5 = np.zeros(5, dtype=int)
    plot_orbit_visualization('S_n', orbit_ids_s5, 5)

    # O_h(3**3): octahedral group on 3x3x3 cube
    # Orbit structure by Chebyshev distance from center
    n = 3
    orbit_ids_oh = np.zeros(n ** 3, dtype=int)
    center = (n - 1) / 2.0  # = 1.0
    for z in range(n):
        for y in range(n):
            for x in range(n):
                idx = z * n * n + y * n + x
                # Chebyshev distance from center determines orbit class
                dist = max(abs(x - center), abs(y - center), abs(z - center))
                if dist <= 0.5:
                    orbit_ids_oh[idx] = 0  # center (1 position)
                elif dist <= 1.0:
                    orbit_ids_oh[idx] = 1  # face centers (6 positions)
                elif dist <= 1.5:
                    orbit_ids_oh[idx] = 2  # edge centers (12 positions)
                else:
                    orbit_ids_oh[idx] = 3  # corners (8 positions)
    plot_orbit_visualization('O_h', orbit_ids_oh, 27, n_cube=3)

    print(f'\nAll group plots saved to: {OUT}')


if __name__ == '__main__':
    run_all_plots()
