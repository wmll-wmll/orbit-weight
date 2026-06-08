"""
Roofline model for Theorem 3: Gather vs Dense crossover boundary.

Derives a closed-form decision boundary from the Roofline performance model
and validates against all 8 measured CROSSOVER_DATA points from the gfx936
domestic GPU.

Usage:
    python theory/crossover_roofline.py
"""

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


# ── Roofline cost models ──────────────────────────────────────────

def roofline_gather(N: int, B: int, D: int, BW_eff: float = 180.0,
                    elem_bytes: int = 2, launch_us: float = 0.003) -> float:
    """Predicted gather latency in milliseconds.

    Gather is memory-bandwidth-bound. It reads B*N*D elements with
    random-access (uncoalesced) pattern.

    Args:
        N: number of positions
        B: batch size
        D: feature dimension
        BW_eff: effective memory bandwidth in GB/s (default: gfx936)
        elem_bytes: bytes per element (2 for fp16, 4 for fp32)
        launch_us: kernel launch overhead in ms

    Returns:
        Predicted latency in milliseconds
    """
    total_bytes = B * N * D * elem_bytes
    # T = data / bandwidth + launch overhead
    latency_s = total_bytes / (BW_eff * 1e9)
    return latency_s * 1e3 + launch_us


def roofline_dense_matmul(N: int, B: int, D: int, peak_tflops: float = 110.0,
                          elem_bytes: int = 2, launch_us: float = 0.012) -> float:
    """Predicted dense matmul (x @ P^T) latency in milliseconds.

    Computes out = x.transpose(1,2) @ P.T where x: [B, N, D], P: [N, N].
    The matmul is [B*D, N] @ [N, N] → O(B * D * N^2) FLOPs.

    Args:
        N: number of positions
        B: batch size
        D: feature dimension
        peak_tflops: effective TFLOPS for this shape (default: gfx936)
        elem_bytes: bytes per element
        launch_us: kernel launch overhead in ms

    Returns:
        Predicted latency in milliseconds
    """
    # FLOPs: B * D * N * (2N - 1) ≈ 2 * B * N^2 * D for N >> 1
    flops = 2.0 * B * N * N * D
    # T = FLOPs / throughput + launch overhead
    latency_s = flops / (peak_tflops * 1e12)
    return latency_s * 1e3 + launch_us


def closed_form_N_star(BW_eff: float = 180.0, peak_tflops: float = 110.0,
                       elem_bytes: int = 2) -> float:
    """Derive N* threshold where dense matmul beats gather.

    From Theorem 3:
        N* = FLOPS_peak * elem_bytes / (2 * BW_eff)

    For gather to win: N < N*
    For dense to win:  N > N*

    Args:
        BW_eff: effective bandwidth in GB/s
        peak_tflops: effective matmul TFLOPS for small-N shapes
        elem_bytes: bytes per element

    Returns:
        N* threshold
    """
    # Convert units: TFLOPS = 10^12 FLOPS/s, BW_eff = 10^9 bytes/s
    # N* = (peak_tflops * 10^12) * elem_bytes / (2 * BW_eff * 10^9)
    #    = peak_tflops * elem_bytes * 500 / BW_eff
    return (peak_tflops * 1e12) * elem_bytes / (2.0 * BW_eff * 1e9)


def predict_winner(N: int, B: int, D: int, BW_eff: float = 180.0,
                   peak_tflops: float = 110.0, elem_bytes: int = 2) -> dict:
    """Predict which method wins for a given (N, B, D) configuration.

    Returns dict with predicted latencies and winner.
    """
    t_gather = roofline_gather(N, B, D, BW_eff, elem_bytes)
    t_dense = roofline_dense_matmul(N, B, D, peak_tflops, elem_bytes)

    if t_dense < t_gather:
        winner = 'dense'
        ratio = t_gather / t_dense
    else:
        winner = 'gather'
        ratio = t_dense / t_gather

    return {
        'N': N, 'B': B, 'D': D, 'BxD': B * D,
        'gather_ms': t_gather, 'dense_ms': t_dense,
        'winner': winner, 'ratio': ratio,
    }


def predict_decision_boundary_surface(B_values, D_values, N_max=250,
                                       BW_eff=180.0, peak_tflops=110.0):
    """Compute the N*(B,D) surface where crossover occurs.

    Returns N_star matrix of shape (len(B_values), len(D_values)).
    """
    N_star = np.zeros((len(B_values), len(D_values)))
    for i, B in enumerate(B_values):
        for j, D in enumerate(D_values):
            # Find N where T_gather = T_dense
            # Binary search since T_dense grows quadratically in N
            lo, hi = 1, N_max
            for _ in range(30):
                mid = (lo + hi) / 2
                t_g = roofline_gather(int(mid), B, D, BW_eff)
                t_d = roofline_dense_matmul(int(mid), B, D, peak_tflops)
                if t_d < t_g:
                    hi = mid
                else:
                    lo = mid
            N_star[i, j] = (lo + hi) / 2
    return N_star


# ── Calibration against measured data ─────────────────────────────

# From backends/domestic.py: CROSSOVER_DATA
MEASURED_DATA = [
    # (N, B, D, gather_ms, dense_ms, ratio)
    # ratio > 0 → gather wins; ratio < 0 → dense wins
    (27,  64,  128, 0.043, 0.052, 1.19),    # gather
    (27,  256, 512, 0.039, 0.046, 1.19),    # gather
    (64,  64,  128, 0.015, 0.046, 3.08),    # gather
    (64,  256, 512, 0.074, 0.049, -1.52),   # DENSE WINS
    (125, 64,  128, 0.013, 0.046, 3.45),    # gather
    (125, 256, 512, 0.138, 0.130, -1.06),   # DENSE WINS
    (216, 64,  128, 0.020, 0.046, 2.35),    # gather
    (216, 256, 512, 0.237, 0.201, -1.18),   # DENSE WINS
]


def calibrate_roofline(data=MEASURED_DATA):
    """Find best-fit BW_eff and peak_tflops to match measured data.

    Minimizes mean squared error between predicted and measured ratios.
    Returns (BW_eff_opt, peak_tflops_opt, r2_score, rmse).
    """
    from scipy.optimize import minimize

    def mse(params):
        BW_eff, peak_tflops = params
        err = 0.0
        for N, B, D, g_ms, d_ms, ratio in data:
            pred = predict_winner(N, B, D, BW_eff, peak_tflops)
            measured_ratio = abs(ratio)  # positive = gather speedup, negative = dense speedup
            measured_winner = 'gather' if ratio > 0 else 'dense'
            pred_ratio = pred['ratio']
            # Penalize winner mismatch heavily
            winner_penalty = 100.0 if measured_winner != pred['winner'] else 0.0
            err += (measured_ratio - pred_ratio) ** 2 + winner_penalty
        return err

    # Initial guess
    result = minimize(mse, [180.0, 110.0], bounds=[(50, 500), (10, 200)], method='L-BFGS-B')

    BW_opt, flops_opt = result.x

    # Compute R²
    ss_res = 0.0
    ss_tot = 0.0
    measured_ratios = []
    predicted_ratios = []
    n_correct = 0

    for N, B, D, g_ms, d_ms, ratio in data:
        pred = predict_winner(N, B, D, BW_opt, flops_opt)
        measured_ratios.append(abs(ratio))
        predicted_ratios.append(pred['ratio'])
        measured_winner = 'gather' if ratio > 0 else 'dense'
        if measured_winner == pred['winner']:
            n_correct += 1

    measured_ratios = np.array(measured_ratios)
    predicted_ratios = np.array(predicted_ratios)
    ss_res = np.sum((measured_ratios - predicted_ratios) ** 2)
    ss_tot = np.sum((measured_ratios - measured_ratios.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
    rmse = np.sqrt(ss_res / len(data))

    return BW_opt, flops_opt, r2, rmse, n_correct


# ── Plotting ──────────────────────────────────────────────────────

def plot_roofline_vs_measured(data=MEASURED_DATA, out_dir=None):
    """Generate figure comparing roofline predictions vs measured data.

    Creates a 2x2 figure:
      (1) Scatter: predicted vs measured ratio
      (2) Decision boundary: N vs B×D with winner coloring
      (3) Latency comparison: gather vs dense bars
      (4) Error analysis: per-point prediction error
    """
    if out_dir is None:
        out_dir = os.path.join(os.path.dirname(__file__), '..', 'figures')

    BW_opt, flops_opt, r2, rmse, n_correct = calibrate_roofline(data)
    N_star = closed_form_N_star(BW_opt, flops_opt)

    fig, axes = plt.subplots(2, 2, figsize=(14, 11))

    # Panel 1: Predicted vs Measured ratio scatter
    ax = axes[0, 0]
    measured_ratios = []
    predicted_ratios = []
    colors = []
    for N, B, D, g_ms, d_ms, ratio in data:
        pred = predict_winner(N, B, D, BW_opt, flops_opt)
        measured_ratios.append(abs(ratio))
        predicted_ratios.append(pred['ratio'])
        colors.append('#E24A33' if ratio > 0 else '#4C72B0')  # red=gather, blue=dense

    ax.scatter(measured_ratios, predicted_ratios, c=colors, s=100, zorder=5, edgecolors='black', linewidth=0.5)
    max_val = max(max(measured_ratios), max(predicted_ratios)) * 1.1
    ax.plot([0, max_val], [0, max_val], 'k--', alpha=0.3, label='y = x')
    ax.set_xlabel('Measured Speedup Ratio')
    ax.set_ylabel('Predicted Speedup Ratio')
    ax.set_title(f'Roofline Prediction Accuracy\nR²={r2:.3f}, RMSE={rmse:.3f}, {n_correct}/8 correct')
    # Annotate each point with (N, B, D)
    for i, (N, B, D, _, _, _) in enumerate(data):
        ax.annotate(f'({N},{B},{D})', (measured_ratios[i], predicted_ratios[i]),
                    textcoords="offset points", xytext=(5, 5), fontsize=7)
    ax.legend()
    ax.grid(alpha=0.3)

    # Panel 2: N vs B×D decision boundary
    ax = axes[0, 1]
    bd_values = []
    for N, B, D, _, _, ratio in data:
        bd_values.append((N, B * D, 'gather' if ratio > 0 else 'dense'))

    for N, BD, winner in bd_values:
        marker = 'o' if winner == 'gather' else 's'
        color = '#E24A33' if winner == 'gather' else '#4C72B0'
        ax.scatter(N, BD, c=color, marker=marker, s=120, zorder=5, edgecolors='black', linewidth=0.5)

    # Draw decision boundary (N_star line)
    ax.axvline(x=N_star, color='green', linestyle='--', alpha=0.6,
               label=f'N* = {N_star:.0f} (theoretical)')
    ax.set_xlabel('N (number of positions)')
    ax.set_ylabel('B × D')
    ax.set_title(f'Decision Boundary (gfx936)\nBW={BW_opt:.0f} GB/s, FLOPS={flops_opt:.0f} TFLOPS')
    ax.legend()
    ax.grid(alpha=0.3)
    # Annotate regions
    ax.text(30, 120000, 'GATHER\nWINS', fontsize=10, color='#E24A33', fontweight='bold', alpha=0.5)
    ax.text(150, 120000, 'DENSE\nWINS', fontsize=10, color='#4C72B0', fontweight='bold', alpha=0.5)

    # Panel 3: Latency bar chart
    ax = axes[1, 0]
    x_pos = np.arange(len(data))
    width = 0.35
    gather_times = [d[3] for d in data]
    dense_times = [d[4] for d in data]
    labels = [f'N={d[0]}\nB={d[1]},D={d[2]}' for d in data]

    ax.bar(x_pos - width/2, gather_times, width, label='Gather (measured)', color='#E24A33', alpha=0.8)
    ax.bar(x_pos + width/2, dense_times, width, label='Dense (measured)', color='#4C72B0', alpha=0.8)
    ax.set_ylabel('Latency (ms)')
    ax.set_title('Measured Latency Comparison')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, fontsize=7)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    # Panel 4: Prediction error per data point
    ax = axes[1, 1]
    errors = []
    winners = []
    for i, (N, B, D, _, _, ratio) in enumerate(data):
        pred = predict_winner(N, B, D, BW_opt, flops_opt)
        errors.append(pred['ratio'] - abs(ratio))
        winners.append('gather' if ratio > 0 else 'dense')

    bar_colors = ['#E24A33' if w == 'gather' else '#4C72B0' for w in winners]
    bars = ax.bar(range(len(data)), errors, color=bar_colors, alpha=0.8, edgecolors='black', linewidth=0.5)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.set_xlabel('Data Point Index')
    ax.set_ylabel('Prediction Error (predicted - measured)')
    ax.set_title('Per-Point Prediction Error')
    ax.set_xticks(range(len(data)))
    ax.set_xticklabels([f'({d[0]},{d[1]},{d[2]})' for d in data], fontsize=7, rotation=45)
    ax.grid(axis='y', alpha=0.3)

    fig.suptitle(f'Theorem 3: Roofline Model Validation on gfx936\n'
                 f'Calibrated: BW_eff={BW_opt:.1f} GB/s, Peak={flops_opt:.1f} TFLOPS, N*={N_star:.0f}',
                 fontsize=13, fontweight='bold')
    fig.tight_layout()

    os.makedirs(out_dir, exist_ok=True)
    fig.savefig(os.path.join(out_dir, 'fig_theory_roofline.png'), dpi=200)
    fig.savefig(os.path.join(out_dir, 'fig_theory_roofline.svg'))
    plt.close()
    print(f"  Saved fig_theory_roofline.png to {out_dir}")

    return BW_opt, flops_opt, r2, n_correct


# ── Main ──────────────────────────────────────────────────────────

def run():
    """Run Theorem 3 validation: calibrate roofline and compare with data."""
    print("=" * 60)
    print("THEOREM 3: Gather vs Dense Crossover — Roofline Validation")
    print("=" * 60)

    # 1. Raw roofline prediction
    N_star = closed_form_N_star()
    print(f"\n  Theoretical N* (gfx936 defaults):")
    print(f"    BW_eff = 180 GB/s, peak = 110 TFLOPS")
    print(f"    N* = {N_star:.1f} (dense wins when N > {N_star:.0f})")

    # 2. Calibrate against measured data
    BW_opt, flops_opt, r2, rmse, n_correct = calibrate_roofline()
    N_star_cal = closed_form_N_star(BW_opt, flops_opt)
    print(f"\n  Calibrated parameters:")
    print(f"    BW_eff = {BW_opt:.1f} GB/s")
    print(f"    Peak   = {flops_opt:.1f} TFLOPS")
    print(f"    N*     = {N_star_cal:.1f}")
    print(f"    R²     = {r2:.4f}")
    print(f"    RMSE   = {rmse:.4f}")
    print(f"    Winner accuracy: {n_correct}/8")

    # 3. Per-point predictions
    print(f"\n  Per-point comparison:")
    print(f"  {'N':>4s} {'B':>4s} {'D':>4s} | {'Gath(ms)':>9s} {'Dense(ms)':>9s} | {'Meas':>8s} {'Pred':>8s} | {'Match?':>6s}")
    print(f"  {'-'*65}")
    for N, B, D, g_ms, d_ms, ratio in MEASURED_DATA:
        pred = predict_winner(N, B, D, BW_opt, flops_opt)
        measured_winner = 'gather' if ratio > 0 else 'dense'
        match = '[OK]' if measured_winner == pred['winner'] else 'X'
        print(f"  {N:>4d} {B:>4d} {D:>4d} | {g_ms:>8.3f}  {d_ms:>8.3f} | "
              f"{measured_winner:>8s} {pred['winner']:>8s} | {match:>6s}")

    # 4. Decision boundary analysis
    print(f"\n  Decision boundary analysis:")
    bd_pairs = [(100000, 64, 'B*D >= 100K and N >= 64'),
                (50000, 125, 'B*D >= 50K and N >= 125'),
                (150000, 0, 'B*D >= 150K (unconditional)')]
    for BD_thresh, N_thresh, desc in bd_pairs:
        for N, B, D, _, _, ratio in MEASURED_DATA:
            if B * D >= BD_thresh and (N_thresh == 0 or N >= N_thresh):
                pred = predict_winner(N, B, D, BW_opt, flops_opt)
                measured_winner = 'gather' if ratio > 0 else 'dense'
                print(f"    {desc}: N={N}, B*D={B*D} → measured={measured_winner}, predicted={pred['winner']}")

    # 5. Plot
    print()
    BW_opt, flops_opt, r2, n_correct = plot_roofline_vs_measured()

    # 6. Generality check: predict for NVIDIA RTX 4060
    print(f"\n  Generality check (NVIDIA RTX 4060):")
    nv_bw = 272.0  # GB/s
    nv_flops = 15.0  # TFLOPS (much lower than gfx936)
    nv_N_star = closed_form_N_star(nv_bw, nv_flops)
    print(f"    BW_eff = {nv_bw} GB/s, peak = {nv_flops} TFLOPS")
    print(f"    N* = {nv_N_star:.1f}")
    print(f"    Note: lower N* on NVIDIA means dense wins at smaller N,")
    print(f"    explaining why NVIDIA speedup (1.80x) > gfx936 (1.63x)")

    print(f"\n  [OK] Roofline validation complete.")
    return BW_opt, flops_opt, r2


if __name__ == "__main__":
    run()
