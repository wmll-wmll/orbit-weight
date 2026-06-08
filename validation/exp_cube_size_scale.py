"""Experiment 5: Throughput scaling across cube sizes.

Question: How does the StandardMLP vs CubeMLP throughput ratio change
as cube size grows (3³→4³→5³)?

Uses CUDA events for precise GPU timing, multiple runs with warmup.
"""

import torch
import numpy as np
from validation.runner import print_header
from models.mlp import make_standard_mlp, make_cube_mlp


def benchmark_model(model, x, target, n_warmup=50, n_repeat=100):
    """Benchmark forward+backward+update using CUDA events. Returns (mean_ms, std_ms)."""
    device = x.device
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)

    # Warmup
    for _ in range(n_warmup):
        opt.zero_grad()
        loss = torch.nn.functional.mse_loss(model(x), target)
        loss.backward()
        opt.step()
    torch.cuda.synchronize()

    # Measure with CUDA events
    times = []
    starter = torch.cuda.Event(enable_timing=True)
    ender = torch.cuda.Event(enable_timing=True)
    for _ in range(n_repeat):
        opt.zero_grad()
        starter.record()
        loss = torch.nn.functional.mse_loss(model(x), target)
        loss.backward()
        opt.step()
        ender.record()
        torch.cuda.synchronize()
        times.append(starter.elapsed_time(ender))

    return np.mean(times), np.std(times, ddof=1)


def run(device: str = "cuda"):
    cube_sizes = [
        (3, 27, "3x3x3 N=27"),
        (4, 64, "4x4x4 N=64"),
        (5, 125, "5x5x5 N=125"),
    ]
    d_models = [64, 128, 256]
    n_layers = 6
    batch = 256

    print_header("EXPERIMENT 5: Throughput Scaling vs Cube Size")
    print("Forward+backward+update (CUDA event timing, mean +/- std over 100 runs)")
    print(f"n_layers={n_layers}, batch={batch}")
    print()

    # Store all ratios for summary
    all_ratios = {}

    for d_model in d_models:
        n_params = n_layers * (d_model * d_model + d_model * 2) * 4 // 1024
        print(f"=== d_model={d_model} (~{n_params}K params) ===")
        print(f"{'Cube':<16s} {'N/D':>6s} | {'Standard':>18s} | {'CubeMLP':>18s} | {'Ratio':>8s} | {'Gap':>7s}")
        print("-" * 90)

        for n_cube, N, label in cube_sizes:
            torch.manual_seed(42)
            std_model = make_standard_mlp(N, d_model, n_layers).to(device)
            cube_model = make_cube_mlp(N, d_model, n_layers, n_cube=n_cube).to(device)

            x = torch.randn(batch, N, d_model, device=device)
            target = torch.randn(batch, N, d_model, device=device)

            mu_std, sig_std = benchmark_model(std_model, x, target)
            mu_cube, sig_cube = benchmark_model(cube_model, x, target)

            ratio = mu_std / mu_cube
            gap_pct = (1 - ratio) * 100  # positive = CubeMLP slower
            nd_ratio = N / d_model
            key = (d_model, N)
            all_ratios[key] = (ratio, gap_pct)

            print(f"{label:<16s} {nd_ratio:>5.2f} | {mu_std:>7.2f} +/-{sig_std:>5.2f}ms | "
                  f"{mu_cube:>7.2f} +/-{sig_cube:>5.2f}ms | {ratio:>7.3f}x | {gap_pct:>+5.1f}%")

            del std_model, cube_model
            torch.cuda.empty_cache()

        print()

    # Summary matrix
    print("=== Summary: Ratio (Standard/CubeMLP) Matrix ===")
    print(f"{'':<12s}", end="")
    for _, N, label in cube_sizes:
        print(f" | {label:<14s}", end="")
    print(f" | {'Mean':<8s}")
    print("-" * 65)

    for d_model in d_models:
        print(f"{'d_model='+str(d_model):<12s}", end="")
        ratios_for_d = []
        for _, N, _ in cube_sizes:
            ratio, gap = all_ratios[(d_model, N)]
            ratios_for_d.append(ratio)
            print(f" | {ratio:.3f}x ({gap:+.1f}%)", end="")
        mean_ratio = np.mean(ratios_for_d)
        print(f" | {mean_ratio:.3f}x")

    # Trend analysis
    print(f"\n=== Trend Analysis ===")
    for d_model in d_models:
        ratios_for_d = [all_ratios[(d_model, N)][0] for _, N, _ in cube_sizes]
        ns = [N for _, N, _ in cube_sizes]
        print(f"d_model={d_model}: ratios = {[f'{r:.3f}' for r in ratios_for_d]}, "
              f"N range: {ns[0]}->{ns[-1]}, "
              f"delta = {ratios_for_d[-1] - ratios_for_d[0]:+.3f}")

    return all_ratios


if __name__ == "__main__":
    run()
