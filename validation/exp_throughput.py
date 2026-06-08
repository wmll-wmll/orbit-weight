"""Experiment 3: Training throughput at scale.

Question: What's the real training throughput gap between StandardMLP and CubeMLP?

Hypothesis (revised from discovery): On NVIDIA, CubeMLP will be slower due to
gather overhead. But the gap should be small (<20%) because gather is applied
AFTER compute on contiguous data (fused approach).

The real value: on domestic GPUs where gather is less optimized, the same
"compute then gather" approach would show a BIGGER advantage.
"""

import torch
import numpy as np
from validation.runner import ExperimentRunner, print_header
from models.mlp import make_standard_mlp, make_cube_mlp


def run(device: str = "cuda"):
    runner = ExperimentRunner(device=device)
    N = 27

    print_header("EXPERIMENT 3: Training Throughput at Scale")
    print("Forward+backward+update throughput for realistic model sizes.")

    configs = [
        (128, 6, 256, "Small (~0.5M params)"),
        (256, 8, 128, "Medium (~2M params)"),
        (512, 12, 64, "Large (~12M params)"),
    ]

    print(f"\n{'Config':<25s} | {'Standard':>10s} | {'CubeMLP':>10s} | {'Ratio':>8s} | {'Winner'}")
    print("-" * 70)

    for d_model, n_layers, B, desc in configs:
        std = make_standard_mlp(N, d_model, n_layers).to(device)
        cube = make_cube_mlp(N, d_model, n_layers, n_cube=3).to(device)

        x = torch.randn(B, N, d_model, device=device)
        target = torch.randn(B, N, d_model, device=device)

        mu_std, _ = runner.measure_throughput(std, x, target)
        mu_cube, _ = runner.measure_throughput(cube, x, target)
        ratio = mu_std / mu_cube
        winner = "CubeMLP" if ratio > 1.0 else "StandardMLP"

        print(f"{desc:<25s} | {mu_std:>7.1f}ms | {mu_cube:>7.1f}ms | "
              f"{ratio:>6.2f}x | {winner}")

    return


if __name__ == "__main__":
    run()
