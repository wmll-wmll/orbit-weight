"""Benchmark: permutation strategy latency across backends.

Compares 3 permutation strategies:
  - gather: torch.gather (fast on NVIDIA, slow on MUXI)
  - dense: matmul(x, P^T) (competitive at small N)
  - sparse: sparse.mm(P, x_flat) (competitive for sparse P)

Goal: Find crossover points where matmul-based approaches beat gather,
especially under MUXI assumptions (gather penalty = 1.5x, 2x, 3x).
"""

import torch
import numpy as np
import time
from validation.runner import print_header
from backends import nvidia
from cube.cube3d import CubePermutations
from cube.perm_matrix import PermutationMatrix


def benchmark_fn(fn, *args, n_warmup=30, n_repeat=100):
    """Benchmark a function call using CUDA events. Returns (mean_ms, std_ms)."""
    # Warmup
    for _ in range(n_warmup):
        fn(*args)
    torch.cuda.synchronize()

    starter = torch.cuda.Event(enable_timing=True)
    ender = torch.cuda.Event(enable_timing=True)
    times = []
    for _ in range(n_repeat):
        starter.record()
        fn(*args)
        ender.record()
        torch.cuda.synchronize()
        times.append(starter.elapsed_time(ender))

    return np.mean(times), np.std(times, ddof=1)


def make_perm_matrix(N, backend_fn, device="cuda"):
    """Create a permutation matrix (dense and sparse) from a cube rotation."""
    cube = CubePermutations(int(round(N ** (1/3))))
    perm = cube.get_rotation('U')
    # Dense permutation matrix [N, N]
    P = torch.zeros(N, N, device=device)
    P[torch.arange(N), perm] = 1.0
    P_sparse = P.to_sparse()
    return perm.to(device), P, P_sparse


def run(device: str = "cuda"):
    print_header("Permutation Strategy Latency Comparison")
    print("Comparing gather vs dense matmul vs sparse matmul across N and D.")
    print("MUXI backend = same torch ops (placeholder), with modeled penalty.")
    print()

    batch = 256
    n_cubes = [
        (3, 27, "3x3x3"),
        (4, 64, "4x4x4"),
        (5, 125, "5x5x5"),
        (6, 216, "6x6x6"),
    ]
    d_models = [64, 128, 256, 512]

    # Store all results
    results = []

    for n_cube, N, label in n_cubes:
        perm, P_dense, P_sparse = make_perm_matrix(N, nvidia.gather, device)
        print(f"\n--- {label} (N={N}) ---")
        print(f"  Permutation sparsity: {P_dense.sum().item():.0f}/{N} = {P_dense.sum().item()/N:.1f} non-zeros per row")
        print(f"  {'D':>5s} | {'gather':>10s} | {'dense':>10s} | {'sparse':>10s} | "
              f"{'dense/gather':>13s} | {'sparse/gather':>14s}")
        print(f"  {'':->5s}-+-{'':->10s}-+-{'':->10s}-+-{'':->10s}-+-{'':->13s}-+-{'':->14s}")

        for D in d_models:
            x = torch.randn(batch, N, D, device=device)

            # gather
            mu_g, sig_g = benchmark_fn(nvidia.gather, x, perm)

            # dense matmul
            mu_d, sig_d = benchmark_fn(nvidia.permute_dense, x, P_dense)

            # sparse matmul
            mu_s, sig_s = benchmark_fn(nvidia.permute_sparse, x, P_sparse)

            results.append({
                'N': N, 'D': D, 'label': label,
                'gather': mu_g, 'dense': mu_d, 'sparse': mu_s,
                'dense_ratio': mu_d / mu_g,
                'sparse_ratio': mu_s / mu_g,
            })

            print(f"  {D:>5d} | {mu_g:>7.3f}ms | {mu_d:>7.3f}ms | {mu_s:>7.3f}ms | "
                  f"{mu_d/mu_g:>11.1f}x | {mu_s/mu_g:>12.1f}x")

    # ═══════════════════════════════════════════════════════════════
    # Summary: crossover analysis
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'='*80}")
    print("CROSSOVER ANALYSIS: When does matmul beat gather?")
    print("=" * 80)
    print(f"  1.0x = same speed. <1.0x = matmul faster. >1.0x = gather faster.")
    print()
    print(f"  On NVIDIA (gather is fast):")
    print(f"  {'N':>5s} | {'D=64':>10s} {'D=128':>10s} {'D=256':>10s} {'D=512':>10s}")
    print(f"  {'':->5s}-+-{'':->10s}-{'':->10s}-{'':->10s}-{'':->10s}")
    for n_cube, N, label in n_cubes:
        print(f"  {N:>5d} |", end="")
        for D in d_models:
            r = [r for r in results if r['N'] == N and r['D'] == D][0]
            marker = " ***" if r['dense_ratio'] < 1.0 else ""
            print(f" {r['dense_ratio']:>9.2f}x{marker}", end="")
        print()

    # Modeled MUXI: gather is 2x slower
    print(f"\n  On MUXI (modeled: gather = 2x slower, matmul = same):")
    print(f"  {'N':>5s} | {'D=64':>10s} {'D=128':>10s} {'D=256':>10s} {'D=512':>10s}")
    print(f"  {'':->5s}-+-{'':->10s}-{'':->10s}-{'':->10s}-{'':->10s}")
    for n_cube, N, label in n_cubes:
        print(f"  {N:>5d} |", end="")
        for D in d_models:
            r = [r for r in results if r['N'] == N and r['D'] == D][0]
            muxi_ratio = r['dense'] / (r['gather'] * 2.0)  # modeled 2x gather penalty
            marker = " ***" if muxi_ratio < 1.0 else ""
            print(f" {muxi_ratio:>9.2f}x{marker}", end="")
        print()

    # MUXI penalty sweep
    print(f"\n  MUXI crossover threshold (smallest gather penalty where dense wins):")
    print(f"  {'N':>5s} | {'D=64':>12s} {'D=128':>12s} {'D=256':>12s} {'D=512':>12s}")
    print(f"  {'':->5s}-+-{'':->12s}-{'':->12s}-{'':->12s}-{'':->12s}")
    for n_cube, N, label in n_cubes:
        print(f"  {N:>5d} |", end="")
        for D in d_models:
            r = [r for r in results if r['N'] == N and r['D'] == D][0]
            # How much slower must gather be for dense to win?
            threshold = r['dense'] / r['gather']  # dense/gather ratio on NVIDIA
            # dense wins when gather_penalty * threshold < 1.0
            # gather_penalty > threshold means dense wins
            if threshold < 1.0:
                print(f" {'dense wins':>12s}", end="")
            else:
                print(f" {threshold:>9.1f}x needed", end="")
        print()

    return results


if __name__ == "__main__":
    run()
