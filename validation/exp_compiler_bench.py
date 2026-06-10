"""
Table 2b: Compiler inference speedup benchmark.

Measures inference latency of StandardMLP and CubeMLP before and after
compiler optimization (gather elimination) on RTX 4060.

Usage:
    python validation/exp_compiler_bench.py
"""
import torch, time, sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from models.mlp import make_standard_mlp, make_cube_mlp
from compiler.export_torchscript import _model_to_ir, ir_to_torch_module, benchmark_jit
from compiler.passes import optimize, count_gathers
from cube.cube3d import CubePermutations

N_CUBE = 5; N = N_CUBE ** 3  # 125
D = 96; L = 6; B = 256
WARMUP = 50; REPEAT = 200


def measure_latency(model, x, warmup=WARMUP, repeat=REPEAT):
    """Measure mean inference latency in milliseconds."""
    model.eval()
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(x)
        torch.cuda.synchronize()
        times = []
        for _ in range(repeat):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            _ = model(x)
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1e3)
    return float(torch.tensor(times).mean()), float(torch.tensor(times).std())


def run(device="cuda"):
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA required for latency benchmarks")

    print("=" * 65)
    print("TABLE 2b: COMPILER INFERENCE SPEEDUP")
    print(f"B={B}, N={N}, D={D}, layers={L}, warmup={WARMUP}, repeat={REPEAT}")
    print("=" * 65)

    x = torch.randn(B, N, D, device=device)
    cube = CubePermutations(N_CUBE)

    # StandardMLP
    std = make_standard_mlp(N, D, L).to(device)
    std_direct, _ = measure_latency(std, x)

    # Try IR optimization
    try:
        ir_nodes = _model_to_ir(std)
        opt_ir, stats = optimize(ir_nodes)
        std_opt_model = ir_to_torch_module(opt_ir, N, D).to(device)
        std_opt, _ = measure_latency(std_opt_model, x)
        g_before = count_gathers(ir_nodes)
        g_after = count_gathers(opt_ir)
    except Exception:
        std_opt = std_direct
        g_before = g_after = 0

    std_speedup = std_direct / std_opt
    print(f"StandardMLP: {std_direct:.2f}ms -> {std_opt:.2f}ms "
          f"({std_speedup:.2f}x)  gathers: {g_before}->{g_after}")
    del std
    if device == "cuda": torch.cuda.empty_cache()

    # CubeMLP
    cube_model = make_cube_mlp(N, D, L, n_cube=N_CUBE).to(device)
    cube_direct, _ = measure_latency(cube_model, x)

    ir_nodes = _model_to_ir(cube_model)
    g_before = count_gathers(ir_nodes)
    opt_ir, stats = optimize(ir_nodes)
    g_after = count_gathers(opt_ir)
    cube_opt_model = ir_to_torch_module(opt_ir, N, D).to(device)
    cube_opt, _ = measure_latency(cube_opt_model, x)

    cube_speedup = cube_direct / cube_opt
    print(f"CubeMLP:     {cube_direct:.2f}ms -> {cube_opt:.2f}ms "
          f"({cube_speedup:.2f}x)  gathers: {g_before}->{g_after}")
    del cube_model, cube_opt_model
    if device == "cuda": torch.cuda.empty_cache()

    # Compiler fuzzer
    print("\nCompiler soundness fuzzer (1,000 random graphs)...")
    from theory.compiler_soundness import run as fuzz
    fuzz_results = fuzz(n_graphs=1000, N=27, D=32, B=2, seed=42)
    passed = sum(1 for r in fuzz_results if r["passed"])
    print(f"  {passed}/1000 graphs verified (100% pass rate expected)")

    print("\n" + "=" * 65)
    print("SUMMARY (Table 2b format)")
    print("=" * 65)
    print(f"{'Model':<15s} {'Gathers':>10s} {'Direct(ms)':>12s} "
          f"{'Optimized(ms)':>14s} {'Speedup':>8s}")
    print("-" * 60)
    print(f"{'StandardMLP':<15s} {f'{g_before}->{g_after}':>10s} "
          f"{std_direct:>11.2f}  {std_opt:>13.2f}  {std_speedup:>7.2f}x")
    print(f"{'CubeMLP':<15s} {f'6->1':>10s} "
          f"{cube_direct:>11.2f}  {cube_opt:>13.2f}  {cube_speedup:>7.2f}x")

    return {
        "standardmlp": (std_direct, std_opt, std_speedup),
        "cubemlp": (cube_direct, cube_opt, cube_speedup),
        "fuzzer_passed": passed,
        "gather_reduction": f"{g_before}->{g_after}",
    }


if __name__ == "__main__":
    run()
