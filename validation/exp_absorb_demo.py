"""Hard Metric 2: End-to-end Absorb Demo.

Proves that CubeMLP's permutation overhead can be eliminated at inference.

Pipeline:
  1. Build CubeMLP (6 layers, each: LN→Linear→GELU→gather(p_i))
  2. Convert to compiler IR → optimize (Reorder + Absorb + Chain)
  3. Extract optimized model (6 gathers → 1 gather at end)
  4. Verify output equivalence (must be exact)
  5. Zero-gather variant: drop final gather (for invariant heads)
  6. Benchmark all 4 variants vs StandardMLP

Key claim: After Absorb, CubeMLP inference = StandardMLP structure.
"""

import torch
import numpy as np
from validation.runner import print_header
from models.mlp import make_standard_mlp, make_cube_mlp, _GatherLayer
from compiler.ir import (
    GatherOp, ElementWiseOp, LinearOp, OpType,
    compose_permutations,
)
from compiler.passes import optimize, count_gathers, describe_graph


def model_to_ir(model: torch.nn.Sequential):
    """Convert a flat Sequential CubeMLP to compiler IR nodes.

    Pattern (flat): LN → Linear → GELU → _GatherLayer → LN → ...
    Each child is a single module.
    """
    nodes = []
    for child in model.children():
        if isinstance(child, torch.nn.LayerNorm):
            nodes.append(ElementWiseOp(OpType.LAYER_NORM, params={
                'weight': child.weight.data.clone(),
                'bias': child.bias.data.clone(),
            }))
        elif isinstance(child, torch.nn.Linear):
            nodes.append(LinearOp(
                weight=child.weight.data.clone(),
                bias=child.bias.data.clone() if child.bias is not None else None,
                per_position=False,
            ))
        elif isinstance(child, torch.nn.GELU):
            nodes.append(ElementWiseOp(OpType.GELU))
        elif isinstance(child, _GatherLayer):
            nodes.append(GatherOp(child.perm.clone()))
    return nodes


def ir_to_model(nodes, D: int):
    """Build a PyTorch Sequential model from IR nodes."""
    layers = []
    for node in nodes:
        if isinstance(node, ElementWiseOp):
            if node.op_type == OpType.LAYER_NORM:
                m = torch.nn.LayerNorm(D, elementwise_affine=True)
                if 'weight' in node.params:
                    m.weight.data.copy_(node.params['weight'])
                    m.bias.data.copy_(node.params['bias'])
                layers.append(m)
            elif node.op_type == OpType.GELU:
                layers.append(torch.nn.GELU())
            # ReLU and Dropout not used in this demo
        elif isinstance(node, LinearOp):
            m = torch.nn.Linear(D, D)
            m.weight.data.copy_(node.weight)
            if node.bias is not None:
                m.bias.data.copy_(node.bias)
            layers.append(m)
        elif isinstance(node, GatherOp):
            layers.append(_GatherLayer(node.perm))
    return torch.nn.Sequential(*layers)


def verify_equivalence(model_a, model_b, x):
    """Return (max_abs_diff, max_rel_diff)."""
    model_a.eval()
    model_b.eval()
    with torch.no_grad():
        out_a = model_a(x)
        out_b = model_b(x)
    max_diff = (out_a - out_b).abs().max().item()
    rel_denom = out_a.abs().max().item() + 1e-12
    rel_diff = (out_a - out_b).abs().max().item() / rel_denom
    return max_diff, rel_diff


def benchmark_forward(model, x, n_warmup=50, n_repeat=200):
    """Forward-only latency using CUDA events. Returns (mean_ms, std_ms)."""
    model.eval()
    with torch.no_grad():
        for _ in range(n_warmup):
            _ = model(x)
    torch.cuda.synchronize()

    starter = torch.cuda.Event(enable_timing=True)
    ender = torch.cuda.Event(enable_timing=True)
    times = []
    with torch.no_grad():
        for _ in range(n_repeat):
            starter.record()
            _ = model(x)
            ender.record()
            torch.cuda.synchronize()
            times.append(starter.elapsed_time(ender))

    return np.mean(times), np.std(times, ddof=1)


def count_gather_layers(model):
    """Count _GatherLayer instances in a PyTorch model."""
    return sum(1 for m in model.modules() if isinstance(m, _GatherLayer))


def run(device: str = "cuda"):
    N_CUBE = 5
    N = N_CUBE ** 3  # 125
    D = 128
    N_LAYERS = 6
    BATCH = 256

    print_header("HARD METRIC 2: Absorb — Zero Overhead at Inference")
    print(f"Config: {N_CUBE}^3={N} positions, d_model={D}, {N_LAYERS} layers, batch={BATCH}")
    print(f"Claim: Compiler reduces 6 gathers → 1 (or 0 for invariant head)")
    print()

    # ── Step 1: Build models ──────────────────────────────────
    torch.manual_seed(42)
    cube_model = make_cube_mlp(N, D, N_LAYERS, n_cube=N_CUBE).to(device)
    std_model = make_standard_mlp(N, D, N_LAYERS, dropout=0.0).to(device)

    # ── Step 2: IR → optimize → extract ───────────────────────
    ir_before = model_to_ir(cube_model)
    n_gathers_before = count_gathers(ir_before)

    ir_after, stats = optimize(ir_before)

    optimized_model = ir_to_model(ir_after, D).to(device)
    n_gathers_after = count_gathers(ir_after)

    # Build zero-gather variant (for permutation-invariant heads)
    ir_zero = [n for n in ir_after if not isinstance(n, GatherOp)]
    zero_model = ir_to_model(ir_zero, D).to(device)

    # Show IR transformation
    print("  IR optimization:")
    print(f"    Nodes before: {len(ir_before)}  ({describe_graph(ir_before)[:130]})")
    print(f"    Nodes after:  {len(ir_after)}  ({describe_graph(ir_after)[:130]})")
    print(f"    Gathers: {n_gathers_before} → {n_gathers_after}")
    print(f"    Pass stats: {stats}")
    print(f"    Zero-gather nodes: {len(ir_zero)}")

    # ── Step 3: Correctness ───────────────────────────────────
    x = torch.randn(BATCH, N, D, device=device)

    # 3a: Original CubeMLP vs Optimized (must be exact)
    max_diff, rel_diff = verify_equivalence(cube_model, optimized_model, x)
    equiv_holds = max_diff < 1e-4
    print(f"\n  Correctness verification:")
    print(f"    CubeMLP vs Optimized: max_abs={max_diff:.2e}, rel={rel_diff:.2e}  "
          f"{'PASS' if equiv_holds else 'FAIL'}")

    # 3b: Zero-gather model mean invariance
    out_opt = optimized_model(x)          # [B, N, D] with gather at end
    out_zero = zero_model(x)              # [B, N, D] without gather
    mean_diff = (out_opt.mean(dim=1) - out_zero.mean(dim=1)).abs().max().item()
    invariant_holds = mean_diff < 1e-4
    print(f"    Mean-invariance (pooled after gather vs pooled before): "
          f"diff={mean_diff:.2e}  {'PASS' if invariant_holds else 'FAIL'}")

    # 3c: Verify that the final gather maps output positions correctly
    # Get the composed gather from optimized model
    final_gather_node = [n for n in ir_after if isinstance(n, GatherOp)]
    if final_gather_node:
        perm_total = final_gather_node[0].perm
        # out_opt[b, i, :] should == out_zero[b, perm_total[i], :]
        out_opt_from_zero = torch.gather(
            out_zero, 1,
            perm_total.unsqueeze(0).unsqueeze(-1).expand(BATCH, -1, D).to(device)
        )
        gather_diff = (out_opt - out_opt_from_zero).abs().max().item()
        print(f"    Gather correctness (optimized vs manual gather of zero-model): "
              f"diff={gather_diff:.2e}  {'PASS' if gather_diff < 1e-4 else 'FAIL'}")

    # 3d: Structure check — zero-gather model should have identical structure to StandardMLP
    zero_types = [type(m).__name__ for m in zero_model.children()]
    std_types = [type(m).__name__ for m in std_model.children()]
    structure_match = zero_types == std_types
    print(f"    Structure match (zero-gather vs StandardMLP): "
          f"{'PASS' if structure_match else 'DIFF'}")
    if not structure_match:
        # Show first difference
        for j, (a, b) in enumerate(zip(zero_types, std_types)):
            if a != b:
                print(f"      First diff at pos {j}: {a} vs {b}")
                break

    # ── Step 4: Benchmark ─────────────────────────────────────
    print(f"\n  {'='*75}")
    print(f"  Inference Latency (forward only, {N_LAYERS} layers, D={D}, N={N})")
    print(f"  {'='*75}")
    print(f"  {'Model':<32s} | {'Latency':>16s} | {'Gathers':>8s} | {'Ratio vs Std':>13s}")
    print(f"  {'-'*75}")

    models_to_bench = [
        ("StandardMLP (baseline)", std_model),
        ("CubeMLP (original, 6 gathers)", cube_model),
        ("CubeMLP (optimized, 1 gather)", optimized_model),
        ("CubeMLP (absorbed, 0 gathers)", zero_model),
    ]

    results = {}
    for name, model in models_to_bench:
        mu, sig = benchmark_forward(model, x)
        n_g = count_gather_layers(model)
        results[name] = (mu, sig, n_g)

    base_mu = results["StandardMLP (baseline)"][0]
    for name, (mu, sig, n_g) in results.items():
        ratio = mu / base_mu
        print(f"  {name:<32s} | {mu:>7.3f} ± {sig:>5.3f}ms | {n_g:>7d} | {ratio:>11.3f}x")

    # ── Step 5: Key numbers ───────────────────────────────────
    cube_mu = results["CubeMLP (original, 6 gathers)"][0]
    opt_mu = results["CubeMLP (optimized, 1 gather)"][0]
    zero_mu = results["CubeMLP (absorbed, 0 gathers)"][0]

    gather_cost_6 = (cube_mu - zero_mu) / zero_mu * 100
    gather_cost_1 = (opt_mu - zero_mu) / zero_mu * 100
    zero_vs_std = (zero_mu - base_mu) / base_mu * 100

    print(f"\n  {'='*75}")
    print(f"  VERDICT")
    print(f"  {'='*75}")
    print(f"  StandardMLP baseline:        {base_mu:.3f}ms")
    print(f"  CubeMLP (6 gathers):         {cube_mu:.3f}ms  (+{gather_cost_6:.1f}% vs zero)")
    print(f"  CubeMLP (optimized, 1 gath): {opt_mu:.3f}ms  (+{gather_cost_1:.1f}% vs zero)")
    print(f"  CubeMLP (absorbed, 0 gath):  {zero_mu:.3f}ms  ({zero_vs_std:+.1f}% vs StandardMLP)")
    print()

    metric2_pass = abs(zero_vs_std) < 3.0 and equiv_holds and structure_match
    if metric2_pass:
        print(f"  >>> HARD METRIC 2: PASS <<<")
        print(f"  Absorb eliminates ALL gather overhead. CubeMLP → StandardMLP at inference.")
    else:
        print(f"  Hard Metric 2: MARGINAL (zero_vs_std={zero_vs_std:+.1f}%)")

    return {
        'n_gathers': (n_gathers_before, n_gathers_after, 0),
        'equiv_holds': equiv_holds,
        'invariant_holds': invariant_holds,
        'structure_match': structure_match,
        'gather_cost_6_pct': gather_cost_6,
        'gather_cost_1_pct': gather_cost_1,
        'zero_vs_std_pct': zero_vs_std,
        'metric2_pass': metric2_pass,
    }


if __name__ == "__main__":
    run()
