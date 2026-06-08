"""Export compiler-optimized IR graphs to TorchScript for production deployment.

Converts the compiler IR (GatherOp, ElementWiseOp, LinearOp) back to concrete
nn.Module instances, then uses torch.jit.script() to produce a deployable
TorchScript module. Includes a benchmarking harness to quantify the latency
impact of gather elimination on both standard MLPs and cube-structured MLPs.

Typical usage:
    >>> results = run(device="cuda")
    >>> print(results['standard_opt_stats']['graph_after'])
"""

from typing import List, Tuple, Dict, Optional, Union
import time
import torch
import torch.nn as nn

from compiler.ir import (
    Node, GatherOp, ElementWiseOp, LinearOp, OpType,
)
from compiler.passes import optimize, count_gathers, describe_graph
from models.mlp import (
    _GatherLayer, OrbitLinear,
    make_standard_mlp, make_cube_mlp,
)


# ═══════════════════════════════════════════════════════════════
# Per-position linear module (TorchScript-compatible)
# ═══════════════════════════════════════════════════════════════

class _PerPositionLinear(nn.Module):
    """Per-position linear transform: output[b,i,:] = input[b,i,:] @ W[i] + b[i].

    Uses torch.matmul with broadcasting for TorchScript compatibility.
    Equivalent to OrbitLinear when each position is its own orbit.
    """

    def __init__(self, weight: torch.Tensor, bias: Optional[torch.Tensor] = None):
        super().__init__()
        self.weight = nn.Parameter(weight.clone().contiguous())
        self.has_bias = bias is not None
        if self.has_bias:
            self.bias = nn.Parameter(bias.clone().contiguous())
        else:
            # TorchScript requires the attribute to exist; use a dummy buffer.
            self.register_buffer('bias', torch.zeros(1), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, N, D], weight: [N, D, D]
        # matmul with broadcast: [B, N, 1, D] @ [1, N, D, D] -> [B, N, 1, D]
        out = torch.matmul(x.unsqueeze(2), self.weight.unsqueeze(0)).squeeze(2)
        if self.has_bias:
            out = out + self.bias
        return out


# ═══════════════════════════════════════════════════════════════
# IR -> nn.Module conversion
# ═══════════════════════════════════════════════════════════════

def ir_to_torch_module(nodes: List[Node], N: int, D: int) -> nn.Module:
    """Convert a list of IR nodes to an nn.Module (nn.Sequential).

    Mapping:
        GatherOp                  -> _GatherLayer(perm)
        ElementWiseOp(LAYER_NORM) -> nn.LayerNorm(D)     (with params if present)
        ElementWiseOp(GELU)       -> nn.GELU()
        ElementWiseOp(RELU)       -> nn.ReLU()
        ElementWiseOp(DROPOUT)    -> nn.Dropout(p)
        LinearOp(shared)          -> nn.Linear(D, D)     (with weight/bias)
        LinearOp(per_position)    -> _PerPositionLinear   (with weight/bias)

    Args:
        nodes: list of IR Node objects (GatherOp, ElementWiseOp, LinearOp).
        N: number of positions (only needed for context, unused in construction).
        D: feature dimension.

    Returns:
        nn.Sequential containing the corresponding concrete layers.
    """
    layers: List[nn.Module] = []

    for node in nodes:
        if isinstance(node, GatherOp):
            # Permutation gather
            layers.append(_GatherLayer(node.perm))

        elif isinstance(node, ElementWiseOp):
            if node.op_type == OpType.LAYER_NORM:
                ln = nn.LayerNorm(D)
                if 'weight' in node.params:
                    ln.weight.data.copy_(node.params['weight'])
                if 'bias' in node.params:
                    ln.bias.data.copy_(node.params['bias'])
                layers.append(ln)

            elif node.op_type == OpType.GELU:
                layers.append(nn.GELU())

            elif node.op_type == OpType.RELU:
                layers.append(nn.ReLU())

            elif node.op_type == OpType.DROPOUT:
                p = node.params.get('p', 0.0)
                layers.append(nn.Dropout(p))

        elif isinstance(node, LinearOp):
            if node.per_position:
                layers.append(_PerPositionLinear(node.weight, node.bias))
            else:
                linear = nn.Linear(D, D)
                if node.weight is not None:
                    linear.weight.data.copy_(node.weight)
                if node.bias is not None:
                    linear.bias.data.copy_(node.bias)
                layers.append(linear)

    return nn.Sequential(*layers)


# ═══════════════════════════════════════════════════════════════
# Model -> IR conversion (for the optimization pipeline)
# ═══════════════════════════════════════════════════════════════

def _model_to_ir(module: nn.Module) -> List[Node]:
    """Walk an nn.Module tree and convert each recognised layer to its IR node.

    Handles nn.Sequential, nn.ModuleList, and flat sequential structures.
    Unrecognised layers are skipped (they remain as-is in the original model,
    but this path is intended for models built by make_standard_mlp / make_cube_mlp).

    Args:
        module: an nn.Module (typically nn.Sequential).

    Returns:
        List of IR Node objects.
    """
    nodes: List[Node] = []

    for _name, child in module.named_children():
        if isinstance(child, (nn.Sequential, nn.ModuleList)):
            # Recurse into containers
            nodes.extend(_model_to_ir(child))

        elif isinstance(child, nn.LayerNorm):
            nodes.append(ElementWiseOp(
                OpType.LAYER_NORM,
                {
                    'weight': child.weight.data.clone(),
                    'bias': child.bias.data.clone(),
                },
            ))

        elif isinstance(child, nn.GELU):
            nodes.append(ElementWiseOp(OpType.GELU))

        elif isinstance(child, nn.ReLU):
            nodes.append(ElementWiseOp(OpType.RELU))

        elif isinstance(child, nn.Dropout):
            nodes.append(ElementWiseOp(
                OpType.DROPOUT, {'p': child.p},
            ))

        elif isinstance(child, nn.Linear):
            nodes.append(LinearOp(
                weight=child.weight.data.clone(),
                bias=child.bias.data.clone() if child.bias is not None else None,
                per_position=False,
            ))

        elif isinstance(child, _GatherLayer):
            nodes.append(GatherOp(child.perm.clone()))

        elif isinstance(child, OrbitLinear):
            # Expand orbit-shared weights into full per-position tensors
            W = child.weight.data[child.orbit_ids]    # [N, D, D]
            b = child.bias.data[child.orbit_ids]      # [N, D]
            nodes.append(LinearOp(weight=W.clone(), bias=b.clone(), per_position=True))

        # Other layer types are silently skipped (e.g. residual connections
        # are not represented in the IR).

    return nodes


# ═══════════════════════════════════════════════════════════════
# Export pipeline
# ═══════════════════════════════════════════════════════════════

def export_optimized_model(
    model: nn.Module,
    example_input: torch.Tensor,
    optimize_first: bool = True,
) -> Tuple[torch.jit.ScriptModule, Dict]:
    """Convert a model to TorchScript, optionally applying the IR optimizer first.

    Args:
        model: an nn.Module (e.g. from make_standard_mlp or make_cube_mlp).
        example_input: a tensor [B, N, D] used for shape inference.
        optimize_first: if True, convert model -> IR -> optimize() -> nn.Module
            before scripting. If False, script the model directly.

    Returns:
        (scripted_module, stats_dict) where stats_dict contains:
            - n_gathers_before: int
            - n_gathers_after: int
            - graph_before: str (human-readable IR before optimization)
            - graph_after: str (human-readable IR after optimization)
    """
    n_gathers_before = 0
    n_gathers_after = 0
    graph_before = ""
    graph_after = ""

    if optimize_first:
        # 1. Convert model to IR
        nodes = _model_to_ir(model)
        n_gathers_before = count_gathers(nodes)
        graph_before = describe_graph(nodes)

        # 2. Run the optimization passes to fixed point
        nodes, _opt_stats = optimize(nodes)
        n_gathers_after = count_gathers(nodes)
        graph_after = describe_graph(nodes)

        # 3. Rebuild nn.Module from optimized IR
        B, N, D = example_input.shape
        optimized_model = ir_to_torch_module(nodes, N, D)
    else:
        optimized_model = model
        # Count gathers in the raw model for reporting
        nodes = _model_to_ir(model)
        n_gathers_before = count_gathers(nodes)
        n_gathers_after = n_gathers_before
        graph_before = describe_graph(nodes)
        graph_after = graph_before

    # 4. TorchScript the final module
    optimized_model.eval()
    optimized_model.to(example_input.device)

    scripted = torch.jit.script(optimized_model)

    stats = {
        'n_gathers_before': n_gathers_before,
        'n_gathers_after': n_gathers_after,
        'graph_before': graph_before,
        'graph_after': graph_after,
    }

    return scripted, stats


# ═══════════════════════════════════════════════════════════════
# Benchmarking
# ═══════════════════════════════════════════════════════════════

def benchmark_jit(
    jit_model: torch.jit.ScriptModule,
    x: torch.Tensor,
    n_warmup: int = 30,
    n_repeat: int = 100,
) -> float:
    """Measure mean latency of a TorchScript model in milliseconds.

    Performs warmup runs to stabilise GPU clocks / CUDA contexts, then
    times n_repeat forward passes with synchronisation.

    Args:
        jit_model: the scripted module to benchmark.
        x: input tensor [B, N, D] on the target device.
        n_warmup: number of untimed warmup iterations.
        n_repeat: number of timed iterations.

    Returns:
        Mean latency in milliseconds.
    """
    jit_model.eval()

    with torch.no_grad():
        # Warmup
        for _ in range(n_warmup):
            _ = jit_model(x)

        if x.is_cuda:
            torch.cuda.synchronize()

        start = time.perf_counter()
        for _ in range(n_repeat):
            _ = jit_model(x)
        if x.is_cuda:
            torch.cuda.synchronize()
        end = time.perf_counter()

    latency_s = (end - start) / n_repeat
    return latency_s * 1000.0  # ms


# ═══════════════════════════════════════════════════════════════
# End-to-end runner
# ═══════════════════════════════════════════════════════════════

def run(device: str = "cuda") -> Dict:
    """Build, export, and benchmark StandardMLP and CubeMLP models.

    For each model architecture, two TorchScript variants are produced:
    one directly scripted from the nn.Module, and one that passes through
    the IR optimizer first. Latency of all four variants is measured and
    a comparison table is printed.

    Args:
        device: target device string ("cuda" or "cpu").

    Returns:
        Dict with keys:
            - standard_raw_latency_ms: float
            - standard_opt_latency_ms: float
            - cube_raw_latency_ms: float
            - cube_opt_latency_ms: float
            - standard_raw_stats: dict
            - standard_opt_stats: dict
            - cube_raw_stats: dict
            - cube_opt_stats: dict
    """
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU.")
        device = "cpu"

    N, D, n_layers = 27, 128, 6
    batch_size = 4

    print(f"Building models (N={N}, D={D}, layers={n_layers}) on {device} ...")

    # ---- Build models ---------------------------------------------------
    std_model = make_standard_mlp(N, D, n_layers).to(device)
    cube_model = make_cube_mlp(N, D, n_layers).to(device)

    example_input = torch.randn(batch_size, N, D, device=device)

    # ---- Export (raw + optimized) ---------------------------------------
    print("Exporting standard MLP (raw) ...")
    std_raw_jit, std_raw_stats = export_optimized_model(
        std_model, example_input, optimize_first=False,
    )
    print("Exporting standard MLP (optimized) ...")
    std_opt_jit, std_opt_stats = export_optimized_model(
        std_model, example_input, optimize_first=True,
    )

    print("Exporting cube MLP (raw) ...")
    cube_raw_jit, cube_raw_stats = export_optimized_model(
        cube_model, example_input, optimize_first=False,
    )
    print("Exporting cube MLP (optimized) ...")
    cube_opt_jit, cube_opt_stats = export_optimized_model(
        cube_model, example_input, optimize_first=True,
    )

    # ---- Benchmark ------------------------------------------------------
    print("Benchmarking ...")
    std_raw_ms = benchmark_jit(std_raw_jit, example_input)
    std_opt_ms = benchmark_jit(std_opt_jit, example_input)
    cube_raw_ms = benchmark_jit(cube_raw_jit, example_input)
    cube_opt_ms = benchmark_jit(cube_opt_jit, example_input)

    # ---- Print comparison table -----------------------------------------
    def _speedup(opt_ms: float, raw_ms: float) -> str:
        if raw_ms > 0:
            ratio = raw_ms / opt_ms
            return f"{ratio:.2f}x"
        return "N/A"

    print()
    print("=" * 78)
    print("  TorchScript Export & Optimization Benchmark")
    print("=" * 78)
    print(f"  {'Model':<20} {'Gathers raw':>11} {'Gathers opt':>11} "
          f"{'Raw ms':>9} {'Opt ms':>9} {'Speedup':>8}")
    print("  " + "-" * 76)
    print(f"  {'StandardMLP':<20} {std_raw_stats['n_gathers_after']:>11} "
          f"{std_opt_stats['n_gathers_after']:>11} "
          f"{std_raw_ms:>9.3f} {std_opt_ms:>9.3f} "
          f"{_speedup(std_opt_ms, std_raw_ms):>8}")
    print(f"  {'CubeMLP':<20} {cube_raw_stats['n_gathers_after']:>11} "
          f"{cube_opt_stats['n_gathers_after']:>11} "
          f"{cube_raw_ms:>9.3f} {cube_opt_ms:>9.3f} "
          f"{_speedup(cube_opt_ms, cube_raw_ms):>8}")
    print("=" * 78)

    # ---- Print IR graphs ------------------------------------------------
    print()
    print("--- StandardMLP IR (raw) ---")
    print(std_raw_stats['graph_before'])
    print()
    print("--- StandardMLP IR (optimized) ---")
    print(std_opt_stats['graph_after'])
    print()
    print("--- CubeMLP IR (raw) ---")
    print(cube_raw_stats['graph_before'])
    print()
    print("--- CubeMLP IR (optimized) ---")
    print(cube_opt_stats['graph_after'])

    # ---- Build results --------------------------------------------------
    return {
        'standard_raw_latency_ms': std_raw_ms,
        'standard_opt_latency_ms': std_opt_ms,
        'cube_raw_latency_ms': cube_raw_ms,
        'cube_opt_latency_ms': cube_opt_ms,
        'standard_raw_stats': std_raw_stats,
        'standard_opt_stats': std_opt_stats,
        'cube_raw_stats': cube_raw_stats,
        'cube_opt_stats': cube_opt_stats,
    }


# ═══════════════════════════════════════════════════════════════
# CLI entry point
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys
    dev = sys.argv[1] if len(sys.argv) > 1 else "cuda"
    results = run(device=dev)
