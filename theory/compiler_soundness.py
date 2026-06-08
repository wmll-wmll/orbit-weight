"""
Exhaustive fuzzer for compiler soundness (Theorem 2 empirical validation).

Generates random IR graphs, applies optimize(), and verifies that the
original and optimized graphs produce identical outputs.

Usage:
    python theory/compiler_soundness.py          # 10K random graphs
    python theory/compiler_soundness.py --quick  # 1K random graphs (faster)
"""

import torch
import sys
import os
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from compiler.ir import (
    GatherOp, ElementWiseOp, LinearOp, OpType,
    compose_permutations, invert_permutation,
)
from compiler.passes import optimize, count_gathers, describe_graph


# ── Reference interpreter (same as test_equivariance.py) ──────────

def gather_ref(x: torch.Tensor, perm: torch.Tensor) -> torch.Tensor:
    """Reference gather implementation."""
    indices = perm.unsqueeze(0).unsqueeze(-1).expand(x.size(0), -1, x.size(2))
    return torch.gather(x, 1, indices)


def interpret(nodes: list, x: torch.Tensor) -> torch.Tensor:
    """Execute a list of IR nodes on input tensor x."""
    for node in nodes:
        if isinstance(node, GatherOp):
            x = gather_ref(x, node.perm)
        elif isinstance(node, ElementWiseOp):
            if node.op_type == OpType.LAYER_NORM:
                weight = node.params.get("weight", torch.ones(x.size(-1)))
                bias = node.params.get("bias", torch.zeros(x.size(-1)))
                mean = x.mean(dim=-1, keepdim=True)
                var = x.var(dim=-1, keepdim=True, unbiased=False)
                x = (x - mean) / torch.sqrt(var + 1e-5)
                x = x * weight + bias
            elif node.op_type == OpType.GELU:
                x = torch.nn.functional.gelu(x)
            elif node.op_type == OpType.RELU:
                x = torch.nn.functional.relu(x)
            elif node.op_type == OpType.DROPOUT:
                pass  # no-op during eval
        elif isinstance(node, LinearOp):
            if node.weight is not None:
                W = node.weight
                b = node.bias
                if node.per_position:
                    # W: [N, D, d_out]
                    out = torch.einsum('bnd,ndo->bno', x, W.to(x.device, x.dtype))
                    if b is not None:
                        out = out + b.to(x.device, x.dtype)
                    x = out
                else:
                    # W: [D, d_out] — shared weight
                    x = torch.matmul(x, W.to(x.device, x.dtype).T)
                    if b is not None:
                        x = x + b.to(x.device, x.dtype)
    return x


# ── Random graph generators ───────────────────────────────────────

def random_permutation(N: int) -> torch.Tensor:
    """Generate a random permutation of N elements."""
    return torch.randperm(N)


def random_linear_weights(N: int, D: int, per_position: bool = False):
    """Generate random LinearOp weights.

    Args:
        N: number of positions
        D: feature dimension
        per_position: if True, generates [N, D, D]; else [D, D]

    Returns:
        (weight, bias)
    """
    if per_position:
        W = torch.randn(N, D, D) / (D ** 0.5)
        b = torch.randn(N, D) * 0.1
    else:
        W = torch.randn(D, D) / (D ** 0.5)
        b = torch.randn(D) * 0.1
    return W, b


def generate_random_graph(N: int = 27, D: int = 64,
                           min_nodes: int = 3, max_nodes: int = 12) -> list:
    """Generate a random well-formed IR graph.

    Args:
        N: number of positions
        D: feature dimension
        min_nodes: minimum number of nodes
        max_nodes: maximum number of nodes

    Returns:
        List of IR Node objects
    """
    import random
    n_nodes = random.randint(min_nodes, max_nodes)
    nodes = []

    node_types = ['gather', 'elementwise', 'linear_shared', 'linear_pp']
    weights = [2, 3, 2, 1]  # sampling weights (element-wise most common)

    for _ in range(n_nodes):
        t = random.choices(node_types, weights=weights, k=1)[0]

        if t == 'gather':
            perm = random_permutation(N)
            nodes.append(GatherOp(perm))
        elif t == 'elementwise':
            op = random.choice([OpType.LAYER_NORM, OpType.GELU, OpType.RELU])
            nodes.append(ElementWiseOp(op))
        elif t == 'linear_shared':
            W, b = random_linear_weights(N, D, per_position=False)
            nodes.append(LinearOp(weight=W, bias=b, per_position=False))
        elif t == 'linear_pp':
            W, b = random_linear_weights(N, D, per_position=True)
            nodes.append(LinearOp(weight=W, bias=b, per_position=True))

    return nodes


def generate_diverse_graph(N: int = 27, D: int = 64) -> list:
    """Generate a graph with all three node types present.

    Ensures the graph exercises all optimization passes.
    """
    nodes = [
        GatherOp(random_permutation(N)),
        ElementWiseOp(OpType.LAYER_NORM),
        ElementWiseOp(OpType.GELU),
        GatherOp(random_permutation(N)),
    ]
    # 50% chance of per-position linear (triggers AbsorbPass)
    import random
    if random.random() < 0.5:
        W, b = random_linear_weights(N, D, per_position=True)
        nodes.append(LinearOp(weight=W, bias=b, per_position=True))
    else:
        W, b = random_linear_weights(N, D, per_position=False)
        nodes.append(LinearOp(weight=W, bias=b, per_position=False))

    # Sometimes add another gather to test ChainPass
    if random.random() < 0.3:
        nodes.append(GatherOp(random_permutation(N)))

    return nodes


# ── Verification ──────────────────────────────────────────────────

def verify_graph(graph: list, x: torch.Tensor, atol: float = 1e-5) -> dict:
    """Verify that optimize() preserves semantics for a given graph.

    Args:
        graph: list of IR nodes
        x: input tensor [B, N, D]
        atol: absolute tolerance for equivalence

    Returns:
        dict with verification results
    """
    result = {
        'graph_before': describe_graph(graph),
        'n_gathers_before': count_gathers(graph),
        'passed': False,
        'max_diff': float('inf'),
        'stats': {},
    }

    # Reference output (before optimization)
    with torch.no_grad():
        out_before = interpret(graph, x.clone())

    # Optimize
    try:
        optimized, stats = optimize(list(graph))  # copy to avoid mutation
        result['stats'] = stats
        result['graph_after'] = describe_graph(optimized)
        result['n_gathers_after'] = count_gathers(optimized)

        with torch.no_grad():
            out_after = interpret(optimized, x.clone())

        max_diff = (out_before - out_after).abs().max().item()
        result['max_diff'] = max_diff
        result['passed'] = max_diff < atol

    except Exception as e:
        result['error'] = str(e)

    return result


# ── Main fuzzer ───────────────────────────────────────────────────

def run(n_graphs: int = 10000, N: int = 27, D: int = 64,
        B: int = 4, atol: float = 1e-5, seed: int = 42):
    """Run the compiler soundness fuzzer.

    Generates n_graphs random IR graphs, applies optimize(), and verifies
    that the original and optimized graphs produce identical outputs.

    Args:
        n_graphs: number of random graphs to test
        N: number of positions
        D: feature dimension
        B: batch size
        atol: absolute tolerance
        seed: random seed
    """
    import random
    random.seed(seed)
    torch.manual_seed(seed)

    print("=" * 60)
    print("THEOREM 2: Compiler Soundness — Random Graph Fuzzer")
    print("=" * 60)
    print(f"Testing {n_graphs} random graphs (N={N}, D={D}, B={B})")
    print(f"Tolerance: {atol:.1e}")
    print()

    results = []
    n_passed = 0
    n_failed = 0
    n_error = 0
    total_gathers_before = 0
    total_gathers_after = 0

    for i in range(n_graphs):
        # 80% diverse graphs, 20% fully random
        if random.random() < 0.8:
            graph = generate_diverse_graph(N, D)
        else:
            graph = generate_random_graph(N, D)

        x = torch.randn(B, N, D)
        r = verify_graph(graph, x, atol)
        results.append(r)

        if r['passed']:
            n_passed += 1
        elif 'error' in r:
            n_error += 1
        else:
            n_failed += 1

        total_gathers_before += r.get('n_gathers_before', 0)
        total_gathers_after += r.get('n_gathers_after', 0)

        # Progress indicator
        if (i + 1) % 1000 == 0:
            print(f"  [{i+1:>5d}/{n_graphs}] passed={n_passed} "
                  f"failed={n_failed} errors={n_error} "
                  f"max_diff={r.get('max_diff', 0):.2e}")

    # Summary
    print(f"\n{'='*60}")
    print(f"RESULTS")
    print(f"{'='*60}")
    print(f"  Total graphs tested:  {n_graphs}")
    print(f"  Passed:               {n_passed} ({100*n_passed/n_graphs:.1f}%)")
    print(f"  Failed (diff > tol):  {n_failed}")
    print(f"  Errors (exceptions):  {n_error}")
    print(f"  Avg gathers before:   {total_gathers_before/n_graphs:.2f}")
    print(f"  Avg gathers after:    {total_gathers_after/n_graphs:.2f}")
    if total_gathers_before > 0:
        reduction = (1 - total_gathers_after / max(total_gathers_before, 1)) * 100
        print(f"  Gather reduction:     {reduction:.1f}%")

    # Show worst-case diff
    if n_failed > 0:
        worst = max(r for r in results if not r['passed'] and 'error' not in r)
        print(f"\n  Worst failure:")
        print(f"    max_diff:  {worst['max_diff']:.2e}")
        print(f"    before:    {worst['graph_before']}")
        print(f"    after:     {worst.get('graph_after', 'N/A')}")

    # Verdict
    if n_passed == n_graphs:
        print(f"\n[PASS] ALL {n_graphs} GRAPHS PASSED -- compiler is sound")
    else:
        print(f"\n[WARN] {n_failed}/{n_graphs} graphs failed")
        print(f"  This may indicate a compiler bug or numerical precision issue.")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--quick', action='store_true', help='Run only 1000 graphs')
    parser.add_argument('--n-graphs', type=int, default=10000)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    n = 1000 if args.quick else args.n_graphs
    run(n_graphs=n, seed=args.seed)
