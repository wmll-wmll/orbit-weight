"""
Correctness and equivariance tests for the cube permutation system.

Validates:
1. PermutationMatrix.apply == torch.gather (correctness)
2. Cube rotations are valid bijections
3. Rotation equivariance: rotating data then applying a learned function
   should equal applying the function then rotating the result.
4. Sinkhorn layer converges to a valid permutation
5. Compiler passes: Reorder, Absorb, Chain preserve semantics
6. Compiler optimize() pipeline correctness
"""

import torch
import sys
sys.path.insert(0, "e:/operator")

from cube.perm_matrix import PermutationMatrix, gather_ref, verify_equivalence
from cube.cube3d import CubePermutations
from cube.layers import CubePermutation, SoftCubePermutation
from compiler.ir import (
    GatherOp, ElementWiseOp, LinearOp, OpType,
    compose_permutations, invert_permutation,
)
from compiler.passes import (
    reorder_pass, absorb_pass, chain_pass, fuse_pass,
    optimize, count_gathers, describe_graph,
)


def test_permutation_matrix_correctness():
    """Test that PermutationMatrix.apply yields same result as torch.gather."""
    print("Test: PermutationMatrix correctness...", end=" ")
    for N in [8, 27, 64]:
        for B in [1, 16]:
            for D in [32, 128]:
                x = torch.randn(B, N, D)
                perm = torch.randperm(N)

                pm = PermutationMatrix(perm)
                out_pm = pm.apply(x)
                out_gather = gather_ref(x, perm)

                max_diff = (out_pm - out_gather).abs().max().item()
                assert max_diff < 1e-5, f"FAIL at shape ({B},{N},{D}): diff={max_diff:.2e}"

    print("PASS")


def test_cube_rotation_validity():
    """Test that all 12 cube rotations are valid permutations."""
    print("Test: Cube rotation validity...", end=" ")
    cube = CubePermutations(3)

    for name, perm in cube.all_generators().items():
        assert cube.is_valid_permutation(perm), f"FAIL: {name} is not a valid permutation"

    # U rotation affects 8 of 9 top-layer positions (center stays fixed)
    perm_u = cube.rotation_U(clockwise=True)
    affected = (perm_u != torch.arange(27)).sum().item()
    assert affected == 8, f"FAIL: U rotation affects {affected} positions, expected 8"

    print("PASS")


def test_rotation_equivariance():
    """Test rotational equivariance for a simple linear transformation.

    Property: R(f(x)) = f(R(x)) where R is a cube rotation and f is a
    per-position linear transform (1x1 conv / diagonal linear layer).
    """
    print("Test: Rotation equivariance...", end=" ")

    cube = CubePermutations(3)
    B, N, D = 4, 27, 32
    x = torch.randn(B, N, D)

    # A per-position linear transform (weights tied across positions)
    linear = torch.nn.Linear(D, D, bias=False)

    for face in ['U', 'F', 'R']:
        perm = cube.get_rotation(face)

        # Path 1: rotate then apply f
        x_rotated = gather_ref(x, perm)
        y_rotate_then_f = linear(x_rotated)

        # Path 2: apply f then rotate
        y_f = linear(x)
        y_f_then_rotate = gather_ref(y_f, perm)

        max_diff = (y_rotate_then_f - y_f_then_rotate).abs().max().item()
        assert max_diff < 1e-4, f"FAIL: {face} equivariance diff={max_diff:.2e}"

    print("PASS")


def test_composition():
    """Test that permutation composition is associative."""
    print("Test: Composition associativity...", end=" ")

    cube = CubePermutations(3)
    p_u = cube.rotation_U()
    p_r = cube.rotation_R()
    p_f = cube.rotation_F()

    # (U ∘ R) ∘ F == U ∘ (R ∘ F) via composition
    left = cube.compose(cube.compose(p_u, p_r), p_f)
    right = cube.compose(p_u, cube.compose(p_r, p_f))

    assert torch.equal(left, right), "FAIL: composition is not associative"
    print("PASS")


def test_cube_permutation_layer():
    """Test that CubePermutation nn.Module works end-to-end."""
    print("Test: CubePermutation layer...", end=" ")

    layer = CubePermutation(n=3)
    x = torch.randn(16, 27, 128)

    # Single move
    y = layer(x, moves=["U"])
    assert y.shape == x.shape

    # Multiple moves
    y = layer(x, moves=["U", "R", "F"])
    assert y.shape == x.shape

    # No moves (default U)
    y = layer(x)
    assert y.shape == x.shape

    # Verify against raw gather
    cube = CubePermutations(3)
    perm = cube.get_rotation("U")
    y_gather = gather_ref(x, perm)
    assert torch.allclose(y, y_gather, atol=1e-5), "FAIL: layer output != gather"

    print("PASS")


def test_soft_permutation_layer():
    """Test that SoftCubePermutation runs and produces valid output."""
    print("Test: SoftCubePermutation layer...", end=" ")

    layer = SoftCubePermutation(n=3, d_q=16, n_sinkhorn=20)
    x = torch.randn(8, 27, 64)

    # Forward pass (soft)
    y = layer(x)
    assert y.shape == x.shape
    assert torch.isfinite(y).all(), "FAIL: output contains NaN/Inf"

    # Gradient flow
    y.sum().backward()
    assert layer.query.grad is not None, "FAIL: no gradient on query"
    assert torch.isfinite(layer.query.grad).all(), "FAIL: NaN/Inf in query grad"

    # Hard permutation (inference mode)
    y_hard = layer(x, hard=True)
    assert y_hard.shape == x.shape

    print("PASS")


def test_sinkhorn_permutation_convergence():
    """Test that Sinkhorn output is close to doubly stochastic."""
    print("Test: Sinkhorn convergence...", end=" ")

    layer = SoftCubePermutation(n=3, d_q=16, n_sinkhorn=50)
    with torch.no_grad():
        S = layer._compute_similarity()
        P = layer._sinkhorn(S)

    # Rows sum to 1
    row_sums = P.sum(dim=1)
    assert (row_sums - 1.0).abs().max().item() < 0.01, "FAIL: rows not normalized"

    # Columns sum to 1
    col_sums = P.sum(dim=0)
    assert (col_sums - 1.0).abs().max().item() < 0.01, "FAIL: columns not normalized"

    # Values in [0, 1]
    assert P.min().item() >= 0 and P.max().item() <= 1.01

    print("PASS")


def test_large_n_correctness():
    """Test correctness for larger N (4³, 5³)."""
    print("Test: Large N correctness...", end=" ")

    for n in [4, 5]:
        N = n ** 3
        cube = CubePermutations(n)
        perm = cube.rotation_U()

        x = torch.randn(2, N, 32)
        out_pm = PermutationMatrix(perm).apply(x)
        out_gather = gather_ref(x, perm)
        max_diff = (out_pm - out_gather).abs().max().item()
        assert max_diff < 1e-5, f"FAIL at N={N}: diff={max_diff:.2e}"

    print("PASS")


# ═══════════════════════════════════════════════════════════════
# IR interpreter for testing passes
# ═══════════════════════════════════════════════════════════════

def interpret(nodes: list, x: torch.Tensor) -> torch.Tensor:
    """Execute a list of IR nodes on input tensor x.

    This is a reference interpreter used only for testing correctness.
    Production code uses the actual PyTorch ops.
    """
    for node in nodes:
        if isinstance(node, GatherOp):
            x = gather_ref(x, node.perm)
        elif isinstance(node, ElementWiseOp):
            if node.op_type == OpType.LAYER_NORM:
                # Simulate LN with a known weight/bias
                weight = node.params.get("weight", torch.ones(x.size(-1)))
                bias = node.params.get("bias", torch.zeros(x.size(-1)))
                mean = x.mean(dim=-1, keepdim=True)
                var = x.var(dim=-1, keepdim=True, unbiased=False)
                x = (x - mean) / torch.sqrt(var + 1e-5)
                x = x * weight.to(x.device) + bias.to(x.device)
            elif node.op_type == OpType.GELU:
                x = torch.nn.functional.gelu(x)
            elif node.op_type == OpType.RELU:
                x = torch.nn.functional.relu(x)
            elif node.op_type == OpType.DROPOUT:
                pass  # No-op during eval
        elif isinstance(node, LinearOp):
            if node.per_position and node.weight is not None:
                W = node.weight  # [N, D, d_out]
                b = node.bias   # [N, d_out]
                out = torch.einsum('bnd,ndo->bno', x, W.to(x.device, x.dtype))
                if b is not None:
                    out = out + b.to(x.device, x.dtype)
                x = out
    return x


# ═══════════════════════════════════════════════════════════════
# Test 9: Reorder pass — GatherOp commutes past ElementWiseOp
# ═══════════════════════════════════════════════════════════════

def test_reorder_pass():
    """Verify: GatherOp → ElementWiseOp ≡ ElementWiseOp → GatherOp."""
    print("Test: ReorderPass correctness...", end=" ")

    perm = torch.randperm(27)
    x = torch.randn(4, 27, 64)

    # Original: Gather → LN → GELU
    original = [
        GatherOp(perm),
        ElementWiseOp(OpType.LAYER_NORM),
        ElementWiseOp(OpType.GELU),
    ]

    # Run reorder to saturation (like optimize() does)
    reordered = list(original)
    total_n = 0
    for _ in range(3):
        reordered, n = reorder_pass(reordered)
        total_n += n
        if n == 0:
            break
    assert total_n == 2, f"Expected 2 rewrites total, got {total_n}"

    out_orig = interpret(original, x)
    out_reordered = interpret(reordered, x)

    max_diff = (out_orig - out_reordered).abs().max().item()
    assert max_diff < 1e-4, f"ReorderPass mismatch: diff={max_diff:.2e}"

    print("PASS")


# ═══════════════════════════════════════════════════════════════
# Test 10: Absorb pass — GatherOp + per-position LinearOp
# ═══════════════════════════════════════════════════════════════

def test_absorb_pass():
    """Verify: GatherOp → LinearOp(pp) ≡ LinearOp(absorbed) → GatherOp."""
    print("Test: AbsorbPass correctness...", end=" ")

    N, D, d_out = 27, 64, 32
    perm = torch.randperm(N)
    x = torch.randn(4, N, D)
    W = torch.randn(N, D, d_out)
    b = torch.randn(N, d_out)

    # Original: Gather → Linear(per_pos)
    original = [
        GatherOp(perm),
        LinearOp(weight=W, bias=b, per_position=True),
    ]

    # After absorb: Linear(absorbed) → Gather
    absorbed, n = absorb_pass(original)
    assert n == 1, f"Expected 1 rewrite, got {n}"

    out_orig = interpret(original, x)
    out_absorbed = interpret(absorbed, x)

    max_diff = (out_orig - out_absorbed).abs().max().item()
    assert max_diff < 1e-4, f"AbsorbPass mismatch: diff={max_diff:.2e}"

    print("PASS")


# ═══════════════════════════════════════════════════════════════
# Test 11: Chain pass — consecutive gathers merge
# ═══════════════════════════════════════════════════════════════

def test_chain_pass():
    """Verify: GatherOp(p1) → GatherOp(p2) ≡ GatherOp(p1[p2])."""
    print("Test: ChainPass correctness...", end=" ")

    p1 = torch.randperm(27)
    p2 = torch.randperm(27)
    x = torch.randn(4, 27, 64)

    # Two consecutive gathers
    original = [GatherOp(p1), GatherOp(p2)]

    # After chain: single composed gather
    chained, n = chain_pass(original)
    assert n == 1, f"Expected 1 merge, got {n}"
    assert len(chained) == 1
    assert count_gathers(chained) == 1

    out_orig = interpret(original, x)
    out_chained = interpret(chained, x)

    max_diff = (out_orig - out_chained).abs().max().item()
    assert max_diff < 1e-4, f"ChainPass mismatch: diff={max_diff:.2e}"

    print("PASS")


# ═══════════════════════════════════════════════════════════════
# Test 12: Optimize pipeline — full graph optimization
# ═══════════════════════════════════════════════════════════════

def test_optimize_pipeline():
    """Verify the full optimize() pipeline produces correct results."""
    print("Test: Optimize pipeline...", end=" ")

    N, D, d_out = 27, 64, 32
    p1 = torch.randperm(N)
    p2 = torch.randperm(N)
    W = torch.randn(N, D, d_out)
    x = torch.randn(4, N, D)

    # A realistic graph: Gather → LN → GELU → Gather → Linear(pp)
    original = [
        GatherOp(p1),
        ElementWiseOp(OpType.LAYER_NORM),
        ElementWiseOp(OpType.GELU),
        GatherOp(p2),
        LinearOp(weight=W, per_position=True),
    ]

    out_orig = interpret(original, x)

    optimized, stats = optimize(original)
    out_opt = interpret(optimized, x)

    max_diff = (out_orig - out_opt).abs().max().item()
    assert max_diff < 1e-4, f"Optimize mismatch: diff={max_diff:.2e}"

    # Verify optimizations happened
    assert stats["chain_merges"] + stats["reorder_rewrites"] + stats["absorb_rewrites"] > 0, \
        "No optimizations applied"

    print(f"PASS ({stats})")


if __name__ == "__main__":
    print("=" * 60)
    print("CUBE PERMUTATION — CORRECTNESS TEST SUITE")
    print("=" * 60)

    test_permutation_matrix_correctness()
    test_cube_rotation_validity()
    test_rotation_equivariance()
    test_composition()
    test_cube_permutation_layer()
    test_soft_permutation_layer()
    test_sinkhorn_permutation_convergence()
    test_large_n_correctness()
    test_reorder_pass()
    test_absorb_pass()
    test_chain_pass()
    test_optimize_pipeline()

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)
