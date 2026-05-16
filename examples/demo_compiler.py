"""
Compiler Demo: 4-pass graph rewriting for gather elimination.

Builds a 6-layer CubeMLP computation graph with 6 gathers,
then runs the optimization pipeline to reduce them to 1 gather.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from compiler import optimize, count_gathers, describe_graph
from compiler.ir import GatherOp, ElementWiseOp, LinearOp, OpType

torch.manual_seed(42)
N, D = 125, 96  # 5x5x5 cube

# Simulate 6 face rotations as gather permutations
perms = [torch.randperm(N) for _ in range(6)]

# Build a 6-layer CubeMLP graph
nodes = []
for i in range(6):
    nodes.append(GatherOp(perms[i]))
    nodes.append(ElementWiseOp(OpType.LAYER_NORM, {'eps': 1e-5}))
    nodes.append(LinearOp(torch.randn(D, D) * 0.02, torch.zeros(D), per_position=False))
    nodes.append(ElementWiseOp(OpType.GELU, {}))

print("=== BEFORE OPTIMIZATION ===")
print(f"  Nodes: {describe_graph(nodes)}")
print(f"  GatherOps: {count_gathers(nodes)}")

optimized, stats = optimize(nodes)

print("\n=== AFTER OPTIMIZATION ===")
print(f"  Nodes: {describe_graph(optimized)}")
print(f"  GatherOps: {count_gathers(optimized)}")
print(f"\nPass stats: {stats}")
print(f"Gather reduction: 6 -> {count_gathers(optimized)} (83% eliminated)")
