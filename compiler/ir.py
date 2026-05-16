"""Compiler IR for gather-based computation graphs.

Three node types capture the essential structure of a cube MLP layer:
- GatherOp: permute positions along dim=1
- ElementWiseOp: per-position ops that commute with gather (LN, GELU, etc.)
- LinearOp: per-position linear transform (matmul, no cross-position mixing)

A graph is a list of these nodes. The passes in passes.py rewrite graphs
to minimize gather operations — the key optimization for domestic GPUs.
"""

from enum import Enum, auto
from dataclasses import dataclass, field
from typing import Optional, List, Tuple

import torch


class OpType(Enum):
    LAYER_NORM = auto()
    GELU = auto()
    RELU = auto()
    DROPOUT = auto()


class Node:
    """Base class for IR nodes."""
    pass


@dataclass
class GatherOp(Node):
    """Permute positions along dim=1.

    Semantics: output[b, i, d] = input[b, perm[i], d]
    """
    perm: torch.Tensor  # [N] permutation vector

    def __repr__(self):
        return f"GatherOp(N={len(self.perm)})"


@dataclass
class ElementWiseOp(Node):
    """Per-position element-wise operation.

    These commute with GatherOp because they operate independently
    on each position's feature vector.

    Semantics: output[b, i, :] = f(input[b, i, :]) for some f: R^D → R^D
    """
    op_type: OpType
    params: dict = field(default_factory=dict)

    def __repr__(self):
        return f"ElementWiseOp({self.op_type.name})"


@dataclass
class LinearOp(Node):
    """Per-position linear transform.

    Does NOT mix across positions. W is [D, d_out] or None for square transform.
    When per_position=True, each position has its own weight W[i] of shape [D, d_out].

    Semantics: output[b, i, :] = input[b, i, :] @ W[i] + b[i]
    """
    weight: Optional[torch.Tensor] = None  # [D, d_out] or [N, D, d_out]
    bias: Optional[torch.Tensor] = None    # [d_out] or [N, d_out]
    per_position: bool = False

    def __repr__(self):
        shape = list(self.weight.shape) if self.weight is not None else "?"
        pp = "pp" if self.per_position else "shared"
        return f"LinearOp({pp}, W={shape})"


def compose_permutations(p2: torch.Tensor, p1: torch.Tensor) -> torch.Tensor:
    """Compose two permutations: result[i] = p2[p1[i]].

    When applied to data: GatherOp(p2) then GatherOp(p1) gives
    output[i] = input[p2[p1[i]]] = input[(p2 ∘ p1)[i]] = GatherOp(p2[p1])(input)
    """
    return p2[p1]


def invert_permutation(perm: torch.Tensor) -> torch.Tensor:
    """Compute the inverse permutation."""
    inv = torch.empty_like(perm)
    inv[perm] = torch.arange(len(perm), device=perm.device)
    return inv
