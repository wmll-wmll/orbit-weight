from .ir import (
    Node, GatherOp, ElementWiseOp, LinearOp, OpType,
    compose_permutations, invert_permutation,
)
from .passes import (
    reorder_pass, absorb_pass, chain_pass, fuse_pass,
    optimize, count_gathers, describe_graph,
)
