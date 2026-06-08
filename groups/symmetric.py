"""
Symmetric group S_n: all permutations of n elements.

The symmetric group of order n! has n-1 adjacent transpositions as generators.
All n positions form a SINGLE orbit — any position can be permuted to any
other position.

This group serves as the "fully connected" extreme: orbit sharing = uniform
sharing (K=1). It demonstrates that when the group is large enough to connect
all positions, our method correctly reduces to standard weight sharing.

Useful as a sanity check and to contrast with groups that produce non-trivial
orbit structure (Rubik's cube, cyclic with step>1).

Example:
    group = SymmetricGroup(n=5)
    # 1 orbit of 5 elements → uniform sharing is optimal
"""

import torch
from typing import List


class SymmetricGroup:
    """Symmetric group S_n acting on n positions.

    Order = n!
    Generators: n-1 adjacent transpositions (i, i+1).

    The transpositions alone generate all of S_n, and since any position
    can be swapped with its neighbor, all positions belong to a single
    orbit under the group action.
    """

    def __init__(self, n: int):
        if n < 2:
            raise ValueError(f"n must be >= 2, got {n}")
        self.N = n
        self.name = f"S_{n}"

    def get_generators(self) -> List[torch.Tensor]:
        """Return n-1 adjacent transposition generators.

        gen_k swaps positions k and k+1, leaves others unchanged.
        """
        generators = []
        for k in range(self.N - 1):
            perm = list(range(self.N))
            perm[k], perm[k + 1] = perm[k + 1], perm[k]
            generators.append(torch.tensor(perm, dtype=torch.long))
        return generators

    def compute_orbits(self) -> tuple:
        """All n positions form a single orbit.

        Any permutation can be decomposed into adjacent transpositions,
        so all positions are mutually reachable.
        """
        orbit_ids = torch.zeros(self.N, dtype=torch.long)
        return orbit_ids, 1

    def __repr__(self) -> str:
        return f"SymmetricGroup(n={self.N}, name='{self.name}')"
