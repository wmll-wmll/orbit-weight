"""
Cyclic group C_n: 1D cyclic shifts acting on n positions.

The cyclic group of order n has a single generator: rotation by `step` positions.
When step and n are coprime, all positions belong to a single orbit.
When step divides n, positions split into step orbits of size n/step each.

This group models 1D periodic structures (rings, circular sequences).
"""

import torch
from typing import List


class CyclicGroup:
    """Cyclic group C_n acting on n positions.

    Generator: shift by `step` positions (wraparound).
    When step=1 (default), all positions form a single orbit
    (any position can reach any other by repeated shifts).

    Example:
        group = CyclicGroup(n=8, step=1)
        # All 8 positions in 1 orbit → maximum weight sharing
        # Equivalent to a standard 1D conv with periodic padding

        group = CyclicGroup(n=8, step=2)
        # 2 orbits of size 4 (even positions, odd positions)
        # Models a bipartite ring structure
    """

    def __init__(self, n: int, step: int = 1):
        if n < 1:
            raise ValueError(f"n must be >= 1, got {n}")
        self.N = n
        self.step = step
        self.name = f"C_{n}" if step == 1 else f"C_{n}(step={step})"

    def get_generators(self) -> List[torch.Tensor]:
        """Return the single cyclic shift generator.

        gen[i] = (i + step) % n
        """
        gen = torch.tensor([(i + self.step) % self.N for i in range(self.N)],
                           dtype=torch.long)
        # Also include the inverse shift for symmetric BFS
        gen_inv = torch.tensor([(i - self.step) % self.N for i in range(self.N)],
                                dtype=torch.long)
        return [gen, gen_inv]

    def compute_orbits(self) -> tuple:
        """Compute orbit decomposition.

        The number of orbits = gcd(n, step).
        Each orbit size = n / gcd(n, step).
        """
        import math
        g = math.gcd(self.N, self.step)
        n_orbits = g
        orbit_ids = torch.tensor([i % g for i in range(self.N)], dtype=torch.long)
        return orbit_ids, n_orbits

    def __repr__(self) -> str:
        return f"CyclicGroup(n={self.N}, step={self.step}, name='{self.name}')"
