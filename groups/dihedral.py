"""
Dihedral group D_n: symmetries of a regular n-gon.

The dihedral group of order 2n has two generators:
    - rotation (r): cyclic shift by 1 position
    - reflection (s): mirror across axis through vertex 0

All n vertices of a regular polygon belong to a single orbit under D_n
when n >= 3 — any vertex can be mapped to any other via rotation+reflection.

This group models 2D structures with rotational and reflectional symmetry.
"""

import torch
from typing import List


class DihedralGroup:
    """Dihedral group D_n acting on n vertices of a regular polygon.

    Order = 2n (n rotations + n reflections).
    Generators: rotation r (cyclic shift by 1) + reflection s (mirror).

    For n >= 3, all vertices form a SINGLE orbit under D_n.
    This means orbit-based weight sharing reduces to uniform sharing
    for the full dihedral group — demonstrating that the orbit approach
    correctly identifies when uniform sharing IS the right answer.

    For D_n on n×n 2D grid positions (not just polygon vertices),
    additional structure is needed (see octahedral.py for 3D analog).

    Example:
        group = DihedralGroup(n=6)  # hexagon: 1 orbit of 6 vertices
        group = DihedralGroup(n=4)  # square: 1 orbit of 4 vertices
    """

    def __init__(self, n: int):
        if n < 3:
            raise ValueError(f"n must be >= 3 for dihedral group, got {n}")
        self.N = n
        self.name = f"D_{n}"

    def get_generators(self) -> List[torch.Tensor]:
        """Return rotation and reflection generators.

        Rotation r:     r[i] = (i + 1) % n
        Reflection s:   s[i] = (-i) % n  (mirror across axis through vertex 0)
        """
        # Rotation by 1 position
        r = torch.tensor([(i + 1) % self.N for i in range(self.N)],
                         dtype=torch.long)
        # Reflection across axis through vertex 0
        s = torch.tensor([(-i) % self.N for i in range(self.N)],
                         dtype=torch.long)
        # Also include inverses
        r_inv = torch.tensor([(i - 1) % self.N for i in range(self.N)],
                              dtype=torch.long)
        return [r, r_inv, s]  # s is its own inverse

    def compute_orbits(self) -> tuple:
        """All n vertices form a single orbit under D_n (n >= 3).

        Proof: rotation alone connects all vertices. Reflection is
        redundant for orbit computation but provides additional structure.
        """
        orbit_ids = torch.zeros(self.N, dtype=torch.long)
        return orbit_ids, 1

    def __repr__(self) -> str:
        return f"DihedralGroup(n={self.N}, name='{self.name}')"
