"""
Octahedral group O_h: full symmetry group of the cube/octahedron.

The octahedral group is the direct product of the rotational octahedral
group O (24 proper rotations) and the inversion group C_i (2 elements),
giving |O_h| = 48.

For a 3D n×n×n grid, the group action partitions positions into orbits
based on:
    - Distance from center (radius)
    - Face / edge / corner membership
    - For n >= 5, positions within the same geometric category at the
      same distance form a single orbit

This is the natural generalization of the Rubik's cube face rotation group.
While the face rotation group is a subgroup of O (6 face rotations + inverses),
the full O_h group captures all 48 symmetries of the cube.

Example:
    group = OctahedralGroup(n=5)  # 125 positions → ~48 orbits
    group = OctahedralGroup(n=3)  # 27 positions → ~10 orbits
"""

import torch
from typing import List, Tuple
import math


class OctahedralGroup:
    """Full octahedral symmetry group O_h acting on an n×n×n 3D grid.

    The group action is: for each element g in O_h (represented as a
    3×3 signed permutation matrix), map 3D coordinate (x,y,z) to g·(x,y,z),
    then convert back to linear index via z-major ordering.

    Since we work on a finite grid, the action must also handle coordinates
    that fall outside [0, n-1] — these are mapped back using periodic or
    clamped boundary conditions. By default, we use "cube" boundary where
    the action is only defined for the symmetry of the cube itself.

    For O_h on an n×n×n grid, generators are:
        - 90° rotation around z-axis (C4)
        - 90° rotation around x-axis (C4)
        - Reflection through xy-plane (σ_h)
    These three generate the full O_h group of order 48.
    """

    def __init__(self, n: int = 5):
        """
        Args:
            n: grid side length (total positions = n³)
        """
        if n < 2:
            raise ValueError(f"n must be >= 2, got {n}")
        self.n = n
        self.N = n ** 3
        self.name = f"O_h({n}³)"

        # Coordinate grid: z-major ordering (matching cube/cube3d.py)
        self._coords = torch.tensor(
            [[x, y, z] for z in range(n) for y in range(n) for x in range(n)],
            dtype=torch.long,
        )  # [N, 3]

        # Build linear index lookup
        self._idx_of_xyz = {}
        for i, (x, y, z) in enumerate(self._coords.tolist()):
            self._idx_of_xyz[(x, y, z)] = i

    def _coord_to_idx(self, x: int, y: int, z: int) -> int:
        """Convert 3D coordinate to linear index.

        Invalid coordinates (outside [0, n-1]) are clamped.
        """
        x = max(0, min(self.n - 1, x))
        y = max(0, min(self.n - 1, y))
        z = max(0, min(self.n - 1, z))
        return z * self.n * self.n + y * self.n + x

    def _apply_3x3_matrix(self, mat: torch.Tensor) -> torch.Tensor:
        """Apply a 3×3 signed permutation matrix to all grid positions.

        For each position at coordinate c = (x,y,z), the new position is:
            c' = mat @ c  (clamped to [0, n-1])

        Args:
            mat: [3, 3] signed permutation matrix (±1 entries, one per row/col)

        Returns:
            permutation vector of length N
        """
        # Transform all coordinates: [N, 3] @ [3, 3]^T = [N, 3]
        new_coords = self._coords.float() @ mat.float().T
        new_coords = new_coords.long()

        perm = torch.zeros(self.N, dtype=torch.long)
        for i in range(self.N):
            nx, ny, nz = new_coords[i].tolist()
            perm[i] = self._coord_to_idx(nx, ny, nz)

        return perm

    def get_generators(self) -> List[torch.Tensor]:
        """Return generators for the full O_h group.

        Three generators:
            1. C4 rotation around z-axis (90° counterclockwise)
            2. C4 rotation around x-axis (90°)
            3. Reflection through xy-plane (σ_h: z → -z)

        These three generate the full O_h group of order 48.
        """
        # C4_z: (x, y, z) → (-y, x, z)
        C4_z = torch.tensor([[0, -1, 0],
                              [1,  0, 0],
                              [0,  0, 1]], dtype=torch.float32)

        # C4_x: (x, y, z) → (x, -z, y)
        C4_x = torch.tensor([[1,  0, 0],
                              [0,  0, -1],
                              [0,  1, 0]], dtype=torch.float32)

        # σ_h: (x, y, z) → (x, y, n-1-z)
        sigma_h = torch.tensor([[1,  0, 0],
                                 [0,  1, 0],
                                 [0,  0, -1]], dtype=torch.float32)

        generators = []
        for mat in [C4_z, C4_x, sigma_h]:
            generators.append(self._apply_3x3_matrix(mat))
            # Also include inverse
            generators.append(self._apply_3x3_matrix(mat.T))

        return generators

    def compute_orbits(self) -> Tuple[torch.Tensor, int]:
        """Compute orbit decomposition via BFS.

        Uses the 6 generators (3 generators + 3 inverses) to discover
        all positions reachable from each seed.
        """
        from groups.base import compute_orbits_from_generators
        return compute_orbits_from_generators(self.N, self.get_generators())

    def orbit_summary(self) -> dict:
        """Return a summary of orbit structure.

        Returns dict with:
            - n_positions: total number of positions (N)
            - n_orbits: number of orbits (K)
            - compression_ratio: K/N
            - orbit_sizes: list of sizes for each orbit
        """
        orbit_ids, n_orbits = self.compute_orbits()
        sizes = [(orbit_ids == k).sum().item() for k in range(n_orbits)]
        return {
            'n_positions': self.N,
            'n_orbits': n_orbits,
            'compression_ratio': n_orbits / self.N,
            'orbit_sizes': sorted(sizes, reverse=True),
            'unique_sizes': sorted(set(sizes), reverse=True),
        }

    def __repr__(self) -> str:
        orbit_ids, n_orbits = self.compute_orbits()
        return (f"OctahedralGroup(n={self.n}, N={self.N}, "
                f"orbits={n_orbits}, name='{self.name}')")
