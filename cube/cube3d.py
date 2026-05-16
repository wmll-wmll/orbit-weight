"""
3×3×3 Rubik's cube permutation generator using group theory.

Coordinate convention:
    x: left→right  [0, n)
    y: bottom→top  [0, n)
    z: front→back  [0, n)

Linear indexing: idx = z*n² + y*n + x  (z-major for coalesced access on z-first layouts).

Each face rotation is a permutation of the 3D grid positions on that face.
Unaffected positions map to themselves (identity).
"""

import torch
from typing import Tuple, List, Optional, Dict


class CubePermutations:
    """Generate permutation vectors for N×N×N cube face rotations.

    Six generators (U/D/L/R/F/B + inverses) form the full cube group.
    Each generator returns a permutation vector of length N³.
    """

    def __init__(self, n: int = 3):
        """
        Args:
            n: cube side length (default 3 for standard Rubik's cube)
        """
        self.n = n
        self.total = n ** 3

        # Build lookup tables for coordinate conversion
        xyz = torch.tensor(
            [[x, y, z] for z in range(n) for y in range(n) for x in range(n)],
            dtype=torch.long,
        )  # [N³, 3] — z-major ordering
        self._xyz_of_idx = xyz
        self._idx_of_xyz = torch.full((n, n, n), -1, dtype=torch.long)
        for i, (x, y, z) in enumerate(xyz.tolist()):
            self._idx_of_xyz[x, y, z] = i

    def to_flat(self, x: int, y: int, z: int) -> int:
        """3D coordinate to linear index."""
        return z * self.n * self.n + y * self.n + x

    def to_xyz(self, idx: int) -> Tuple[int, int, int]:
        """Linear index to 3D coordinate."""
        xyz = self._xyz_of_idx[idx].tolist()
        return xyz[0], xyz[1], xyz[2]

    # ──── Six basic rotations ────

    def rotation_U(self, clockwise: bool = True) -> torch.Tensor:
        """U face (top, y=n-1) rotation.

        (x, n-1, z) -> (z, n-1, n-1-x) clockwise when viewed from above.
        """
        n = self.n
        perm = torch.arange(self.total).clone()
        y = n - 1
        for x in range(n):
            for z in range(n):
                src = (x, y, z)
                if clockwise:
                    dst = (z, y, n - 1 - x)
                else:
                    dst = (n - 1 - z, y, x)
                perm[self.to_flat(*dst)] = self.to_flat(*src)
        return perm

    def rotation_D(self, clockwise: bool = True) -> torch.Tensor:
        """D face (bottom, y=0) rotation.

        (x, 0, z) -> (n-1-z, 0, x) clockwise when viewed from below.
        """
        n = self.n
        perm = torch.arange(self.total).clone()
        y = 0
        for x in range(n):
            for z in range(n):
                src = (x, y, z)
                if clockwise:
                    dst = (n - 1 - z, y, x)
                else:
                    dst = (z, y, n - 1 - x)
                perm[self.to_flat(*dst)] = self.to_flat(*src)
        return perm

    def rotation_F(self, clockwise: bool = True) -> torch.Tensor:
        """F face (front, z=n-1) rotation.

        (x, y, n-1) -> (n-1-y, x, n-1) clockwise when viewed from front.
        """
        n = self.n
        perm = torch.arange(self.total).clone()
        z = n - 1
        for x in range(n):
            for y in range(n):
                src = (x, y, z)
                if clockwise:
                    dst = (n - 1 - y, x, z)
                else:
                    dst = (y, n - 1 - x, z)
                perm[self.to_flat(*dst)] = self.to_flat(*src)
        return perm

    def rotation_B(self, clockwise: bool = True) -> torch.Tensor:
        """B face (back, z=0) rotation.

        (x, y, 0) -> (y, n-1-x, 0) clockwise when viewed from behind.
        """
        n = self.n
        perm = torch.arange(self.total).clone()
        z = 0
        for x in range(n):
            for y in range(n):
                src = (x, y, z)
                if clockwise:
                    dst = (y, n - 1 - x, z)
                else:
                    dst = (n - 1 - y, x, z)
                perm[self.to_flat(*dst)] = self.to_flat(*src)
        return perm

    def rotation_L(self, clockwise: bool = True) -> torch.Tensor:
        """L face (left, x=0) rotation.

        (0, y, z) -> (0, z, n-1-y) clockwise when viewed from left.
        """
        n = self.n
        perm = torch.arange(self.total).clone()
        x = 0
        for y in range(n):
            for z in range(n):
                src = (x, y, z)
                if clockwise:
                    dst = (x, z, n - 1 - y)
                else:
                    dst = (x, n - 1 - z, y)
                perm[self.to_flat(*dst)] = self.to_flat(*src)
        return perm

    def rotation_R(self, clockwise: bool = True) -> torch.Tensor:
        """R face (right, x=n-1) rotation.

        (n-1, y, z) -> (n-1, n-1-z, y) clockwise when viewed from right.
        """
        n = self.n
        perm = torch.arange(self.total).clone()
        x = n - 1
        for y in range(n):
            for z in range(n):
                src = (x, y, z)
                if clockwise:
                    dst = (x, n - 1 - z, y)
                else:
                    dst = (x, z, n - 1 - y)
                perm[self.to_flat(*dst)] = self.to_flat(*src)
        return perm

    # ──── Named access ────

    def get_rotation(self, name: str) -> torch.Tensor:
        """Get rotation by name: 'U', 'U´', 'U2', 'D', 'F', etc.

        Suffix '´' = counter-clockwise, '2' = double (180°).
        """
        base = name[0].upper()
        if base == 'U':
            fn = self.rotation_U
        elif base == 'D':
            fn = self.rotation_D
        elif base == 'F':
            fn = self.rotation_F
        elif base == 'B':
            fn = self.rotation_B
        elif base == 'L':
            fn = self.rotation_L
        elif base == 'R':
            fn = self.rotation_R
        else:
            raise ValueError(f"Unknown rotation: {name}")

        suffix = name[1:]
        if suffix in ("", "'"):
            clockwise = suffix == ""
            perm = fn(clockwise=clockwise)
            if suffix == "2" or suffix == "2'":
                perm = self.compose(perm, perm)
        elif suffix in ("2", "2'"):
            perm = fn(clockwise=True)
            perm = self.compose(perm, perm)
        else:
            raise ValueError(f"Unknown rotation suffix: {suffix}")

        if suffix == "'":
            perm = fn(clockwise=False)

        return perm

    # ──── Composition ────

    def compose(self, *perms: torch.Tensor) -> torch.Tensor:
        """Compose permutations: perm_last ∘ ... ∘ perm_first."""
        result = perms[0]
        for p in perms[1:]:
            result = result[p]
        return result

    def random_scramble(self, moves: int = 20) -> torch.Tensor:
        """Generate a random scramble permutation as a sequence of random moves."""
        import random
        faces = ['U', 'D', 'F', 'B', 'L', 'R']
        perm = torch.arange(self.total).clone()
        for _ in range(moves):
            face = random.choice(faces)
            clockwise = random.choice([True, False])
            move_perm = self.get_rotation(face if clockwise else face + "'")
            perm = perm[move_perm]
        return perm

    # ──── All generator permutations ────

    def all_generators(self) -> Dict[str, torch.Tensor]:
        """Return all 12 generator permutations (6 faces × 2 directions)."""
        result = {}
        for face in ['U', 'D', 'F', 'B', 'L', 'R']:
            result[face] = self.get_rotation(face)
            result[face + "'"] = self.get_rotation(face + "'")
        return result

    # ──── Validation ────

    def is_valid_permutation(self, perm: torch.Tensor) -> bool:
        """Verify perm is a bijection on [0, total)."""
        return (
            perm.numel() == self.total
            and perm.min().item() >= 0
            and perm.max().item() < self.total
            and len(perm.unique()) == self.total
        )

    def verify_equivariance(self, perm: torch.Tensor, x: torch.Tensor) -> bool:
        """Check that applying the same rotation to data yields consistent results.

        For input x of shape [B, N³, D] where each position corresponds to a cube
        coordinate, applying perm should shuffle the neuron activations consistently
        with the geometric rotation.
        """
        from .perm_matrix import PermutationMatrix

        pm = PermutationMatrix(perm)
        out_direct = pm.apply(x)

        # Verify against torch.gather
        indices = perm.unsqueeze(0).unsqueeze(-1).expand(x.size(0), -1, x.size(2)).to(x.device)
        out_gather = torch.gather(x, 1, indices)

        return torch.allclose(out_direct, out_gather, atol=1e-6)


def demo_cube_permutations(n: int = 3):
    """Print a human-readable summary of U-face rotation."""
    cube = CubePermutations(n)
    perm_u = cube.rotation_U(clockwise=True)
    print(f"U-face rotation ({n}×{n}×{n} cube)")
    print(f"  Total positions: {cube.total}")
    print(f"  Affected positions: {(perm_u != torch.arange(cube.total)).sum().item()}")
    print(f"  Unaffected: {(perm_u == torch.arange(cube.total)).sum().item()}")
    print(f"  Valid permutation: {cube.is_valid_permutation(perm_u)}")
    return cube, perm_u


if __name__ == "__main__":
    demo_cube_permutations(3)
