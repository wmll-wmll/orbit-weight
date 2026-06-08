"""
Base protocol and utilities for group-theoretic orbit decomposition.

Provides:
    - Group protocol: unified interface for all finite groups
    - compute_orbits(): BFS orbit decomposition from generators
    - create_orbit_model(): factory for OrbitMLP from any Group

The BFS orbit decomposition algorithm is identical to the one used in
cube/cube3d.py and validate_dimer.py, factored out for reuse.
"""

import torch
import torch.nn as nn
from typing import List, Tuple, Protocol


class Group(Protocol):
    """Unified interface for all finite groups acting on [N].

    A Group is defined by a set of generator permutations on N positions.
    Orbit decomposition is computed by BFS from each unvisited position
    using the generators as adjacency operators.

    Attributes:
        N: number of positions the group acts on
        name: human-readable group name
    """
    N: int
    name: str

    def get_generators(self) -> List[torch.Tensor]:
        """Return generator permutations for BFS orbit computation.

        Each generator is a permutation vector of length N, where
        gen[i] = image of position i under the generator.

        The BFS uses these as adjacency operators to discover all
        positions reachable from a seed.
        """
        ...

    def compute_orbits(self) -> Tuple[torch.Tensor, int]:
        """Compute orbit decomposition under the group action.

        Returns:
            orbit_ids: [N] tensor mapping each position to its orbit index (0..K-1)
            n_orbits: total number of orbits K
        """
        return compute_orbits_from_generators(self.N, self.get_generators())


def compute_orbits_from_generators(
    N: int,
    generators: List[torch.Tensor],
) -> Tuple[torch.Tensor, int]:
    """BFS orbit decomposition from a list of generator permutations.

    Algorithm:
        1. For each unvisited position, start a BFS
        2. From current position, apply all generators to find neighbors
        3. All reachable positions belong to the same orbit

    Complexity: O(|generators| * N) = O(N) since |generators| is constant.

    Args:
        N: number of positions
        generators: list of permutation tensors, each of length N

    Returns:
        orbit_ids: [N] tensor, orbit_ids[i] = orbit index of position i
        n_orbits: total number of orbits
    """
    orbit_ids = torch.full((N,), -1, dtype=torch.long)
    orbit_id = 0

    for pos in range(N):
        if orbit_ids[pos] >= 0:
            continue
        # Start new orbit
        stack = [pos]
        orbit_ids[pos] = orbit_id
        while stack:
            p = stack.pop()
            for gen in generators:
                neighbor = int(gen[p])
                if orbit_ids[neighbor] < 0:
                    orbit_ids[neighbor] = orbit_id
                    stack.append(neighbor)
        orbit_id += 1

    return orbit_ids, orbit_id


def compute_orbits(N: int, generators: List[torch.Tensor]) -> Tuple[torch.Tensor, int]:
    """Convenience wrapper for compute_orbits_from_generators.

    This is the primary public API. Usage:
        orbit_ids, n_orbits = compute_orbits(N, generators)
    """
    return compute_orbits_from_generators(N, generators)


# ── OrbitLinear layer ─────────────────────────────────────────────

class OrbitLinear(nn.Module):
    """Per-position linear layer with orbit-based weight sharing.

    Each orbit gets its own weight matrix and bias vector.
    Positions within the same orbit share the same parameters.

    Args:
        orbit_ids: [N] tensor mapping positions to orbit indices
        n_orbits: total number of orbits K
        D: feature dimension (input and output are both D)
    """

    def __init__(self, orbit_ids: torch.Tensor, n_orbits: int, D: int):
        super().__init__()
        self.register_buffer('orbit_ids', orbit_ids, persistent=False)
        self.n_orbits = n_orbits
        self.weight = nn.Parameter(torch.randn(n_orbits, D, D) / (D ** 0.5))
        self.bias = nn.Parameter(torch.zeros(n_orbits, D))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with orbit-shared weights.

        Args:
            x: [B, N, D] input tensor

        Returns:
            [B, N, D] output tensor
        """
        # w: [N, D, D] — per-position weight (shared within orbits)
        w = self.weight[self.orbit_ids]
        b = self.bias[self.orbit_ids]  # [N, D]
        # einsum avoids materializing [B, N, D, D] intermediate
        return torch.einsum('bnd,ndm->bnm', x, w) + b.unsqueeze(0)


# ── Model factory ─────────────────────────────────────────────────

def create_orbit_model(
    group: Group,
    D: int,
    n_layers: int,
    use_gathers: bool = False,
    gather_perms: List[torch.Tensor] = None,
) -> nn.Module:
    """Create an OrbitMLP (or OrbitCubeMLP) for any group.

    Architecture: for each layer:
        LayerNorm → OrbitLinear → GELU → (optional: GatherOp)

    Args:
        group: Group instance providing orbit decomposition
        D: feature dimension
        n_layers: number of layers
        use_gathers: if True, adds gather rotations (OrbitCubeMLP style)
        gather_perms: list of permutation tensors for gather layers
                      (required if use_gathers=True, length >= n_layers)

    Returns:
        nn.Sequential model
    """
    orbit_ids, n_orbits = group.compute_orbits()

    layers = []
    for i in range(n_layers):
        layers.extend([
            nn.LayerNorm(D),
            OrbitLinear(orbit_ids, n_orbits, D),
            nn.GELU(),
        ])
        if use_gathers and gather_perms is not None and i < len(gather_perms):
            layers.append(_GatherLayer(gather_perms[i]))

    return nn.Sequential(*layers)


class _GatherLayer(nn.Module):
    """Apply a pre-computed gather along dim=1.

    (Identical to models/mlp.py:_GatherLayer)
    """
    def __init__(self, perm: torch.Tensor):
        super().__init__()
        self.register_buffer('perm', perm, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, D = x.shape
        idx = self.perm.unsqueeze(0).unsqueeze(-1).expand(B, -1, D)
        return torch.gather(x, 1, idx)
