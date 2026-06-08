"""Shared MLP model definitions — single source of truth.

Variants:
- StandardMLP: LN → Linear → GELU → (Dropout), with residual
- CubeMLP: LN → Linear → GELU → Gather(rotate), with residual
- CubeMLP_Fused: LN → Linear → GELU, gather applied after compute

All variants share the same parameter count and FLOPs (modulo gather vs dropout).
"""

import torch
import torch.nn as nn

from cube.cube3d import CubePermutations


# ═══════════════════════════════════════════════════════════════
# Standard MLP
# ═══════════════════════════════════════════════════════════════

class StandardMLP(nn.Module):
    """Baseline: standard MLP with optional dropout."""

    def __init__(self, n_positions: int, d_model: int, n_layers: int,
                 dropout: float = 0.0, activation=nn.GELU):
        super().__init__()
        self.layers = nn.ModuleList()
        for _ in range(n_layers):
            self.layers.append(nn.Sequential(
                nn.LayerNorm(d_model),
                nn.Linear(d_model, d_model),
                activation(),
            ))
            if dropout > 0:
                self.layers[-1].append(nn.Dropout(dropout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x) + x
        return x


# ═══════════════════════════════════════════════════════════════
# Cube MLP (fused — gather after compute)
# ═══════════════════════════════════════════════════════════════

class _GatherLayer(nn.Module):
    """Apply a pre-computed gather along dim=1."""
    def __init__(self, perm: torch.Tensor):
        super().__init__()
        self.register_buffer('perm', perm, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, D = x.shape
        idx = self.perm.unsqueeze(0).unsqueeze(-1).expand(B, -1, D)
        return torch.gather(x, 1, idx)


class CubeMLP(nn.Module):
    """Cube-structured MLP: cube rotations replace dropout as regularization.

    Each layer: LN → Linear → GELU → gather(cube_rotation)

    The gather happens AFTER compute on contiguous data. This is the
    "fused" approach that the compiler IR would produce after applying
    the ReorderPass.
    """

    def __init__(self, n_positions: int, d_model: int, n_layers: int,
                 n_cube: int = 3, activation=nn.GELU):
        super().__init__()
        assert n_positions == n_cube ** 3, \
            f"n_positions ({n_positions}) must equal n_cube^3 ({n_cube**3})"

        cube = CubePermutations(n_cube)
        moves = ['U', 'R', 'F', 'D', 'L', 'B']

        self.layers = nn.ModuleList()
        for i in range(n_layers):
            block = nn.Sequential(
                nn.LayerNorm(d_model),
                nn.Linear(d_model, d_model),
                activation(),
            )
            perm = cube.get_rotation(moves[i % 6])
            block.append(_GatherLayer(perm))
            self.layers.append(block)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x) + x
        return x


# ═══════════════════════════════════════════════════════════════
# Sequential constructor (for simpler API, used by bench_validation.py)
# ═══════════════════════════════════════════════════════════════

def make_standard_mlp(n_pos: int, d_model: int, n_layers: int,
                      dropout: float = 0.0) -> nn.Module:
    """Build StandardMLP as nn.Sequential (no residuals)."""
    layers = []
    for _ in range(n_layers):
        layers.extend([
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
        ])
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
    return nn.Sequential(*layers)


def make_cube_mlp(n_pos: int, d_model: int, n_layers: int,
                  n_cube: int = 3) -> nn.Module:
    """Build CubeMLP as nn.Sequential (no residuals).

    LN → Linear → GELU → gather(rotate)
    """
    assert n_pos == n_cube ** 3
    cube = CubePermutations(n_cube)
    moves = ['U', 'R', 'F', 'D', 'L', 'B']
    layers = []
    for i in range(n_layers):
        layers.extend([
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
        ])
        perm = cube.get_rotation(moves[i % 6])
        layers.append(_GatherLayer(perm))
    return nn.Sequential(*layers)


# ═══════════════════════════════════════════════════════════════
# OrbitLinear layer and orbit-based model constructor
# ═══════════════════════════════════════════════════════════════

class OrbitLinear(nn.Module):
    """Per-position linear layer with orbit-based weight sharing.

    Each orbit has its own weight matrix and bias vector.
    Positions sharing the same orbit index use the same parameters.

    This is the core building block for orbit-shared models.
    When orbit_ids maps all positions to the same orbit (K=1),
    it reduces to standard shared-weight Linear.
    When each position is its own orbit (K=N),
    it reduces to per-position independent Linear.

    Args:
        orbit_ids: [N] tensor mapping positions to orbit indices (0..K-1)
        n_orbits: total number of orbits K
        D: feature dimension (input = output = D)
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
            x: [B, N, D]

        Returns:
            [B, N, D]
        """
        w = self.weight[self.orbit_ids]  # [N, D, D]
        b = self.bias[self.orbit_ids]    # [N, D]
        # einsum avoids [B, N, D, D] intermediate
        return torch.einsum('bnd,ndm->bnm', x, w) + b.unsqueeze(0)


def make_orbit_mlp(n_pos: int, d_model: int, n_layers: int,
                   orbit_ids: torch.Tensor = None,
                   n_orbits: int = None,
                   n_cube: int = None,
                   use_gathers: bool = False) -> nn.Module:
    """Build OrbitMLP (or OrbitCubeMLP) as nn.Sequential.

    LN → OrbitLinear → GELU → (optional: gather)

    If orbit_ids is None, computes orbits from the cube face rotation group
    using n_cube (requires n_pos == n_cube**3).

    Args:
        n_pos: number of positions N
        d_model: feature dimension D
        n_layers: number of layers
        orbit_ids: [N] tensor, orbit assignment per position (optional)
        n_orbits: number of orbits K (required if orbit_ids is provided)
        n_cube: cube side length (used if orbit_ids is None)
        use_gathers: if True, adds gather rotations after each GELU

    Returns:
        nn.Sequential model
    """
    if orbit_ids is None:
        if n_cube is None:
            raise ValueError("Either orbit_ids or n_cube must be provided")
        assert n_pos == n_cube ** 3
        cube = CubePermutations(n_cube)
        orbit_ids = cube.get_orbit_ids()
        n_orbits = cube.get_orbit_count()
    else:
        assert n_orbits is not None, "n_orbits required when orbit_ids provided"

    layers = []
    if use_gathers and n_cube is not None:
        cube = CubePermutations(n_cube)
        moves = ['U', 'R', 'F', 'D', 'L', 'B']

    for i in range(n_layers):
        layers.extend([
            nn.LayerNorm(d_model),
            OrbitLinear(orbit_ids, n_orbits, d_model),
            nn.GELU(),
        ])
        if use_gathers and n_cube is not None:
            perm = cube.get_rotation(moves[i % 6])
            layers.append(_GatherLayer(perm))

    return nn.Sequential(*layers)
