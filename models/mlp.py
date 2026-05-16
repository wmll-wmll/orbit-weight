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
