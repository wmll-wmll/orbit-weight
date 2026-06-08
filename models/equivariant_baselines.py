"""Equivariant baseline models for comparison with orbit weight sharing.

Implements four SOTA baselines that isolate different aspects of equivariance:
- GCNNBaseline:     G-CNN on C4 group with rotated weight sharing
- DeepSetsBaseline: Permutation-invariant Deep Sets architecture
- SE3TransformerAdapter: Grid-adapted attention with learnable geometric priors
- RandomBaseline:   Random position groupings (isolates algebraic structure effect)

All models follow the interface:
    model = BaselineClass(n_positions, d_model, n_layers, ...)
    output = model(x)           # x: [B, N, D], output: [B, N, D] or [B, D]
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ═══════════════════════════════════════════════════════════════
# Grid utility helpers
# ═══════════════════════════════════════════════════════════════

def _infer_grid(n_positions: int):
    """Infer grid dimensionality and shape from position count.

    Tries 3D cube, then 2D square, then falls back to 1D.

    Returns:
        (ndim, shape): e.g. (3, (D, H, W)), (2, (H, W)), or (1, (N,))
    """
    c = round(n_positions ** (1.0 / 3.0))
    if c ** 3 == n_positions:
        return 3, (c, c, c)
    s = int(math.isqrt(n_positions))
    if s * s == n_positions:
        return 2, (s, s)
    return 1, (n_positions,)


def _build_grid_coords(n_positions: int):
    """Build [N, ndim] float tensor of normalised grid coordinates.

    Coordinates are centered and scaled to [-1, 1] in each dimension.
    """
    ndim, shape = _infer_grid(n_positions)
    if ndim == 3:
        D, H, W = shape
        z = torch.arange(D, dtype=torch.float)
        y = torch.arange(H, dtype=torch.float)
        x = torch.arange(W, dtype=torch.float)
        Z, Y, X = torch.meshgrid(z, y, x, indexing='ij')
        coords = torch.stack([Z.reshape(-1), Y.reshape(-1), X.reshape(-1)], dim=1)
    elif ndim == 2:
        H, W = shape
        y = torch.arange(H, dtype=torch.float)
        x = torch.arange(W, dtype=torch.float)
        Y, X = torch.meshgrid(y, x, indexing='ij')
        coords = torch.stack([Y.reshape(-1), X.reshape(-1)], dim=1)
    else:
        coords = torch.arange(n_positions, dtype=torch.float).unsqueeze(1)
    # Normalise to [-1, 1]
    if coords.max() > 1.0:
        coords = 2.0 * coords / (coords.max(dim=0).values + 1e-8) - 1.0
    return coords, ndim, shape


def _cube_orbit_count(n: int) -> int:
    """Return the number of position orbits under the cube face-rotation group.

    An n x n x n cube partitions into at most 4 orbit types:
      corners (8), edges (12(n-2)), face-centres (6(n-2)^2), interior ((n-2)^3).
    """
    if n <= 1:
        return 1
    count = 1  # corners always present for n >= 2
    if n > 2:
        if 12 * (n - 2) > 0:
            count += 1  # edges
        if 6 * (n - 2) ** 2 > 0:
            count += 1  # face centres
        if (n - 2) ** 3 > 0:
            count += 1  # interior
    return count


# ═══════════════════════════════════════════════════════════════
# 1. GCNNBaseline -- G-CNN on C4 group
# ═══════════════════════════════════════════════════════════════

class GCNNBaseline(nn.Module):
    """G-CNN baseline on the C4 group (4 discrete rotations).

    For a 2D grid input flattened to [B, N, D]:
      - Builds C4 rotation permutations (0, 90, 180, 270 degrees).
      - Each layer applies the same MLP to the original view and three
        rotated views, rotates each output back to the canonical frame,
        and averages -- implementing weight-tying across rotations.
      - Residual connections around each layer.

    This captures the key G-CNN idea (shared weights across group-
    transformed copies) without the full group-convolution machinery.

    Args:
        n_positions: total spatial positions (must be a perfect square).
        d_model: feature dimension.
        n_layers: number of G-CNN layers.
        activation: activation class (default: nn.GELU).
    """

    def __init__(self, n_positions: int, d_model: int, n_layers: int,
                 activation=nn.GELU):
        super().__init__()
        ndim, shape = _infer_grid(n_positions)
        if ndim != 2:
            raise ValueError(
                f"GCNNBaseline requires a 2D grid, but n_positions={n_positions} "
                f"was inferred as {ndim}D (shape={shape})."
            )
        H, W = shape
        if H != W:
            raise ValueError(
                f"GCNNBaseline requires a square grid, got {H}x{W}."
            )
        n = H

        # ---- Build C4 gather-index buffers ----
        yg, xg = torch.meshgrid(torch.arange(n), torch.arange(n), indexing='ij')
        # perm[dst] = src  (consistent with _GatherLayer convention used in mlp.py)
        perm_r0 = torch.arange(n_positions)                               # identity
        # 90 deg CW:  (x_src, y_src) -> (n-1-y_src, x_src)
        perm_r90 = ((n - 1 - xg) * n + yg).reshape(-1)
        # 180 deg:    (x_src, y_src) -> (n-1-x_src, n-1-y_src)
        perm_r180 = ((n - 1 - yg) * n + (n - 1 - xg)).reshape(-1)
        # 270 deg CW: (x_src, y_src) -> (y_src, n-1-x_src)
        perm_r270 = (xg * n + (n - 1 - yg)).reshape(-1)

        self.register_buffer('perm_r0', perm_r0)
        self.register_buffer('perm_r90', perm_r90)
        self.register_buffer('perm_r180', perm_r180)
        self.register_buffer('perm_r270', perm_r270)

        # Inverse gather indices for rotating back:
        # inv(90 deg)=270 deg, inv(180 deg)=180 deg, inv(270 deg)=90 deg
        self.register_buffer('perm_r90_inv', perm_r270)
        self.register_buffer('perm_r180_inv', perm_r180)
        self.register_buffer('perm_r270_inv', perm_r90)

        # ---- Layers ----
        self.layers = nn.ModuleList()
        for _ in range(n_layers):
            self.layers.append(nn.Sequential(
                nn.LayerNorm(d_model),
                nn.Linear(d_model, d_model),
                activation(),
            ))

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _gather(self, x: torch.Tensor, perm: torch.Tensor) -> torch.Tensor:
        """Gather positions along dim=1 using pre-built perm [N]."""
        B, N, D = x.shape
        idx = perm.unsqueeze(0).unsqueeze(-1).expand(B, -1, D)
        return torch.gather(x, 1, idx)

    def _rotated_pool(self, x: torch.Tensor, layer: nn.Module) -> torch.Tensor:
        """Apply *layer* to x and its three C4 rotations, rotate back, average."""
        out_r0 = layer(x)

        x_r90 = self._gather(x, self.perm_r90)
        out_r90 = self._gather(layer(x_r90), self.perm_r90_inv)

        x_r180 = self._gather(x, self.perm_r180)
        out_r180 = self._gather(layer(x_r180), self.perm_r180_inv)

        x_r270 = self._gather(x, self.perm_r270)
        out_r270 = self._gather(layer(x_r270), self.perm_r270_inv)

        return (out_r0 + out_r90 + out_r180 + out_r270) / 4.0

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: [B, N, D] input features.

        Returns:
            [B, N, D] output features.
        """
        for layer in self.layers:
            x = self._rotated_pool(x, layer) + x
        return x


# ═══════════════════════════════════════════════════════════════
# 2. DeepSetsBaseline -- permutation-invariant Deep Sets
# ═══════════════════════════════════════════════════════════════

class DeepSetsBaseline(nn.Module):
    """Deep Sets baseline -- permutation-invariant by construction.

    Architecture (Zaheer et al. 2017):
        rho(x_i): per-position shared MLP (LN -> Linear -> GELU, n_layers deep)
        pool:     sum over positions
        phi:      final linear projection

    Args:
        n_positions: total number of positions (used only for sizing context;
                     the model is permutation-invariant w.r.t. this dimension).
        d_model: feature dimension.
        n_layers: depth of the per-position rho network.
        pool_output: if True (default) returns [B, D] after phi(sum(rho(x))).
                     if False returns [B, N, D] from rho alone.
        activation: activation class (default: nn.GELU).
    """

    def __init__(self, n_positions: int, d_model: int, n_layers: int,
                 pool_output: bool = True, activation=nn.GELU):
        super().__init__()
        self.pool_output = pool_output

        # rho: per-position MLP with residual connections
        self.rho_layers = nn.ModuleList()
        for _ in range(n_layers):
            self.rho_layers.append(nn.Sequential(
                nn.LayerNorm(d_model),
                nn.Linear(d_model, d_model),
                activation(),
            ))

        # phi: post-pooling projection (single linear layer)
        self.phi = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: [B, N, D] input features.

        Returns:
            [B, D] if pool_output=True (default), else [B, N, D].
        """
        for layer in self.rho_layers:
            x = layer(x) + x
        if self.pool_output:
            x = x.sum(dim=1)          # [B, D]
            x = self.phi(x)           # [B, D]
        return x


# ═══════════════════════════════════════════════════════════════
# 3. SE3TransformerAdapter -- grid-adapted attention with geometric priors
# ═══════════════════════════════════════════════════════════════

class SE3TransformerAdapter(nn.Module):
    """Grid-adapted SE(3)-Transformer baseline.

    Replaces spherical-harmonic embeddings with learnable relative-position
    encodings based on pairwise Euclidean distance (RBF kernel).
    Multi-head self-attention over positions is biased by a learned
    function of inter-position distance, injecting a geometric prior.

    This captures the key equivariance mechanism (distance-based attention
    bias) without the full tensor-product / Clebsch-Gordan complexity.

    Args:
        n_positions: number of spatial positions.
        d_model: feature dimension (must be divisible by n_heads).
        n_layers: number of transformer layers.
        n_heads: number of attention heads (default: 4).
        n_rbf: number of RBF kernels for distance encoding (default: 16).
        dropout: attention dropout rate (default: 0.0).
        activation: activation class for FFN (default: nn.GELU).
    """

    def __init__(self, n_positions: int, d_model: int, n_layers: int,
                 n_heads: int = 4, n_rbf: int = 16, dropout: float = 0.0,
                 activation=nn.GELU):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(
                f"d_model ({d_model}) must be divisible by n_heads ({n_heads})."
            )

        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        # ---- Grid geometry ----
        coords, ndim, shape = _build_grid_coords(n_positions)
        self.register_buffer('coords', coords)       # [N, ndim]
        self.ndim = ndim

        # Pairwise distances [N, N]
        diffs = coords.unsqueeze(0) - coords.unsqueeze(1)   # [N, N, ndim]
        dists = torch.norm(diffs, dim=-1)                    # [N, N]
        self.register_buffer('pairwise_dists', dists)

        # ---- RBF distance encoding ----
        max_dist = dists.max().item()
        self.register_buffer(
            'rbf_centres',
            torch.linspace(0.0, max_dist, n_rbf),
        )
        # RBF bandwidth: spacing between adjacent centres
        sigma = max_dist / max(n_rbf - 1, 1) if n_rbf > 1 else 1.0
        self.rbf_sigma = sigma

        # Precompute RBF features [N, N, n_rbf] -- deterministic
        rbf = torch.exp(
            -((dists.unsqueeze(-1) - self.rbf_centres) ** 2)
            / (2.0 * sigma ** 2 + 1e-8)
        )
        self.register_buffer('rbf_features', rbf)

        # MLP: RBF features -> per-head attention bias  [N, N, n_rbf] -> [N, N, n_heads]
        self.bias_mlp = nn.Sequential(
            nn.Linear(n_rbf, d_model // n_heads),
            nn.ReLU(),
            nn.Linear(d_model // n_heads, n_heads),
        )

        # ---- Transformer layers ----
        self.layers = nn.ModuleList()
        for _ in range(n_layers):
            self.layers.append(_SE3TransformerLayer(
                d_model, n_heads, dropout, activation,
            ))

    def _get_attn_bias(self) -> torch.Tensor:
        """Compute per-head attention bias from RBF features.

        Returns:
            [n_heads, N, N] bias to add to pre-softmax logits.
        """
        bias = self.bias_mlp(self.rbf_features)   # [N, N, n_heads]
        return bias.permute(2, 0, 1)               # [n_heads, N, N]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: [B, N, D] input features.

        Returns:
            [B, N, D] output features.
        """
        attn_bias = self._get_attn_bias()  # [n_heads, N, N]
        for layer in self.layers:
            x = layer(x, attn_bias) + x
        return x


class _SE3TransformerLayer(nn.Module):
    """Single transformer layer with distance-biased multi-head attention."""

    def __init__(self, d_model: int, n_heads: int, dropout: float,
                 activation):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        self.norm1 = nn.LayerNorm(d_model)
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout_attn = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            activation(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(4 * d_model, d_model),
        )
        self.dropout_ffn = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor,
                attn_bias: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: [B, N, D] input.
            attn_bias: [n_heads, N, N] geometric bias.

        Returns:
            [B, N, D] (pre-residual -- caller adds the residual).
        """
        B, N, D = x.shape

        # Multi-head self-attention with geometric bias
        x_norm = self.norm1(x)
        q, k, v = self.qkv(x_norm).chunk(3, dim=-1)            # each [B, N, D]

        q = q.view(B, N, self.n_heads, self.d_head).transpose(1, 2)  # [B, h, N, d]
        k = k.view(B, N, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(B, N, self.n_heads, self.d_head).transpose(1, 2)

        attn_logits = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_head)
        attn_logits = attn_logits + attn_bias.unsqueeze(0)           # broadcast batch
        attn_weights = F.softmax(attn_logits, dim=-1)
        attn_weights = self.dropout_attn(attn_weights)

        out = torch.matmul(attn_weights, v)                          # [B, h, N, d]
        out = out.transpose(1, 2).contiguous().view(B, N, D)
        out = self.out_proj(out)

        # Feed-forward network
        out = out + self.dropout_ffn(self.ffn(self.norm2(out)))
        return out


# ═══════════════════════════════════════════════════════════════
# 4. RandomBaseline -- random position groupings
# ═══════════════════════════════════════════════════════════════

class RandomBaseline(nn.Module):
    """Random-baseline with the same parameter count as OrbitMLP.

    Uses fixed random position-group assignments instead of algebraically
    derived orbit IDs.  This isolates the effect of algebraic structure
    (orbit weight sharing) from raw parameter count: if OrbitMLP outperforms
    RandomBaseline, it is due to the group structure, not just having more
    parameters than a standard shared-weight MLP.

    Internally uses _GroupLinear (identical pattern to OrbitLinear) with
    randomly assigned group_ids.

    Args:
        n_positions: number of spatial positions.
        d_model: feature dimension.
        n_layers: number of layers.
        n_groups: number of random groups K (default: inferred to match
                  the cube orbit count when n_positions is a perfect cube,
                  otherwise n_positions // 4 clamped to [1, n_positions]).
        seed: random seed for deterministic group assignment (default: 42).
        activation: activation class (default: nn.GELU).
    """

    def __init__(self, n_positions: int, d_model: int, n_layers: int,
                 n_groups: int = None, seed: int = 42,
                 activation=nn.GELU):
        super().__init__()

        # ---- Determine number of random groups ----
        if n_groups is None:
            ndim, shape = _infer_grid(n_positions)
            if ndim == 3:
                n_groups = _cube_orbit_count(shape[0])
            else:
                n_groups = max(1, min(n_positions, n_positions // 4))
        n_groups = max(1, min(n_positions, int(n_groups)))

        # ---- Fixed random assignment ----
        gen = torch.Generator()
        gen.manual_seed(seed)
        group_ids = torch.randint(0, n_groups, (n_positions,), generator=gen)
        self.register_buffer('group_ids', group_ids)
        self.n_groups = n_groups

        # ---- Layers (LN -> GroupLinear -> activation) ----
        self.layers = nn.ModuleList()
        for _ in range(n_layers):
            self.layers.append(nn.Sequential(
                nn.LayerNorm(d_model),
                _GroupLinear(group_ids, n_groups, d_model),
                activation(),
            ))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: [B, N, D] input features.

        Returns:
            [B, N, D] output features.
        """
        for layer in self.layers:
            x = layer(x) + x
        return x


class _GroupLinear(nn.Module):
    """Per-position linear layer with group-based weight sharing.

    Positions sharing the same group index use the same weight matrix
    and bias vector.  Identical pattern to OrbitLinear.

    Args:
        group_ids: [N] tensor mapping each position to a group (0..K-1).
        n_groups: total number of groups K.
        d_model: feature dimension (input = output = D).
    """

    def __init__(self, group_ids: torch.Tensor, n_groups: int, d_model: int):
        super().__init__()
        self.register_buffer('group_ids', group_ids, persistent=False)
        self.n_groups = n_groups
        self.weight = nn.Parameter(
            torch.randn(n_groups, d_model, d_model) / math.sqrt(d_model)
        )
        self.bias = nn.Parameter(torch.zeros(n_groups, d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: [B, N, D].

        Returns:
            [B, N, D].
        """
        w = self.weight[self.group_ids]   # [N, D, D]
        b = self.bias[self.group_ids]     # [N, D]
        return torch.einsum('bnd,ndm->bnm', x, w) + b.unsqueeze(0)
