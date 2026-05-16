"""
Trainable Rubik's cube layers for neural network integration.

Three variants:
- CubePermutation: discrete cube rotation (fast, deterministic, for data aug)
- FusedPermLN: fused permutation + LayerNorm (target for MUXI backend)
- SoftCubePermutation: differentiable soft permutation via Sinkhorn normalization
"""

import torch
import torch.nn as nn
from typing import Optional, List

from .cube3d import CubePermutations
from .perm_matrix import PermutationMatrix


class CubePermutation(nn.Module):
    """Discrete cube rotation layer.

    Applies a Rubik's cube face rotation as a permutation of neuron positions.
    Useful for data augmentation, feature mixing, and rotation-equivariant layers.

    Example:
        layer = CubePermutation(n=3)
        x = torch.randn(16, 27, 128)  # [B, 27, D]
        y = layer(x, moves=["U", "R"])  # apply U then R rotation
    """

    def __init__(self, n: int = 3):
        super().__init__()
        self.n = n
        self._cube = CubePermutations(n)

    def forward(
        self,
        x: torch.Tensor,
        moves: Optional[List[str]] = None,
    ) -> torch.Tensor:
        """Apply cube rotation(s) to input tensor.

        Args:
            x: [B, N³, D] input tensor
            moves: list of move strings like ["U", "R'", "F2"].
                   If None, applies a pre-set rotation (default: U clockwise).

        Returns:
            [B, N³, D] rotated tensor
        """
        if x.size(1) != self._cube.total:
            raise ValueError(
                f"Input dim 1 must be {self._cube.total} (N³ for N={self.n}), "
                f"got {x.size(1)}"
            )

        if moves is None:
            perm = self._cube.rotation_U(clockwise=True)
        else:
            perms = [self._cube.get_rotation(m) for m in moves]
            perm = self._cube.compose(*perms)

        pm = PermutationMatrix(perm.to(x.device), format="dense")
        return pm.apply(x)


class FusedPermLN(nn.Module):
    """Fused permutation + LayerNorm layer.

    Semantically: LN ∘ Gather(perm). By computing LN on contiguous data
    first and then gathering, we avoid the gather's memory round-trip
    for the LN computation. This is the pattern the compiler's
    ReorderPass produces.

    On MUXI GPUs, this can be implemented as a true single-kernel fusion.
    On NVIDIA, this is a simulated fusion (two separate ops) that still
    benefits from better cache locality.
    """

    def __init__(self, perm: torch.Tensor, normalized_shape: int):
        super().__init__()
        self.register_buffer('perm', perm, persistent=False)
        self.ln = nn.LayerNorm(normalized_shape)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # LN first (on contiguous data) → then gather
        y = self.ln(x)
        B, N, D = y.shape
        idx = self.perm.unsqueeze(0).unsqueeze(-1).expand(B, -1, D)
        return torch.gather(y, 1, idx)


class SoftCubePermutation(nn.Module):
    """Differentiable soft permutation layer via Sinkhorn normalization.

    Learns a doubly-stochastic matrix close to a permutation, enabling
    gradient-based optimization of the "rotation" behavior.

    Architecture:
        - N learnable query vectors q_i ∈ R^{d_q}
        - Pairwise similarity: S[i,j] = -||q_i - q_j||²
        - Sinkhorn iteration: alternate row/col normalization → doubly stochastic P
        - Forward: output = x @ P^T (soft gather via matmul)
        - At inference, can snap to hard permutation via Hungarian algorithm.

    Math:
        P = Sinkhorn(exp(-||q_i - q_j||² / τ))
        output[b,i,d] = Σ_j P[i,j] * x[b,j,d]
    """

    def __init__(
        self,
        n: int = 3,
        d_q: int = 16,
        n_sinkhorn: int = 20,
        temperature: float = 1.0,
    ):
        """
        Args:
            n: cube side length (total positions = n³)
            d_q: dimension of learnable query vectors
            n_sinkhorn: number of Sinkhorn iterations
            temperature: softmax temperature (lower → closer to hard permutation)
        """
        super().__init__()
        self.n = n
        self.N = n ** 3
        self.d_q = d_q
        self.n_sinkhorn = n_sinkhorn
        self.temperature = temperature

        # Learnable query vectors — one per position
        self.query = nn.Parameter(torch.randn(self.N, d_q) * 0.02)

    def _compute_similarity(self) -> torch.Tensor:
        """Compute pairwise similarity matrix S ∈ R^{N×N}.

        S[i,j] = -||q_i - q_j||² / τ
        """
        # ||q_i - q_j||² = ||q_i||² + ||q_j||² - 2 q_i^T q_j
        q_norm = (self.query ** 2).sum(dim=1, keepdim=True)  # [N, 1]
        dist_sq = q_norm + q_norm.T - 2 * (self.query @ self.query.T)
        return -dist_sq / self.temperature

    def _sinkhorn(self, S: torch.Tensor) -> torch.Tensor:
        """Sinkhorn-Knopp normalization → doubly stochastic matrix.

        Args:
            S: [N, N] log-probability (similarity) matrix

        Returns:
            [N, N] doubly stochastic matrix (rows and columns sum to 1)
        """
        N = S.size(0)
        P = torch.softmax(S, dim=0)  # Start with column normalization

        for _ in range(self.n_sinkhorn):
            # Row normalization
            row_sum = P.sum(dim=1, keepdim=True).clamp(min=1e-12)
            P = P / row_sum
            # Column normalization
            col_sum = P.sum(dim=0, keepdim=True).clamp(min=1e-12)
            P = P / col_sum

        return P

    def forward(self, x: torch.Tensor, hard: bool = False) -> torch.Tensor:
        """Apply soft permutation to input.

        Args:
            x: [B, N, D] input tensor
            hard: if True, use Hungarian algorithm to snap to hard permutation
                  (non-differentiable, for inference only)

        Returns:
            [B, N, D] softly permuted tensor
        """
        if x.size(1) != self.N:
            raise ValueError(f"Input dim 1 must be {self.N}, got {x.size(1)}")

        S = self._compute_similarity()
        P = self._sinkhorn(S)

        if hard:
            # Hungarian algorithm for hard permutation (inference only)
            from scipy.optimize import linear_sum_assignment
            cost = -(P.detach().cpu().numpy())
            _, col_idx = linear_sum_assignment(cost)
            P = torch.zeros(self.N, self.N, device=x.device, dtype=x.dtype)
            P[torch.arange(self.N), torch.tensor(col_idx, device=x.device)] = 1.0

        # output = x @ P^T: [B, N, D] @ [N, N]^T = [B, N, D]
        # More efficiently: x.transpose(1,2) gives [B, D, N], then
        # (P @ x.transpose(1,2)).transpose(1,2) = ... actually:
        # output[b,i,d] = Σ_j P[i,j] * x[b,j,d]
        # = (P @ x[b].T).T for each b
        # = batched: (P @ x.transpose(1,2)).transpose(1,2)
        out = torch.matmul(P.unsqueeze(0), x)  # [1,N,N] @ [B,N,D] → [B,N,D]
        return out

    def get_hard_permutation(self) -> torch.Tensor:
        """Return the nearest hard permutation (for analysis/debugging)."""
        with torch.no_grad():
            S = self._compute_similarity()
            P = self._sinkhorn(S)
            from scipy.optimize import linear_sum_assignment
            cost = -(P.cpu().numpy())
            _, col_idx = linear_sum_assignment(cost)
            return torch.tensor(col_idx, dtype=torch.long)

    def regularization_loss(self) -> torch.Tensor:
        """Entropy regularization to encourage near-permutation behavior."""
        S = self._compute_similarity()
        P = self._sinkhorn(S)
        # Negative entropy: -Σ P log P → minimize to push toward hard permutation
        entropy = -(P * torch.log(P.clamp(min=1e-12))).sum(dim=1).mean()
        return -entropy  # Lower entropy = more permutation-like
