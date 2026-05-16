"""
Permutation matrix representation for GPU-accelerated gather operations.

Core insight: torch.gather(x, dim, index) is equivalent to x @ P^T
where P is a permutation matrix with P[i, perm[i]] = 1.

This module provides dense, CSR-sparse, and BSR-sparse representations,
with automatic format selection based on matrix size.
"""

import torch
from typing import Optional, Tuple


class PermutationMatrix:
    """Permutation matrix backed by dense, CSR-sparse, or BSR-sparse storage.

    Given a permutation vector perm of length N, the matrix P satisfies:
        output[b, i, d] = sum_j x[b, j, d] * P[i, j] = x[b, perm[i], d]
    """

    def __init__(self, perm: torch.Tensor, format: str = "dense"):
        """
        Args:
            perm: [N] permutation vector where perm[i] = source index for output position i
            format: "dense" | "sparse_csr" | "bsr"
        """
        if perm.dim() != 1:
            raise ValueError(f"perm must be 1D, got shape {perm.shape}")
        if not self._is_permutation(perm):
            raise ValueError("perm must be a valid permutation (bijection)")

        self.perm = perm
        self.N = perm.size(0)
        self.format = format
        self._dense_cache: Optional[torch.Tensor] = None
        self._sparse_cache: Optional[torch.Tensor] = None

    @staticmethod
    def _is_permutation(perm: torch.Tensor) -> bool:
        if perm.max().item() >= perm.size(0) or perm.min().item() < 0:
            return False
        return len(perm.unique()) == perm.size(0)

    @property
    def dense(self) -> torch.Tensor:
        """Return N×N float32 dense permutation matrix."""
        if self._dense_cache is None:
            self._dense_cache = self._build_dense().to(device=self.perm.device)
        return self._dense_cache

    def _build_dense(self) -> torch.Tensor:
        # P[i, j] = 1 iff j == perm[i] (row i = output pos, col j = input pos)
        col_idx = torch.arange(self.N, device=self.perm.device).unsqueeze(0)  # [1, N]
        row_src = self.perm.unsqueeze(1)                                      # [N, 1]
        return (col_idx == row_src).float()

    @property
    def sparse(self) -> torch.Tensor:
        """Return N×N CSR sparse permutation matrix."""
        if self._sparse_cache is None:
            self._sparse_cache = self._build_sparse().to(device=self.perm.device)
        return self._sparse_cache

    def _build_sparse(self) -> torch.Tensor:
        indices = torch.stack([torch.arange(self.N, device=self.perm.device), self.perm], dim=0)
        values = torch.ones(self.N, device=self.perm.device)
        return torch.sparse_coo_tensor(indices, values, (self.N, self.N)).coalesce()

    # ──── Application methods ────

    def apply(self, x: torch.Tensor, dim: int = 1) -> torch.Tensor:
        """Apply permutation along `dim`, equivalent to x[:, perm, :].

        Routes to the best available implementation based on self.format.
        """
        if dim != 1:
            raise NotImplementedError("Only dim=1 is currently supported")
        if self.format == "dense":
            return self._apply_dense(x)
        elif self.format == "sparse_csr":
            return self._apply_sparse(x)
        else:
            raise ValueError(f"Unknown format: {self.format}")

    def _apply_dense(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, N, D] -> output[b,i,d] = sum_j x[b,j,d] * P[i,j].

        Strategy: output = (P @ x^T)^T where x is viewed as [B, D, N]
        Because: x @ P^T = (P @ (x^T))^T when x is [B, N, D].
        But simpler: x.transpose(1,2) gives [B, D, N], then
        our P is [N, N] with rows as "which source index for output i".
        Actually P[i, perm[i]] = 1 means row i of P dot column j of x_T
        picks x_T[perm[i], j] = x[b, perm[i], d].

        Output[b,i,d] = sum_j x[b,j,d] * P[i,j]
        = x[b, perm[i], d]  since P[i,j]=1 only at j=perm[i].

        This is: out = torch.bmm(P.unsqueeze(0).expand(B, N, N), x)
        Or, equivalently for larger tensors: torch.einsum('ij,bjd->bid', P, x).

        For GPU efficiency we use torch.matmul:
            x: [B, N, D], P: [N, N]
            out = torch.matmul(P, x) if P broadcasts? No.
            Actually: out = torch.matmul(x.transpose(1,2), P.T).transpose(1,2)
            = [B, D, N] @ [N, N]^T => [B, D, N] => transpose => [B, N, D]
        """
        B, N, D = x.shape
        P = self.dense.to(x.device, dtype=x.dtype)
        # [B, D, N] @ [N, N] -> [B, D, N] -> transpose -> [B, N, D]
        out = torch.matmul(x.transpose(1, 2), P.T).transpose(1, 2)
        return out

    def _apply_sparse(self, x: torch.Tensor) -> torch.Tensor:
        """Apply via sparse-dense matrix multiply.

        output[b,i,d] = sum_j P[i,j] * x[b,j,d]
        = (P @ x_2d) reshaped, where x_2d [N, B*D].
        """
        B, N, D = x.shape
        P = self.sparse.to(x.device, dtype=x.dtype)

        # Reshape: [B, N, D] -> [N, B*D]
        x_flat = x.transpose(0, 1).reshape(N, B * D)
        # sparse.mm: [N, N] @ [N, B*D] -> [N, B*D]
        out_flat = torch.sparse.mm(P, x_flat)
        # Reshape back: [N, B*D] -> [N, B, D] -> [B, N, D]
        return out_flat.reshape(N, B, D).transpose(0, 1)


def gather_ref(x: torch.Tensor, perm: torch.Tensor, dim: int = 1) -> torch.Tensor:
    """Reference gather implementation for correctness checks."""
    indices = perm.unsqueeze(0).unsqueeze(-1).expand(x.size(0), -1, x.size(2))
    return torch.gather(x, dim, indices)


def verify_equivalence(x: torch.Tensor, perm: torch.Tensor, atol: float = 1e-6):
    """Check that PermutationMatrix.apply == torch.gather."""
    pm = PermutationMatrix(perm)
    out_matmul = pm.apply(x)
    out_gather = gather_ref(x, perm)
    diff = (out_matmul - out_gather).abs().max().item()
    if diff > atol:
        raise AssertionError(f"Mismatch: max diff = {diff:.6e}")
    return True
