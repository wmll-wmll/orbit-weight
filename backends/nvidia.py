"""NVIDIA CUDA backend — reference implementation.

Uses PyTorch's native ops which are already heavily optimized on NVIDIA GPUs.
gather is fast here. The value of our approach shows on domestic GPUs where
gather is NOT optimized.
"""

import torch
from typing import Optional


def gather(x: torch.Tensor, perm: torch.Tensor, dim: int = 1) -> torch.Tensor:
    """Permute positions along dim using torch.gather."""
    indices = perm.unsqueeze(0).unsqueeze(-1).expand(x.size(0), -1, x.size(2))
    return torch.gather(x, dim, indices.to(x.device))


def permute_dense(x: torch.Tensor, P: torch.Tensor) -> torch.Tensor:
    """Apply permutation via dense matmul: x @ P^T."""
    # [B, D, N] @ [N, N]^T → [B, D, N] → [B, N, D]
    return torch.matmul(x.transpose(1, 2), P.T.to(x.device, x.dtype)).transpose(1, 2)


def permute_sparse(x: torch.Tensor, P: torch.Tensor) -> torch.Tensor:
    """Apply permutation via sparse matmul."""
    B, N, D = x.shape
    P = P.to(x.device, x.dtype).coalesce()
    x_flat = x.transpose(0, 1).reshape(N, B * D)
    out_flat = torch.sparse.mm(P, x_flat)
    return out_flat.reshape(N, B, D).transpose(0, 1)


def fused_perm_ln(
    x: torch.Tensor,
    perm: torch.Tensor,
    ln: torch.nn.LayerNorm,
) -> torch.Tensor:
    """Fused permute + LayerNorm (simulated — not a true fused kernel on NVIDIA).

    Since LN is element-wise, this just reorders: LN then gather.
    """
    y = ln(x)
    return gather(y, perm)


def synchronize():
    torch.cuda.synchronize()


def memory_allocated_mb() -> float:
    return torch.cuda.memory_allocated() / (1024 ** 2)
