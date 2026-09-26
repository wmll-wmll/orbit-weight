import math
import numpy as np
import torch
from torch import nn
D=64
L=4
N_GRID=4

def corrected_grid_orbits(n=N_GRID):
    """Cube-centred action of the 48 signed permutation matrices on {0..n-1}^3.

    g_R(c) = h + R (c - h),  h = (n-1)/2.  Every generator is a bijection.
    Returns (orbit_label[64], K).
    """
    half = (n - 1) / 2.0
    coords = np.array([[x, y, z] for z in range(n) for y in range(n) for x in range(n)])
    centred = coords - half
    orbit = ((coords == 0) | (coords == n - 1)).sum(axis=1).astype(np.int64)
    return orbit, int(np.unique(orbit).size)

def voxel_index(frac, n=N_GRID):
    """half-open voxel convention c = floor(n * f), f wrapped into [0,1)."""
    f = np.asarray(frac, dtype=np.float64)
    f = f - np.floor(f)
    c = np.clip(np.floor(f * n).astype(np.int64), 0, n - 1)
    return c[:, 2] * n * n + c[:, 1] * n + c[:, 0], c

def cube4_label_from_coords(frac):
    """Orbit label of the cube-centred action = number of components in {0, n-1}."""
    _, c = voxel_index(frac)
    return ((c <= 0) | (c >= N_GRID - 1)).sum(axis=1).astype(np.int64)

def geo4_label_from_coords(frac):
    lin, _ = voxel_index(frac)
    lab = np.zeros(len(lin), dtype=np.int64)
    lab[lin >= 8] = 1
    lab[lin >= 32] = 2
    lab[lin >= 56] = 3
    return lab

class LabelLinear(nn.Module):
    def __init__(self, K):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(K, D, D) / math.sqrt(D))
        self.bias = nn.Parameter(torch.zeros(K, D))

    def forward(self, x, oid):
        w = self.weight[oid]
        b = self.bias[oid]
        return torch.einsum("bnd,bndm->bnm", x, w) + b

class LabelBlock(nn.Module):
    def __init__(self, K):
        super().__init__()
        self.ln = nn.LayerNorm(D)
        self.ol = LabelLinear(K)
        self.act = nn.GELU()

    def forward(self, x, oid):
        return self.act(self.ol(self.ln(x), oid))

def head():
    return nn.Sequential(nn.Linear(D, D // 2), nn.GELU(), nn.Linear(D // 2, 1))

def count_params(m):
    return sum(p.numel() for p in m.parameters())

def find_wide_H(target_total):
    h = (D * (D // 2) + D // 2) + (D // 2 + 1)
    best = None
    for H in range(16, 2000):
        # WideMLP: LN(D)->DxH, (LN(H)->HxH) x (L-2), LN(H)->HxD ; LayerNorm params included
        back = (2 * D + D * H + H) + (L - 2) * (2 * H + H * H + H) + (2 * H + H * D + D)
        t = back + h
        if best is None or abs(t - target_total) < abs(best[1] - target_total):
            best = (H, t)
    return best[0], best[1]

class OrbitMLP(nn.Module):
    """label-selected affine blocks; K=1 degenerates to a shared-transform MLP."""
    def __init__(self, K):
        super().__init__()
        self.K = K
        self.blocks = nn.ModuleList([LabelBlock(K) for _ in range(L)])
        self.head = head()

    def forward(self, x, oid, mask=None):
        for b in self.blocks:
            x = b(x, oid)
        if mask is not None:
            x = (x * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True).clamp(min=1)
        else:
            x = x.mean(1)
        return self.head(x).squeeze(-1)

class WideMLP(nn.Module):
    def __init__(self, H):
        super().__init__()
        self.blocks = nn.ModuleList([
            nn.Sequential(nn.LayerNorm(D), nn.Linear(D, H), nn.GELU()),
            nn.Sequential(nn.LayerNorm(H), nn.Linear(H, H), nn.GELU()),
            nn.Sequential(nn.LayerNorm(H), nn.Linear(H, H), nn.GELU()),
            nn.Sequential(nn.LayerNorm(H), nn.Linear(H, D), nn.GELU()),
        ])
        self.head = head()

    def forward(self, x, oid=None, mask=None):
        for b in self.blocks:
            x = b(x)
        if mask is not None:
            x = (x * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True).clamp(min=1)
        else:
            x = x.mean(1)
        return self.head(x).squeeze(-1)

class FeatMLP(nn.Module):
    """standard architecture on an arbitrary per-atom feature vector projected to D."""
    def __init__(self, n_in):
        super().__init__()
        self.proj = nn.Parameter(torch.randn(n_in, D) / math.sqrt(n_in), requires_grad=False)
        self.blocks = nn.ModuleList([
            nn.Sequential(nn.LayerNorm(D), nn.Linear(D, D), nn.GELU()) for _ in range(L)])
        self.head = head()

    def forward(self, x, oid=None, mask=None):
        x = x @ self.proj
        for b in self.blocks:
            x = b(x)
        if mask is not None:
            x = (x * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True).clamp(min=1)
        else:
            x = x.mean(1)
        return self.head(x).squeeze(-1)