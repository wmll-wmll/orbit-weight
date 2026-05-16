"""
Orbit Demo: Group-theoretic orbit decomposition for Rubik's Cube.

Computes orbit partition of N=125 positions under the face rotation group G,
then builds and trains a minimal OrbitMLP vs StandardMLP to show the effect.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
from cube.cube3d import CubePermutations

torch.manual_seed(42)
n, N, D = 5, 125, 32

# ── 1. Orbit decomposition ──
cp = CubePermutations(n)
gen_dict = cp.all_generators()
generators = list(gen_dict.values())

visited = torch.zeros(N, dtype=torch.bool)
orbit_ids = torch.zeros(N, dtype=torch.long)
current_orbit = 0

for seed in range(N):
    if visited[seed]:
        continue
    queue = [seed]
    visited[seed] = True
    while queue:
        v = queue.pop(0)
        orbit_ids[v] = current_orbit
        for gen in generators:
            w = gen[v].item()
            if not visited[w]:
                visited[w] = True
                queue.append(w)
    current_orbit += 1

K = orbit_ids.max().item() + 1
print(f"Cube {n}x{n}x{n}: N={N}, K={K} orbits, ratio={K/N:.1%}")

# ── 2. Show orbit sizes ──
from collections import Counter
sizes = Counter(orbit_ids.tolist())
print(f"Orbit size distribution: {dict(Counter(sizes.values()))}")

# ── 3. Build minimal OrbitLinear ──
class OrbitLinear(torch.nn.Module):
    def __init__(self, N, K, orbit_ids, D):
        super().__init__()
        self.N, self.K = N, K
        self.register_buffer('orbit_ids', orbit_ids)
        self.W = torch.nn.Parameter(torch.randn(K, D, D) * 0.02)
        self.b = torch.nn.Parameter(torch.zeros(K, D))

    def forward(self, x):
        B = x.shape[0]
        y = torch.zeros(B, self.N, D, device=x.device)
        for k in range(self.K):
            mask = (self.orbit_ids == k)
            if mask.any():
                y[:, mask, :] = x[:, mask, :] @ self.W[k].T + self.b[k]
        return y

# Quick correctness check
ol = OrbitLinear(N, K, orbit_ids, D)
x = torch.randn(4, N, D)
y = ol(x)
print(f"\nOrbitLinear: input {list(x.shape)} -> output {list(y.shape)}")
print(f"Parameters: {sum(p.numel() for p in ol.parameters()):,}")
print(f"vs PerPosition: {N * (D*D + D):,} parameters")
print(f"Compression: {sum(p.numel() for p in ol.parameters()) / (N * (D*D + D)):.1%}")
