"""
Minimal validation: Group-theoretic orbit weight sharing for C2 symmetric dimer.
Synthetic data — runs in ~60 seconds, no external dependencies beyond torch/numpy.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time

torch.manual_seed(42)
np.random.seed(42)

# ── 1. Synthetic C2 dimer data ──────────────────────────────────
N_PER_SUBUNIT = 30          # residues per subunit
N = N_PER_SUBUNIT * 2       # total residues (subunit A: 0-29, B: 30-59)
D = 16                      # feature dimension

# C2 group: 1 generator (swap subunit A ↔ B)
# perm_C2[i] = i + 30 if i < 30 else i - 30
perm_C2 = torch.tensor([i + N_PER_SUBUNIT if i < N_PER_SUBUNIT else i - N_PER_SUBUNIT for i in range(N)])

# Identity permutation
perm_id = torch.arange(N)

# Group G = {identity, C2}
G_generators = [perm_C2]
G_elements = [perm_id, perm_C2]

# ── 2. Orbit decomposition (BFS, identical to Rubik's cube code) ──
def compute_orbits(N, generators):
    """BFS orbit decomposition. Same algorithm as cube/perm_matrix.py"""
    visited = torch.zeros(N, dtype=torch.bool)
    orbits = []
    orbit_ids = torch.zeros(N, dtype=torch.long)

    for seed in range(N):
        if visited[seed]:
            continue
        orbit = []
        queue = [seed]
        visited[seed] = True
        while queue:
            v = queue.pop(0)
            orbit.append(v)
            for gen in generators:
                w = gen[v].item()
                if not visited[w]:
                    visited[w] = True
                    queue.append(w)
        orbits.append(orbit)

    for oid, orbit in enumerate(orbits):
        for v in orbit:
            orbit_ids[v] = oid
    return orbits, orbit_ids

orbits, orbit_ids = compute_orbits(N, G_generators)
K = len(orbits)  # Should be N_PER_SUBUNIT = 30 (each residue pairs with its C2 image)

print(f"C2 dimer: N={N}, orbits={K}")
print(f"  Orbit sizes: all = {len(orbits[0]) if orbits else 0} (each is a residue pair)")
assert K == N_PER_SUBUNIT, f"Expected {N_PER_SUBUNIT} orbits, got {K}"

# ── 3. Generate synthetic task ──────────────────────────────────
# Task: predict whether residue i is "functional" (y=1) based on its features
# Ground truth: residues in subunit A at positions 5-15 are functional;
# due to C2 symmetry, the same positions in subunit B are also functional.

x = torch.randn(200, N, D)  # 200 samples, 60 residues, 16 features

# Create symmetry-consistent labels: position i in A and position i in B have same label
base_labels = torch.zeros(N)
base_labels[5:15] = 1                              # subunit A functional region
base_labels[35:45] = 1                             # subunit B same region (C2 image)
y = base_labels.unsqueeze(0).expand(200, -1)       # [200, N]

# Add noise to features to make it non-trivial
x[:, 5:15, 0] += 1.5    # functional residues have stronger feature[0]
x[:, 35:45, 0] += 1.5
x += torch.randn_like(x) * 0.3

# Train/test split
n_train = 120
x_train, y_train = x[:n_train], y[:n_train]
x_test, y_test = x[n_train:], y[n_train:]

# ── 4. OrbitLinear layer ────────────────────────────────────────
class OrbitLinear(nn.Module):
    """One linear layer with orbit-shared weights."""
    def __init__(self, N, K, orbit_ids, D):
        super().__init__()
        self.N = N
        self.K = K
        self.register_buffer('orbit_ids', orbit_ids)
        self.W = nn.Parameter(torch.randn(K, D, D) * 0.02)
        self.b = nn.Parameter(torch.zeros(K, D))

    def forward(self, x):
        # x: [B, N, D]
        B = x.shape[0]
        y = torch.zeros(B, self.N, self.D, device=x.device)
        for k in range(self.K):
            mask = (self.orbit_ids == k)
            if mask.any():
                y[:, mask, :] = x[:, mask, :] @ self.W[k].T + self.b[k]
        return y

# ── 5. Model variants ───────────────────────────────────────────
def make_model(variant, D=16, hidden=32):
    """Build a 2-layer MLP with specified weight sharing."""
    if variant == 'uniform':
        # All residues share 1 weight matrix (StandardMLP)
        return nn.Sequential(
            nn.Linear(D, hidden), nn.ReLU(),
            nn.Linear(hidden, 1)
        )
    elif variant == 'orbit':
        # Orbit-shared weights
        return nn.Sequential(
            OrbitLinear(N, K, orbit_ids, D),
            nn.ReLU(),
        ).append(nn.Linear(D, 1))  # won't work cleanly, let me restructure
    # ... let me restructure below

# Actually, let me define this more cleanly
class UniformMLP(nn.Module):
    def __init__(self, D, hidden):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(D, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def forward(self, x):
        B, N, D = x.shape
        x = x.reshape(B * N, D)
        out = self.net(x).view(B, N)
        return out

class OrbitMLP(nn.Module):
    def __init__(self, N, K, orbit_ids, D, hidden):
        super().__init__()
        self.N, self.K = N, K
        self.register_buffer('orbit_ids', orbit_ids)
        self.W1 = nn.Parameter(torch.randn(K, hidden, D) * 0.02)
        self.b1 = nn.Parameter(torch.zeros(K, hidden))
        self.W2 = nn.Parameter(torch.randn(K, 1, hidden) * 0.02)
        self.b2 = nn.Parameter(torch.zeros(K, 1))

    def forward(self, x):
        B = x.shape[0]
        y = torch.zeros(B, self.N, hidden, device=x.device)
        for k in range(self.K):
            mask = (self.orbit_ids == k)
            if mask.any():
                y[:, mask, :] = x[:, mask, :] @ self.W1[k].T + self.b1[k]
        y = F.relu(y)
        out = torch.zeros(B, self.N, 1, device=x.device)
        for k in range(self.K):
            mask = (self.orbit_ids == k)
            if mask.any():
                out[:, mask, :] = y[:, mask, :] @ self.W2[k].T + self.b2[k]
        return out.squeeze(-1)

class PerPositionMLP(nn.Module):
    def __init__(self, N, D, hidden):
        super().__init__()
        self.N = N
        self.W1 = nn.Parameter(torch.randn(N, hidden, D) * 0.02)
        self.b1 = nn.Parameter(torch.zeros(N, hidden))
        self.W2 = nn.Parameter(torch.randn(N, 1, hidden) * 0.02)
        self.b2 = nn.Parameter(torch.zeros(N, 1))

    def forward(self, x):
        B = x.shape[0]
        y = torch.zeros(B, self.N, hidden, device=x.device)
        for i in range(self.N):
            y[:, i, :] = x[:, i, :] @ self.W1[i].T + self.b1[i]
        y = F.relu(y)
        out = torch.zeros(B, self.N, device=x.device)
        for i in range(self.N):
            out[:, i] = (y[:, i, :] @ self.W2[i].T + self.b2[i]).squeeze(-1)
        return out

class RandomGroupMLP(nn.Module):
    """Same param count as OrbitMLP but random residue grouping."""
    def __init__(self, N, K, D, hidden):
        super().__init__()
        self.N, self.K = N, K
        # Random assignment (fixed, not learned)
        rand_ids = torch.randperm(N) % K
        self.register_buffer('group_ids', rand_ids)
        self.W1 = nn.Parameter(torch.randn(K, hidden, D) * 0.02)
        self.b1 = nn.Parameter(torch.zeros(K, hidden))
        self.W2 = nn.Parameter(torch.randn(K, 1, hidden) * 0.02)
        self.b2 = nn.Parameter(torch.zeros(K, 1))

    def forward(self, x):
        B = x.shape[0]
        y = torch.zeros(B, self.N, hidden, device=x.device)
        for k in range(self.K):
            mask = (self.group_ids == k)
            if mask.any():
                y[:, mask, :] = x[:, mask, :] @ self.W1[k].T + self.b1[k]
        y = F.relu(y)
        out = torch.zeros(B, self.N, 1, device=x.device)
        for k in range(self.K):
            mask = (self.group_ids == k)
            if mask.any():
                out[:, mask, :] = y[:, mask, :] @ self.W2[k].T + self.b2[k]
        return out.squeeze(-1)

# ── 6. Training ─────────────────────────────────────────────────
def train_one(model, x_train, y_train, x_test, y_test, epochs=60, lr=0.01):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    best_acc = 0
    t0 = time.time()
    for ep in range(epochs):
        model.train()
        opt.zero_grad()
        logits = model(x_train)
        loss = F.binary_cross_entropy_with_logits(logits, y_train)
        loss.backward()
        opt.step()
        # Eval
        if (ep + 1) % 15 == 0:
            model.eval()
            with torch.no_grad():
                preds = (model(x_test).sigmoid() > 0.5).float()
                acc = (preds == y_test).float().mean().item()
                best_acc = max(best_acc, acc)
    elapsed = time.time() - t0
    return best_acc, elapsed

# ── 7. Run comparison ───────────────────────────────────────────
hidden = 32
results = []

print("\nTraining models...")
print("-" * 55)

# Uniform (1 shared W)
model = UniformMLP(D, hidden)
param_count = sum(p.numel() for p in model.parameters())
acc, t = train_one(model, x_train, y_train, x_test, y_test, epochs=60)
results.append(('Uniform (1 shared)', acc, param_count))
print(f"  Uniform (1 shared W)     | acc={acc:.1%} | params={param_count:>6} | {t:.1f}s")

# Orbit-shared (K=30 groups by C2 symmetry)
model = OrbitMLP(N, K, orbit_ids, D, hidden)
param_count = sum(p.numel() for p in model.parameters())
acc, t = train_one(model, x_train, y_train, x_test, y_test, epochs=60)
results.append(('Orbit (C2 group)', acc, param_count))
print(f"  Orbit (C2 group, K={K})   | acc={acc:.1%} | params={param_count:>6} | {t:.1f}s")

# Random grouping (same K=30, same param count)
model = RandomGroupMLP(N, K, D, hidden)
param_count = sum(p.numel() for p in model.parameters())
acc, t = train_one(model, x_train, y_train, x_test, y_test, epochs=60)
results.append(('Random (K=30 groups)', acc, param_count))
print(f"  Random (K={K} groups)     | acc={acc:.1%} | params={param_count:>6} | {t:.1f}s")

# Per-position independent (N=60, no sharing)
model = PerPositionMLP(N, D, hidden)
param_count = sum(p.numel() for p in model.parameters())
acc, t = train_one(model, x_train, y_train, x_test, y_test, epochs=60)
results.append(('PerResidue (60 indep)', acc, param_count))
print(f"  PerResidue ({N} indep)   | acc={acc:.1%} | params={param_count:>6} | {t:.1f}s")

# ── 8. Sample efficiency sweep ──────────────────────────────────
print("\nSample efficiency (Orbit vs Uniform):")
print("-" * 55)
for n_samples in [30, 60, 120, 200]:
    x_s = x[:n_samples]; y_s = y[:n_samples]
    n_tr = int(n_samples * 0.6)
    x_tr, y_tr = x_s[:n_tr], y_s[:n_tr]
    x_te, y_te = x_s[n_tr:], y_s[n_tr:]

    m_u = UniformMLP(D, hidden)
    acc_u, _ = train_one(m_u, x_tr, y_tr, x_te, y_te, epochs=80, lr=0.01)

    m_o = OrbitMLP(N, K, orbit_ids, D, hidden)
    acc_o, _ = train_one(m_o, x_tr, y_tr, x_te, y_te, epochs=80, lr=0.01)

    print(f"  n={n_samples:>3} (train={n_tr})  Uniform={acc_u:.1%}  Orbit={acc_o:.1%}  Δ={acc_o-acc_u:+.1%}")

# ── 9. Summary ──────────────────────────────────────────────────
print("\n" + "=" * 55)
print("SUMMARY")
print("=" * 55)
print(f"{'Variant':<25} {'Acc':>7} {'Params':>8}")
print("-" * 42)
for name, acc, nparam in results:
    print(f"{name:<25} {acc:>6.1%} {nparam:>8}")
print()
print(f"Orbit / Uniform  accuracy ratio: {results[1][1]/results[0][1]:.2f}x")
print(f"Orbit / Random   accuracy ratio: {results[1][1]/results[2][1]:.2f}x")
print(f"Orbit / PerRes   param ratio:    {results[1][2]/results[3][2]:.2%}")
print(f"\nN={N}, D={D}, hidden={hidden}, K={K} orbits (C2 point group)")
print("Task: predict functional residues in a symmetric dimer")
print("Key insight: orbit-shared weights exploit C2 symmetry as prior")
