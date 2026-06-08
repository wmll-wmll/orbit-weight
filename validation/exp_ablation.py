"""Ablation study: isolate the contribution of each component.

Models tested:
  1. StandardMLP        — shared weight, no gathers  (baseline)
  2. CubeMLP            — shared weight + gathers    (gathers only)
  3. OrbitMLP           — orbit-shared weights, NO gathers (orbits only)
  4. OrbitCubeMLP       — orbit-shared + gathers     (full model)
  5. RandomOrbitMLP     — random grouping + gathers  (same #orbits, wrong grouping)
  6. PerPositionMLP     — independent per-position, no gathers (upper bound)

Key questions:
  Q1: Do gathers help?  (CubeMLP vs StandardMLP, OrbitCubeMLP vs OrbitMLP)
  Q2: Do orbits help?   (OrbitMLP vs StandardMLP, OrbitCubeMLP vs CubeMLP)
  Q3: Is group theory better than random? (OrbitCubeMLP vs RandomOrbitMLP)
  Q4: Is full per-position overkill? (OrbitCubeMLP vs PerPositionMLP)
"""

import torch
import torch.nn as nn
import numpy as np
from validation.runner import print_header


# ======================================================================
# Orbit computation (shared with exp_group_theory)
# ======================================================================

def compute_orbits(n_cube: int, device="cpu"):
    from cube.cube3d import CubePermutations
    cube = CubePermutations(n_cube)
    N = n_cube ** 3
    generators = list(cube.all_generators().values())
    orbit_ids = torch.full((N,), -1, dtype=torch.long)
    orbit_id = 0
    for pos in range(N):
        if orbit_ids[pos] >= 0:
            continue
        stack = [pos]
        orbit_ids[pos] = orbit_id
        while stack:
            p = stack.pop()
            for g in generators:
                neighbor = int(g[p])
                if orbit_ids[neighbor] < 0:
                    orbit_ids[neighbor] = orbit_id
                    stack.append(neighbor)
        orbit_id += 1
    return orbit_ids.to(device), orbit_id


def make_random_orbits(orbit_ids, device="cpu"):
    """Shuffle orbit assignments to create random groups of same sizes."""
    import random
    random.seed(1234)
    N = len(orbit_ids)
    new_ids = torch.full((N,), -1, dtype=torch.long)
    # Get the size distribution
    sizes = [(orbit_ids == i).sum().item() for i in range(orbit_ids.max().item() + 1)]
    # Shuffle positions
    positions = list(range(N))
    random.shuffle(positions)
    # Assign positions to orbits of same sizes
    idx = 0
    for oid, sz in enumerate(sizes):
        for _ in range(sz):
            new_ids[positions[idx]] = oid
            idx += 1
    return new_ids.to(device)


# ======================================================================
# Layer components
# ======================================================================

class OrbitLinear(nn.Module):
    """Per-position linear with orbit-based weight sharing."""
    def __init__(self, orbit_ids, n_orbits, D):
        super().__init__()
        self.register_buffer('orbit_ids', orbit_ids, persistent=False)
        self.weight = nn.Parameter(torch.randn(n_orbits, D, D) / (D ** 0.5))
        self.bias = nn.Parameter(torch.zeros(n_orbits, D))

    def forward(self, x):
        # x: [B, N, D]; weight: [n_orbits, D, D]; bias: [n_orbits, D]
        w = self.weight[self.orbit_ids]  # [N, D, D]
        b = self.bias[self.orbit_ids]    # [N, D]
        # einsum avoids broadcasting w to [B, N, D, D]
        return torch.einsum('bnd,ndm->bnm', x, w) + b.unsqueeze(0)


class PerPositionLinear(nn.Module):
    """Independent weight per position."""
    def __init__(self, N, D):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(N, D, D) / (D ** 0.5))
        self.bias = nn.Parameter(torch.zeros(N, D))

    def forward(self, x):
        return torch.einsum('bnd,ndm->bnm', x, self.weight) + self.bias.unsqueeze(0)


# ======================================================================
# Model builders
# ======================================================================

def build_model(name, orbit_ids, n_orbits, N, D, n_layers, n_cube, device):
    from cube.cube3d import CubePermutations
    from models.mlp import _GatherLayer

    cube = CubePermutations(n_cube)
    moves = ['U', 'R', 'F', 'D', 'L', 'B']

    if name == 'StandardMLP':
        from models.mlp import make_standard_mlp
        return make_standard_mlp(N, D, n_layers, dropout=0.0).to(device)

    elif name == 'CubeMLP':
        from models.mlp import make_cube_mlp
        return make_cube_mlp(N, D, n_layers, n_cube=n_cube).to(device)

    elif name in ('OrbitMLP', 'OrbitCubeMLP', 'RandomOrbitMLP'):
        use_gather = name != 'OrbitMLP'
        is_random = name == 'RandomOrbitMLP'
        oids = make_random_orbits(orbit_ids, device) if is_random else orbit_ids
        n_orb = n_orbits

        layers = []
        for i in range(n_layers):
            layers.extend([
                nn.LayerNorm(D),
                OrbitLinear(oids, n_orb, D),
                nn.GELU(),
            ])
            if use_gather:
                perm = cube.get_rotation(moves[i % 6])
                layers.append(_GatherLayer(perm))
        return nn.Sequential(*layers).to(device)

    elif name == 'PerPositionMLP':
        layers = []
        for i in range(n_layers):
            layers.extend([
                nn.LayerNorm(D),
                PerPositionLinear(N, D),
                nn.GELU(),
            ])
        return nn.Sequential(*layers).to(device)

    else:
        raise ValueError(f"Unknown model: {name}")


def count_params(model):
    return sum(p.numel() for p in model.parameters())

def linear_params(model):
    return sum(p.numel() for m in model.modules()
               if isinstance(m, (nn.Linear, OrbitLinear, PerPositionLinear))
               for p in m.parameters())


# ======================================================================
# Task 1: Position reconstruction
# ======================================================================

class PerPosHead(nn.Module):
    def __init__(self, D, n_classes):
        super().__init__()
        self.cls = nn.Linear(D, n_classes)
    def forward(self, x):
        return self.cls(x)  # [B, N, n_classes]


def test_position_recon(model, N, D, N_CUBE, device, n_train=400, n_test=200, n_epochs=80):
    from tasks.spatial import make_position_data

    data_tr, labels_tr = make_position_data(
        n_train, n_cube=N_CUBE, d_model=D, noise=0.15, n_moves=3, seed=42)
    data_te, labels_te = make_position_data(
        n_test, n_cube=N_CUBE, d_model=D, noise=0.15, n_moves=3, seed=99)

    data_tr = data_tr.to(device); labels_tr = labels_tr.to(device)
    data_te = data_te.to(device); labels_te = labels_te.to(device)

    head = PerPosHead(D, N).to(device)
    opt = torch.optim.AdamW(list(model.parameters()) + list(head.parameters()),
                            lr=1e-3, weight_decay=0.01)
    best_acc = 0.0
    for epoch in range(n_epochs):
        model.train(); head.train()
        opt.zero_grad()
        logits = head(model(data_tr))
        loss = nn.functional.cross_entropy(logits.reshape(-1, N), labels_tr.reshape(-1))
        loss.backward()
        opt.step()

        model.eval(); head.eval()
        with torch.no_grad():
            pred = head(model(data_te)).argmax(dim=-1)
            acc = (pred == labels_te).float().mean().item()
            best_acc = max(best_acc, acc)

    return best_acc


# ======================================================================
# Task 2: Rotation generalization
# ======================================================================

class PoolHead(nn.Module):
    def __init__(self, D, n_classes):
        super().__init__()
        self.cls = nn.Linear(D, n_classes)
    def forward(self, x):
        return self.cls(x.mean(dim=1))


def test_rotation_generalize(model, N, D, N_CUBE, device,
                              train_faces=['U','R','F'],
                              test_faces=['D','L','B'],
                              n_pretrain=2000, n_finetune=20, n_test=500,
                              n_pretrain_epochs=80, n_finetune_epochs=100):
    from tasks.spatial import make_rotation_data

    # Pre-train
    data_pt, labels_pt = make_rotation_data(
        n_pretrain, n_cube=N_CUBE, d_model=D, noise=0.2, faces=train_faces, seed=42)
    data_pt = data_pt.to(device); labels_pt = labels_pt.to(device)

    head = PoolHead(D, 3).to(device)
    opt = torch.optim.AdamW(list(model.parameters()) + list(head.parameters()),
                            lr=1e-3, weight_decay=0.01)
    for _ in range(n_pretrain_epochs):
        model.train(); head.train()
        opt.zero_grad()
        loss = nn.functional.cross_entropy(head(model(data_pt)), labels_pt)
        loss.backward()
        opt.step()

    # Pre-train accuracy
    model.eval(); head.eval()
    with torch.no_grad():
        pred = head(model(data_pt)).argmax(dim=1)
        pretrain_acc = (pred == labels_pt).float().mean().item()

    # Fine-tune
    data_ft, labels_ft = make_rotation_data(
        n_finetune, n_cube=N_CUBE, d_model=D, noise=0.2, faces=test_faces, seed=99)
    data_ft = data_ft.to(device); labels_ft = labels_ft.to(device)

    data_te, labels_te = make_rotation_data(
        n_test, n_cube=N_CUBE, d_model=D, noise=0.2, faces=test_faces, seed=88)
    data_te = data_te.to(device); labels_te = labels_te.to(device)

    head_ft = PoolHead(D, 6).to(device)
    head_ft.cls.weight.data[:3].copy_(head.cls.weight.data)
    head_ft.cls.bias.data[:3].copy_(head.cls.bias.data)

    opt_ft = torch.optim.AdamW(list(model.parameters()) + list(head_ft.parameters()),
                               lr=1e-3, weight_decay=0.01)
    best_ft = 0.0
    best_ft_zero = 0.0  # best from epoch 0

    for epoch in range(n_finetune_epochs):
        model.train(); head_ft.train()
        opt_ft.zero_grad()
        loss = nn.functional.cross_entropy(head_ft(model(data_ft)), labels_ft)
        loss.backward()
        opt_ft.step()

        model.eval(); head_ft.eval()
        with torch.no_grad():
            pred = head_ft(model(data_te)).argmax(dim=1)
            acc = (pred == labels_te).float().mean().item()
            best_ft = max(best_ft, acc)
            if epoch == 0:
                best_ft_zero = acc

    return pretrain_acc, best_ft_zero, best_ft


# ======================================================================
# Main
# ======================================================================

def run(device: str = "cuda"):
    N_CUBE = 5
    N = N_CUBE ** 3
    D = 48
    N_LAYERS = 4

    print_header("ABLATION STUDY: What Matters?")
    print(f"Config: {N_CUBE}^3={N} positions, D={D}, {N_LAYERS} layers")
    print()

    orbit_ids, n_orbits = compute_orbits(N_CUBE)
    print(f"Orbits: {n_orbits} (from face rotation group)")
    print(f"Compression: {n_orbits}/{N} = {n_orbits/N:.1%}")
    print()

    model_names = [
        'StandardMLP',
        'CubeMLP',
        'OrbitMLP',
        'OrbitCubeMLP',
        'RandomOrbitMLP',
        'PerPositionMLP',
    ]

    # ── Build all models and measure params ───────────────────
    print(f"  {'='*80}")
    print(f"  MODEL SUMMARY")
    print(f"  {'='*80}")
    print(f"  {'Model':<20s} | {'Params':>10s} | {'Linear P':>10s} | Gather | Orbit |")
    print(f"  {'-'*65}")

    models = {}
    for name in model_names:
        torch.manual_seed(42)
        m = build_model(name, orbit_ids, n_orbits, N, D, N_LAYERS, N_CUBE, device)
        models[name] = m
        tp = count_params(m)
        lp = linear_params(m)
        has_gather = 'MLP' in name and name not in ('StandardMLP', 'OrbitMLP', 'PerPositionMLP')
        has_orbit = 'Orbit' in name or 'PerPosition' in name
        has_random = 'Random' in name
        print(f"  {name:<20s} | {tp:>10,} | {lp:>10,} | "
              f"{'Y' if has_gather else '-':>6s} | "
              f"{'Rand' if has_random else 'Y' if has_orbit else '-':>5s} |")

    # ── Task 1: Position reconstruction ───────────────────────
    print(f"\n  {'='*80}")
    print(f"  TASK 1: Position Reconstruction (400 train, 200 test)")
    print(f"  {'='*80}")
    print(f"  {'Model':<20s} | {'Test Acc':>10s} | {'Params':>10s} |")
    print(f"  {'-'*45}")

    recon_results = {}
    for name in model_names:
        torch.manual_seed(42)
        m = build_model(name, orbit_ids, n_orbits, N, D, N_LAYERS, N_CUBE, device)
        acc = test_position_recon(m, N, D, N_CUBE, device)
        recon_results[name] = acc
        tp = count_params(m)
        print(f"  {name:<20s} | {acc:>9.2%} | {tp:>10,} |")
        del m; torch.cuda.empty_cache()

    # ── Task 2: Rotation generalization ───────────────────────
    print(f"\n  {'='*80}")
    print(f"  TASK 2: Rotation Generalization")
    print(f"  Pre-train on {{U,R,F}} (2000 samples) → Test on {{D,L,B}} (500 samples)")
    print(f"  {'='*80}")
    print(f"  {'Model':<20s} | {'Pre-train':>10s} | {'FT Zero':>10s} | {'FT Best':>10s} |")
    print(f"  {'-'*55}")

    gen_results = {}
    for name in model_names:
        torch.manual_seed(42)
        m = build_model(name, orbit_ids, n_orbits, N, D, N_LAYERS, N_CUBE, device)
        pt_acc, ft_zero, ft_best = test_rotation_generalize(m, N, D, N_CUBE, device)
        gen_results[name] = (pt_acc, ft_zero, ft_best)
        print(f"  {name:<20s} | {pt_acc:>9.2%} | {ft_zero:>9.2%} | {ft_best:>9.2%} |")
        del m; torch.cuda.empty_cache()

    # ── Summary table ─────────────────────────────────────────
    print(f"\n  {'='*80}")
    print(f"  ABLATION ANSWER KEY")
    print(f"  {'='*80}")
    print()

    # Q1: Gathers?
    std_recon = recon_results['StandardMLP']
    cube_recon = recon_results['CubeMLP']
    orbit_recon = recon_results['OrbitMLP']
    orbitcube_recon = recon_results['OrbitCubeMLP']
    print(f"  Q1: Do gathers help position reconstruction?")
    print(f"      StandardMLP → CubeMLP:  {std_recon:.2%} → {cube_recon:.2%} (Δ={cube_recon-std_recon:+.2%})")
    print(f"      OrbitMLP → OrbitCubeMLP: {orbit_recon:.2%} → {orbitcube_recon:.2%} (Δ={orbitcube_recon-orbit_recon:+.2%})")

    print(f"  Q2: Do orbit-shared weights help?")
    print(f"      CubeMLP → OrbitCubeMLP: {cube_recon:.2%} → {orbitcube_recon:.2%} (Δ={orbitcube_recon-cube_recon:+.2%})")
    print(f"      StandardMLP → OrbitMLP:  {std_recon:.2%} → {orbit_recon:.2%} (Δ={orbit_recon-std_recon:+.2%})")

    random_acc = recon_results['RandomOrbitMLP']
    print(f"  Q3: Is GROUP-THEORETIC orbit better than RANDOM orbit?")
    print(f"      RandomOrbitMLP: {random_acc:.2%} vs OrbitCubeMLP: {orbitcube_recon:.2%} (Δ={orbitcube_recon-random_acc:+.2%})")

    perpos_acc = recon_results['PerPositionMLP']
    print(f"  Q4: Is full per-position overkill vs orbit-shared?")
    print(f"      PerPositionMLP: {perpos_acc:.2%} ({count_params(models['PerPositionMLP']):,}p)")
    print(f"      OrbitCubeMLP:   {orbitcube_recon:.2%} ({count_params(models['OrbitCubeMLP']):,}p)")
    perpos_tp = count_params(models['PerPositionMLP'])
    orbitcube_tp = count_params(models['OrbitCubeMLP'])
    print(f"      Param ratio: {orbitcube_tp/perpos_tp:.1%}, Acc ratio: {orbitcube_recon/max(perpos_acc, 0.001):.1%}")

    # Generalization summary
    print(f"\n  Generalization (zero-shot to unseen rotations):")
    for name in model_names:
        _, ftz, _ = gen_results[name]
        print(f"      {name:<20s}: {ftz:.2%}")

    return recon_results, gen_results


if __name__ == "__main__":
    run()
