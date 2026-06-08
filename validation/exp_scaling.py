"""Scaling experiment: model size and sample size sweep.

Tests how orbit sharing scales compared to baselines.

Key comparisons:
  - OrbitMLP (orbit-shared) vs PerPositionMLP (full per-position) @ different D
  - Sample efficiency: how many samples does each need?
  - Orbit groups naturally scale with cube size

Uses PerPosVotingHead to preserve spatial signal.
Task: 8-class shape classification with random composite rotations (rotation-invariant).
"""

import torch
import torch.nn as nn
import numpy as np
from validation.runner import print_header
from validation.exp_ablation import build_model, count_params, compute_orbits


class PerPosVotingHead(nn.Module):
    """Per-position classification with voting aggregation."""
    def __init__(self, D, n_classes):
        super().__init__()
        self.cls = nn.Linear(D, n_classes)

    def forward(self, x):
        return self.cls(x).mean(dim=1)  # [B, N, n_classes] -> [B, n_classes]


def generate_shape_data(n_samples, n_cube, D, noise, shapes, seed):
    """Generate rotation-invariant shape classification data.

    Each sample: a shape with random composite rotations applied,
    coordinate-weighted features, and noise.
    """
    from cube.cube3d import CubePermutations

    cube = CubePermutations(n_cube)
    N = n_cube ** 3
    all_faces = ['U', 'R', 'F', 'D', 'L', 'B']
    n_classes = len(shapes)

    rng = np.random.RandomState(seed)

    coords = torch.tensor(
        [[x, y, z] for z in range(n_cube) for y in range(n_cube) for x in range(n_cube)],
        dtype=torch.float32)
    coords_norm = coords / max(n_cube - 1, 1)

    samples_per_class = n_samples // n_classes
    n_actual = samples_per_class * n_classes

    data = torch.zeros(n_actual, N, D)
    labels = torch.zeros(n_actual, dtype=torch.long)

    idx = 0
    for cls_idx, (shape_name, base_mask) in enumerate(shapes.items()):
        for i in range(samples_per_class):
            # Random composite rotation (0-4 random face moves)
            perm = torch.arange(N)
            n_rots = rng.randint(0, 5)
            for _ in range(n_rots):
                face = all_faces[rng.randint(0, 6)]
                clockwise = rng.randint(0, 2) == 0
                move_perm = cube.get_rotation(face if clockwise else face + "'")
                perm = move_perm[perm]

            rotated = base_mask[perm]

            # Spatial features: mask + coordinate-weighted mask + noise
            feature = torch.zeros(N, D)
            feature[:, 0] = rotated
            feature[:, 1] = rotated * coords_norm[:, 0]
            feature[:, 2] = rotated * coords_norm[:, 1]
            feature[:, 3] = rotated * coords_norm[:, 2]
            if D > 4:
                feature[:, 4:] = 0.02 * torch.randn(N, D - 4)
            feature += noise * torch.randn(N, D)

            data[idx] = feature
            labels[idx] = cls_idx
            idx += 1

    # Shuffle
    perm = torch.randperm(n_actual)
    return data[perm], labels[perm]


def make_shapes(n_cube):
    """Create 8 shape masks for a cube of given size."""
    N = n_cube ** 3
    shapes = {}

    # Full cube
    shapes['cube'] = torch.ones(N)

    # Sphere
    shapes['sphere'] = torch.tensor([
        1.0 if ((x/(n_cube-1)-0.5)**2 + (y/(n_cube-1)-0.5)**2 + (z/(n_cube-1)-0.5)**2)**0.5 <= 0.4
        else 0.0
        for z in range(n_cube) for y in range(n_cube) for x in range(n_cube)])

    # Cross (3 orthogonal bars)
    shapes['cross'] = torch.tensor([
        1.0 if (abs(x/(n_cube-1)-0.5) <= 0.15 or
                abs(y/(n_cube-1)-0.5) <= 0.15 or
                abs(z/(n_cube-1)-0.5) <= 0.15)
        else 0.0
        for z in range(n_cube) for y in range(n_cube) for x in range(n_cube)])

    # Hollow cube (surface only)
    shapes['hollow'] = torch.tensor([
        1.0 if (x == 0 or x == n_cube-1 or y == 0 or y == n_cube-1 or z == 0 or z == n_cube-1)
        else 0.0
        for z in range(n_cube) for y in range(n_cube) for x in range(n_cube)])

    # L-shape
    shapes['Lshape'] = torch.tensor([
        1.0 if (x <= 1 and y <= 1) or (z <= 1 and y <= 1) else 0.0
        for z in range(n_cube) for y in range(n_cube) for x in range(n_cube)])

    # Corner piece
    shapes['corner'] = torch.tensor([
        1.0 if x <= 1 or y <= 1 or z <= 1 else 0.0
        for z in range(n_cube) for y in range(n_cube) for x in range(n_cube)])

    # Checkerboard
    shapes['checker'] = torch.tensor([
        1.0 if (x + y + z) % 2 == 0 else 0.0
        for z in range(n_cube) for y in range(n_cube) for x in range(n_cube)])

    # Ring (torus along z)
    shapes['ring'] = torch.tensor([
        1.0 if 1.0 <= ((x-(n_cube-1)/2)**2 + (y-(n_cube-1)/2)**2)**0.5 <= n_cube/2-0.5
        and abs(z-(n_cube-1)/2) <= 0.5
        else 0.0
        for z in range(n_cube) for y in range(n_cube) for x in range(n_cube)])

    return shapes


def train_epochs(model, head, data_tr, labels_tr, data_te, labels_te,
                 n_epochs, lr=1e-3, batch_size=256):
    """Train with mini-batches, return best test accuracy."""
    opt = torch.optim.AdamW(list(model.parameters()) + list(head.parameters()),
                            lr=lr, weight_decay=0.01)
    n_train = len(data_tr)
    best = 0.0

    for ep in range(n_epochs):
        # Shuffle and mini-batch train
        perm = torch.randperm(n_train)
        model.train(); head.train()
        for start in range(0, n_train, batch_size):
            idx = perm[start:start+batch_size]
            opt.zero_grad()
            loss = nn.functional.cross_entropy(head(model(data_tr[idx])), labels_tr[idx])
            loss.backward()
            opt.step()

        # Eval every 20 epochs
        if ep % 20 == 0:
            model.eval(); head.eval()
            with torch.no_grad():
                # Eval in batches to avoid OOM
                preds = []
                for s in range(0, len(data_te), batch_size):
                    idx = torch.arange(s, min(s+batch_size, len(data_te)))
                    preds.append(head(model(data_te[idx])).argmax(dim=1))
                pred = torch.cat(preds)
                best = max(best, (pred == labels_te).float().mean().item())

    return best


def run_one(name, N, D, n_layers, n_cube, orbit_ids, n_orbits,
            shapes, n_train, n_test, device, n_epochs=80):
    """Single training run."""
    data_tr, labels_tr = generate_shape_data(n_train, n_cube, D, 0.15, shapes, 42)
    data_te, labels_te = generate_shape_data(n_test, n_cube, D, 0.15, shapes, 99)

    data_tr = data_tr.to(device); labels_tr = labels_tr.to(device)
    data_te = data_te.to(device); labels_te = labels_te.to(device)

    torch.manual_seed(42)
    model = build_model(name, orbit_ids, n_orbits, N, D, n_layers, n_cube, device)
    head = PerPosVotingHead(D, len(shapes)).to(device)

    best = train_epochs(model, head, data_tr, labels_tr, data_te, labels_te, n_epochs)

    tp = count_params(model)
    del model, head
    torch.cuda.empty_cache()
    return best, tp


def run(device: str = "cuda"):
    print_header("SCALING EXPERIMENT (PerPos Voting Head)")
    print()
    print("Task: 8-class shape classification (rotation-invariant)")
    print("Head: PerPosVotingHead — each position votes, spatial signal preserved")
    print()

    configs = [
        # (n_cube, N, D, n_layers, n_train, label)
        (5, 125, 48, 4, 3200, "Small  (D=48,  4L, 3.2K samples)"),
        (5, 125, 96, 6, 6400, "Medium (D=96,  6L, 6.4K samples)"),
        (5, 125, 128, 6, 9600, "Large  (D=128, 6L, 9.6K samples)"),
    ]

    shapes = make_shapes(5)
    models_to_test = ['StandardMLP', 'OrbitMLP', 'OrbitCubeMLP', 'PerPositionMLP']

    for n_cube, N, D, n_layers, n_train, label in configs:
        orbit_ids, n_orbits = compute_orbits(n_cube)
        n_test = n_train // 4

        print(f"--- {label} ---")
        print(f"    Orbits: {n_orbits}/{N} = {n_orbits/N:.1%}")
        print(f"    {'Model':<20s} | {'Test Acc':>10s} | {'Params':>12s}")
        print(f"    {'-'*48}")

        for model_name in models_to_test:
            acc, tp = run_one(model_name, N, D, n_layers, n_cube,
                              orbit_ids, n_orbits, shapes,
                              n_train, n_test, device, n_epochs=100)
            print(f"    {model_name:<20s} | {acc:>9.2%} | {tp:>12,}")

        print()

    # Sample efficiency sweep
    print("--- Sample Efficiency Sweep (D=96, 6L, 8 shapes) ---")
    n_cube, N, D, n_layers = 5, 125, 96, 6
    orbit_ids, n_orbits = compute_orbits(n_cube)

    sample_sizes = [800, 1600, 3200, 6400]
    print(f"    {'Samples':>8s} | {'StandardMLP':>12s} | {'OrbitMLP':>12s} | {'PerPositionMLP':>12s}")
    print(f"    {'-'*55}")

    for n_tr in sample_sizes:
        accs = {}
        for model_name in ['StandardMLP', 'OrbitMLP', 'PerPositionMLP']:
            acc, _ = run_one(model_name, N, D, n_layers, n_cube,
                             orbit_ids, n_orbits, shapes,
                             n_tr, n_tr // 4, device, n_epochs=80)
            accs[model_name] = acc
        print(f"    {n_tr:>7d} | {accs['StandardMLP']:>11.2%} | "
              f"{accs['OrbitMLP']:>11.2%} | {accs['PerPositionMLP']:>11.2%}")

    print(f"\n    Chance: {1/8:.0%}")
    return


if __name__ == "__main__":
    run()
