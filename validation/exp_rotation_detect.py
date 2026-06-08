"""Rotation detection: predict which rotation was applied to a shape.

Uses PerPosVotingHead to preserve spatial signal — each position votes
on which rotation occurred. This tests if the model can learn the cube's
geometry: positions that CHANGED vs didn't change are the key signal.

Hypothesis: OrbitMLP > StandardMLP because orbit-shared weights can
learn per-orbit rotation-detection patterns, while StandardMLP must
use the same weights for all positions.
"""

import torch
import torch.nn as nn
import numpy as np
from validation.runner import print_header
from validation.exp_ablation import build_model, count_params, compute_orbits
from validation.exp_scaling import PerPosVotingHead


def generate_rotation_detection_data(n_samples, n_cube, D, noise, seed):
    """Generate (data, labels) for rotation detection.

    Each sample: a random base pattern + one of 6 face rotations.
    The model must predict WHICH rotation was applied.

    Features: coordinate-encoded pattern so spatial structure matters.
    Each position gets [pattern, pattern*x, pattern*y, pattern*z, ...].
    """
    from cube.cube3d import CubePermutations

    cube = CubePermutations(n_cube)
    N = n_cube ** 3
    all_faces = ['U', 'R', 'F', 'D', 'L', 'B']
    n_rot_classes = 6

    rng = np.random.RandomState(seed)

    coords = torch.tensor(
        [[x, y, z] for z in range(n_cube) for y in range(n_cube) for x in range(n_cube)],
        dtype=torch.float32)
    coords_norm = coords / max(n_cube - 1, 1)

    # Create 4 distinct base patterns (one per "shape")
    n_shapes = 4
    shapes = {
        'cube': torch.ones(N),
        'sphere': torch.tensor([
            1.0 if ((x/(n_cube-1)-0.5)**2+(y/(n_cube-1)-0.5)**2+(z/(n_cube-1)-0.5)**2)**0.5 <= 0.4
            else 0.0
            for z in range(n_cube) for y in range(n_cube) for x in range(n_cube)]),
        'cross': torch.tensor([
            1.0 if (abs(x/(n_cube-1)-0.5) <= 0.15 or
                    abs(y/(n_cube-1)-0.5) <= 0.15 or
                    abs(z/(n_cube-1)-0.5) <= 0.15)
            else 0.0
            for z in range(n_cube) for y in range(n_cube) for x in range(n_cube)]),
        'hollow': torch.tensor([
            1.0 if (x == 0 or x == n_cube-1 or
                    y == 0 or y == n_cube-1 or
                    z == 0 or z == n_cube-1)
            else 0.0
            for z in range(n_cube) for y in range(n_cube) for x in range(n_cube)]),
    }

    samples_per_class = n_samples // (n_shapes * n_rot_classes)
    n_actual = samples_per_class * n_shapes * n_rot_classes

    data = torch.zeros(n_actual, N, D)
    labels = torch.zeros(n_actual, dtype=torch.long)

    idx = 0
    for shape_name, base_mask in shapes.items():
        for rot_idx, face in enumerate(all_faces):
            perm = cube.get_rotation(face)
            rotated = base_mask[perm]

            for i in range(samples_per_class):
                feature = torch.zeros(N, D)
                feature[:, 0] = rotated
                feature[:, 1] = rotated * coords_norm[:, 0]
                feature[:, 2] = rotated * coords_norm[:, 1]
                feature[:, 3] = rotated * coords_norm[:, 2]
                if D > 4:
                    feature[:, 4:] = 0.02 * torch.randn(N, D - 4)
                feature += noise * torch.randn(N, D)

                data[idx] = feature
                labels[idx] = rot_idx  # Classify the ROTATION
                idx += 1

    # Shuffle
    perm = torch.randperm(n_actual)
    return data[perm], labels[perm]


def run(device: str = "cuda"):
    print_header("ROTATION DETECTION (PerPos Voting Head)")
    print()
    print("Task: Given a shape with a rotation applied, detect which rotation.")
    print("Head: PerPosVotingHead — each position votes, spatial signal preserved.")
    print()

    n_cube = 5
    N = n_cube ** 3
    D = 64
    n_layers = 6
    n_train = 2400  # 4 shapes x 6 rotations x 100 samples
    n_test = 600
    n_epochs = 100

    orbit_ids, n_orbits = compute_orbits(n_cube)

    print(f"Config: {n_cube}^3={N} positions, D={D}, {n_layers} layers")
    print(f"Orbits: {n_orbits}/{N} = {n_orbits/N:.1%} compression")
    print(f"Train: {n_train}, Test: {n_test}")
    print(f"Chance (6 rotations): {1/6:.1%}")
    print()

    data_tr, labels_tr = generate_rotation_detection_data(n_train, n_cube, D, 0.1, 42)
    data_te, labels_te = generate_rotation_detection_data(n_test, n_cube, D, 0.1, 99)

    print(f"Train data: {data_tr.shape}, Test data: {data_te.shape}")
    print(f"Train label dist: {labels_tr.bincount().tolist()}")
    print()

    models_to_test = ['StandardMLP', 'CubeMLP', 'OrbitMLP', 'OrbitCubeMLP', 'PerPositionMLP']

    print(f"    {'Model':<20s} | {'Test Acc':>10s} | {'Params':>12s}")
    print(f"    {'-'*48}")

    for name in models_to_test:
        torch.manual_seed(42)
        model = build_model(name, orbit_ids, n_orbits, N, D, n_layers, n_cube, device)
        head = PerPosVotingHead(D, 6).to(device)

        d_tr = data_tr.to(device); l_tr = labels_tr.to(device)
        d_te = data_te.to(device); l_te = labels_te.to(device)

        opt = torch.optim.AdamW(list(model.parameters()) + list(head.parameters()),
                                lr=1e-3, weight_decay=0.01)

        best = 0.0
        for ep in range(n_epochs):
            model.train(); head.train()
            opt.zero_grad()
            loss = nn.functional.cross_entropy(head(model(d_tr)), l_tr)
            loss.backward()
            opt.step()
            if ep % 20 == 0:
                model.eval(); head.eval()
                with torch.no_grad():
                    pred = head(model(d_te)).argmax(dim=1)
                    best = max(best, (pred == l_te).float().mean().item())

        tp = count_params(model)
        print(f"    {name:<20s} | {best:>9.2%} | {tp:>12,}")
        del model, head
        torch.cuda.empty_cache()

    print()
    return


if __name__ == "__main__":
    run()
