"""Experiment 2: Rotation equivariance.

Question: When inputs are rotated, does CubeMLP maintain accuracy better?

Setup: Train on ORIGINAL spatial patterns, test on both original AND
cube-rotated versions. Use PerPositionHead so rotation matters.

Key design: The task must be LEARNABLE (not noise). Use 6 classes,
low noise, Gaussian blob spatial patterns. StandardMLP should learn
to ~80%+ on original data. The DROP on rotated data measures
rotation robustness.

Hypothesis: CubeMLP shows less accuracy drop on rotated data because
its architecture already applies cube rotations internally.
"""

import torch
import numpy as np
from validation.runner import ExperimentRunner, print_header
from models.mlp import make_standard_mlp, make_cube_mlp
from models.heads import PerPositionHead, ModelWrapper
from tasks.spatial import make_spatial_classification
from cube.cube3d import CubePermutations


def run(device: str = "cuda"):
    runner = ExperimentRunner(device=device)
    N, D, N_CLASSES = 27, 128, 6
    N_TRAIN, N_TEST = 800, 500
    cube = CubePermutations(3)

    print_header("EXPERIMENT 2: Rotation Equivariance")
    print("Train on original data, test on original AND cube-rotated data.")
    print("6 classes, noise=0.2 — learnable but needs spatial reasoning.")

    data_train, labels_train = make_spatial_classification(
        N_TRAIN, n_cube=3, d_model=D, n_classes=N_CLASSES, noise=0.2, seed=42
    )
    data_test_orig, labels_test = make_spatial_classification(
        N_TEST, n_cube=3, d_model=D, n_classes=N_CLASSES, noise=0.2, seed=99
    )

    # Create rotated test sets
    rotations = {
        'U': cube.rotation_U(clockwise=True),
        'R': cube.rotation_R(clockwise=True),
        'F': cube.rotation_F(clockwise=True),
    }

    data_tests = {'orig': data_test_orig}
    for name, perm in rotations.items():
        idx = perm.unsqueeze(0).unsqueeze(-1).expand(N_TEST, -1, D)
        data_tests[name] = torch.gather(data_test_orig, 1, idx)

    data_train = data_train.to(device)
    labels_train = labels_train.to(device)
    labels_test = labels_test.to(device)
    for k in data_tests:
        data_tests[k] = data_tests[k].to(device)

    models_def = {
        "StandardMLP": lambda: ModelWrapper(
            make_standard_mlp(N, D, 6, dropout=0.0),
            PerPositionHead(D, N_CLASSES),
        ),
        "StandardMLP+Drop": lambda: ModelWrapper(
            make_standard_mlp(N, D, 6, dropout=0.1),
            PerPositionHead(D, N_CLASSES),
        ),
        "CubeMLP": lambda: ModelWrapper(
            make_cube_mlp(N, D, 6, n_cube=3),
            PerPositionHead(D, N_CLASSES),
        ),
    }

    print(f"\n{'Model':<20s}", end="")
    for name in ['Orig'] + list(rotations.keys()):
        print(f" | {'Acc '+name:>10s}", end="")
    print(f" | {'Avg Drop':>10s} | {'Robustness':>12s}")
    print("-" * 105)

    for name, factory in models_def.items():
        test_accs = {k: [] for k in data_tests}
        for seed in range(3):
            torch.manual_seed(seed)
            model = factory().to(device)
            runner.train_model(
                model, data_train, labels_train,
                n_epochs=100, batch_size=64, lr=3e-4, weight_decay=0.01,
            )
            model.eval()
            with torch.no_grad():
                for rot_name, data_test in data_tests.items():
                    pred = model(data_test).argmax(dim=1)
                    acc = (pred == labels_test).float().mean().item()
                    test_accs[rot_name].append(acc)

        orig_mean = np.mean(test_accs['orig'])
        rot_means = {k: np.mean(v) for k, v in test_accs.items() if k != 'orig'}

        print(f"{name:<20s}", end="")
        print(f" | {orig_mean:>9.1%}", end="")
        avg_drop = 0
        for rot_name in rotations:
            m = rot_means[rot_name]
            print(f" | {m:>9.1%}", end="")
            avg_drop += (orig_mean - m)
        avg_drop /= len(rotations)
        robustness = (orig_mean - avg_drop) / orig_mean if orig_mean > 0 else 0
        print(f" | {avg_drop:>9.1%} | {robustness:>11.2%} retained")

    return


if __name__ == "__main__":
    run()
