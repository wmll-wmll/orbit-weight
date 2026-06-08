"""Experiment 1: Sample efficiency on position reconstruction.

Task: given a scrambled input, predict each position's ORIGINAL index.
This inherently requires understanding permutations — exactly the inductive
bias that CubeMLP has built in.

Hypothesis: CubeMLP reaches target accuracy with fewer training samples.
"""

import torch
import numpy as np
from validation.runner import ExperimentRunner, print_header
from models.mlp import make_standard_mlp, make_cube_mlp
from models.heads import PerPositionNoPoolHead, ModelWrapper
from tasks.spatial import make_position_data


def run(device: str = "cuda"):
    runner = ExperimentRunner(device=device)
    N, D = 27, 128
    N_POS_CLASSES = N  # predict which of 27 positions
    N_TEST = 500

    print_header("EXPERIMENT 1: Sample Efficiency (Position Reconstruction)")
    print("Task: given scrambled input, predict each position's original index.")
    print("Chance: 1/27 = 3.7%, Target: 50% accuracy.")

    data_test, labels_test = make_position_data(
        N_TEST, n_cube=3, d_model=D, noise=0.15, n_moves=3, seed=99
    )
    data_test = data_test.to(device)
    labels_test = labels_test.to(device)

    train_sizes = [40, 80, 160, 320, 640, 1280]
    n_epochs_map = {40: 300, 80: 250, 120: 200, 160: 200, 320: 150, 640: 120, 1280: 100}

    models_def = {
        "StandardMLP": lambda: ModelWrapper(
            make_standard_mlp(N, D, 4, dropout=0.0),
            PerPositionNoPoolHead(D, N_POS_CLASSES),
        ),
        "CubeMLP": lambda: ModelWrapper(
            make_cube_mlp(N, D, 4, n_cube=3),
            PerPositionNoPoolHead(D, N_POS_CLASSES),
        ),
    }

    print(f"\n{'Model':<15s}", end="")
    for sz in train_sizes:
        print(f" | n={sz:>4d}", end="")
    print(f" | {'Min to 30%':>12s} | {'Min to 50%':>12s}")
    print("-" * 95)

    for name, factory in models_def.items():
        results = []
        for n_train in train_sizes:
            accs = []
            n_ep = n_epochs_map.get(n_train, 150)
            for seed in range(3):
                data_train, labels_train = make_position_data(
                    n_train, n_cube=3, d_model=D, noise=0.15, n_moves=3, seed=seed
                )
                data_train = data_train.to(device)
                labels_train = labels_train.to(device)

                model = factory().to(device)
                runner.train_model(
                    model, data_train, labels_train,
                    n_epochs=n_ep,
                    batch_size=min(n_train, 64),
                    lr=1e-3, weight_decay=0.01,
                )
                model.eval()
                with torch.no_grad():
                    logits = model(data_test)  # [B, N, N]
                    pred = logits.argmax(dim=-1)  # [B, N]
                    acc = (pred == labels_test).float().mean().item()
                accs.append(acc)

            results.append(np.mean(accs))

        min_30 = None
        min_50 = None
        for sz, acc in zip(train_sizes, results):
            if acc >= 0.30 and min_30 is None:
                min_30 = sz
            if acc >= 0.50 and min_50 is None:
                min_50 = sz

        print(f"{name:<15s}", end="")
        for acc in results:
            print(f" | {acc:>5.1%}", end="")
        s30 = f"{min_30}" if min_30 is not None else f">{train_sizes[-1]}"
        s50 = f"{min_50}" if min_50 is not None else f">{train_sizes[-1]}"
        print(f" | {s30:>12s} | {s50:>12s}")

    return results


if __name__ == "__main__":
    run()
