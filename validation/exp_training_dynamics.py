"""Experiment 4: Training dynamics (loss vs wall-clock time).

Question: Who converges faster on tasks that need spatial reasoning?

Task: Rotation prediction — given a cube-structured input that has been
rotated, predict WHICH face rotation was applied (6-class classification).

This REQUIRES understanding the cube geometry:
- U rotation moves the top face (8/27 positions change)
- The model must detect WHICH positions moved and how
- StandardMLP must learn this from scratch
- CubeMLP already has cube rotations built into its architecture

Hypothesis: CubeMLP converges faster (lower wall-clock time to target accuracy)
because its internal rotations provide a useful inductive bias.
"""

import torch
import numpy as np
from validation.runner import ExperimentRunner, print_header, find_time_to_accuracy
from models.mlp import make_standard_mlp, make_cube_mlp
from models.heads import PerPositionHead, ModelWrapper
from tasks.spatial import make_rotation_data


def run(device: str = "cuda"):
    runner = ExperimentRunner(device=device)
    N, D = 27, 128
    FACES = ['U', 'R', 'F', 'D', 'L', 'B']
    N_CLASSES = 6
    N_TRAIN, N_TEST = 1000, 500
    N_EPOCHS = 100

    print_header("EXPERIMENT 4: Training Dynamics (loss vs wall-clock)")
    print(f"Task: 6-way rotation classification on cube-structured data.")
    print(f"Chance: {1/N_CLASSES:.1%}. StandardMLP must learn cube geometry from data.")
    print(f"CubeMLP has cube rotations built in as architectural prior.")

    data_train, labels_train = make_rotation_data(
        N_TRAIN, n_cube=3, d_model=D, noise=0.3, faces=FACES, seed=42
    )
    data_test, labels_test = make_rotation_data(
        N_TEST, n_cube=3, d_model=D, noise=0.3, faces=FACES, seed=99
    )

    models_def = {
        "StandardMLP (no dropout)": lambda: ModelWrapper(
            make_standard_mlp(N, D, 6, dropout=0.0),
            PerPositionHead(D, N_CLASSES),
        ),
        "StandardMLP (dropout=0.1)": lambda: ModelWrapper(
            make_standard_mlp(N, D, 6, dropout=0.1),
            PerPositionHead(D, N_CLASSES),
        ),
        "CubeMLP (fused)": lambda: ModelWrapper(
            make_cube_mlp(N, D, 6, n_cube=3),
            PerPositionHead(D, N_CLASSES),
        ),
    }

    history = {}
    for name, factory in models_def.items():
        torch.manual_seed(42)
        model = factory().to(device)
        print(f"\n  Training: {name}")
        history[name] = runner.train_model(
            model, data_train, labels_train,
            n_epochs=N_EPOCHS, batch_size=64, lr=1e-3, weight_decay=0.01,
            scheduler_cls=torch.optim.lr_scheduler.CosineAnnealingLR,
            verbose=True, label=name,
            data_test=data_test, labels_test=labels_test,
        )

    # Summary
    print(f"\n{'Model':<30s} | {'Final Acc':>10s} | {'Time to 30%':>14s} | {'Time to 50%':>14s} | {'Time to 70%':>14s} | {'Total Time':>12s}")
    print("-" * 100)

    baseline_t30 = None
    for name, log in history.items():
        final_acc = log[-1]["test_acc"]
        t30 = find_time_to_accuracy(log, 0.30)
        t50 = find_time_to_accuracy(log, 0.50)
        t70 = find_time_to_accuracy(log, 0.70)
        if baseline_t30 is None and t30 is not None:
            baseline_t30 = t30

        t30_str = f"{t30:.1f}s" if t30 else "N/A"
        t50_str = f"{t50:.1f}s" if t50 else "N/A"
        t70_str = f"{t70:.1f}s" if t70 else "N/A"
        print(f"{name:<30s} | {final_acc:>9.1%} | {t30_str:>14s} | {t50_str:>14s} | {t70_str:>14s} | {log[-1]['wall_time_s']:>10.1f}s")

    # Key metric: convergence speedup
    if baseline_t30:
        print(f"\n  Convergence speedup vs StandardMLP (no dropout):")
        for name, log in history.items():
            t30 = find_time_to_accuracy(log, 0.30)
            t50 = find_time_to_accuracy(log, 0.50)
            if t30 and t50:
                print(f"    {name}: to 30%: {baseline_t30/t30:.2f}x, to 50%: {baseline_t30/t50:.2f}x")

    return history


if __name__ == "__main__":
    run()
