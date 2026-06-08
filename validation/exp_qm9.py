"""
QM9 molecular force field prediction experiment.

Uses the synthetic molecular data from tasks/qm9.py (or real QM9 if available)
to evaluate orbit weight sharing on a real-world scientific task.

Task: predict per-voxel force vectors (3D) from voxelized molecular input.

Usage:
    python validation/exp_qm9.py
"""

import torch
import torch.nn as nn
import numpy as np
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from groups.octahedral import OctahedralGroup
from models.mlp import make_standard_mlp, make_orbit_mlp
from validation.runner import print_header, ExperimentRunner


# ======================================================================
# PerPosition regression head
# ======================================================================

class ForceHead(nn.Module):
    """Predict per-position 3D force vectors."""
    def __init__(self, d_model: int, n_out: int = 3):
        super().__init__()
        self.linear = nn.Linear(d_model, n_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)  # [B, N, 3]


# ======================================================================
# Main experiment
# ======================================================================

def run(device: str = "cuda"):
    from tasks.qm9 import make_qm9_data

    N_CUBE = 5
    N = N_CUBE ** 3  # 125
    D = 64
    N_LAYERS = 4
    N_TRAIN = 800
    N_TEST = 200
    N_EPOCHS = 80

    print_header("QM9 MOLECULAR FORCE FIELD PREDICTION")
    print(f"Config: {N_CUBE}^3={N} voxels, D={D}, layers={N_LAYERS}")
    print(f"Train: {N_TRAIN}, Test: {N_TEST}")
    print()

    # Generate data
    print("Generating molecular data...")
    data_tr, targets_tr = make_qm9_data(N_TRAIN, n_voxel=N_CUBE, d_model=D,
                                         noise=0.1, seed=42)
    data_te, targets_te = make_qm9_data(N_TEST, n_voxel=N_CUBE, d_model=D,
                                         noise=0.1, seed=99)
    print(f"  Train data: {list(data_tr.shape)}, targets: {list(targets_tr.shape)}")
    print(f"  Test data:  {list(data_te.shape)}, targets: {list(targets_te.shape)}")
    print()

    device_obj = torch.device(device if torch.cuda.is_available() else "cpu")
    data_tr = data_tr.to(device_obj); targets_tr = targets_tr.to(device_obj)
    data_te = data_te.to(device_obj); targets_te = targets_te.to(device_obj)

    # Compute orbits under O_h (molecular symmetry)
    oh = OctahedralGroup(N_CUBE)
    orbit_ids, n_orbits = oh.compute_orbits()
    print(f"O_h({N_CUBE}) orbits: K={n_orbits} (compression: {n_orbits}/{N} = {n_orbits/N:.1%})")
    print()

    # Define models
    models_def = {
        'StandardMLP': lambda: make_standard_mlp(N, D, N_LAYERS),
        'OrbitMLP': lambda: make_orbit_mlp(N, D, N_LAYERS, orbit_ids=orbit_ids,
                                            n_orbits=n_orbits, use_gathers=False),
        'OrbitCubeMLP': lambda: make_orbit_mlp(N, D, N_LAYERS, orbit_ids=orbit_ids,
                                                n_orbits=n_orbits, n_cube=N_CUBE,
                                                use_gathers=True),
    }

    print(f"  {'='*70}")
    print(f"  TRAINING MODELS")
    print(f"  {'='*70}")

    results = {}
    for name, factory in models_def.items():
        torch.manual_seed(42)
        model = factory().to(device_obj)
        head = ForceHead(D, 3).to(device_obj)

        n_params = sum(p.numel() for p in model.parameters())
        n_params_total = n_params + sum(p.numel() for p in head.parameters())

        opt = torch.optim.AdamW(
            list(model.parameters()) + list(head.parameters()),
            lr=1e-3, weight_decay=0.01)

        best_mae = float('inf')
        for epoch in range(N_EPOCHS):
            model.train(); head.train()
            opt.zero_grad()
            pred = head(model(data_tr))
            loss = nn.functional.mse_loss(pred, targets_tr)
            loss.backward()
            opt.step()

            model.eval(); head.eval()
            with torch.no_grad():
                pred_te = head(model(data_te))
                mae = (pred_te - targets_te).abs().mean().item()

                if mae < best_mae:
                    best_mae = mae

        results[name] = {
            'model_params': n_params,
            'total_params': n_params_total,
            'best_mae': best_mae,
            'final_loss': loss.item(),
        }
        print(f"  {name:<20s}: params={n_params_total:>8,}, best_mae={best_mae:.4f}")

        del model, head
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Summary
    print(f"\n  {'='*70}")
    print(f"  SUMMARY: QM9 Force Field Prediction")
    print(f"  {'='*70}")
    print(f"  {'Model':<20s} {'Params':>10s} {'MAE':>10s} {'vs Standard':>12s}")
    print(f"  {'-'*55}")

    baseline_mae = results['StandardMLP']['best_mae']
    for name, r in results.items():
        ratio = r['best_mae'] / baseline_mae if baseline_mae > 0 else 1.0
        vs = f"{ratio:.2f}x" if name != 'StandardMLP' else '--'
        print(f"  {name:<20s} {r['total_params']:>10,} {r['best_mae']:>10.4f} {vs:>12s}")

    best_name = min(results, key=lambda k: results[k]['best_mae'])
    print(f"\n  Best model: {best_name} (MAE={results[best_name]['best_mae']:.4f})")

    return results


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    run(device)
