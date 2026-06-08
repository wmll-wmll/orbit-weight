"""
Multi-group validation experiment: orbit weight sharing across four finite groups.

Groups tested:
  - CyclicGroup (C_n)     — 1D cyclic shifts
  - DihedralGroup (D_n)   — 2D rotations + reflections
  - SymmetricGroup (S_n)  — all permutations (fully connected)
  - OctahedralGroup (O_h) — full 3D cubic symmetry

For each group, we create a synthetic classification task where labels
are invariant under the group action. The data generation uses orbit
prototypes: each orbit has a unique feature signature, and positions
within the same orbit share the same prototype (plus noise).

Three models are compared per group:
  1. OrbitMLP    — OrbitLinear with true orbit decomposition
  2. UniformMLP  — standard weight sharing (nn.Linear, equivalent to K=1)
  3. RandomMLP   — OrbitLinear with random orbit assignment (same K, wrong grouping)

Key questions:
  Q1: When true K > 1, does orbit structure beat uniform sharing?
  Q2: Is group-theoretic orbit decomposition better than random grouping?
  Q3: For K = 1 groups, does OrbitMLP correctly reduce to uniform sharing?

For groups where all positions form a single orbit (D_n, S_n), the
per-position task is a sample-level binary classification: all positions
in a sample share the same label. This tests whether the model can
learn that uniformity is the correct inductive bias.
"""

import torch
import torch.nn as nn
import numpy as np
import random as py_random

from validation.runner import print_header, ExperimentRunner
from groups import CyclicGroup, DihedralGroup, SymmetricGroup, OctahedralGroup
from groups.base import OrbitLinear


# ======================================================================
# Task generation
# ======================================================================

def generate_orbit_task(group, D: int, n_samples: int,
                        noise: float = 0.3, seed: int = 42):
    """Generate a synthetic per-position classification task.

    For groups with K >= 2 orbits:
      - Each orbit gets a random feature prototype in R^D.
      - Each position i gets prototype[orbit_ids[i]] + Gaussian noise.
      - Labels are the orbit IDs (K-class per-position classification).

    For groups with K == 1 orbit:
      - Two global prototypes (class 0 and class 1) are generated.
      - Each sample picks a class c uniformly; all N positions get
        prototype[c] + noise. This creates a per-position binary task
        where the signal is sample-level but evaluated per-position.

    Returns:
        data:   [n_samples, N, D]  feature tensor
        labels: [n_samples, N]     label tensor (int64, 0..n_classes-1)
        n_classes: number of output classes
    """
    rng = torch.Generator()
    rng.manual_seed(seed)

    orbit_ids, n_orbits = group.compute_orbits()
    N = group.N

    if n_orbits >= 2:
        # K >= 2: per-orbit prototypes
        prototypes = torch.randn(n_orbits, D, generator=rng) / (D ** 0.5)
        n_classes = n_orbits

        data = torch.zeros(n_samples, N, D)
        labels = torch.zeros(n_samples, N, dtype=torch.long)

        for s in range(n_samples):
            proto_noisy = prototypes + noise * torch.randn(
                n_orbits, D, generator=rng) / (D ** 0.5)
            for i in range(N):
                oid = int(orbit_ids[i])
                data[s, i] = proto_noisy[oid] + noise * torch.randn(
                    D, generator=rng) / (D ** 0.5)
                labels[s, i] = oid
    else:
        # K == 1: sample-level binary task
        n_classes = 2
        prototypes = torch.randn(n_classes, D, generator=rng) / (D ** 0.5)

        data = torch.zeros(n_samples, N, D)
        labels = torch.zeros(n_samples, N, dtype=torch.long)

        for s in range(n_samples):
            c = s % n_classes  # balanced classes
            proto = prototypes[c] + noise * torch.randn(D, generator=rng) / (D ** 0.5)
            for i in range(N):
                data[s, i] = proto + noise * torch.randn(D, generator=rng) / (D ** 0.5)
            labels[s, :] = c

    return data, labels, n_classes


# ======================================================================
# Model builders
# ======================================================================

def make_random_orbit_ids(orbit_ids: torch.Tensor, n_orbits: int,
                          seed: int = 1234) -> torch.Tensor:
    """Shuffle orbit assignments to create random groups of same sizes.

    When n_orbits == 1, uses n_orbits = max(2, N // 4) to ensure
    a genuinely different model from the uniform baseline.
    """
    py_random.seed(seed)
    N = len(orbit_ids)

    if n_orbits == 1:
        n_orbits = max(2, N // 4)

    # Get the size distribution from true orbits
    sizes = []
    for oid in range(int(orbit_ids.max().item()) + 1):
        sz = int((orbit_ids == oid).sum().item())
        sizes.append(sz)

    # If we need more orbits than true, create additional small groups
    while len(sizes) < n_orbits:
        sizes.append(max(1, N // n_orbits))
    # If we need fewer, merge
    while len(sizes) > n_orbits:
        sizes[-2] += sizes[-1]
        sizes.pop()

    # Normalize to sum to N
    total = sum(sizes)
    sizes = [max(1, int(s * N / total)) for s in sizes]
    # Adjust last to make exact N
    sizes[-1] = N - sum(sizes[:-1])
    if sizes[-1] <= 0:
        sizes = sizes[:-1]

    positions = list(range(N))
    py_random.shuffle(positions)

    new_ids = torch.full((N,), -1, dtype=torch.long)
    idx = 0
    for oid, sz in enumerate(sizes):
        for _ in range(sz):
            if idx < N:
                new_ids[positions[idx]] = oid
                idx += 1

    # Any remaining positions go to last orbit
    new_ids[new_ids < 0] = n_orbits - 1
    return new_ids


class PerPositionHead(nn.Module):
    """Linear classifier applied per-position: [B,N,D] -> [B,N,C]."""
    def __init__(self, D: int, n_classes: int):
        super().__init__()
        self.cls = nn.Linear(D, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.cls(x)  # [B, N, n_classes]


def build_orbit_mlp(orbit_ids: torch.Tensor, n_orbits: int,
                    D: int, n_layers: int) -> nn.Module:
    """OrbitMLP: LN -> OrbitLinear -> GELU repeated n_layers times."""
    layers = []
    for _ in range(n_layers):
        layers.extend([
            nn.LayerNorm(D),
            OrbitLinear(orbit_ids, n_orbits, D),
            nn.GELU(),
        ])
    return nn.Sequential(*layers)


def build_uniform_mlp(D: int, n_layers: int) -> nn.Module:
    """UniformMLP: LN -> nn.Linear -> GELU repeated n_layers times.

    nn.Linear(D,D) on [B,N,D] applies the same weight to all N positions,
    equivalent to OrbitLinear with K=1.
    """
    layers = []
    for _ in range(n_layers):
        layers.extend([
            nn.LayerNorm(D),
            nn.Linear(D, D),
            nn.GELU(),
        ])
    return nn.Sequential(*layers)


def build_random_mlp(orbit_ids: torch.Tensor, n_orbits: int,
                     D: int, n_layers: int, seed: int = 1234) -> nn.Module:
    """RandomMLP: same architecture as OrbitMLP but with random orbit groups."""
    rand_ids = make_random_orbit_ids(orbit_ids, n_orbits, seed=seed)
    n_rand = int(rand_ids.max().item()) + 1
    return build_orbit_mlp(rand_ids, n_rand, D, n_layers)


# ======================================================================
# Training and evaluation
# ======================================================================

class ModelWithHead(nn.Module):
    """Wrapper that composes a backbone with a per-position classification head.

    The runner expects model(data) -> logits. This wrapper chains
    backbone -> head so ExperimentRunner.train_model() can be used directly.
    """
    def __init__(self, backbone: nn.Module, head: nn.Module):
        super().__init__()
        self.backbone = backbone
        self.head = head

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(x))


def count_params(model: nn.Module) -> int:
    """Total trainable parameters (backbone only, excludes head)."""
    # Exclude the head if wrapped in ModelWithHead
    if isinstance(model, ModelWithHead):
        return sum(p.numel() for p in model.backbone.parameters())
    return sum(p.numel() for p in model.parameters())


def train_and_eval(backbone: nn.Module,
                   data_tr: torch.Tensor, labels_tr: torch.Tensor,
                   data_te: torch.Tensor, labels_te: torch.Tensor,
                   n_classes: int,
                   n_epochs: int = 100,
                   batch_size: int = 64,
                   lr: float = 1e-3,
                   weight_decay: float = 0.01,
                   device: str = "cuda",
                   verbose: bool = False) -> float:
    """Train model on per-position classification using ExperimentRunner.

    Returns best test accuracy across all epochs.
    """
    head = PerPositionHead(data_tr.size(-1), n_classes)
    model = ModelWithHead(backbone, head)

    runner = ExperimentRunner(device=device)
    history = runner.train_model(
        model, data_tr, labels_tr,
        n_epochs=n_epochs,
        batch_size=batch_size,
        lr=lr,
        weight_decay=weight_decay,
        data_test=data_te,
        labels_test=labels_te,
        verbose=verbose,
        label="",
    )

    return max((h["test_acc"] for h in history), default=0.0)


# ======================================================================
# Main
# ======================================================================

def run(device: str = "cuda"):
    D = 64
    N_LAYERS = 3
    N_TRAIN = 600
    N_TEST = 200
    N_EPOCHS = 100
    NOISE = 0.3

    # ── Define groups ────────────────────────────────────────────
    groups = [
        CyclicGroup(n=12, step=2),       # C_12, step=2 -> 2 orbits
        DihedralGroup(n=8),              # D_8 -> 1 orbit (on 8 vertices)
        SymmetricGroup(n=16),            # S_16 -> 1 orbit
        OctahedralGroup(n=5),            # O_h(5^3) -> orbits vary
    ]

    # ── Header ───────────────────────────────────────────────────
    print_header("BROAD GROUP VALIDATION: Orbit Weight Sharing Across Four Groups")
    print(f"Config: D={D}, {N_LAYERS} layers, {N_TRAIN} train / {N_TEST} test samples")
    print(f"Noise={NOISE}, epochs={N_EPOCHS}, lr=1e-3, weight_decay=0.01")
    print()

    all_results = {}

    for group in groups:
        orbit_ids, n_orbits = group.compute_orbits()
        N = group.N
        gname = group.name

        print_header(f"Group: {gname}  (N={N}, orbits={n_orbits})")

        # ── Orbit summary ────────────────────────────────────────
        orbit_sizes = sorted(
            [(orbit_ids == k).sum().item() for k in range(n_orbits)],
            reverse=True)
        print(f"  Orbit sizes: {orbit_sizes[:10]}")
        if len(orbit_sizes) > 10:
            print(f"               ... and {len(orbit_sizes)-10} more")
        print(f"  Compression ratio: {n_orbits}/{N} = {n_orbits/N:.1%}")
        print()

        # ── Generate task ────────────────────────────────────────
        data_tr, labels_tr, n_classes = generate_orbit_task(
            group, D, N_TRAIN, noise=NOISE, seed=42)
        data_te, labels_te, _ = generate_orbit_task(
            group, D, N_TEST, noise=NOISE, seed=99)

        print(f"  Task: {n_classes}-class per-position classification")
        print(f"  Train: {data_tr.shape}, Test: {data_te.shape}")
        if n_orbits == 1 and n_classes == 2:
            print(f"  (K=1 group -> sample-level binary task, all positions share label)")
        print()

        # ── Build models ─────────────────────────────────────────
        torch.manual_seed(42)

        model_orbit = build_orbit_mlp(orbit_ids, n_orbits, D, N_LAYERS)
        model_uniform = build_uniform_mlp(D, N_LAYERS)
        model_random = build_random_mlp(orbit_ids, n_orbits, D, N_LAYERS, seed=1234)

        n_rand_orbits = 1
        for m in model_random.modules():
            if isinstance(m, OrbitLinear):
                n_rand_orbits = m.n_orbits
                break

        models = [
            ("OrbitMLP",   model_orbit,   n_orbits),
            ("UniformMLP", model_uniform, 1),
            ("RandomMLP",  model_random,  n_rand_orbits),
        ]

        # ── Model summary ────────────────────────────────────────
        print(f"  {'Model':<14s} | {'Params':>10s} | {'Orbits':>7s} |")
        print(f"  {'-'*36}")
        for name, m, ko in models:
            tp = count_params(m)
            print(f"  {name:<14s} | {tp:>10,} | {ko:>7} |")
        print()

        # ── Train and evaluate ───────────────────────────────────
        print(f"  {'Model':<14s} | {'Test Acc':>10s} | {'Params':>10s} | {'Orbits':>7s} |")
        print(f"  {'-'*49}")

        group_results = []
        for name, m, ko in models:
            torch.manual_seed(42)
            acc = train_and_eval(
                m, data_tr, labels_tr, data_te, labels_te,
                n_classes, n_epochs=N_EPOCHS, device=device, verbose=False)
            tp = count_params(m)
            group_results.append({
                "group": gname,
                "model": name,
                "accuracy": acc,
                "params": tp,
                "n_orbits": ko,
                "n_positions": N,
                "true_orbits": n_orbits,
                "n_classes": n_classes,
            })
            print(f"  {name:<14s} | {acc:>9.2%} | {tp:>10,} | {ko:>7} |")
            # Free memory
            del m

        all_results[gname] = group_results
        print()
        torch.cuda.empty_cache()

    # ── Global summary table ─────────────────────────────────────
    print_header("CROSS-GROUP SUMMARY")
    print(f"  {'Group':<20s} | {'Model':<12s} | {'Acc':>8s} | {'Params':>8s} | "
          f"{'Orbits':>6s} | {'N_pos':>6s} |")
    print(f"  {'-'*78}")

    for gname, results in all_results.items():
        for r in results:
            print(f"  {r['group']:<20s} | {r['model']:<12s} | "
                  f"{r['accuracy']:>7.2%} | {r['params']:>8,} | "
                  f"{r['n_orbits']:>6} | {r['n_positions']:>6} |")

    # ── Answer key ───────────────────────────────────────────────
    print()
    print_header("BROAD GROUP ANSWER KEY")
    print()

    for gname, results in all_results.items():
        orbit_acc = next(r["accuracy"] for r in results if r["model"] == "OrbitMLP")
        uniform_acc = next(r["accuracy"] for r in results if r["model"] == "UniformMLP")
        random_acc = next(r["accuracy"] for r in results if r["model"] == "RandomMLP")
        true_K = next(r["true_orbits"] for r in results if r["model"] == "OrbitMLP")
        rand_K = next(r["n_orbits"] for r in results if r["model"] == "RandomMLP")

        print(f"  {gname} (true K={true_K}):")
        print(f"    Orbit   vs Uniform:  {orbit_acc:.2%} vs {uniform_acc:.2%}  "
              f"(Delta={orbit_acc-uniform_acc:+.2%})")
        print(f"    Orbit   vs Random:   {orbit_acc:.2%} vs {random_acc:.2%}  "
              f"(Delta={orbit_acc-random_acc:+.2%}, random K={rand_K})")

        if true_K == 1:
            print(f"    NOTE: K=1 group -> OrbitMLP == UniformMLP (both share 1 weight)")

        orbit_p = next(r["params"] for r in results if r["model"] == "OrbitMLP")
        uniform_p = next(r["params"] for r in results if r["model"] == "UniformMLP")
        print(f"    Params: Orbit={orbit_p:,}  Uniform={uniform_p:,}  "
              f"(ratio={orbit_p/max(uniform_p,1):.1f}x)")

        print()

    return all_results


if __name__ == "__main__":
    run()
