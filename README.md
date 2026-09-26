# Orbit-Weight: Group-Theoretic Orbit Weight Sharing + Compiler-Driven Gather Elimination

[![License](https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10%2B-green.svg)]()

**Algebraic Structure as a Principled Prior for Neural Architecture Design**

A principled approach to neural network weight sharing based on group-theoretic orbit decomposition, combined with a minimal compiler IR that systematically eliminates gather operations. Covers theory (three theorems with formal proofs), systems (compiler with 10,000-graph verified soundness), and applications (crystallography, synthetic benchmarks, domestic GPU optimization).

---

## Revision status - COMMAT-D-26-03017R1 (updated 2026-09-24)

**The crystallography evidence released under `revision_v3/` is superseded.** The corrected
release that accompanies the rebuttal is mirrored in
[`revision_20260924/`](revision_20260924/): the executable code and all JSON evidence of the
`Data_and_code_20260924` archive.

| Quantity | `revision_v3/` (superseded) | `revision_20260924/` (current) |
|---|---|---|
| StructMLP formation-energy MAE (eV/atom) | 0.2576 | **0.2486** |
| HierarchyMLP formation-energy MAE (eV/atom) | 0.2488 | **0.2452** |
| Distinct space groups in the cohort | 158 | **160** |
| Site-permutation max deviation (eV) | 3.58e-7 | **5.25e-7** |
| Site-stabilizer encoding `Sym4` | not present | **0.1962 eV/atom** |
| Archived-projection diagnostic | not present | **0.1917 -> 0.9655 eV/atom** |

What changed: space-group metadata was re-derived from the exported structures (spglib,
`symprec = 1e-3 A`; the 133 mismatching records are itemised in
`revision_20260924/results/structure_audit.json`), and the pair-distance cache was rebuilt with
correct minimum-image conventions (1,495 structures affected, maximum 19.39 A). The `Sym4`
stabilizer-order control, the archived-projection diagnostic and the resource / robustness
re-checks were added. The corrected release was produced on CPU with batch size 256, Python
3.12.14 and PyTorch 2.14.0; see `revision_20260924/results/PROTOCOL_BEFORE_TRAINING.json`.

Scope of this mirror: code, manifests and JSON only. Checkpoints, prepared feature arrays and
per-crystal predictions (`.pt`, `.npz`, `.npy`, `.jsonl`) as well as the 2.4 MB
`site_symmetry_records.json` ship with the manuscript's `Data_and_code_20260924` archive and are
not duplicated here; `revision_20260924/MANIFEST.json` is the verbatim manifest of that full
archive, so the files mirrored here can be checked file-by-file against it.
`revision_20260924/code/audit_grid_action.py` is self-contained (NumPy only) and reproduces the
corrected four-orbit audit directly.

The historical source of the originally archived implementation remains pinned to commit
[`cb3715fe37ce`](https://github.com/wmll-wmll/orbit-weight/tree/cb3715fe37ce48f5efd20edba09fcf0669f4e979),
the commit referenced in the response letter. `revision/` and `revision_v3/` are retained
unmodified as the historical record.

---

## Quick Start

```bash
pip install torch numpy scipy pyyaml matplotlib

# Run compiler demo (60s)
python examples/demo_compiler.py

# Run orbit decomposition demo (30s)
python examples/demo_orbit.py

# Run C2 dimer validation (60s)
python examples/validate_dimer.py

# Run all correctness tests
python tests/test_equivariance.py

# Verify compiler soundness (10,000 random graphs)
python theory/compiler_soundness.py
```

---

## Repository Structure

```
├── cube/              Group theory layer
│   ├── cube3d.py       Rubik's cube permutation generators (12 ops), get_orbit_ids()
│   ├── perm_matrix.py  Permutation matrix (dense/CSR)
│   └── layers.py       nn.Module: CubePermutation, FusedPermLN, SoftCubePermutation
├── groups/            General finite group library (NEW)
│   ├── base.py         Group protocol, BFS orbit decomposition, OrbitLinear
│   ├── cyclic.py       Cyclic group C_n
│   ├── dihedral.py     Dihedral group D_n
│   ├── symmetric.py    Symmetric group S_n
│   └── octahedral.py   Octahedral group O_h (48 elements)
├── compiler/          Graph IR + optimization passes
│   ├── ir.py           3 node types: GatherOp, ElementWiseOp, LinearOp
│   ├── passes.py       4 passes: Reorder, Absorb, Chain, Fuse + optimize()
│   └── export_torchscript.py  IR-to-TorchScript export pipeline (NEW)
├── theory/            Formal theory + numerical validation (NEW)
│   ├── proofs.md       Three theorems with complete proofs
│   ├── orbit_bounds.py Theorem 1: Rademacher generalization bound validation
│   ├── compiler_soundness.py  Theorem 2: 10,000-graph fuzzer (100% pass rate)
│   └── crossover_roofline.py  Theorem 3: Roofline model calibrated on gfx936
├── models/            Model implementations
│   ├── mlp.py          StandardMLP, CubeMLP, OrbitLinear, make_orbit_mlp()
│   ├── heads.py        PoolHead, PerPositionHead, AttentionHead
│   └── equivariant_baselines.py  GCNN, DeepSets, SE3-T adapter, Random (NEW)
├── tasks/             Data generators
│   ├── crystal.py      Real crystal structures (Materials Project, 158 space groups) (NEW)
│   ├── spatial.py      Rotation prediction, position reconstruction
│   └── shapes3d.py     3D voxel shape classification (8 types)
├── backends/          Device-specific strategies
│   ├── nvidia.py       CUDA reference
│   ├── domestic.py     Biren gfx936 (calibrated crossover + roofline_estimate())
│   └── muxi.py         MUXI MXMACA placeholder
├── validation/        Experiment framework
│   ├── runner.py       ExperimentRunner (bootstrap CI, Cohen's d, Wilcoxon)
│   ├── exp_*.py        15+ experiment scripts
│   ├── plot_*.py       4 plotting modules
│   └── report.py       Auto-generate publication tables
├── config/            Hardware YAML configs
│   ├── rtx4060.yaml    NVIDIA Ada Lovelace
│   ├── domestic.yaml   Biren gfx936 (132 TFLOPS fp16)
│   └── muxi.yaml       MUXI estimated
├── tests/
│   └── test_equivariance.py  All correctness tests (12 tests)
├── examples/
│   ├── demo_compiler.py    Compiler pipeline walkthrough
│   ├── demo_orbit.py       Orbit decomposition demo
│   └── validate_dimer.py   C2 symmetric dimer validation
├── check_gpu.py       GPU capability checker
├── validate_dimer.py  Standalone C2 dimer demo
├── requirements.txt
└── README.md
```

---

## Core Idea

**Problem:** Standard MLPs share one weight matrix across all input positions — ignoring the algebraic structure (symmetry groups) inherent in spatially organized data. Per-position weights are too expensive (N × parameters).

**Solution:** Use the **orbit decomposition** of a finite group G acting on input positions as the blueprint for parameter partitioning. Positions in the same orbit share weights; distinct orbits use independent weights (1 ≤ K ≤ N).

```
Uniform sharing:  1 matrix for all positions      → cheap but weak
Orbit sharing:    K matrices (group-determined)    → principled middle ground
Per-position:     N matrices                       → expensive but expressive
```

**Compiler co-design:** Group actions are implemented as gather operations — a bottleneck on domestic GPUs. Our 4-pass compiler eliminates 83% of gathers while **guaranteeing exact output equivalence** (Theorem 2, verified on 10,000 random graphs).

---

## Three Theorems

| Theorem | Statement | Verification |
|---------|-----------|-------------|
| **1. Orbit Optimality** | Generalization gap ≤ O(√K / √m) | R² > 0.95 on empirical validation |
| **2. Compiler Soundness** | optimize(G)(x) = G(x) for all inputs | **10,000/10,000 fuzzer pass rate** |
| **3. Crossover Boundary** | N* = FLOPS_peak × s / (2 × BW_eff) | 8/8 gfx936 points correctly predicted |

---

## Key Results

### Position Occlusion Experiment (Table 1)

Under 70% position masking during training, orbit weight sharing dramatically outperforms random grouping:

| Group | N | K | OrbitMLP | RandomMLP | Orbit−Random |
|-------|---|---|----------|-----------|-------------|
| Cube(5) | 125 | 48 | **75.6%** | 45.2% | **+30.3 pp** |
| O_h(3) | 27 | 15 | **70.1%** | 49.4% | **+20.7 pp** |

The gap **grows with occlusion severity** — the signature of gradient sharing through correct algebraic structure.

### Crystal Formation Energy (Table 2)

On 2,736 real crystals (Materials Project, 158 space groups):

| Model | K | MAE |
|-------|---|------|
| StandardMLP | 1 | 0.8719 |
| RandomMLP | 25 | 0.8697 |
| **OrbitMLP (O_h proxy)** | 25 | **0.8235** |
| PerPositionMLP | 40 | 0.8081 |

OrbitMLP beats StandardMLP by **5.5%** and RandomMLP by **5.3%** on real scientific data.

### Compiler Speedup

| Model | Gathers Before | Gathers After | Speedup (RTX 4060) |
|-------|---------------|---------------|---------------------|
| StandardMLP | 0 | 0 | 1.09× (overhead) |
| CubeMLP | 6 | 1 | **1.13×** |

Projected 1.5–1.8× on domestic GPUs (gfx936) based on measured crossover data.

---

## Running Experiments

```bash
# Multi-group occlusion experiment (Table 1)
python validation/exp_broad_groups.py

# Crystal data download + formation energy (Table 2a)
python tasks/crystal.py

# Compiler speedup benchmark (Table 2b)
python compiler/export_torchscript.py

# All correctness tests
python tests/test_equivariance.py

# Compiler soundness fuzzer
python theory/compiler_soundness.py --quick    # 1,000 graphs
python theory/compiler_soundness.py            # 10,000 graphs

# Theory validation
python theory/orbit_bounds.py
python theory/crossover_roofline.py

# Full experiment suite
python validation/exp_throughput.py
python validation/exp_ablation.py
python validation/exp_sample_efficiency.py
python validation/exp_rotation_equivariance.py
python validation/exp_group_theory.py
```

---

## Compiler Passes

| Pass | Transform | Effect |
|------|-----------|--------|
| **Reorder** | GatherOp → ElementWiseOp  ⇒  ElementWiseOp → GatherOp | Push gathers deeper |
| **Absorb** | GatherOp(perm) → LinearOp(W)  ⇒  LinearOp(W') → GatherOp(perm) | Eliminate gathers via weight permutation |
| **Chain** | GatherOp(p₁) → GatherOp(p₂)  ⇒  GatherOp(p₁∘p₂) | Compose consecutive gathers |
| **Fuse** | Detect GatherOp + ElementWiseOp | Mark kernel fusion targets |

---

## Applications

| Domain | Symmetry Group | Method |
|--------|---------------|--------|
| Crystal structure prediction | 230 space groups (via spglib) | Orbit sharing |
| Protein dimer contact prediction | C2 point group | Orbit sharing |
| MoE model inference | Expert routing | Gather elimination |
| Multi-view 3D perception | SE(3) camera group | Both |

---

## Hardware Support

| Backend | Status | Notes |
|---------|--------|-------|
| NVIDIA CUDA | Full | Reference implementation |
| Biren gfx936 (HIP) | Calibrated | 8 crossover data points, decision boundary |
| MUXI MXMACA | Placeholder | Migration guide included |

---

## Citation

```bibtex
@article{orbit-weight-sharing,
  title={Orbit Weight Sharing: Group-Theoretic Parameter Partitioning with Compiler Co-Design for Neural Networks},
  author={Liang, N.},
  note={Submitted to Communications AI \& Computing},
  year={2026}
}
```

## License

CC BY-NC-SA 4.0 — non-commercial research/education use only. See [LICENSE](LICENSE).
