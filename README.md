# Orbit-Weight: Group-Theoretic Orbit Weight Sharing + Compiler-Driven Gather Elimination

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10%2B-green.svg)]()

**面向国产GPU的结构化权重共享与编译器推理加速**

A principled approach to neural network weight sharing based on group-theoretic orbit decomposition, combined with a minimal compiler IR that systematically eliminates gather operations for domestic GPU inference acceleration.

---

## Quick Start

```bash
# Install deps
pip install torch numpy scipy pyyaml

# Run compiler demo (60s)
python examples/demo_compiler.py

# Run orbit decomposition demo (30s)
python examples/demo_orbit.py

# Run C2 dimer validation (60s)
python examples/validate_dimer.py

# Run all tests
python tests/test_equivariance.py
```

---

## Two Core Contributions

### 1. Orbit-Shared Weighting

**Problem:** Standard MLPs share one weight matrix across all input positions — ignoring spatial structure. Per-position weights are too expensive (N × parameters).

**Solution:** Decompose positions into orbits under a symmetry group. Same-orbit positions share weights; different orbits use independent weights.

```
Uniform sharing:  1 matrix for all 125 positions  → cheap but weak
Orbit sharing:   48 matrices for 48 orbits        → principled middle ground
Per-position:   125 matrices for 125 positions    → expensive but expressive
```

**Key result:** Orbit-shared models reach 100% accuracy with 8× fewer training samples than standard MLPs. Group-theoretic orbits outperform equal-parameter random grouping by 44%.

### 2. Compiler-Driven Gather Elimination

**Problem:** Gather operations are the bottleneck on domestic GPUs (Biren gfx936, MUXI) where dense matmul is heavily optimized but scatter/gather is not.

**Solution:** A minimal IR with 3 node types (`GatherOp`, `ElementWiseOp`, `LinearOp`) and 4 graph-rewriting passes:

```
Before: Gather → LN → Linear → GELU → Gather → LN → Linear → GELU → ...  (6 gathers)
After:  LN → Linear → GELU → LN → Linear → GELU → ... → Gather           (1 gather)
```

| Pass | Transform | Effect |
|------|-----------|--------|
| **Reorder** | Push `GatherOp` past `ElementWiseOp` | Move gathers deeper |
| **Absorb** | Absorb `GatherOp` into `LinearOp` by permuting weights | Eliminate gathers |
| **Chain** | Merge consecutive `GatherOp(p1) → GatherOp(p2)` | Compose permutations |
| **Fuse** | Detect fusable `GatherOp → ElementWiseOp` patterns | Mark fusion targets |

**Key result:** 6 gathers → 1 gather (83% reduction). 1.63× inference speedup on domestic GPU, 1.80× on NVIDIA.

---

## Architecture

```
├── cube/              Group theory layer
│   ├── cube3d.py       Rubik's cube permutation generators (12 ops)
│   ├── perm_matrix.py  Permutation matrix (dense/CSR/BSR)
│   └── layers.py       nn.Module: CubePermutation, FusedPermLN, SoftCubePermutation
├── compiler/          Graph IR + optimization passes
│   ├── ir.py           3 node types: GatherOp, ElementWiseOp, LinearOp
│   └── passes.py       4 passes: Reorder, Absorb, Chain, Fuse + optimize()
├── models/            Reference model implementations
│   ├── mlp.py          StandardMLP, CubeMLP, model builders
│   └── heads.py        PoolHead, PerPositionHead, AttentionHead
├── backends/          Device-specific strategies
│   ├── nvidia.py       CUDA reference
│   ├── domestic.py     Biren gfx936 (HIP/DTK 25.04, calibrated crossover data)
│   └── muxi.py         MUXI MXMACA placeholder
├── config/            Hardware configuration
│   ├── rtx4060.yaml    NVIDIA Ada Lovelace
│   ├── domestic.yaml   Biren gfx936 (132 TFLOPS fp16 + measured data)
│   └── muxi.yaml       MUXI estimated config
├── examples/          Runnable demos
│   ├── demo_compiler.py    Compiler pipeline walkthrough
│   ├── demo_orbit.py       Orbit decomposition demo
│   └── validate_dimer.py   C2 symmetric dimer validation
├── tests/             Test suite
│   └── test_equivariance.py  All correctness tests (8 tests)
├── figures/           Experiment figures
└── paper/             Paper sources (Markdown + generator)
```

---

## Key Results

### Ablation Study (125-way position reconstruction)

| Model | Accuracy | Parameters |
|-------|----------|------------|
| StandardMLP (uniform) | 1.52% | 55,872 |
| OrbitMLP (48 orbits) | **15.06%** | 2,681,856 |
| RandomOrbitMLP (48 random) | 10.05% | 2,681,856 |
| PerPositionMLP (125 indep) | 18.52% | 6,984,000 |

- Orbit > Random by **+4.44pp (44% relative)** — same params, structured grouping wins
- Orbit achieves **78.3%** of PerPosition accuracy with **38.4%** parameters

### Sample Efficiency

| Training samples | StandardMLP | OrbitMLP |
|-----------------|------------|----------|
| 400 | 72.92% | **100.00%** |
| 3200 | 86.62% | 100.00% |

Orbit reaches 100% with **8× fewer samples** than StandardMLP can achieve at all.

### Inference Speedup

| Platform | Unoptimized | Optimized | Speedup |
|----------|-------------|-----------|---------|
| Biren gfx936 | 3.17ms | 1.94ms | **1.63×** |
| NVIDIA 4060 | 3.64ms | 2.02ms | **1.80×** |

### Gather vs Dense Crossover (gfx936, measured)

| Condition | Winner |
|-----------|--------|
| B×D < 100,000 | Gather (1.2–3.5×) |
| B×D ≥ 100,000, N ≥ 64 | Dense matmul (1.1–1.5×) |
| B×D ≥ 150,000 | Unconditionally dense |

---

## Applications

| Domain | Symmetry Group | Method |
|--------|---------------|--------|
| Crystal structure prediction | 230 space groups | Orbit sharing |
| Virus capsid assembly | Icosahedral point group | Orbit sharing |
| Molecular conformations | Molecular point groups (C₂ᵥ, D₆ₕ, Oₕ) | Orbit sharing |
| MoE model inference | Expert routing | Gather elimination |
| Transformer/LLM inference | Token rearrangement | Gather elimination |
| Multi-view 3D perception | SE(3) camera group | Both |

---

## Run Experiments

```bash
# Full experiment suite (requires GPU)
python ../魔方ai/validation/exp_throughput.py
python ../魔方ai/validation/exp_ablation.py
python ../魔方ai/validation/exp_sample_efficiency.py
python ../魔方ai/validation/exp_rotation_equivariance.py
python ../魔方ai/validation/exp_group_theory.py
python ../魔方ai/validation/exp_cube_size_scale.py
python ../魔方ai/validation/exp_training_dynamics.py
python ../魔方ai/validation/exp_backend_gather.py

# Generate figures
python ../魔方ai/validation/plot_results.py
python ../魔方ai/validation/plot_crossover.py
python ../魔方ai/validation/plot_2x2.py

# Generate paper
cd paper && python gen.py all     # HTML + LaTeX + PDF
cd paper && python gen_docx.py all  # Word documents
```

---

## Hardware Support

| Backend | Status | Notes |
|---------|--------|-------|
| NVIDIA CUDA | Full | Reference implementation, gather is fast here |
| Biren gfx936 (HIP) | Calibrated | 8 measured crossover data points, decision boundary validated |
| MUXI MXMACA | Placeholder | Migration guide included, awaiting hardware access |

---

## Citation

```bibtex
@article{orbit-sharing-2025,
  title={Orbit-Shared Weighting via Group Theory and Compiler-Driven Gather Elimination for Domestic GPU Deployment},
  note={Group-theoretic orbit decomposition + compiler IR for gather elimination on domestic GPUs},
  year={2025}
}
```

## License

Apache 2.0 — see [LICENSE](LICENSE) for details.

---

## Related Work

This repository is part of a larger research project. See also:
- **Competition entry** (XH-202608): `competition_work/` — MoE/MLA/NSA operator optimization for TileLang + MXMACA
- **Full paper**: `魔方ai/paper/` — Chinese and English versions with all figures
- **Extended experiments**: `魔方ai/validation/` — 15 experiment scripts with bootstrap CI analysis

---

*Maintained as open-source research. Contributions and discussions welcome.*
