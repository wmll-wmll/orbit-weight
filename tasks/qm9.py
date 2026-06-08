"""QM9 molecular force field data loader.

The QM9 dataset contains ~134K small organic molecules with DFT-computed
properties. This module provides a SYNTHETIC substitute that captures the
essential structure: molecules with 3D atom positions and symmetry-determined
properties, voxelized to a grid.

Two task formulations:
1. Force prediction: given a voxelized molecule, predict per-voxel "forces"
   (synthetic gradients derived from molecular energy).
2. Energy prediction: given a voxelized molecule, predict the scalar energy.

Both tasks use random octahedral rotations to inject symmetry that the
model must learn to handle — testing rotation equivariance.
"""

import torch
import numpy as np
from typing import Tuple, Optional
from groups.octahedral import OctahedralGroup


# ═══════════════════════════════════════════════════════════════════
# Task 1: Per-atom force prediction
# ═══════════════════════════════════════════════════════════════════

def make_qm9_data(
    n_samples: int,
    n_voxel: int = 5,
    d_model: int = 48,
    noise: float = 0.1,
    seed: int = 42,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generate synthetic QM9-like data for force field prediction.

    Each sample represents a small molecule voxelized to an n_voxel^3 grid.
    The task is to predict per-voxel "forces" (synthetic gradients derived
    from the molecular energy functional).

    Generation steps:
    1. Create 1-8 "atoms" at random 3D positions within [0, n_voxel)
    2. Each atom gets an "atomic number" feature (random float in [0,1])
    3. Voxelize: for each grid cell, sum features of atoms contained within
    4. Apply a random rotation from the octahedral group O_h
    5. Add Gaussian noise

    Args:
        n_samples: number of molecules
        n_voxel: grid side length (total voxels = n_voxel^3)
        d_model: feature dimension (voxel feature channels)
        noise: std of Gaussian noise added to voxel features
        seed: random seed

    Returns:
        data: [n_samples, N, d_model] — voxelized molecule features
        targets: [n_samples, N, 3] — per-voxel force vectors (x, y, z)
    """
    torch.manual_seed(seed)
    np_rng = np.random.RandomState(seed)
    N = n_voxel ** 3

    # Build coordinate grid in z-major ordering:
    # idx = z * n_voxel^2 + y * n_voxel + x
    coords = torch.tensor(
        [[x, y, z] for z in range(n_voxel) for y in range(n_voxel) for x in range(n_voxel)],
        dtype=torch.float32,
    )  # [N, 3]

    # Pre-compute coordinate grid as numpy for vectorized voxelization
    coords_np = np.array([[x, y, z] for z in range(n_voxel) for y in range(n_voxel) for x in range(n_voxel)],
                         dtype=np.float32)  # [N, 3]

    # Random projection matrix to expand atomic number (scalar) to d_model features.
    # Each atom's single "atomic number" is projected via a learned-lookalike
    # random embedding to produce d_model dimensions.
    atomic_embedding = torch.randn(d_model) / (d_model ** 0.5)

    # Use a random energy functional: each voxel's contribution to the
    # molecular energy is a learned 3D potential field. Forces are the
    # negative gradient of this field w.r.t. position.
    # We define a synthetic potential V(x,y,z) = sum of Gaussians centered
    # at random "interaction sites."
    n_interaction_sites = 8
    site_centers_np = np_rng.uniform(0, n_voxel - 1, size=(n_interaction_sites, 3)).astype(np.float32)
    site_strengths_np = np_rng.randn(n_interaction_sites).astype(np.float32)
    site_widths_np = np_rng.uniform(0.5, 1.5, size=n_interaction_sites).astype(np.float32)

    # Compute the potential values at all grid points
    # V(x) = sum_k strength_k * exp(-|x - center_k|^2 / (2 * width_k^2))
    grid_potential_np = np.zeros(N, dtype=np.float32)
    grid_force_np = np.zeros((N, 3), dtype=np.float32)  # -grad V
    for k in range(n_interaction_sites):
        diff = coords_np - site_centers_np[k]  # [N, 3]
        dist_sq = (diff ** 2).sum(axis=1)  # [N]
        w2 = site_widths_np[k] ** 2
        gauss = np.exp(-dist_sq / (2 * w2))
        grid_potential_np += site_strengths_np[k] * gauss
        # Force = -dV/dx = strength * gauss * diff / w2  (chain rule)
        grad_factor = (site_strengths_np[k] / w2) * gauss  # [N]
        grid_force_np[:, 0] += grad_factor * diff[:, 0]
        grid_force_np[:, 1] += grad_factor * diff[:, 1]
        grid_force_np[:, 2] += grad_factor * diff[:, 2]

    grid_potential = torch.from_numpy(grid_potential_np).float()  # [N]
    grid_force = torch.from_numpy(grid_force_np).float()  # [N, 3]

    # Initialize the octahedral group for random rotations
    oh_group = OctahedralGroup(n=n_voxel)
    generators = oh_group.get_generators()

    # Pre-compute a set of unique O_h group elements by composing generators
    oh_elements = _compute_oh_elements(generators, N)

    data = torch.zeros(n_samples, N, d_model)
    targets = torch.zeros(n_samples, N, 3)

    for i in range(n_samples):
        # Step 1: Create 1-8 random atoms
        n_atoms = np_rng.randint(1, 9)
        atom_positions = np_rng.uniform(0, n_voxel, size=(n_atoms, 3)).astype(np.float32)
        atom_numbers = np_rng.uniform(0, 1, size=n_atoms).astype(np.float32)

        # Step 2: Voxelize — sum atomic features at each grid cell
        # Each atom contributes its atomic number to the cell containing it
        voxel_occupancy = np.zeros(N, dtype=np.float32)
        for a in range(n_atoms):
            ax, ay, az = atom_positions[a]
            # Determine which voxel cell the atom falls into
            vx = int(np.floor(ax))
            vy = int(np.floor(ay))
            vz = int(np.floor(az))
            # Clamp to valid range
            vx = max(0, min(n_voxel - 1, vx))
            vy = max(0, min(n_voxel - 1, vy))
            vz = max(0, min(n_voxel - 1, vz))
            idx = vz * n_voxel * n_voxel + vy * n_voxel + vx
            voxel_occupancy[idx] += atom_numbers[a]

        # Normalize occupancy to avoid large dynamic range
        voxel_occupancy = voxel_occupancy / max(n_atoms, 1.0)
        voxel_features = torch.from_numpy(voxel_occupancy).float()  # [N]

        # Step 3: Expand to d_model using random projection
        # The atomic number at each voxel is embedded into d_model dimensions
        features = voxel_features.unsqueeze(1) * atomic_embedding.unsqueeze(0)  # [N, d_model]

        # Step 4: Apply a random rotation from O_h
        rotation_perm = oh_elements[np_rng.randint(0, len(oh_elements))]
        rotated_features = features[rotation_perm]  # [N, d_model]
        rotated_forces = grid_force[rotation_perm]  # [N, 3]

        # Step 5: Add Gaussian noise
        data[i] = rotated_features + noise * torch.randn(N, d_model)
        targets[i] = rotated_forces + noise * 0.01 * torch.randn(N, 3)

    return data, targets


# ═══════════════════════════════════════════════════════════════════
# Task 2: Molecular energy prediction
# ═══════════════════════════════════════════════════════════════════

def make_qm9_energy_data(
    n_samples: int,
    n_voxel: int = 5,
    d_model: int = 48,
    noise: float = 0.1,
    seed: int = 42,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generate synthetic QM9-like data for molecular energy prediction.

    Similar to make_qm9_data but targets are scalar energies per molecule
    rather than per-voxel force vectors.

    The energy is computed as the sum of the potential over all voxels,
    modulated by the atom occupancy pattern — different atom arrangements
    yield different total energies.

    Args:
        n_samples: number of molecules
        n_voxel: grid side length (total voxels = n_voxel^3)
        d_model: feature dimension
        noise: std of Gaussian noise added to voxel features
        seed: random seed

    Returns:
        data: [n_samples, N, d_model] — voxelized molecule features
        targets: [n_samples] — scalar molecular energy
    """
    torch.manual_seed(seed)
    np_rng = np.random.RandomState(seed)
    N = n_voxel ** 3

    coords_np = np.array(
        [[x, y, z] for z in range(n_voxel) for y in range(n_voxel) for x in range(n_voxel)],
        dtype=np.float32,
    )  # [N, 3]

    atomic_embedding = torch.randn(d_model) / (d_model ** 0.5)

    # Define the synthetic potential field V(x,y,z)
    n_interaction_sites = 8
    site_centers_np = np_rng.uniform(0, n_voxel - 1, size=(n_interaction_sites, 3)).astype(np.float32)
    site_strengths_np = np_rng.randn(n_interaction_sites).astype(np.float32)
    site_widths_np = np_rng.uniform(0.5, 1.5, size=n_interaction_sites).astype(np.float32)

    grid_potential_np = np.zeros(N, dtype=np.float32)
    for k in range(n_interaction_sites):
        diff = coords_np - site_centers_np[k]
        dist_sq = (diff ** 2).sum(axis=1)
        w2 = site_widths_np[k] ** 2
        grid_potential_np += site_strengths_np[k] * np.exp(-dist_sq / (2 * w2))

    grid_potential = torch.from_numpy(grid_potential_np).float()  # [N]

    # Initialize octahedral group for random rotations
    oh_group = OctahedralGroup(n=n_voxel)
    generators = oh_group.get_generators()
    oh_elements = _compute_oh_elements(generators, N)

    data = torch.zeros(n_samples, N, d_model)
    targets = torch.zeros(n_samples)

    for i in range(n_samples):
        # Step 1: Create 1-8 random atoms
        n_atoms = np_rng.randint(1, 9)
        atom_positions = np_rng.uniform(0, n_voxel, size=(n_atoms, 3)).astype(np.float32)
        atom_numbers = np_rng.uniform(0, 1, size=n_atoms).astype(np.float32)

        # Step 2: Voxelize
        voxel_occupancy = np.zeros(N, dtype=np.float32)
        for a in range(n_atoms):
            ax, ay, az = atom_positions[a]
            vx = max(0, min(n_voxel - 1, int(np.floor(ax))))
            vy = max(0, min(n_voxel - 1, int(np.floor(ay))))
            vz = max(0, min(n_voxel - 1, int(np.floor(az))))
            idx = vz * n_voxel * n_voxel + vy * n_voxel + vx
            voxel_occupancy[idx] += atom_numbers[a]

        voxel_occupancy = voxel_occupancy / max(n_atoms, 1.0)
        voxel_features = torch.from_numpy(voxel_occupancy).float()  # [N]

        # Step 3: Expand to d_model
        features = voxel_features.unsqueeze(1) * atomic_embedding.unsqueeze(0)  # [N, d_model]

        # Step 4: Apply random O_h rotation
        rotation_perm = oh_elements[np_rng.randint(0, len(oh_elements))]
        rotated_features = features[rotation_perm]  # [N, d_model]

        # Step 5: Add noise to features
        data[i] = rotated_features + noise * torch.randn(N, d_model)

        # Energy = sum over voxels of (occupancy * potential)
        # The rotation permutes the occupancy pattern relative to the fixed
        # potential field, so the total energy varies per sample.
        energy = (voxel_features * grid_potential).sum()
        targets[i] = energy + noise * 0.01 * torch.randn(1)

    return data, targets


# ═══════════════════════════════════════════════════════════════════
# Task 3: Placeholder for real QM9 download
# ═══════════════════════════════════════════════════════════════════

def download_qm9(data_dir: str = "qm9_data") -> str:
    """Placeholder for downloading the real QM9 dataset.

    Creates the data directory and prints instructions for obtaining
    the actual QM9 dataset (134K molecules, ~3.4 GB compressed).

    The real QM9 dataset can be downloaded from:
        https://figshare.com/collections/quantum_chemistry_structures_and_properties_of_134_kilo_molecules/978904

    Or via deepchem:
        import deepchem as dc
        tasks, datasets, transformers = dc.molnet.load_qm9()
        train, valid, test = datasets

    Args:
        data_dir: path to create for QM9 data storage

    Returns:
        data_dir: the created directory path
    """
    import os

    os.makedirs(data_dir, exist_ok=True)

    instructions = f"""
================================================================================
QM9 Dataset Download Instructions
================================================================================

The QM9 dataset (134K molecules with DFT-computed properties) is not included
in this repository due to its size (~3.4 GB compressed).

To download the real QM9 dataset:

Option 1: Direct download
    Visit: https://figshare.com/collections/quantum_chemistry_structures_and_properties_of_134_kilo_molecules/978904
    Download files: dsgdb9nsd.xyz.tar.bz2, uncharted.tar.bz2
    Extract into: {os.path.abspath(data_dir)}/

Option 2: Using deepchem
    ```python
    import deepchem as dc
    tasks, datasets, transformers = dc.molnet.load_qm9()
    train, valid, test = datasets
    ```

Option 3: Using torch-geometric
    ```python
    from torch_geometric.datasets import QM9
    dataset = QM9(root='{os.path.abspath(data_dir)}')
    ```

Until the real dataset is downloaded, use the synthetic generators:
    from tasks.qm9 import make_qm9_data, make_qm9_energy_data
    data, forces = make_qm9_data(n_samples=1000)       # force prediction
    data, energy = make_qm9_energy_data(n_samples=1000)  # energy prediction

================================================================================
"""
    print(instructions)
    return os.path.abspath(data_dir)


# ═══════════════════════════════════════════════════════════════════
# Helper: compute O_h group elements by composing generators
# ═══════════════════════════════════════════════════════════════════

def _compute_oh_elements(generators, N: int) -> list:
    """Compute all unique O_h group elements via BFS on the Cayley graph.

    Starting from the identity permutation, apply all generators (and their
    inverses) repeatedly to discover all 48 elements of the octahedral group.

    Args:
        generators: list of permutation tensors from OctahedralGroup
        N: number of positions

    Returns:
        list of permutation tensors (all unique group elements)
    """
    identity = torch.arange(N, dtype=torch.long)
    # Build the full generator set (forward + inverse)
    all_gens = list(generators)
    # Add inverses by computing inverse permutations
    for gen in generators:
        inv_perm = torch.empty(N, dtype=torch.long)
        inv_perm[gen] = torch.arange(N, dtype=torch.long)
        all_gens.append(inv_perm)

    # BFS on the Cayley graph
    seen = {tuple(identity.tolist())}
    elements = [identity]
    queue = [identity]

    while queue:
        current = queue.pop(0)
        for gen in all_gens:
            neighbor = gen[current]
            key = tuple(neighbor.tolist())
            if key not in seen:
                seen.add(key)
                elements.append(neighbor)
                queue.append(neighbor)

    return elements


# ═══════════════════════════════════════════════════════════════════
# Quick test
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("Testing QM9 synthetic data generators...")
    print()

    # Test force prediction
    data, forces = make_qm9_data(n_samples=16, n_voxel=5, d_model=48, noise=0.1, seed=42)
    print(f"Force prediction:")
    print(f"  data:    {data.shape}     (expected: [16, 125, 48])")
    print(f"  targets: {forces.shape}  (expected: [16, 125, 3])")
    print(f"  data range:    [{data.min().item():.3f}, {data.max().item():.3f}]")
    print(f"  target range:  [{forces.min().item():.3f}, {forces.max().item():.3f}]")
    print()

    # Test energy prediction
    data_e, energy = make_qm9_energy_data(n_samples=16, n_voxel=5, d_model=48, noise=0.1, seed=42)
    print(f"Energy prediction:")
    print(f"  data:    {data_e.shape}     (expected: [16, 125, 48])")
    print(f"  targets: {energy.shape}      (expected: [16])")
    print(f"  data range:    [{data_e.min().item():.3f}, {data_e.max().item():.3f}]")
    print(f"  target range:  [{energy.min().item():.3f}, {energy.max().item():.3f}]")
    print()

    # Test download placeholder
    data_dir = download_qm9("qm9_data_test")
    print(f"Data directory: {data_dir}")
