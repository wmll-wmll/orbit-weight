"""Spatial reasoning tasks that REQUIRE understanding of 3D structure.

Two task types:

1. RotationPrediction: Given a cube-structured input, predict which face
   rotation was applied. This forces the model to learn the cube's geometry.
   Hypothesis: CubeMLP should learn faster (built-in rotational prior).

2. PositionReconstruction: Given a scrambled input, predict the original
   position of each feature vector. Tests whether the model can "unscramble"
   permutations. Hypothesis: CubeMLP should be more sample-efficient.

Both tasks use controlled difficulty (noise level, number of classes)
so we can find the point where CubeMLP's inductive bias matters.
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional

from cube.cube3d import CubePermutations


# ═══════════════════════════════════════════════════════════════
# Task 1: Rotation prediction
# ═══════════════════════════════════════════════════════════════

def make_rotation_data(
    n_samples: int,
    n_cube: int = 3,
    d_model: int = 128,
    noise: float = 0.3,
    faces: Optional[list] = None,
    seed: int = 42,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generate (data, labels) for rotation prediction.

    Each sample: random base pattern + one of K cube face rotations.
    The model must predict which rotation was applied.

    Args:
        n_samples: number of samples
        n_cube: cube side length (total positions = n_cube³)
        d_model: feature dimension
        noise: std of Gaussian noise added to features
        faces: which rotations to use (default: 6 basic face rotations)
        seed: random seed

    Returns:
        data: [n_samples, N, d_model] — base pattern with rotation applied
        labels: [n_samples] — which rotation was applied (0..K-1)
    """
    if faces is None:
        faces = ['U', 'R', 'F', 'D', 'L', 'B']

    torch.manual_seed(seed)
    N = n_cube ** 3
    K = len(faces)
    cube = CubePermutations(n_cube)

    # Pre-compute rotation permutations
    perms = [cube.get_rotation(f) for f in faces]

    # Shared base pattern — same spatial structure for all samples
    coords = torch.tensor(
        [[x, y, z] for z in range(n_cube) for y in range(n_cube) for x in range(n_cube)],
        dtype=torch.float32,
    )  # [N, 3]
    proj = torch.randn(3, d_model) / (3 ** 0.5)
    base_pattern = coords @ proj  # [N, D] — same for ALL samples

    # Pre-compute all 12 rotation permutations (6 faces × 2 directions)
    all_face_perms = {}
    for f in faces:
        all_face_perms[f] = cube.get_rotation(f)
        all_face_perms[f + "'"] = cube.get_rotation(f + "'")

    # For each class, create a 2-move composite permutation
    # This means more positions change (up to 16/27), making the
    # signal stronger and harder to memorize by noise pattern alone.
    import random
    rng = random.Random(seed)
    class_perms = []
    # Create K distinct composite permutations
    composite_names = []
    for k in range(K):
        # Each class = two consecutive rotations on different faces
        f1 = faces[k % len(faces)]
        f2 = faces[(k + 1) % len(faces)]
        p1 = all_face_perms[f1]
        p2 = all_face_perms[f2]
        composite = cube.compose(p1, p2)
        # Verify uniqueness
        attempts = 0
        while any(torch.equal(composite, cp) for cp in class_perms) and attempts < 20:
            f1 = rng.choice(faces)
            f2 = rng.choice(faces)
            composite = cube.compose(all_face_perms[f1], all_face_perms[f2])
            attempts += 1
        class_perms.append(composite)
        composite_names.append(f"{f1}+{f2}")

    # Each sample: apply the composite rotation + different noise
    data = torch.zeros(n_samples, N, d_model)
    labels = torch.randint(0, K, (n_samples,))

    for i in range(n_samples):
        k = labels[i].item()
        data[i] = base_pattern[class_perms[k]] + noise * torch.randn(N, d_model)

    return data, labels


# ═══════════════════════════════════════════════════════════════
# Task 2: Position reconstruction
# ═══════════════════════════════════════════════════════════════

def make_position_data(
    n_samples: int,
    n_cube: int = 3,
    d_model: int = 128,
    noise: float = 0.2,
    n_moves: int = 3,
    seed: int = 42,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generate (data, labels) for position reconstruction.

    Each sample: base pattern → random scramble (n_moves moves) → add noise.
    The model must predict, for each position, its INDEX in the original pattern.

    This is equivalent to learning the inverse permutation.

    Args:
        n_samples: number of samples
        n_cube: cube side length
        d_model: feature dimension
        noise: std of Gaussian noise
        n_moves: number of random cube moves in the scramble
        seed: random seed

    Returns:
        data: [n_samples, N, d_model] — scrambled pattern
        labels: [n_samples, N] — original position index for each output position
    """
    torch.manual_seed(seed)
    N = n_cube ** 3
    cube = CubePermutations(n_cube)
    move_names = ['U', 'R', 'F', 'D', 'L', 'B']

    # Shared base pattern — each position has a unique but CONSISTENT signature
    base = torch.randn(N, d_model) * 0.3
    pos_signal = torch.randn(N, d_model)
    base_pattern = base + pos_signal  # [N, D] — same for ALL samples

    data = torch.zeros(n_samples, N, d_model)
    labels = torch.zeros(n_samples, N, dtype=torch.long)

    for i in range(n_samples):
        # Generate a random scramble
        perm = torch.arange(N)
        for _ in range(n_moves):
            face = move_names[torch.randint(0, 6, (1,)).item()]
            clockwise = torch.randint(0, 2, (1,)).item()
            move_name = face if clockwise else face + "'"
            move_perm = cube.get_rotation(move_name)
            perm = perm[move_perm]

        # Apply scramble to shared base pattern
        data[i] = base_pattern[perm] + noise * torch.randn(N, d_model)
        labels[i] = perm

    return data, labels


# ═══════════════════════════════════════════════════════════════
# Task 3: Spatial classification (harder version)
# ═══════════════════════════════════════════════════════════════

def make_spatial_classification(
    n_samples: int,
    n_cube: int = 3,
    d_model: int = 128,
    n_classes: int = 10,
    noise: float = 0.5,
    seed: int = 42,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Classification where each class has a distinct 3D spatial pattern.

    Unlike the old synthetic data (which was essentially random), this
    data embeds class identity in the SPATIAL ARRANGEMENT of features.
    A model must understand the 3D structure to classify correctly.

    Each class c is defined by a 3D Gaussian "blob" in coordinate space.
    The feature at position (x,y,z) is determined by the blob's intensity
    at that coordinate. Different classes have blobs at different locations.

    Args:
        n_samples: number of samples
        n_cube: cube side length
        d_model: feature dimension
        n_classes: number of classes
        noise: std of Gaussian noise
        seed: random seed

    Returns:
        data: [n_samples, N, d_model]
        labels: [n_samples]
    """
    torch.manual_seed(seed)
    N = n_cube ** 3

    # Each class has a "center" in 3D coordinate space
    centers = torch.rand(n_classes, 3) * (n_cube - 1)

    # Each class also has a "signature" vector in feature space
    signatures = torch.randn(n_classes, d_model) * 2.0

    # Build coordinate grid
    coords = torch.tensor(
        [[x, y, z] for z in range(n_cube) for y in range(n_cube) for x in range(n_cube)],
        dtype=torch.float32,
    )  # [N, 3]

    data = torch.zeros(n_samples, N, d_model)
    labels = torch.randint(0, n_classes, (n_samples,))

    for i in range(n_samples):
        c = labels[i].item()
        # Gaussian "intensity" at each position based on distance to class center
        dist_sq = ((coords - centers[c].unsqueeze(0)) ** 2).sum(dim=1)  # [N]
        intensity = torch.exp(-dist_sq / (n_cube ** 2))  # [N]

        # Feature at each position: intensity-weighted signature + noise
        data[i] = intensity.unsqueeze(1) * signatures[c].unsqueeze(0)
        data[i] += noise * torch.randn(N, d_model)

    return data, labels
