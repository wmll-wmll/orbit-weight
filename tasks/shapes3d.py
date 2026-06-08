"""3D shape classification benchmark.

Generates 3D voxel shapes (5x5x5) and applies cube rotations.
Task: classify the shape regardless of applied rotation.

This is a clean test of rotation invariance/equivariance.
"""

import torch
import numpy as np
from cube.cube3d import CubePermutations


def make_shape_cube(n_cube=5):
    """Generate a 3D cube shape (all voxels filled)."""
    N = n_cube ** 3
    return torch.ones(N)


def make_shape_sphere(n_cube=5):
    """Generate a 3D sphere shape (voxels within radius)."""
    center = (n_cube - 1) / 2
    radius = n_cube / 2 - 0.5
    coords = []
    for z in range(n_cube):
        for y in range(n_cube):
            for x in range(n_cube):
                d = ((x - center) ** 2 + (y - center) ** 2 + (z - center) ** 2) ** 0.5
                coords.append(1.0 if d <= radius else 0.0)
    return torch.tensor(coords, dtype=torch.float32)


def make_shape_cross(n_cube=5):
    """Generate a 3D cross shape (three orthogonal bars through center)."""
    center = (n_cube - 1) / 2
    coords = []
    for z in range(n_cube):
        for y in range(n_cube):
            for x in range(n_cube):
                on_x = abs(y - center) <= 0.5 and abs(z - center) <= 0.5
                on_y = abs(x - center) <= 0.5 and abs(z - center) <= 0.5
                on_z = abs(x - center) <= 0.5 and abs(y - center) <= 0.5
                coords.append(1.0 if (on_x or on_y or on_z) else 0.0)
    return torch.tensor(coords, dtype=torch.float32)


def make_shape_L(n_cube=5):
    """Generate an L-shape (two bars meeting at corner)."""
    coords = []
    for z in range(n_cube):
        for y in range(n_cube):
            for x in range(n_cube):
                bar_x = (x <= 1) & (y <= 1)
                bar_z = (z <= 1) & (y <= 1)
                coords.append(1.0 if (bar_x or bar_z) else 0.0)
    return torch.tensor(coords, dtype=torch.float32)


def make_shape_corner(n_cube=5):
    """Generate a corner piece (3 orthogonal faces meeting at corner)."""
    coords = []
    for z in range(n_cube):
        for y in range(n_cube):
            for x in range(n_cube):
                val = 1.0 if (x <= 1) or (y <= 1) or (z <= 1) else 0.0
                coords.append(val)
    return torch.tensor(coords, dtype=torch.float32)


def make_shape_checker(n_cube=5):
    """Checkerboard pattern (alternating voxels)."""
    coords = []
    for z in range(n_cube):
        for y in range(n_cube):
            for x in range(n_cube):
                coords.append(1.0 if (x + y + z) % 2 == 0 else 0.0)
    return torch.tensor(coords, dtype=torch.float32)


def make_shape_hollow(n_cube=5):
    """Hollow cube (only surface voxels)."""
    coords = []
    for z in range(n_cube):
        for y in range(n_cube):
            for x in range(n_cube):
                on_surface = (x == 0 or x == n_cube - 1 or
                              y == 0 or y == n_cube - 1 or
                              z == 0 or z == n_cube - 1)
                coords.append(1.0 if on_surface else 0.0)
    return torch.tensor(coords, dtype=torch.float32)


def make_shape_ring(n_cube=5):
    """Ring/torus shape along z-axis."""
    center = (n_cube - 1) / 2
    r_inner = 1.0
    r_outer = n_cube / 2 - 0.5
    coords = []
    for z in range(n_cube):
        for y in range(n_cube):
            for x in range(n_cube):
                d = ((x - center) ** 2 + (y - center) ** 2) ** 0.5
                in_ring = r_inner <= d <= r_outer and abs(z - center) <= 0.5
                coords.append(1.0 if in_ring else 0.0)
    return torch.tensor(coords, dtype=torch.float32)


SHAPE_GENERATORS = {
    'cube': make_shape_cube,
    'sphere': make_shape_sphere,
    'cross': make_shape_cross,
    'Lshape': make_shape_L,
    'corner': make_shape_corner,
    'checker': make_shape_checker,
    'hollow': make_shape_hollow,
    'ring': make_shape_ring,
}


def make_shape_classification(
    n_samples: int,
    n_cube: int = 5,
    d_model: int = 48,
    noise: float = 0.1,
    shapes: list = None,
    seed: int = 42,
):
    """Generate 3D shape classification data.

    Each sample is a shape pattern, possibly rotated, with noise.
    The shape pattern fills the first 1 dimension (expanded to d_model).

    Args:
        n_samples: total samples (will be split evenly across shapes)
        n_cube: cube side length
        d_model: feature dimension
        noise: Gaussian noise std
        shapes: list of shape names (subset of SHAPE_GENERATORS keys)
        seed: random seed

    Returns:
        data: [n_samples, N, d_model] — voxel features
        labels: [n_samples] — shape class index (0..len(shapes)-1)
    """
    if shapes is None:
        shapes = list(SHAPE_GENERATORS.keys())

    N = n_cube ** 3
    n_classes = len(shapes)
    samples_per_class = n_samples // n_classes
    n_samples = samples_per_class * n_classes

    rng = np.random.RandomState(seed)
    torch.manual_seed(seed)

    data = torch.zeros(n_samples, N, d_model)
    labels = torch.zeros(n_samples, dtype=torch.long)

    cube = CubePermutations(n_cube)
    all_faces = ['U', 'R', 'F', 'D', 'L', 'B']

    for cls_idx, shape_name in enumerate(shapes):
        shape_fn = SHAPE_GENERATORS[shape_name]
        base_shape = shape_fn(n_cube)  # [N] binary

        for i in range(samples_per_class):
            idx = cls_idx * samples_per_class + i

            # Apply random rotation
            n_rots = rng.randint(0, 5)
            perm = torch.arange(N)
            for _ in range(n_rots):
                face = all_faces[rng.randint(0, 6)]
                clockwise = rng.randint(0, 2) == 0
                move_perm = cube.get_rotation(face if clockwise else face + "'")
                perm = move_perm[perm]

            rotated = base_shape[perm]

            # Expand to d_model dimensions (shape mask replicated across features)
            feature = rotated.unsqueeze(1).expand(-1, d_model).clone()
            feature += noise * torch.randn(N, d_model)

            data[idx] = feature
            labels[idx] = cls_idx

    return data, labels


if __name__ == "__main__":
    # Quick test
    data, labels = make_shape_classification(160, n_cube=5, d_model=32, noise=0.1, seed=42)
    print(f"Data: {data.shape}, Labels: {labels.shape}")
    print(f"Classes: {labels.bincount()}")
    for i, name in enumerate(SHAPE_GENERATORS.keys()):
        print(f"  Class {i}: {name}")
