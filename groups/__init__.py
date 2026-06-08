"""
General group theory module for orbit-based weight sharing.

Extends the Rubik's Cube face rotation group to arbitrary finite groups
acting on a set of N positions. Provides a unified API for orbit computation
and OrbitLinear model construction.

Supported groups:
    - CyclicGroup (C_n)         — 1D cyclic shifts
    - DihedralGroup (D_n)       — 2D rotations + reflections
    - SymmetricGroup (S_n)      — all permutations (fully connected)
    - OctahedralGroup (O_h)     — 3D octahedral/cubic symmetry
    - RubikCubeGroup            — face rotations (from cube/cube3d.py)

Usage:
    from groups import CyclicGroup, compute_orbits, create_orbit_model

    group = CyclicGroup(n=8, step=1)
    orbit_ids, n_orbits = group.compute_orbits()
    model = create_orbit_model(group, D=64, n_layers=4)
"""

from .base import Group, compute_orbits, create_orbit_model
from .cyclic import CyclicGroup
from .dihedral import DihedralGroup
from .symmetric import SymmetricGroup
from .octahedral import OctahedralGroup
