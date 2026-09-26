#!/usr/bin/env python3
"""Reproduce the corrected and archived 4x4x4 grid-action audit with NumPy only."""
from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "work_results" / "grid_audit_20260916.json"


def coordinates(n=4):
    return np.array([[x, y, z] for z in range(n) for y in range(n) for x in range(n)])


def linear_index(c, n=4):
    return c[:, 2] * n * n + c[:, 1] * n + c[:, 0]


def signed_permutation_matrices():
    out = []
    for perm in itertools.permutations(range(3)):
        for signs in itertools.product((1, -1), repeat=3):
            matrix = np.zeros((3, 3), dtype=int)
            for i, j in enumerate(perm):
                matrix[i, j] = signs[i]
            out.append(matrix)
    return out


def bfs_labels(generators, n_positions):
    labels = np.full(n_positions, -1, dtype=int)
    orbit = 0
    for start in range(n_positions):
        if labels[start] >= 0:
            continue
        labels[start] = orbit
        stack = [start]
        while stack:
            position = stack.pop()
            for generator in generators:
                neighbour = int(generator[position])
                if labels[neighbour] < 0:
                    labels[neighbour] = orbit
                    stack.append(neighbour)
        orbit += 1
    return labels


def main():
    n = 4
    coords = coordinates(n)
    matrices = signed_permutation_matrices()
    centre = (n - 1) / 2

    corrected = []
    for matrix in matrices:
        transformed = np.rint((coords - centre) @ matrix.T + centre).astype(int)
        corrected.append(linear_index(transformed, n))
    lookup = {tuple(mapping.tolist()) for mapping in corrected}
    closure = sum(
        tuple(a[b].tolist()) in lookup for a in corrected for b in corrected
    )
    corrected_labels = ((coords == 0) | (coords == n - 1)).sum(axis=1)
    corrected_sizes = sorted(
        (int(np.count_nonzero(corrected_labels == label))
         for label in np.unique(corrected_labels)), reverse=True
    )

    c4z = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    c4x = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
    mirror = np.array([[1, 0, 0], [0, 1, 0], [0, 0, -1]])
    archived_generators = []
    for matrix in (c4z, c4x, mirror):
        for candidate in (matrix, matrix.T):
            transformed = np.clip(coords @ candidate.T, 0, n - 1).astype(int)
            archived_generators.append(linear_index(transformed, n))
    archived_labels = bfs_labels(archived_generators, n ** 3)
    archived_sizes = sorted(
        (int(np.count_nonzero(archived_labels == label))
         for label in np.unique(archived_labels)), reverse=True
    )

    archived_all = []
    for matrix in matrices:
        transformed = np.clip(coords @ matrix.T, 0, n - 1).astype(int)
        archived_all.append(linear_index(transformed, n))

    report = {
        "group": "48 signed permutation matrices",
        "n_grid": 64,
        "corrected": {
            "gauge": "cube-centred",
            "bijective_maps": int(sum(np.unique(x).size == 64 for x in corrected)),
            "closure_ok": int(closure),
            "closure_total": 48 * 48,
            "n_orbits": int(np.unique(corrected_labels).size),
            "orbit_sizes": corrected_sizes,
        },
        "archived": {
            "gauge": "origin-centred and clipped to [0,3]",
            "non_bijective_maps": int(sum(np.unique(x).size != 64 for x in archived_all)),
            "stored_generator_distinct_images": sorted(
                {int(np.unique(x).size) for x in archived_generators}
            ),
            "classes_from_archived_bfs": int(np.unique(archived_labels).size),
            "class_sizes": archived_sizes,
            "status": "not an orbit decomposition because the generators are not bijections",
        },
    }
    OUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
