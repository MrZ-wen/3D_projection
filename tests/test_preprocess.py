from __future__ import annotations

import numpy as np
import open3d as o3d

from pointcloud_projection.preprocess import (
    apply_manual_rotation,
    estimate_principal_axis_pca,
)


def test_estimate_principal_axis_pca_finds_dominant_direction() -> None:
    t = np.linspace(-5.0, 5.0, 100)
    direction = np.array([1.0, 2.0, 3.0], dtype=float)
    direction = direction / np.linalg.norm(direction)
    points = t[:, None] * direction[None, :]
    axis = estimate_principal_axis_pca(points)

    assert abs(float(np.dot(axis, direction))) > 0.999


def test_apply_manual_rotation_rotates_points_around_z() -> None:
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.array([[1.0, 0.0, 0.0]], dtype=float))

    rotated, metadata = apply_manual_rotation(pcd, rotate_z_deg=90.0)
    point = np.asarray(rotated.points)[0]

    assert np.allclose(point, [0.0, 1.0, 0.0], atol=1e-6)
    assert metadata["rotate_z_deg"] == 90.0
