from __future__ import annotations

import numpy as np
import open3d as o3d

from pointcloud_projection.projection import (
    build_density_artifacts,
    build_density_point_cloud,
    compute_density_grid,
    compute_output_shape,
    grid_to_grayscale_uint8,
    make_projection_image,
    project_points_to_xy,
)


def test_project_points_to_xy_sets_z_to_zero() -> None:
    points = np.array([[1.0, 2.0, 3.0], [-1.0, 0.5, -8.0]])
    projected = project_points_to_xy(points)
    assert np.allclose(projected[:, 2], 0.0)
    assert np.allclose(projected[:, :2], points[:, :2])


def test_compute_density_grid_counts_multiple_points_same_cell() -> None:
    points_xy = np.array(
        [
            [0.0, 0.0],
            [0.0, 0.0],
            [1.0, 1.0],
        ]
    )
    grid, _, _, _ = compute_density_grid(points_xy, grid_size=4)
    assert int(grid.sum()) == 3
    assert int(grid.max()) == 2


def test_grayscale_and_projection_image_have_expected_range() -> None:
    grid = np.array([[0, 1], [3, 7]], dtype=np.int32)
    image = grid_to_grayscale_uint8(grid)
    assert image.dtype == np.uint8
    assert int(image.min()) == 0
    assert int(image.max()) == 255

    projection = make_projection_image(
        np.array([[0.0, 0.0], [1.0, 1.0]]),
        image_size=8,
    )
    assert projection.dtype == np.uint8
    assert int(projection.max()) == 255


def test_compute_output_shape_preserves_aspect_ratio() -> None:
    points_xy = np.array([[0.0, 0.0], [10.0, 2.0]])
    width, height = compute_output_shape(points_xy, max_size=100)
    assert width == 100
    assert 64 <= height < width


def test_density_point_cloud_preserves_all_projected_points() -> None:
    points = np.array(
        [
            [0.0, 0.0, 1.0],
            [0.1, 0.0, 2.0],
            [2.0, 1.0, 3.0],
        ],
        dtype=float,
    )
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    artifacts = build_density_artifacts(pcd, grid_size=32)
    density_pcd = build_density_point_cloud(
        artifacts.projected_points,
        artifacts.point_density_uint8,
    )

    assert len(density_pcd.points) == len(points)
    assert np.allclose(np.asarray(density_pcd.points)[:, 2], 0.0)
