from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import open3d as o3d
from PIL import Image


@dataclass
class DensityArtifacts:
    projected_points: np.ndarray
    projected_colors: np.ndarray | None
    density_grid: np.ndarray
    density_grid_uint8: np.ndarray
    xy_min: np.ndarray
    xy_max: np.ndarray


def project_points_to_xy(points: np.ndarray) -> np.ndarray:
    projected = np.asarray(points, dtype=float).copy()
    if projected.size == 0:
        return projected.reshape((-1, 3))
    projected[:, 2] = 0.0
    return projected


def normalize_xy_to_grid(
    points_xy: np.ndarray,
    grid_size: int,
    xy_min: np.ndarray | None = None,
    xy_max: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if len(points_xy) == 0:
        empty = np.zeros((0, 2), dtype=int)
        if xy_min is None:
            xy_min = np.zeros(2, dtype=float)
        if xy_max is None:
            xy_max = np.ones(2, dtype=float)
        return empty, np.asarray(xy_min, dtype=float), np.asarray(xy_max, dtype=float)

    xy = np.asarray(points_xy, dtype=float)
    xy_min = np.min(xy, axis=0) if xy_min is None else np.asarray(xy_min, dtype=float)
    xy_max = np.max(xy, axis=0) if xy_max is None else np.asarray(xy_max, dtype=float)
    span = np.maximum(xy_max - xy_min, 1e-9)
    scaled = (xy - xy_min) / span
    indices = np.rint(scaled * (grid_size - 1)).astype(int)
    indices = np.clip(indices, 0, grid_size - 1)
    return indices, xy_min, xy_max


def compute_density_grid(points_xy: np.ndarray, grid_size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    grid = np.zeros((grid_size, grid_size), dtype=np.int32)
    indices, xy_min, xy_max = normalize_xy_to_grid(points_xy, grid_size)
    if len(indices):
        x_idx = indices[:, 0]
        y_idx = indices[:, 1]
        np.add.at(grid, (grid_size - 1 - y_idx, x_idx), 1)
    return grid, xy_min, xy_max


def grid_to_grayscale_uint8(grid: np.ndarray, log_scale: bool = True) -> np.ndarray:
    grid_float = np.asarray(grid, dtype=float)
    if log_scale:
        grid_float = np.log1p(grid_float)
    max_value = float(grid_float.max()) if grid_float.size else 0.0
    if max_value <= 0:
        return np.zeros_like(grid_float, dtype=np.uint8)
    normalized = grid_float / max_value
    return np.rint(normalized * 255.0).astype(np.uint8)


def build_density_artifacts(
    pcd: o3d.geometry.PointCloud,
    grid_size: int,
    log_scale: bool = True,
) -> DensityArtifacts:
    points = np.asarray(pcd.points, dtype=float)
    colors = np.asarray(pcd.colors, dtype=float) if pcd.has_colors() else None
    projected_points = project_points_to_xy(points)
    density_grid, xy_min, xy_max = compute_density_grid(projected_points[:, :2], grid_size=grid_size)
    density_grid_uint8 = grid_to_grayscale_uint8(density_grid, log_scale=log_scale)
    return DensityArtifacts(
        projected_points=projected_points,
        projected_colors=colors,
        density_grid=density_grid,
        density_grid_uint8=density_grid_uint8,
        xy_min=xy_min,
        xy_max=xy_max,
    )


def build_projected_point_cloud(points: np.ndarray, colors: np.ndarray | None = None) -> o3d.geometry.PointCloud:
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.asarray(points, dtype=float))
    if colors is not None and len(colors) == len(points):
        pcd.colors = o3d.utility.Vector3dVector(np.asarray(colors, dtype=float))
    return pcd


def build_density_point_cloud(
    density_grid_uint8: np.ndarray,
    xy_min: np.ndarray,
    xy_max: np.ndarray,
) -> o3d.geometry.PointCloud:
    grid = np.asarray(density_grid_uint8, dtype=np.uint8)
    if grid.size == 0:
        return o3d.geometry.PointCloud()

    nonzero = np.argwhere(grid > 0)
    if len(nonzero) == 0:
        return o3d.geometry.PointCloud()

    height, width = grid.shape
    span = np.maximum(np.asarray(xy_max, dtype=float) - np.asarray(xy_min, dtype=float), 1e-9)
    x_step = span[0] / max(width - 1, 1)
    y_step = span[1] / max(height - 1, 1)

    points = []
    colors = []
    for row, col in nonzero:
        x = float(xy_min[0] + col * x_step)
        y = float(xy_min[1] + (height - 1 - row) * y_step)
        gray = float(grid[row, col]) / 255.0
        points.append([x, y, 0.0])
        colors.append([gray, gray, gray])

    return build_projected_point_cloud(np.asarray(points, dtype=float), np.asarray(colors, dtype=float))


def save_png(image_array: np.ndarray, output_path: str | Path) -> str:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.asarray(image_array, dtype=np.uint8), mode="L").save(path)
    return str(path)


def make_projection_image(points_xy: np.ndarray, image_size: int) -> np.ndarray:
    canvas = np.zeros((image_size, image_size), dtype=np.uint8)
    indices, _, _ = normalize_xy_to_grid(points_xy, image_size)
    if len(indices):
        x_idx = indices[:, 0]
        y_idx = indices[:, 1]
        canvas[image_size - 1 - y_idx, x_idx] = 255
    return canvas
