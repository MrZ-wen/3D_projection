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
    density_indices: np.ndarray
    point_density_uint8: np.ndarray
    xy_min: np.ndarray
    xy_max: np.ndarray


def project_points_to_xy(points: np.ndarray) -> np.ndarray:
    projected = np.asarray(points, dtype=float).copy()
    if projected.size == 0:
        return projected.reshape((-1, 3))
    projected[:, 2] = 0.0
    return projected


def compute_output_shape(points_xy: np.ndarray, max_size: int, min_size: int = 64) -> tuple[int, int]:
    max_size = max(int(max_size), 1)
    min_size = max(1, min(int(min_size), max_size))
    if len(points_xy) == 0:
        return max_size, max_size

    xy = np.asarray(points_xy, dtype=float)
    span = np.ptp(xy, axis=0)
    x_span = max(float(span[0]), 1e-9)
    y_span = max(float(span[1]), 1e-9)
    dominant = max(x_span, y_span)

    width = max(int(np.ceil(max_size * x_span / dominant)), min_size)
    height = max(int(np.ceil(max_size * y_span / dominant)), min_size)
    return width, height


def normalize_xy_to_grid(
    points_xy: np.ndarray,
    width: int,
    height: int,
    xy_min: np.ndarray | None = None,
    xy_max: np.ndarray | None = None,
    padding: int = 1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if len(points_xy) == 0:
        empty = np.zeros((0, 2), dtype=int)
        if xy_min is None:
            xy_min = np.zeros(2, dtype=float)
        if xy_max is None:
            xy_max = np.ones(2, dtype=float)
        return empty, np.asarray(xy_min, dtype=float), np.asarray(xy_max, dtype=float)

    xy = np.asarray(points_xy, dtype=float)
    width = max(int(width), 1)
    height = max(int(height), 1)
    padding = max(int(padding), 0)
    xy_min = np.min(xy, axis=0) if xy_min is None else np.asarray(xy_min, dtype=float)
    xy_max = np.max(xy, axis=0) if xy_max is None else np.asarray(xy_max, dtype=float)
    span = np.maximum(xy_max - xy_min, 1e-9)
    scaled = (xy - xy_min) / span
    usable_width = max(width - 1 - 2 * padding, 1)
    usable_height = max(height - 1 - 2 * padding, 1)
    indices = np.empty((len(xy), 2), dtype=int)
    indices[:, 0] = padding + np.rint(scaled[:, 0] * usable_width).astype(int)
    indices[:, 1] = padding + np.rint(scaled[:, 1] * usable_height).astype(int)
    indices[:, 0] = np.clip(indices[:, 0], 0, width - 1)
    indices[:, 1] = np.clip(indices[:, 1], 0, height - 1)
    return indices, xy_min, xy_max


def compute_density_grid(
    points_xy: np.ndarray, grid_size: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    width, height = compute_output_shape(points_xy, grid_size)
    grid = np.zeros((height, width), dtype=np.int32)
    indices, xy_min, xy_max = normalize_xy_to_grid(points_xy, width, height)
    if len(indices):
        x_idx = indices[:, 0]
        y_idx = indices[:, 1]
        np.add.at(grid, (height - 1 - y_idx, x_idx), 1)
    return grid, indices, xy_min, xy_max


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
    density_grid, density_indices, xy_min, xy_max = compute_density_grid(
        projected_points[:, :2],
        grid_size=grid_size,
    )
    density_grid_uint8 = grid_to_grayscale_uint8(density_grid, log_scale=log_scale)
    if len(density_indices):
        point_density_uint8 = density_grid_uint8[
            density_grid_uint8.shape[0] - 1 - density_indices[:, 1],
            density_indices[:, 0],
        ]
    else:
        point_density_uint8 = np.zeros((0,), dtype=np.uint8)
    return DensityArtifacts(
        projected_points=projected_points,
        projected_colors=colors,
        density_grid=density_grid,
        density_grid_uint8=density_grid_uint8,
        density_indices=density_indices,
        point_density_uint8=point_density_uint8,
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
    projected_points: np.ndarray,
    point_density_uint8: np.ndarray,
) -> o3d.geometry.PointCloud:
    points = np.asarray(projected_points, dtype=float)
    point_density_uint8 = np.asarray(point_density_uint8, dtype=np.uint8)
    if len(points) == 0:
        return o3d.geometry.PointCloud()
    if len(point_density_uint8) != len(points):
        return o3d.geometry.PointCloud()
    gray = point_density_uint8.astype(float) / 255.0
    colors = np.repeat(gray[:, None], 3, axis=1)
    return build_projected_point_cloud(points, colors)


def save_png(image_array: np.ndarray, output_path: str | Path) -> str:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.asarray(image_array, dtype=np.uint8), mode="L").save(path)
    return str(path)


def make_projection_image(points_xy: np.ndarray, image_size: int) -> np.ndarray:
    width, height = compute_output_shape(points_xy, image_size)
    canvas = np.zeros((height, width), dtype=np.uint8)
    indices, _, _ = normalize_xy_to_grid(points_xy, width, height)
    if len(indices):
        x_idx = indices[:, 0]
        y_idx = indices[:, 1]
        canvas[height - 1 - y_idx, x_idx] = 255
    return canvas
