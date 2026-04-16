from __future__ import annotations

import copy
from dataclasses import asdict, dataclass

import numpy as np
import open3d as o3d


@dataclass
class PreprocessConfig:
    nb_neighbors: int = 20
    std_ratio: float = 5.0
    black_filter: int = 0
    black_threshold: float = 0.2
    slicing_ratio: float = 0.10
    adjustment: float = 0.0


def rotation_matrix_from_vectors(vec1: np.ndarray, vec2: np.ndarray) -> np.ndarray:
    """Return a rotation matrix that aligns ``vec1`` to ``vec2``."""
    a = (vec1 / np.linalg.norm(vec1)).reshape(3)
    b = (vec2 / np.linalg.norm(vec2)).reshape(3)
    v = np.cross(a, b)
    if np.any(v):
        c = np.dot(a, b)
        s = np.linalg.norm(v)
        kmat = np.array(
            [[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]],
            dtype=float,
        )
        return np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s**2))
    return np.eye(3)


def sort_index(values: list[float], reverse: bool = True) -> list[int]:
    return sorted(range(len(values)), reverse=reverse, key=lambda i: values[i])


def get_bottom_slice(points: np.ndarray, slicing_ratio: float) -> np.ndarray:
    z_sorted = np.sort(points[:, 2])
    slice_idx = int(len(z_sorted) * slicing_ratio)
    z_mid = z_sorted[slice_idx]
    z_mask = (points[:, 2] <= z_mid) & (points[:, 2] >= z_sorted[0])
    return points[z_mask]


def get_top_slice(points: np.ndarray, slicing_ratio: float) -> np.ndarray:
    z_sorted = np.sort(points[:, 2])
    if len(z_sorted) == 0:
        return points
    slice_idx = max(0, len(z_sorted) - 1 - int(len(z_sorted) * slicing_ratio))
    z_mid = z_sorted[slice_idx]
    z_mask = (points[:, 2] >= z_mid) & (points[:, 2] <= z_sorted[-1])
    return points[z_mask]


def get_end_slice(points: np.ndarray, slicing_ratio: float, end: str = "top") -> np.ndarray:
    if end == "bottom":
        return get_bottom_slice(points, slicing_ratio)
    return get_top_slice(points, slicing_ratio)


def get_basal_anchor_slice(
    points: np.ndarray, slicing_ratio: float, end: str = "top"
) -> tuple[np.ndarray, float]:
    if len(points) == 0:
        return points, 0.0

    z_values = np.asarray(points[:, 2], dtype=float)
    z_max = float(np.max(z_values))
    z_min = float(np.min(z_values))
    z_span = max(z_max - z_min, 1e-6)

    initial_ratio = float(np.clip(max(0.008, slicing_ratio * 0.15), 0.008, 0.03))
    max_ratio = float(
        np.clip(max(initial_ratio * 2.0, min(slicing_ratio, 0.08)), initial_ratio, 0.08)
    )
    target_fraction = float(np.clip(slicing_ratio * 0.08, 0.003, 0.012))
    target_count = int(np.clip(len(points) * target_fraction, 24, 256))

    current_ratio = initial_ratio
    if end == "bottom":
        anchor_slice = points[z_values <= z_min + current_ratio * z_span]
    else:
        anchor_slice = points[z_values >= z_max - current_ratio * z_span]

    while len(anchor_slice) < target_count and current_ratio < max_ratio - 1e-9:
        current_ratio = min(max_ratio, current_ratio * 1.6)
        if end == "bottom":
            anchor_slice = points[z_values <= z_min + current_ratio * z_span]
        else:
            anchor_slice = points[z_values >= z_max - current_ratio * z_span]

    if len(anchor_slice) >= 12:
        return anchor_slice, current_ratio

    fallback_ratio = max(slicing_ratio, 0.03)
    fallback_slice = get_end_slice(points, slicing_ratio=fallback_ratio, end=end)
    if len(fallback_slice):
        return fallback_slice, fallback_ratio
    return points, 1.0


def _compute_end_band_stats(points: np.ndarray) -> dict[str, float | np.ndarray]:
    if len(points) == 0:
        return {
            "count": 0.0,
            "density": 0.0,
            "spread": 0.0,
            "centroid": np.zeros(2, dtype=float),
        }
    xspan = float(np.max(points[:, 0]) - np.min(points[:, 0]))
    yspan = float(np.max(points[:, 1]) - np.min(points[:, 1]))
    spread = max(xspan, yspan)
    area = max(xspan * yspan, 1e-6)
    centroid = np.median(points[:, :2], axis=0).astype(float)
    return {
        "count": float(len(points)),
        "density": float(len(points) / area),
        "spread": float(spread),
        "centroid": centroid,
    }


def score_end_basal_likelihood(
    points: np.ndarray,
    end: str,
    slicing_ratio: float,
    band_count: int = 4,
) -> dict[str, float]:
    if len(points) == 0:
        return {
            "total_count": 0.0,
            "mean_density": 0.0,
            "mean_spread": 0.0,
            "occupied_fraction": 0.0,
            "continuity": 0.0,
            "score": 0.0,
        }

    z_values = np.asarray(points[:, 2], dtype=float)
    z_min = float(np.min(z_values))
    z_max = float(np.max(z_values))
    z_span = max(z_max - z_min, 1e-6)
    total_ratio = float(np.clip(max(0.04, slicing_ratio * 1.2), 0.04, 0.18))
    total_thickness = total_ratio * z_span
    band_thickness = total_thickness / max(band_count, 1)

    bands: list[dict[str, float | np.ndarray]] = []
    for idx in range(band_count):
        if end == "bottom":
            band_min = z_min + idx * band_thickness
            band_max = z_min + (idx + 1) * band_thickness
            mask = (z_values >= band_min) & (z_values <= band_max)
        else:
            band_max = z_max - idx * band_thickness
            band_min = z_max - (idx + 1) * band_thickness
            mask = (z_values >= band_min) & (z_values <= band_max)
        bands.append(_compute_end_band_stats(points[mask]))

    nonempty = [band for band in bands if float(band["count"]) > 0]
    total_count = float(np.sum([band["count"] for band in bands], dtype=float))
    mean_density = (
        float(np.mean([band["density"] for band in nonempty], dtype=float)) if nonempty else 0.0
    )
    mean_spread = (
        float(np.mean([band["spread"] for band in nonempty], dtype=float)) if nonempty else 0.0
    )
    occupied_fraction = float(len(nonempty) / max(band_count, 1))

    continuity_scores: list[float] = []
    for band_a, band_b in zip(bands[:-1], bands[1:]):
        if float(band_a["count"]) <= 0 or float(band_b["count"]) <= 0:
            continue
        scale = max(1.0, 0.5 * (float(band_a["spread"]) + float(band_b["spread"])))
        shift = float(np.linalg.norm(np.asarray(band_a["centroid"]) - np.asarray(band_b["centroid"])))
        shift_score = float(np.clip(1.0 - shift / scale, 0.0, 1.0))
        spread_score = float(
            min(float(band_a["spread"]), float(band_b["spread"]))
            / max(float(band_a["spread"]), float(band_b["spread"]), 1e-6)
        )
        density_score = float(
            min(float(band_a["density"]), float(band_b["density"]))
            / max(float(band_a["density"]), float(band_b["density"]), 1e-6)
        )
        continuity_scores.append(0.50 * shift_score + 0.30 * spread_score + 0.20 * density_score)

    continuity = float(np.mean(continuity_scores, dtype=float)) if continuity_scores else 0.0
    score = (
        0.24 * np.log1p(total_count)
        + 0.18 * np.log1p(mean_density)
        + 0.28 * np.log1p(mean_spread)
        + 0.14 * occupied_fraction
        + 0.16 * continuity
    )
    return {
        "total_count": total_count,
        "mean_density": mean_density,
        "mean_spread": mean_spread,
        "occupied_fraction": occupied_fraction,
        "continuity": continuity,
        "score": float(score),
    }


def choose_basal_end(points: np.ndarray, slicing_ratio: float) -> tuple[str, dict[str, float], dict[str, float]]:
    top_stats = score_end_basal_likelihood(points, end="top", slicing_ratio=slicing_ratio)
    bottom_stats = score_end_basal_likelihood(points, end="bottom", slicing_ratio=slicing_ratio)
    if bottom_stats["score"] > top_stats["score"]:
        return "bottom", top_stats, bottom_stats
    return "top", top_stats, bottom_stats


def remove_black_points(
    pcd: o3d.geometry.PointCloud, black_threshold: float
) -> tuple[o3d.geometry.PointCloud, np.ndarray]:
    colors = np.asarray(pcd.colors)
    black_mask = np.all(colors <= black_threshold, axis=1)
    filtered = pcd.select_by_index(np.where(~black_mask)[0])
    return filtered, black_mask


def statistical_outlier_removal(
    pcd: o3d.geometry.PointCloud, nb_neighbors: int, std_ratio: float
) -> o3d.geometry.PointCloud:
    pcd_np = np.asarray(pcd.points)
    if len(pcd_np) == 0:
        return copy.deepcopy(pcd)

    effective_neighbors = min(max(nb_neighbors, 1), max(len(pcd_np) - 1, 1))
    pcd_tree = o3d.geometry.KDTreeFlann(pcd)
    distances = []
    for i in range(len(pcd_np)):
        _, indices, _ = pcd_tree.search_knn_vector_3d(pcd_np[i], effective_neighbors + 1)
        neighbors = pcd_np[indices[1:]]
        if len(neighbors) == 0:
            distances.append(0.0)
            continue
        distances.append(np.mean(np.linalg.norm(neighbors - pcd_np[i], axis=1)))

    mean_dist = np.mean(distances)
    std_dist = np.std(distances)
    if std_dist <= 1e-12:
        return copy.deepcopy(pcd)
    inliers = np.where(np.abs(distances - mean_dist) < std_ratio * std_dist)[0]
    return pcd.select_by_index(inliers)


def clean_point_cloud(
    pcd: o3d.geometry.PointCloud, config: PreprocessConfig
) -> o3d.geometry.PointCloud:
    pcd_filtered = statistical_outlier_removal(pcd, config.nb_neighbors, config.std_ratio)

    if config.black_filter == 0 or not pcd_filtered.has_colors():
        pcd_selected = pcd_filtered
    else:
        pcd_selected, _ = remove_black_points(pcd_filtered, config.black_threshold)

    pcd_cleaned = copy.deepcopy(pcd_selected)
    model_center = pcd_cleaned.get_center()
    pcd_cleaned.translate(-1 * model_center)
    return pcd_cleaned


def anchor_basal_to_z_axis_origin(
    pcd: o3d.geometry.PointCloud, slicing_ratio: float, basal_end: str = "top"
) -> o3d.geometry.PointCloud:
    pcd_anchored = copy.deepcopy(pcd)
    points = np.asarray(pcd_anchored.points, dtype=float)
    if len(points) == 0:
        return pcd_anchored

    top_slice, _ = get_basal_anchor_slice(
        points,
        slicing_ratio=max(slicing_ratio, 0.03),
        end=basal_end,
    )
    if len(top_slice) == 0:
        top_slice = points

    top_center_xy = np.median(top_slice[:, :2], axis=0)
    basal_z = float(np.max(points[:, 2])) if basal_end == "top" else float(np.min(points[:, 2]))
    translation = np.array([-top_center_xy[0], -top_center_xy[1], -basal_z], dtype=float)
    pcd_anchored.translate(translation)

    anchored_points = np.asarray(pcd_anchored.points, dtype=float)
    if basal_end == "top":
        residual = float(np.max(anchored_points[:, 2]))
        if residual > 1e-6:
            pcd_anchored.translate((0.0, 0.0, -residual))
    else:
        residual = float(np.min(anchored_points[:, 2]))
        if residual < -1e-6:
            pcd_anchored.translate((0.0, 0.0, -residual))
    return pcd_anchored


def align_point_cloud_to_z(
    pcd: o3d.geometry.PointCloud, config: PreprocessConfig
) -> tuple[o3d.geometry.PointCloud, dict[str, object]]:
    pcd_rotated = copy.deepcopy(pcd)
    model_center = pcd_rotated.get_center()
    pcd_rotated.translate(-1 * model_center)

    obb = pcd_rotated.get_oriented_bounding_box()
    np_points = np.asarray(obb.get_box_points())

    edge_lengths = [
        np.linalg.norm(np_points[0] - np_points[1]),
        np.linalg.norm(np_points[0] - np_points[2]),
        np.linalg.norm(np_points[0] - np_points[3]),
    ]
    idx_sorted = sort_index(edge_lengths)

    if idx_sorted[0] == 0:
        center_0 = np.mean(np_points[[0, 2, 3, 5]], axis=0)
        center_1 = np.mean(np_points[[1, 4, 6, 7]], axis=0)
    elif idx_sorted[0] == 1:
        center_0 = np.mean(np_points[[0, 1, 3, 6]], axis=0)
        center_1 = np.mean(np_points[[2, 4, 5, 7]], axis=0)
    else:
        center_0 = np.mean(np_points[[0, 1, 2, 7]], axis=0)
        center_1 = np.mean(np_points[[3, 4, 5, 6]], axis=0)

    target_z = np.array([0.0, 0.0, 1.0])
    center_vector = np.array(
        [
            center_0[0] - center_1[0],
            center_0[1] - center_1[1],
            center_0[2] - center_1[2],
        ]
    )
    rotation_matrix = rotation_matrix_from_vectors(center_vector, target_z)
    pcd_rotated.rotate(rotation_matrix, center=(0, 0, 0))

    basal_end, top_stats, bottom_stats = choose_basal_end(
        np.asarray(pcd_rotated.points),
        slicing_ratio=max(config.slicing_ratio, 0.03),
    )

    flipped = False
    if basal_end == "bottom":
        flip_matrix = pcd_rotated.get_rotation_matrix_from_xyz((np.pi, 0, 0))
        pcd_rotated.rotate(flip_matrix, center=(0, 0, 0))
        flipped = True

    if config.adjustment != 0:
        adjust_matrix = pcd_rotated.get_rotation_matrix_from_xyz(
            (0, config.adjustment * np.pi / 2, 0)
        )
        pcd_rotated.rotate(adjust_matrix, center=(0, 0, 0))

    pcd_rotated = anchor_basal_to_z_axis_origin(
        pcd_rotated,
        config.slicing_ratio,
        basal_end="top",
    )

    metadata = {
        "basal_end_before_flip": basal_end,
        "flipped": flipped,
        "top_stats": top_stats,
        "bottom_stats": bottom_stats,
        "config": asdict(config),
    }
    return pcd_rotated, metadata


def preprocess_point_cloud(
    pcd: o3d.geometry.PointCloud, config: PreprocessConfig
) -> tuple[o3d.geometry.PointCloud, o3d.geometry.PointCloud, dict[str, object]]:
    pcd_cleaned = clean_point_cloud(pcd, config)
    pcd_aligned, metadata = align_point_cloud_to_z(pcd_cleaned, config)
    return pcd_cleaned, pcd_aligned, metadata
