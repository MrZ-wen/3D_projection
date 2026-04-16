from __future__ import annotations

from pathlib import Path

import numpy as np
import open3d as o3d


def load_point_cloud(input_path: str | Path, sample_points: int = 200000) -> tuple[o3d.geometry.PointCloud, dict[str, object]]:
    path = Path(input_path)
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    suffix = path.suffix.lower()
    metadata: dict[str, object] = {
        "input_path": str(path),
        "input_format": suffix.lstrip("."),
    }

    if suffix == ".ply":
        pcd = o3d.io.read_point_cloud(str(path))
        if len(np.asarray(pcd.points)) == 0:
            raise ValueError(f"Failed to read points from PLY: {path}")
        metadata["source_type"] = "point_cloud"
        return pcd, metadata

    if suffix == ".stl":
        mesh = o3d.io.read_triangle_mesh(str(path))
        if mesh.is_empty():
            raise ValueError(f"Failed to read mesh from STL: {path}")
        mesh.compute_vertex_normals()
        pcd = mesh.sample_points_uniformly(number_of_points=max(sample_points, 1000))
        metadata["source_type"] = "triangle_mesh"
        metadata["sample_points"] = max(sample_points, 1000)
        return pcd, metadata

    raise ValueError(f"Unsupported input format: {suffix}. Expected .ply or .stl")


def save_point_cloud(pcd: o3d.geometry.PointCloud, output_path: str | Path, write_ascii: bool = True) -> str:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ok = o3d.io.write_point_cloud(str(path), pcd, write_ascii=write_ascii)
    if not ok:
        raise RuntimeError(f"Failed to write point cloud: {path}")
    return str(path)


def convert_input_to_ply(
    input_path: str | Path,
    output_path: str | Path,
    sample_points: int = 200000,
    write_ascii: bool = True,
) -> tuple[o3d.geometry.PointCloud, dict[str, object], Path]:
    pcd, metadata = load_point_cloud(input_path, sample_points=sample_points)
    saved_path = Path(save_point_cloud(pcd, output_path, write_ascii=write_ascii))
    return pcd, metadata, saved_path
