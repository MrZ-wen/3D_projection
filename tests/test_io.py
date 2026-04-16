from __future__ import annotations

from pathlib import Path

import numpy as np
import open3d as o3d

from pointcloud_projection.io import convert_input_to_ply


def test_convert_input_to_ply_writes_output_file(tmp_path: Path) -> None:
    source_path = tmp_path / "source.ply"
    output_path = tmp_path / "converted" / "source_converted.ply"

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(
        np.array([[0.0, 0.0, 0.0], [1.0, 2.0, 3.0]], dtype=float)
    )
    o3d.io.write_point_cloud(str(source_path), pcd, write_ascii=True)

    converted_pcd, metadata, saved_path = convert_input_to_ply(source_path, output_path)

    assert saved_path == output_path
    assert saved_path.exists()
    assert metadata["input_format"] == "ply"
    assert len(converted_pcd.points) == 2
