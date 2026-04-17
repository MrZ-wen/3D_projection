from __future__ import annotations

from pathlib import Path

import numpy as np
import open3d as o3d

from pointcloud_projection.io import convert_input_to_ply, load_point_cloud, save_point_cloud


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


def test_load_point_cloud_uses_ascii_temp_path_for_non_ascii_input(
    tmp_path: Path, monkeypatch
) -> None:
    source_path = tmp_path / "中文样本.ply"
    source_path.write_text("placeholder", encoding="utf-8")

    captured = {}

    def fake_read_point_cloud(path_str: str):
        captured["path"] = path_str
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(np.array([[0.0, 0.0, 0.0]], dtype=float))
        return pcd

    monkeypatch.setattr(o3d.io, "read_point_cloud", fake_read_point_cloud)

    pcd, metadata = load_point_cloud(source_path)

    assert len(pcd.points) == 1
    assert metadata["input_format"] == "ply"
    assert Path(captured["path"]).name.isascii()


def test_save_point_cloud_uses_ascii_temp_path_for_non_ascii_output(
    tmp_path: Path, monkeypatch
) -> None:
    output_path = tmp_path / "输出目录" / "结果文件.ply"
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.array([[0.0, 0.0, 0.0]], dtype=float))

    captured = {}

    def fake_write_point_cloud(path_str: str, point_cloud, write_ascii: bool = True):
        captured["path"] = path_str
        Path(path_str).write_text("ply", encoding="utf-8")
        return True

    monkeypatch.setattr(o3d.io, "write_point_cloud", fake_write_point_cloud)

    saved = save_point_cloud(pcd, output_path)

    assert saved == str(output_path)
    assert output_path.exists()
    assert Path(captured["path"]).name.isascii()


def test_load_point_cloud_uses_poisson_disk_sampling_for_stl(
    tmp_path: Path, monkeypatch
) -> None:
    source_path = tmp_path / "mesh.stl"
    source_path.write_text("solid mesh", encoding="utf-8")

    class FakeMesh:
        def is_empty(self) -> bool:
            return False

        def compute_vertex_normals(self) -> None:
            return None

        def sample_points_poisson_disk(self, number_of_points: int):
            captured["number_of_points"] = number_of_points
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(np.array([[0.0, 0.0, 0.0]], dtype=float))
            return pcd

    captured = {}

    def fake_read_triangle_mesh(path_str: str):
        captured["path"] = path_str
        return FakeMesh()

    monkeypatch.setattr(o3d.io, "read_triangle_mesh", fake_read_triangle_mesh)

    pcd, metadata = load_point_cloud(source_path, sample_points=12345)

    assert len(pcd.points) == 1
    assert metadata["source_type"] == "triangle_mesh"
    assert metadata["sample_points"] == 12345
    assert captured["number_of_points"] == 12345
