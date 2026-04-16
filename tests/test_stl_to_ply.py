from __future__ import annotations

from pathlib import Path

import pytest

from pointcloud_projection.stl_to_ply import collect_stl_files


def test_collect_stl_files_accepts_single_stl_file(tmp_path: Path) -> None:
    file_path = tmp_path / "mesh.stl"
    file_path.write_text("solid mesh", encoding="utf-8")

    files = collect_stl_files(file_path)

    assert files == [file_path]


def test_collect_stl_files_filters_directory_entries(tmp_path: Path) -> None:
    stl_a = tmp_path / "a.stl"
    stl_b = tmp_path / "b.STL"
    txt = tmp_path / "ignore.txt"
    stl_a.write_text("solid a", encoding="utf-8")
    stl_b.write_text("solid b", encoding="utf-8")
    txt.write_text("ignore", encoding="utf-8")

    files = collect_stl_files(tmp_path)

    assert files == [stl_a, stl_b]


def test_collect_stl_files_rejects_non_stl_file(tmp_path: Path) -> None:
    file_path = tmp_path / "mesh.ply"
    file_path.write_text("ply", encoding="utf-8")

    with pytest.raises(ValueError):
        collect_stl_files(file_path)
