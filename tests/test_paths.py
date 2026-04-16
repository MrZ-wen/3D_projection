from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from pointcloud_projection.paths import sort_files_by_import_time


def test_sort_files_by_import_time_uses_ctime_then_name(
    tmp_path: Path, monkeypatch
) -> None:
    first = tmp_path / "b.ply"
    second = tmp_path / "a.ply"
    third = tmp_path / "c.ply"
    for path in (first, second, third):
        path.write_text("ply", encoding="utf-8")

    ctime_map = {
        first: 10,
        second: 10,
        third: 20,
    }

    original_stat = Path.stat

    def fake_stat(self: Path, *args, **kwargs):
        if self in ctime_map:
            return SimpleNamespace(st_ctime_ns=ctime_map[self])
        return original_stat(self, *args, **kwargs)

    monkeypatch.setattr(Path, "stat", fake_stat)

    files = sort_files_by_import_time([third, first, second])

    assert files == [second, first, third]
