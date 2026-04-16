from __future__ import annotations

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def resolve_project_path(path: Path) -> Path:
    if path.is_absolute():
        return path
    return (PROJECT_ROOT / path).resolve()


def format_project_path(path: Path) -> str:
    return os.path.relpath(path.resolve(), PROJECT_ROOT.resolve())


def sort_files_by_import_time(paths: list[Path]) -> list[Path]:
    return sorted(paths, key=lambda path: (path.stat().st_ctime_ns, path.name.lower()))
