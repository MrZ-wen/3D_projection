from __future__ import annotations

import argparse
import json
from pathlib import Path

from pointcloud_projection.io import load_point_cloud, save_point_cloud
from pointcloud_projection.paths import format_project_path, resolve_project_path

DEFAULT_OUTPUT_DIR = Path("stl_to_ply_output")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert STL files to sampled PLY point clouds."
    )
    parser.add_argument(
        "--input",
        "-i",
        required=True,
        help="Input STL file or a directory containing STL files.",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory for converted PLY files, relative to the project root by default.",
    )
    parser.add_argument(
        "--sample-points",
        type=int,
        default=200000,
        help="Number of points sampled from each STL mesh.",
    )
    return parser.parse_args()


def collect_stl_files(input_path: Path) -> list[Path]:
    if not input_path.exists():
        raise FileNotFoundError(f"Input path not found: {input_path}")
    if input_path.is_file():
        if input_path.suffix.lower() != ".stl":
            raise ValueError(f"Unsupported input file: {input_path}")
        return [input_path]
    files = sorted(
        path for path in input_path.iterdir() if path.is_file() and path.suffix.lower() == ".stl"
    )
    if not files:
        raise ValueError(f"No STL files found in: {input_path}")
    return files


def convert_one_file(input_path: Path, output_dir: Path, sample_points: int) -> dict[str, object]:
    pcd, metadata = load_point_cloud(input_path, sample_points=sample_points)
    metadata["input_path"] = format_project_path(input_path)

    output_path = output_dir / f"{input_path.stem}.ply"
    saved_path = Path(save_point_cloud(pcd, output_path))

    summary = {
        "input": metadata,
        "parameters": {
            "sample_points": sample_points,
        },
        "outputs": {
            "converted_ply": format_project_path(saved_path),
        },
        "stats": {
            "point_count": len(pcd.points),
        },
    }

    summary_path = output_dir / f"{input_path.stem}_stl_to_ply_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    summary["outputs"]["summary_json"] = format_project_path(summary_path)

    print(f"Converted PLY: {format_project_path(saved_path)}")
    print(f"Summary JSON: {format_project_path(summary_path)}")
    return summary


def main() -> int:
    args = parse_args()
    input_path = resolve_project_path(Path(args.input))
    output_dir = resolve_project_path(Path(args.output_dir))
    output_dir.mkdir(parents=True, exist_ok=True)

    input_files = collect_stl_files(input_path)
    run_summary: dict[str, object] = {
        "input_path": format_project_path(input_path),
        "output_dir": format_project_path(output_dir),
        "file_count": len(input_files),
        "files": [],
    }
    for file_path in input_files:
        print(f"Converting: {file_path}")
        file_summary = convert_one_file(file_path, output_dir, args.sample_points)
        run_summary["files"].append(file_summary)

    batch_summary_path = output_dir / "batch_summary.json"
    batch_summary_path.write_text(json.dumps(run_summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Batch summary JSON: {format_project_path(batch_summary_path)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
