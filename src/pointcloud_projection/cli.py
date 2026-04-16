from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from pointcloud_projection.io import load_point_cloud, save_point_cloud
from pointcloud_projection.paths import format_project_path, resolve_project_path
from pointcloud_projection.preprocess import PreprocessConfig, preprocess_point_cloud
from pointcloud_projection.projection import (
    build_density_artifacts,
    build_density_point_cloud,
    build_projected_point_cloud,
    make_projection_image,
    save_png,
)

DEFAULT_OUTPUT_DIR = Path("output")
SUPPORTED_SUFFIXES = {".ply", ".stl"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preprocess a 3D model and project it to the XY plane."
    )
    parser.add_argument(
        "--input",
        "-i",
        required=True,
        help="Input STL/PLY file or a directory containing STL/PLY files.",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory for cleaned, aligned, projected, and density outputs, relative to the project root by default.",
    )
    parser.add_argument(
        "--sample-points",
        type=int,
        default=200000,
        help="Number of points sampled from STL meshes.",
    )
    parser.add_argument(
        "--grid-size",
        type=int,
        default=512,
        help="Grid size used for density estimation.",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=1024,
        help="Output image size for projection PNG.",
    )
    parser.add_argument(
        "--no-log-density",
        action="store_true",
        help="Disable log scaling when mapping density to grayscale.",
    )
    parser.add_argument("--nb-neighbors", type=int, default=20)
    parser.add_argument("--std-ratio", type=float, default=5.0)
    parser.add_argument("--black-filter", type=int, default=0)
    parser.add_argument("--black-threshold", type=float, default=0.2)
    parser.add_argument("--slicing-ratio", type=float, default=0.10)
    parser.add_argument("--adjustment", type=float, default=0.0)
    return parser.parse_args()


def collect_input_files(input_path: Path) -> list[Path]:
    if not input_path.exists():
        raise FileNotFoundError(f"Input path not found: {input_path}")
    if input_path.is_file():
        if input_path.suffix.lower() not in SUPPORTED_SUFFIXES:
            raise ValueError(f"Unsupported input file: {input_path}")
        return [input_path]
    files = sorted(
        path for path in input_path.iterdir() if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES
    )
    if not files:
        raise ValueError(f"No STL or PLY files found in: {input_path}")
    return files


def process_one_file(input_path: Path, output_dir: Path, args: argparse.Namespace) -> dict[str, object]:
    config = PreprocessConfig(
        nb_neighbors=args.nb_neighbors,
        std_ratio=args.std_ratio,
        black_filter=args.black_filter,
        black_threshold=args.black_threshold,
        slicing_ratio=args.slicing_ratio,
        adjustment=args.adjustment,
    )

    raw_pcd, input_metadata = load_point_cloud(input_path, sample_points=args.sample_points)
    input_metadata["input_path"] = format_project_path(input_path)
    cleaned_pcd, aligned_pcd, preprocess_metadata = preprocess_point_cloud(raw_pcd, config)

    artifacts = build_density_artifacts(
        aligned_pcd,
        grid_size=args.grid_size,
        log_scale=not args.no_log_density,
    )
    projected_pcd = build_projected_point_cloud(
        artifacts.projected_points,
        artifacts.projected_colors,
    )
    density_pcd = build_density_point_cloud(
        artifacts.density_grid_uint8,
        artifacts.xy_min,
        artifacts.xy_max,
    )
    projection_image = make_projection_image(artifacts.projected_points[:, :2], args.image_size)

    stem = input_path.stem
    cleaned_path = Path(save_point_cloud(cleaned_pcd, output_dir / f"{stem}_cleaned.ply"))
    aligned_path = Path(save_point_cloud(aligned_pcd, output_dir / f"{stem}_aligned.ply"))
    projected_path = Path(save_point_cloud(projected_pcd, output_dir / f"{stem}_projected_xy.ply"))
    density_ply_path = Path(save_point_cloud(density_pcd, output_dir / f"{stem}_density_xy.ply"))
    projection_png_path = Path(save_png(projection_image, output_dir / f"{stem}_projection.png"))
    density_png_path = Path(save_png(artifacts.density_grid_uint8, output_dir / f"{stem}_density.png"))

    summary = {
        "input": input_metadata,
        "preprocess": preprocess_metadata,
        "parameters": {
            "grid_size": args.grid_size,
            "image_size": args.image_size,
            "sample_points": args.sample_points,
            "log_density": not args.no_log_density,
            "preprocess_config": asdict(config),
        },
        "outputs": {
            "cleaned_ply": format_project_path(cleaned_path),
            "aligned_ply": format_project_path(aligned_path),
            "projected_xy_ply": format_project_path(projected_path),
            "density_xy_ply": format_project_path(density_ply_path),
            "projection_png": format_project_path(projection_png_path),
            "density_png": format_project_path(density_png_path),
        },
        "stats": {
            "raw_point_count": len(raw_pcd.points),
            "cleaned_point_count": len(cleaned_pcd.points),
            "aligned_point_count": len(aligned_pcd.points),
            "projected_point_count": len(projected_pcd.points),
            "density_nonzero_cells": int((artifacts.density_grid > 0).sum()),
            "density_max_count": int(artifacts.density_grid.max()) if artifacts.density_grid.size else 0,
        },
    }

    summary_path = output_dir / f"{stem}_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    summary["outputs"]["summary_json"] = format_project_path(summary_path)
    print(f"Cleaned PLY: {format_project_path(cleaned_path)}")
    print(f"Aligned PLY: {format_project_path(aligned_path)}")
    print(f"Projected XY PLY: {format_project_path(projected_path)}")
    print(f"Projection PNG: {format_project_path(projection_png_path)}")
    print(f"Density PNG: {format_project_path(density_png_path)}")
    print(f"Density XY PLY: {format_project_path(density_ply_path)}")
    print(f"Summary JSON: {format_project_path(summary_path)}")
    return summary


def main() -> int:
    args = parse_args()
    input_path = resolve_project_path(Path(args.input))
    output_dir = resolve_project_path(Path(args.output_dir))
    output_dir.mkdir(parents=True, exist_ok=True)

    input_files = collect_input_files(input_path)
    run_summary: dict[str, object] = {
        "input_path": format_project_path(input_path),
        "output_dir": format_project_path(output_dir),
        "file_count": len(input_files),
        "files": [],
    }
    for file_path in input_files:
        print(f"Processing: {file_path}")
        file_summary = process_one_file(file_path, output_dir, args)
        run_summary["files"].append(file_summary)

    batch_summary_path = output_dir / "batch_summary.json"
    batch_summary_path.write_text(json.dumps(run_summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Batch summary JSON: {format_project_path(batch_summary_path)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
