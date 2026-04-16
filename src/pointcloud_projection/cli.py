from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from pointcloud_projection.io import convert_input_to_ply, save_point_cloud
from pointcloud_projection.paths import (
    format_project_path,
    resolve_project_path,
    sort_files_by_import_time,
)
from pointcloud_projection.preprocess import (
    PreprocessConfig,
    align_point_cloud_to_z,
    clean_point_cloud,
    preprocess_point_cloud,
)
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
    parser.add_argument(
        "--adjustment",
        type=float,
        default=0.0,
        help="Legacy quarter-turn adjustment around Y in units of pi/2.",
    )
    parser.add_argument("--rotate-x-deg", type=float, default=0.0)
    parser.add_argument("--rotate-y-deg", type=float, default=0.0)
    parser.add_argument("--rotate-z-deg", type=float, default=0.0)
    parser.add_argument(
        "--interactive-rotate",
        action="store_true",
        help="Enable single-file interactive rotation tuning in the terminal.",
    )
    return parser.parse_args()


def collect_input_files(input_path: Path) -> list[Path]:
    if not input_path.exists():
        raise FileNotFoundError(f"Input path not found: {input_path}")
    if input_path.is_file():
        if input_path.suffix.lower() not in SUPPORTED_SUFFIXES:
            raise ValueError(f"Unsupported input file: {input_path}")
        return [input_path]
    files = sort_files_by_import_time(
        [
            path
            for path in input_path.iterdir()
            if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES
        ]
    )
    if not files:
        raise ValueError(f"No STL or PLY files found in: {input_path}")
    return files


def build_preprocess_config(
    args: argparse.Namespace,
    rotate_x_deg: float | None = None,
    rotate_y_deg: float | None = None,
    rotate_z_deg: float | None = None,
) -> PreprocessConfig:
    return PreprocessConfig(
        nb_neighbors=args.nb_neighbors,
        std_ratio=args.std_ratio,
        black_filter=args.black_filter,
        black_threshold=args.black_threshold,
        slicing_ratio=args.slicing_ratio,
        adjustment=args.adjustment,
        rotate_x_deg=args.rotate_x_deg if rotate_x_deg is None else rotate_x_deg,
        rotate_y_deg=args.rotate_y_deg if rotate_y_deg is None else rotate_y_deg,
        rotate_z_deg=args.rotate_z_deg if rotate_z_deg is None else rotate_z_deg,
    )


def build_projection_outputs(
    aligned_pcd,
    args: argparse.Namespace,
) -> tuple[dict[str, object], object, object, object]:
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
        artifacts.projected_points,
        artifacts.point_density_uint8,
    )
    projection_image = make_projection_image(artifacts.projected_points[:, :2], args.image_size)
    bundle = {
        "artifacts": artifacts,
        "projected_pcd": projected_pcd,
        "density_pcd": density_pcd,
        "projection_image": projection_image,
    }
    return bundle, projected_pcd, density_pcd, projection_image


def save_final_outputs(
    *,
    input_path: Path,
    output_dir: Path,
    args: argparse.Namespace,
    config: PreprocessConfig,
    input_metadata: dict[str, object],
    converted_input_path: Path,
    cleaned_pcd,
    aligned_pcd,
    preprocess_metadata: dict[str, object],
    interactive_rotate: bool = False,
) -> dict[str, object]:
    bundle, projected_pcd, density_pcd, projection_image = build_projection_outputs(aligned_pcd, args)
    artifacts = bundle["artifacts"]

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
            "converted_input_ply": format_project_path(converted_input_path),
            "cleaned_ply": format_project_path(cleaned_path),
            "aligned_ply": format_project_path(aligned_path),
            "projected_xy_ply": format_project_path(projected_path),
            "density_xy_ply": format_project_path(density_ply_path),
            "projection_png": format_project_path(projection_png_path),
            "density_png": format_project_path(density_png_path),
        },
        "mode": {
            "interactive_rotate": interactive_rotate,
        },
        "stats": {
            "raw_point_count": int(input_metadata.get("raw_point_count", 0)),
            "cleaned_point_count": len(cleaned_pcd.points),
            "aligned_point_count": len(aligned_pcd.points),
            "projected_point_count": len(projected_pcd.points),
            "density_point_count": len(density_pcd.points),
            "density_nonzero_cells": int((artifacts.density_grid > 0).sum()),
            "density_max_count": int(artifacts.density_grid.max()) if artifacts.density_grid.size else 0,
            "projection_image_shape": list(projection_image.shape),
            "density_image_shape": list(artifacts.density_grid_uint8.shape),
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


def save_interactive_preview(
    input_path: Path,
    preview_dir: Path,
    aligned_pcd,
    args: argparse.Namespace,
) -> dict[str, str]:
    bundle, projected_pcd, density_pcd, projection_image = build_projection_outputs(aligned_pcd, args)
    artifacts = bundle["artifacts"]
    stem = input_path.stem
    aligned_preview = Path(save_point_cloud(aligned_pcd, preview_dir / f"{stem}_aligned_preview.ply"))
    projected_preview = Path(
        save_point_cloud(projected_pcd, preview_dir / f"{stem}_projected_xy_preview.ply")
    )
    density_preview = Path(
        save_point_cloud(density_pcd, preview_dir / f"{stem}_density_xy_preview.ply")
    )
    projection_png = Path(save_png(projection_image, preview_dir / f"{stem}_projection_preview.png"))
    density_png = Path(
        save_png(artifacts.density_grid_uint8, preview_dir / f"{stem}_density_preview.png")
    )
    return {
        "aligned_ply": format_project_path(aligned_preview),
        "projected_xy_ply": format_project_path(projected_preview),
        "density_xy_ply": format_project_path(density_preview),
        "projection_png": format_project_path(projection_png),
        "density_png": format_project_path(density_png),
    }


def parse_interactive_rotation_command(command: str) -> tuple[str, tuple[float, ...] | None]:
    tokens = command.strip().split()
    if not tokens:
        return "show", None

    action = tokens[0].lower()
    if action in {"save", "quit", "reset", "show", "help"}:
        return action, None
    if action in {"x", "y", "z"} and len(tokens) == 2:
        return action, (float(tokens[1]),)
    if action == "set" and len(tokens) == 4:
        return action, (float(tokens[1]), float(tokens[2]), float(tokens[3]))
    raise ValueError(
        "Unsupported command. Use: x <deg>, y <deg>, z <deg>, set <x> <y> <z>, show, reset, save, quit."
    )


def process_one_file(input_path: Path, output_dir: Path, args: argparse.Namespace) -> dict[str, object]:
    stem = input_path.stem
    raw_pcd, input_metadata, converted_input_path = convert_input_to_ply(
        input_path,
        output_dir / f"{stem}_converted_input.ply",
        sample_points=args.sample_points,
    )
    input_metadata["input_path"] = format_project_path(input_path)
    input_metadata["raw_point_count"] = len(raw_pcd.points)

    config = build_preprocess_config(args)
    cleaned_pcd, aligned_pcd, preprocess_metadata = preprocess_point_cloud(raw_pcd, config)

    return save_final_outputs(
        input_path=input_path,
        output_dir=output_dir,
        args=args,
        config=config,
        input_metadata=input_metadata,
        converted_input_path=converted_input_path,
        cleaned_pcd=cleaned_pcd,
        aligned_pcd=aligned_pcd,
        preprocess_metadata=preprocess_metadata,
        interactive_rotate=False,
    )


def run_interactive_rotate_session(
    input_path: Path,
    output_dir: Path,
    args: argparse.Namespace,
) -> dict[str, object]:
    stem = input_path.stem
    raw_pcd, input_metadata, converted_input_path = convert_input_to_ply(
        input_path,
        output_dir / f"{stem}_converted_input.ply",
        sample_points=args.sample_points,
    )
    input_metadata["input_path"] = format_project_path(input_path)
    input_metadata["raw_point_count"] = len(raw_pcd.points)

    base_config = build_preprocess_config(args)
    cleaned_pcd = clean_point_cloud(raw_pcd, base_config)
    preview_dir = output_dir / "_interactive_preview"
    preview_dir.mkdir(parents=True, exist_ok=True)
    rotations = {
        "rotate_x_deg": float(args.rotate_x_deg),
        "rotate_y_deg": float(args.rotate_y_deg),
        "rotate_z_deg": float(args.rotate_z_deg),
    }

    while True:
        config = build_preprocess_config(
            args,
            rotate_x_deg=rotations["rotate_x_deg"],
            rotate_y_deg=rotations["rotate_y_deg"],
            rotate_z_deg=rotations["rotate_z_deg"],
        )
        aligned_pcd, preprocess_metadata = align_point_cloud_to_z(cleaned_pcd, config)
        preview_outputs = save_interactive_preview(input_path, preview_dir, aligned_pcd, args)

        print(
            "Interactive rotation preview "
            f"(x={rotations['rotate_x_deg']:.2f}, y={rotations['rotate_y_deg']:.2f}, z={rotations['rotate_z_deg']:.2f})"
        )
        print(f"Preview projection PNG: {preview_outputs['projection_png']}")
        print(f"Preview density PNG: {preview_outputs['density_png']}")
        print(
            "Commands: x <deg>, y <deg>, z <deg>, set <x> <y> <z>, show, reset, save, quit"
        )
        command = input("interactive-rotate> ").strip()
        try:
            action, values = parse_interactive_rotation_command(command)
        except ValueError as exc:
            print(exc)
            continue

        if action == "show":
            continue
        if action == "help":
            print("Use x/y/z to increment one axis, set to overwrite all three, save to accept.")
            continue
        if action == "reset":
            rotations = {"rotate_x_deg": 0.0, "rotate_y_deg": 0.0, "rotate_z_deg": 0.0}
            continue
        if action == "save":
            return save_final_outputs(
                input_path=input_path,
                output_dir=output_dir,
                args=args,
                config=config,
                input_metadata=input_metadata,
                converted_input_path=converted_input_path,
                cleaned_pcd=cleaned_pcd,
                aligned_pcd=aligned_pcd,
                preprocess_metadata=preprocess_metadata,
                interactive_rotate=True,
            )
        if action == "quit":
            raise SystemExit("Interactive rotation cancelled without saving.")
        if action in {"x", "y", "z"} and values is not None:
            key = f"rotate_{action}_deg"
            rotations[key] += values[0]
            continue
        if action == "set" and values is not None:
            rotations = {
                "rotate_x_deg": values[0],
                "rotate_y_deg": values[1],
                "rotate_z_deg": values[2],
            }
            continue


def main() -> int:
    args = parse_args()
    input_path = resolve_project_path(Path(args.input))
    output_dir = resolve_project_path(Path(args.output_dir))
    output_dir.mkdir(parents=True, exist_ok=True)

    input_files = collect_input_files(input_path)
    if args.interactive_rotate and len(input_files) != 1:
        raise ValueError("Interactive rotation mode only supports a single input file.")
    run_summary: dict[str, object] = {
        "input_path": format_project_path(input_path),
        "output_dir": format_project_path(output_dir),
        "file_count": len(input_files),
        "files": [],
    }
    if args.interactive_rotate:
        file_path = input_files[0]
        print(f"Processing interactively: {file_path}")
        file_summary = run_interactive_rotate_session(file_path, output_dir, args)
        run_summary["files"].append(file_summary)
    else:
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
