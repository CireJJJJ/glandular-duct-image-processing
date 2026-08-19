from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
os.environ.setdefault("OPENCV_IO_MAX_IMAGE_PIXELS", str(2**40))


def image_paths(folder: Path) -> Iterable[Path]:
    for path in sorted(folder.iterdir()):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            yield path


def read_gray(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Could not read image: {path}")
    return image


def enhance_contrast(
    image: np.ndarray,
    clip_limit: float = 3.6,
    grid_size: tuple[int, int] = (9, 9),
) -> np.ndarray:
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size)
    return clahe.apply(image)


def tile_image(
    image_path: Path,
    output_root: Path,
    tile_size: int,
    step: int,
    minimum_mean: float,
) -> list[dict[str, object]]:
    gray = read_gray(image_path)
    enhanced = enhance_contrast(gray)
    height, width = enhanced.shape

    image_output = output_root / image_path.stem
    image_output.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(image_output / f"{image_path.stem}_enhanced.png"), enhanced)

    horizontal_count = math.ceil(width / step)
    vertical_count = math.ceil(height / step)
    rows: list[dict[str, object]] = []

    for row in range(vertical_count):
        for column in range(horizontal_count):
            left = column * step
            upper = row * step
            right = left + tile_size
            lower = upper + tile_size

            if right > width or lower > height:
                continue

            tile = enhanced[upper:lower, left:right]
            mean_value = float(tile.mean())
            if mean_value < minimum_mean:
                continue

            tile_name = f"{image_path.stem}_{left}_{upper}_{right}_{lower}.png"
            tile_path = image_output / tile_name
            cv2.imwrite(str(tile_path), tile)
            rows.append(
                {
                    "source_image": image_path.name,
                    "tile_path": tile_path.as_posix(),
                    "left": left,
                    "upper": upper,
                    "right": right,
                    "lower": lower,
                    "mean_intensity": round(mean_value, 4),
                }
            )

    return rows


def duct_contours(
    image: np.ndarray,
    minimum_area: float,
    opening_iterations: int,
) -> tuple[np.ndarray, list[np.ndarray]]:
    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    _, binary = cv2.threshold(
        blurred,
        0,
        255,
        cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU,
    )
    kernel = np.ones((3, 3), dtype=np.uint8)
    opened = cv2.morphologyEx(
        binary,
        cv2.MORPH_OPEN,
        kernel,
        iterations=opening_iterations,
    )
    contours, _ = cv2.findContours(
        opened,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    selected = [contour for contour in contours if cv2.contourArea(contour) >= minimum_area]
    return opened, selected


def contour_to_polygon(contour: np.ndarray) -> list[list[int]]:
    epsilon = 0.01 * cv2.arcLength(contour, True)
    simplified = cv2.approxPolyDP(contour, epsilon, True)
    return [[int(x), int(y)] for x, y in simplified.reshape(-1, 2)]


def annotate_image(
    image_path: Path,
    output_root: Path,
    minimum_area: float,
    opening_iterations: int,
) -> dict[str, object]:
    gray = read_gray(image_path)
    opened, contours = duct_contours(gray, minimum_area, opening_iterations)

    shapes = []
    for contour in contours:
        polygon = contour_to_polygon(contour)
        if len(polygon) < 3:
            continue
        shapes.append(
            {
                "label": "duct",
                "points": polygon,
                "group_id": None,
                "description": "",
                "shape_type": "polygon",
                "flags": {},
            }
        )

    annotation = {
        "version": "5.2.1",
        "flags": {},
        "shapes": shapes,
        "imagePath": image_path.name,
        "imageData": None,
        "imageHeight": int(gray.shape[0]),
        "imageWidth": int(gray.shape[1]),
    }

    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / f"{image_path.stem}.json").write_text(
        json.dumps(annotation, indent=2) + "\n",
        encoding="utf-8",
    )

    review = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(review, contours, -1, (0, 255, 0), 2)
    cv2.imwrite(str(output_root / f"{image_path.stem}_review.png"), review)
    cv2.imwrite(str(output_root / f"{image_path.stem}_binary.png"), opened)
    return annotation


def write_manifest(path: Path, rows: list[dict[str, object]]) -> None:
    fields = [
        "source_image",
        "tile_path",
        "left",
        "upper",
        "right",
        "lower",
        "mean_intensity",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare high-resolution glandular-duct images for segmentation review."
    )
    commands = parser.add_subparsers(dest="command", required=True)

    tile = commands.add_parser("tile", help="Enhance and tile high-resolution images.")
    tile.add_argument("input", type=Path)
    tile.add_argument("output", type=Path)
    tile.add_argument("--tile-size", type=int, default=1024)
    tile.add_argument("--step", type=int, default=620)
    tile.add_argument("--minimum-mean", type=float, default=18.0)

    contours = commands.add_parser("contours", help="Propose duct polygons for review.")
    contours.add_argument("input", type=Path)
    contours.add_argument("output", type=Path)
    contours.add_argument("--minimum-area", type=float, default=1024.0)
    contours.add_argument("--opening-iterations", type=int, default=2)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.command == "tile":
        rows: list[dict[str, object]] = []
        for path in image_paths(args.input):
            rows.extend(
                tile_image(
                    path,
                    args.output,
                    args.tile_size,
                    args.step,
                    args.minimum_mean,
                )
            )
        write_manifest(args.output / "crop_manifest.csv", rows)
        print(f"Wrote {len(rows)} tiles to {args.output}")
        return 0

    if args.command == "contours":
        image_count = 0
        polygon_count = 0
        for path in image_paths(args.input):
            annotation = annotate_image(
                path,
                args.output,
                args.minimum_area,
                args.opening_iterations,
            )
            image_count += 1
            polygon_count += len(annotation["shapes"])
        print(f"Reviewed {image_count} images and proposed {polygon_count} polygons")
        return 0

    raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
