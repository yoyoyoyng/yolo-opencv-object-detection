#!/usr/bin/env python3
"""Simple one-class YOLO bounding-box labeler for the H-13 dataset.

Controls in the OpenCV ROI window:
- Drag a rectangle around the entire smart car.
- ENTER or SPACE: confirm the box.
- C: cancel the current selection. The terminal then asks whether the image
  contains no robot, should be retried, or labeling should stop.

The script resumes automatically by skipping images that already have a
matching .txt label file. Use --redo to label every image again.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import cv2

CLASS_ID = 0
CLASS_NAME = "h13_smartcar"
SPLITS = ("train", "val", "test")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
MAX_DISPLAY_WIDTH = 1200
MAX_DISPLAY_HEIGHT = 800


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Label H-13 smart-car images in YOLO format.")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Dataset root containing images/ and labels/ (default: script directory).",
    )
    parser.add_argument(
        "--redo",
        action="store_true",
        help="Relabel images even when a label file already exists.",
    )
    return parser.parse_args()


def collect_images(dataset_root: Path) -> list[tuple[str, Path, Path]]:
    items: list[tuple[str, Path, Path]] = []
    for split in SPLITS:
        image_dir = dataset_root / "images" / split
        label_dir = dataset_root / "labels" / split
        label_dir.mkdir(parents=True, exist_ok=True)
        if not image_dir.is_dir():
            continue
        for image_path in sorted(image_dir.iterdir()):
            if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            label_path = label_dir / f"{image_path.stem}.txt"
            items.append((split, image_path, label_path))
    return items


def resize_for_display(image):
    height, width = image.shape[:2]
    scale = min(MAX_DISPLAY_WIDTH / width, MAX_DISPLAY_HEIGHT / height, 1.0)
    if scale == 1.0:
        return image.copy(), scale
    resized = cv2.resize(
        image,
        (max(1, round(width * scale)), max(1, round(height * scale))),
        interpolation=cv2.INTER_AREA,
    )
    return resized, scale


def save_yolo_box(label_path: Path, box: tuple[float, float, float, float], image_shape) -> None:
    x, y, width, height = box
    image_height, image_width = image_shape[:2]

    x = max(0.0, min(x, image_width - 1.0))
    y = max(0.0, min(y, image_height - 1.0))
    width = max(1.0, min(width, image_width - x))
    height = max(1.0, min(height, image_height - y))

    center_x = (x + width / 2.0) / image_width
    center_y = (y + height / 2.0) / image_height
    norm_width = width / image_width
    norm_height = height / image_height

    label_path.write_text(
        f"{CLASS_ID} {center_x:.6f} {center_y:.6f} {norm_width:.6f} {norm_height:.6f}\n",
        encoding="utf-8",
    )


def save_empty_label(label_path: Path) -> None:
    label_path.write_text("", encoding="utf-8")


def main() -> int:
    args = parse_args()
    dataset_root = args.dataset.expanduser().resolve()
    items = collect_images(dataset_root)

    if not items:
        print(f"No images found under: {dataset_root / 'images'}")
        return 1

    pending = [item for item in items if args.redo or not item[2].exists()]
    done_count = len(items) - len(pending)

    print(f"Dataset: {dataset_root}")
    print(f"Class: {CLASS_NAME} (id={CLASS_ID})")
    print(f"Total images: {len(items)}")
    print(f"Already labeled: {done_count}")
    print(f"Remaining: {len(pending)}")
    print()
    print("For each image:")
    print("  1) Drag one box around the entire robot.")
    print("  2) Press ENTER or SPACE to save.")
    print("  3) Press C to cancel, then choose retry/no-robot/quit in the terminal.")
    print()

    if not pending:
        print("All images already have label files. Use --redo to relabel them.")
        return 0

    window_name = "H13 YOLO Labeling"
    saved_this_run = 0

    try:
        for index, (split, image_path, label_path) in enumerate(pending, start=1):
            image = cv2.imread(str(image_path))
            if image is None:
                print(f"Could not read image, skipping: {image_path}")
                continue

            while True:
                display, scale = resize_for_display(image)
                banner = display.copy()
                text = f"{index}/{len(pending)}  {split}/{image_path.name}"
                cv2.rectangle(banner, (0, 0), (banner.shape[1], 42), (0, 0, 0), -1)
                cv2.putText(
                    banner,
                    text,
                    (12, 29),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

                print(f"[{index}/{len(pending)}] {split}/{image_path.name}")
                selected = cv2.selectROI(
                    window_name,
                    banner,
                    showCrosshair=True,
                    fromCenter=False,
                )

                x, y, width, height = selected
                if width > 0 and height > 0:
                    # Prevent the informational banner from becoming part of a box.
                    y = max(float(y), 42.0)
                    height = max(1.0, float(selected[1] + selected[3]) - y)
                    original_box = (
                        float(x) / scale,
                        float(y) / scale,
                        float(width) / scale,
                        float(height) / scale,
                    )
                    save_yolo_box(label_path, original_box, image.shape)
                    saved_this_run += 1
                    break

                choice = input("Selection canceled: [r]etry, [n]o robot, [q]uit: ").strip().lower()
                if choice in {"n", "no"}:
                    save_empty_label(label_path)
                    saved_this_run += 1
                    break
                if choice in {"q", "quit"}:
                    print("Stopped. Run the script again to resume from the next unlabeled image.")
                    return 0
                print("Retrying this image.")

    finally:
        cv2.destroyAllWindows()

    print()
    print(f"Finished. Label files saved this run: {saved_this_run}")
    print("Next: run `python3 validate_labels.py`.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
