#!/usr/bin/env python3
"""Validate one-class YOLO labels for the H-13 dataset."""

from __future__ import annotations

from pathlib import Path
import sys

SPLITS = ("train", "val", "test")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def main() -> int:
    root = Path(__file__).resolve().parent
    missing: list[Path] = []
    invalid: list[str] = []
    image_count = 0
    object_count = 0
    empty_count = 0

    for split in SPLITS:
        image_dir = root / "images" / split
        label_dir = root / "labels" / split
        if not image_dir.is_dir():
            continue
        for image_path in sorted(image_dir.iterdir()):
            if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            image_count += 1
            label_path = label_dir / f"{image_path.stem}.txt"
            if not label_path.exists():
                missing.append(label_path)
                continue

            lines = [line.strip() for line in label_path.read_text(encoding="utf-8").splitlines() if line.strip()]
            if not lines:
                empty_count += 1
                continue

            for line_number, line in enumerate(lines, start=1):
                parts = line.split()
                if len(parts) != 5:
                    invalid.append(f"{label_path}:{line_number}: expected 5 values, got {len(parts)}")
                    continue
                try:
                    class_id = int(parts[0])
                    values = [float(value) for value in parts[1:]]
                except ValueError:
                    invalid.append(f"{label_path}:{line_number}: non-numeric value")
                    continue
                if class_id != 0:
                    invalid.append(f"{label_path}:{line_number}: class id must be 0")
                cx, cy, width, height = values
                if not (0.0 <= cx <= 1.0 and 0.0 <= cy <= 1.0):
                    invalid.append(f"{label_path}:{line_number}: center must be within [0, 1]")
                if not (0.0 < width <= 1.0 and 0.0 < height <= 1.0):
                    invalid.append(f"{label_path}:{line_number}: width/height must be within (0, 1]")
                object_count += 1

    print(f"Images: {image_count}")
    print(f"Object boxes: {object_count}")
    print(f"Empty/background labels: {empty_count}")
    print(f"Missing label files: {len(missing)}")
    print(f"Invalid label lines: {len(invalid)}")

    if missing:
        print("\nMissing labels (first 20):")
        for path in missing[:20]:
            print(f"- {path.relative_to(root)}")
    if invalid:
        print("\nInvalid labels (first 20):")
        for message in invalid[:20]:
            print(f"- {message}")

    if missing or invalid:
        print("\nValidation failed. Fix the items above before training.")
        return 1

    print("\nValidation passed. The dataset is ready for YOLO training.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
