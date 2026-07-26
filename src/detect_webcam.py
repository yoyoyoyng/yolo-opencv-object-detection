#!/usr/bin/env python3
"""Run real-time YOLO object detection with an OpenCV webcam pipeline.

The default settings reproduce the final project demo:

- model: ``weights/best_chanyong.pt``
- external webcam: camera index ``0``
- capture/display: ``640x480``
- YOLO inference size: ``480``
- one final detection per frame: ``max_det=1``
- compact terminal log, for example::

      0: 480x640 1 chanyong, 5.8ms

Use ``--max-det 20`` when the application must display several objects at once.
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
import sys
import time

import cv2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="YOLO and OpenCV real-time object detection"
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("weights/best_chanyong.pt"),
        help="Path to an Ultralytics YOLO checkpoint (default: weights/best_chanyong.pt).",
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="OpenCV camera index (default: 0).",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.40,
        help="Minimum confidence threshold (default: 0.40).",
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=0.45,
        help="NMS IoU threshold (default: 0.45).",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=480,
        help="YOLO inference image size (default: 480).",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=640,
        help="Requested webcam/display width (default: 640).",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=480,
        help="Requested webcam/display height (default: 480).",
    )
    parser.add_argument(
        "--max-det",
        type=int,
        default=1,
        help=(
            "Maximum final detections per frame. The demo uses 1 to keep only "
            "the highest-confidence target; increase it for multi-object scenes."
        ),
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=1,
        help="Print one compact log every N frames (default: 1).",
    )
    parser.add_argument(
        "--no-log",
        action="store_true",
        help="Disable per-frame terminal logs.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Inference device, for example 0, cpu, or mps. Default: automatic.",
    )
    parser.add_argument(
        "--save",
        type=Path,
        default=None,
        help="Optional MP4 path for saving the annotated stream.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if not 0.0 <= args.conf <= 1.0:
        raise SystemExit("--conf must be between 0 and 1.")
    if not 0.0 <= args.iou <= 1.0:
        raise SystemExit("--iou must be between 0 and 1.")
    if args.imgsz <= 0 or args.width <= 0 or args.height <= 0:
        raise SystemExit("--imgsz, --width, and --height must be positive.")
    if args.max_det <= 0:
        raise SystemExit("--max-det must be at least 1.")
    if args.log_every <= 0:
        raise SystemExit("--log-every must be at least 1.")


def open_camera(index: int) -> cv2.VideoCapture:
    """Prefer Linux V4L2, then fall back to OpenCV's default backend."""
    cap = cv2.VideoCapture(index, cv2.CAP_V4L2)
    if cap.isOpened():
        return cap
    cap.release()
    return cv2.VideoCapture(index)


def class_name(names: dict[int, str] | list[str], class_id: int) -> str:
    if isinstance(names, dict):
        return str(names[class_id])
    return str(names[class_id])


def detection_summary(result, names: dict[int, str] | list[str]) -> str:
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return "(no detections)"

    class_ids = boxes.cls.int().cpu().tolist()
    detected_names = [class_name(names, class_id) for class_id in class_ids]
    counts = Counter(detected_names)

    parts: list[str] = []
    for name, count in counts.items():
        suffix = "s" if count > 1 else ""
        parts.append(f"{count} {name}{suffix}")
    return ", ".join(parts)


def create_writer(path: Path, frame_shape: tuple[int, ...], fps: float) -> cv2.VideoWriter:
    output = path.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    height, width = frame_shape[:2]
    safe_fps = fps if 1.0 <= fps <= 120.0 else 30.0
    writer = cv2.VideoWriter(
        str(output),
        cv2.VideoWriter_fourcc(*"mp4v"),
        safe_fps,
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not create output video: {output}")
    print(f"Saving annotated video to: {output}")
    return writer


def main() -> int:
    args = parse_args()
    validate_args(args)

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit(
            "Ultralytics is not installed. Run: pip install -r requirements.txt"
        ) from exc

    model_path = args.model.expanduser().resolve()
    if not model_path.is_file():
        print(f"Model not found: {model_path}", file=sys.stderr)
        print(
            "Copy best_chanyong.pt into weights/ or pass its path with --model.",
            file=sys.stderr,
        )
        return 1

    print(f"Loading model: {model_path}")
    model = YOLO(str(model_path))

    cap = open_camera(args.camera)
    if not cap.isOpened():
        print(f"Could not open camera index {args.camera}.", file=sys.stderr)
        print("Check the cable and run: python3 src/check_camera.py", file=sys.stderr)
        return 1

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    window_name = "YOLO + OpenCV Object Detection"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, args.width, args.height)

    writer: cv2.VideoWriter | None = None
    frame_number = 0

    print("Detection started.")
    print("Press Q or ESC in the video window to quit; press S to save a frame.")

    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                print("Camera stopped returning frames.", file=sys.stderr)
                break

            predict_kwargs: dict[str, object] = {
                "source": frame,
                "conf": args.conf,
                "iou": args.iou,
                "imgsz": args.imgsz,
                "max_det": args.max_det,
                "verbose": False,
            }
            if args.device is not None:
                predict_kwargs["device"] = args.device

            result = model.predict(**predict_kwargs)[0]
            output = result.plot()

            if not args.no_log and frame_number % args.log_every == 0:
                height, width = frame.shape[:2]
                summary = detection_summary(result, model.names)
                inference_ms = float(result.speed.get("inference", 0.0))
                print(
                    f"0: {height}x{width} {summary}, {inference_ms:.1f}ms",
                    flush=True,
                )

            if args.save is not None and writer is None:
                writer = create_writer(
                    args.save,
                    output.shape,
                    float(cap.get(cv2.CAP_PROP_FPS)),
                )
            if writer is not None:
                writer.write(output)

            cv2.imshow(window_name, output)
            frame_number += 1

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q"), 27):
                break
            if key in (ord("s"), ord("S")):
                screenshot = Path(f"detection_{int(time.time())}.jpg").resolve()
                if cv2.imwrite(str(screenshot), output):
                    print(f"Screenshot saved: {screenshot}")
                else:
                    print(f"Could not save screenshot: {screenshot}", file=sys.stderr)

    except KeyboardInterrupt:
        print("\nDetection stopped by the user.")
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()

    print("Program finished.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
