import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cv.card_common.camera import CameraOptions, open_camera


CARD_WIDTH = 360
CARD_HEIGHT = 520
CARD_ASPECT_RATIO = CARD_HEIGHT / CARD_WIDTH
ASPECT_RATIO_TOLERANCE = 0.35
ALLOWED_LABEL_CHARS = set("A23456789JQKCDHS10")
VALID_SUITS = {"C", "D", "H", "S"}
VALID_RANKS = {"A", "2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Playing-card dataset capture tool.")
    parser.add_argument("--camera", type=int, default=0, help="Camera index for cv2.VideoCapture.")
    parser.add_argument("--backend", default="auto", help="Camera backend: auto, opencv, or rpicam.")
    parser.add_argument("--width", type=int, default=640, help="Requested capture width.")
    parser.add_argument("--height", type=int, default=480, help="Requested capture height.")
    parser.add_argument("--fps", type=int, default=15, help="Requested capture FPS.")
    parser.add_argument("--min-area", type=int, default=12000, help="Minimum contour area for a card candidate.")
    parser.add_argument("--debug", action="store_true", help="Show debug windows at startup.")
    parser.add_argument("--label", default="", help="Initial card label, for example AS or 10H.")
    parser.add_argument(
        "--dataset-dir",
        default=str(Path(__file__).with_name("dataset")),
        help="Dataset root containing raw/, warped/, and meta.jsonl.",
    )
    return parser.parse_args()


def order_quad(points: np.ndarray) -> np.ndarray:
    pts = points.astype(np.float32)
    sums = pts.sum(axis=1)
    diffs = np.diff(pts, axis=1).reshape(-1)
    ordered = np.zeros((4, 2), dtype=np.float32)
    ordered[0] = pts[np.argmin(sums)]
    ordered[2] = pts[np.argmax(sums)]
    ordered[1] = pts[np.argmin(diffs)]
    ordered[3] = pts[np.argmax(diffs)]
    return ordered


def warp_card(frame: np.ndarray, quad: np.ndarray) -> np.ndarray:
    ordered = order_quad(quad)
    target = np.array(
        [
            [0, 0],
            [CARD_WIDTH - 1, 0],
            [CARD_WIDTH - 1, CARD_HEIGHT - 1],
            [0, CARD_HEIGHT - 1],
        ],
        dtype=np.float32,
    )
    matrix = cv2.getPerspectiveTransform(ordered, target)
    return cv2.warpPerspective(frame, matrix, (CARD_WIDTH, CARD_HEIGHT))


def preprocess(frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 60, 140)
    kernel = np.ones((3, 3), dtype=np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)
    return gray, edges


def contour_has_card_shape(contour: np.ndarray) -> bool:
    rect = cv2.minAreaRect(contour)
    width, height = rect[1]
    if width <= 1 or height <= 1:
        return False

    ratio = max(width, height) / min(width, height)
    return abs(ratio - CARD_ASPECT_RATIO) <= ASPECT_RATIO_TOLERANCE


def find_card_quad(frame: np.ndarray, min_area: int) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], float]:
    _, edges = preprocess(frame)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best_quad = None
    best_contour = None
    best_area = 0.0

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area or area <= best_area:
            continue
        if not contour_has_card_shape(contour):
            continue

        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
        if len(approx) == 4 and cv2.isContourConvex(approx):
            candidate = approx.reshape(4, 2)
        else:
            candidate = cv2.boxPoints(cv2.minAreaRect(contour))

        best_area = area
        best_quad = candidate
        best_contour = contour

    return best_quad, best_contour, best_area


def normalize_label(label: str) -> str:
    return label.strip().upper()


def is_valid_label(label: str) -> bool:
    if len(label) < 2:
        return False
    suit = label[-1]
    rank = label[:-1]
    return suit in VALID_SUITS and rank in VALID_RANKS


def draw_overlay(
    frame: np.ndarray,
    quad: Optional[np.ndarray],
    contour: Optional[np.ndarray],
    overlay_enabled: bool,
    label: str,
    status: str,
    sample_count: int,
    flash_text: str,
    flash_until: float,
) -> np.ndarray:
    canvas = frame.copy()
    if overlay_enabled and contour is not None:
        cv2.drawContours(canvas, [contour], -1, (0, 255, 255), 2)
    if overlay_enabled and quad is not None:
        cv2.polylines(canvas, [quad.astype(np.int32)], True, (0, 255, 0), 3)

    cv2.putText(canvas, f"label: {label or '-'}", (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 220, 0), 2, cv2.LINE_AA)
    cv2.putText(canvas, status, (12, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 220, 0), 2, cv2.LINE_AA)
    cv2.putText(
        canvas,
        f"samples for label: {sample_count}",
        (12, 94),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    if flash_text and time.time() < flash_until:
        cv2.rectangle(canvas, (10, 110), (430, 150), (0, 90, 0), -1)
        cv2.putText(canvas, flash_text, (18, 138), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    return canvas


def save_sample(
    dataset_dir: Path,
    label: str,
    frame: np.ndarray,
    warped: np.ndarray,
    quad: np.ndarray,
    area: float,
) -> Tuple[Path, Path, Path]:
    raw_dir = dataset_dir / "raw"
    warped_dir = dataset_dir / "warped" / label
    raw_dir.mkdir(parents=True, exist_ok=True)
    warped_dir.mkdir(parents=True, exist_ok=True)

    stamp = time.strftime("%Y%m%d_%H%M%S")
    raw_path = raw_dir / f"{stamp}_{label}_raw.jpg"
    warped_path = warped_dir / f"{stamp}_{label}_warped.jpg"
    meta_path = dataset_dir / "meta.jsonl"

    cv2.imwrite(str(raw_path), frame)
    cv2.imwrite(str(warped_path), warped)

    record = {
        "timestamp": stamp,
        "label": label,
        "raw_path": str(raw_path.relative_to(dataset_dir)),
        "warped_path": str(warped_path.relative_to(dataset_dir)),
        "quad": quad.astype(float).tolist(),
        "contour_area": area,
        "card_width": CARD_WIDTH,
        "card_height": CARD_HEIGHT,
    }
    with meta_path.open("a", encoding="ascii") as handle:
        handle.write(json.dumps(record) + "\n")

    return raw_path, warped_path, meta_path


def remove_last_meta_record(meta_path: Path, raw_path: Path, warped_path: Path) -> bool:
    if not meta_path.exists():
        return False

    lines = meta_path.read_text(encoding="ascii").splitlines()
    target_raw = raw_path.name
    target_warped = warped_path.name

    for index in range(len(lines) - 1, -1, -1):
        line = lines[index].strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue

        raw_match = os.path.basename(record.get("raw_path", "")) == target_raw
        warped_match = os.path.basename(record.get("warped_path", "")) == target_warped
        if raw_match and warped_match:
            del lines[index]
            content = "\n".join(lines)
            if content:
                content += "\n"
            meta_path.write_text(content, encoding="ascii")
            return True

    return False


def update_label(current: str, key: int) -> str:
    if key in (8, 127):
        return current[:-1]
    if key == ord("-"):
        return ""
    char = chr(key).upper()
    if char not in ALLOWED_LABEL_CHARS:
        return current
    if current == "1" and char != "0":
        return current
    if len(current) >= 3:
        return current
    if current == "1" and char == "0":
        return "10"
    if char == "0" and current != "1":
        return current
    return current + char


def count_samples_for_label(dataset_dir: Path, label: str) -> int:
    if not is_valid_label(label):
        return 0
    label_dir = dataset_dir / "warped" / label
    if not label_dir.exists():
        return 0
    return sum(1 for file_path in label_dir.glob("*_warped.jpg") if file_path.is_file())


def delete_sample(raw_path: Path, warped_path: Path, meta_path: Path) -> bool:
    removed_any = False
    if raw_path.exists():
        raw_path.unlink()
        removed_any = True
    if warped_path.exists():
        warped_path.unlink()
        removed_any = True
    removed_meta = remove_last_meta_record(meta_path, raw_path, warped_path)
    return removed_any or removed_meta


def main() -> int:
    args = parse_args()
    dataset_dir = Path(args.dataset_dir)
    dataset_dir.mkdir(parents=True, exist_ok=True)

    try:
        cap, selected_backend = open_camera(
            CameraOptions(
                camera_index=args.camera,
                backend=args.backend,
                width=args.width,
                height=args.height,
                fps=args.fps,
            )
        )
    except RuntimeError as exc:
        print(f"Failed to open camera: {exc}")
        return 1
    print(f"Using backend: {selected_backend}")

    current_label = normalize_label(args.label)
    overlay_enabled = True
    debug_enabled = args.debug
    flash_text = ""
    flash_until = 0.0
    save_history: list[Tuple[Path, Path, Path, str]] = []

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Camera read failed.")
                return 1

            quad, contour, area = find_card_quad(frame, args.min_area)
            warped = warp_card(frame, quad) if quad is not None else None

            if quad is None:
                status = "no card found"
            elif current_label and not is_valid_label(current_label):
                status = "label incomplete"
            else:
                status = "ready to save" if current_label else "type label then press space"

            sample_count = count_samples_for_label(dataset_dir, current_label)
            preview = draw_overlay(
                frame,
                quad,
                contour,
                overlay_enabled,
                current_label,
                status,
                sample_count,
                flash_text,
                flash_until,
            )
            cv2.imshow("card_dataset_tool", preview)

            if debug_enabled:
                _, edges = preprocess(frame)
                cv2.imshow("card_dataset_edges", edges)
                if warped is not None:
                    cv2.imshow("card_dataset_warped", warped)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                current_label = normalize_label(update_label(current_label, key))
                continue
            if key == 27:
                break
            if key == ord("g"):
                debug_enabled = not debug_enabled
                if not debug_enabled:
                    cv2.destroyWindow("card_dataset_edges")
                    if cv2.getWindowProperty("card_dataset_warped", cv2.WND_PROP_VISIBLE) >= 0:
                        cv2.destroyWindow("card_dataset_warped")
            if key == ord("o"):
                overlay_enabled = not overlay_enabled
            if key == 32:
                if warped is None or quad is None:
                    print("No card available to save.")
                elif not is_valid_label(current_label):
                    print(f"Invalid label: {current_label!r}")
                else:
                    raw_path, warped_path, meta_path = save_sample(dataset_dir, current_label, frame, warped, quad, area)
                    save_history.append((raw_path, warped_path, meta_path, current_label))
                    sample_count = count_samples_for_label(dataset_dir, current_label)
                    flash_text = f"saved {current_label}   total: {sample_count}"
                    flash_until = time.time() + 1.8
                    print(f"Saved raw frame: {raw_path}")
                    print(f"Saved warped card: {warped_path}")
                    print(f"Updated metadata: {meta_path}")
            elif key == ord("u"):
                if not save_history:
                    flash_text = "nothing to undo"
                    flash_until = time.time() + 1.5
                    print("Nothing to undo.")
                else:
                    raw_path, warped_path, meta_path, saved_label = save_history.pop()
                    if delete_sample(raw_path, warped_path, meta_path):
                        sample_count = count_samples_for_label(dataset_dir, saved_label)
                        flash_text = f"undid {saved_label}   total: {sample_count}"
                        flash_until = time.time() + 1.8
                        print(f"Removed raw frame: {raw_path}")
                        print(f"Removed warped card: {warped_path}")
                        print(f"Updated metadata: {meta_path}")
                    else:
                        flash_text = "undo failed"
                        flash_until = time.time() + 1.5
                        print("Undo failed; sample files were not found.")
            elif key != 255:
                current_label = normalize_label(update_label(current_label, key))
    finally:
        cap.release()
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
