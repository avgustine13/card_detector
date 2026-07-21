from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cv.card_common.camera import CameraOptions, open_camera
from cv.card_dataset_tool.app import find_card_quad, warp_card
from cv.card_dataset_tool.cnn_common import load_checkpoint
from cv.card_dataset_tool.live_patch_cnn import predict_card, resolve_device
from cv.card_dataset_tool.patch_preprocess import SUIT_ROI, orient_card_to_corner


RANK_NAMES = {
    "A": "Ace",
    "2": "Two",
    "3": "Three",
    "4": "Four",
    "5": "Five",
    "6": "Six",
    "7": "Seven",
    "8": "Eight",
    "9": "Nine",
    "10": "Ten",
    "J": "Jack",
    "Q": "Queen",
    "K": "King",
}
SUIT_NAMES = {
    "C": "Clubs",
    "D": "Diamonds",
    "H": "Hearts",
    "S": "Spades",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Live card detector with full-name caption overlay.")
    parser.add_argument("--camera", type=int, default=0, help="Camera index.")
    parser.add_argument("--backend", default="auto", help="Camera backend: auto, opencv, or rpicam.")
    parser.add_argument("--width", type=int, default=640, help="Requested capture width.")
    parser.add_argument("--height", type=int, default=480, help="Requested capture height.")
    parser.add_argument("--fps", type=int, default=15, help="Requested capture FPS.")
    parser.add_argument("--min-area", type=int, default=12000, help="Minimum contour area for a card candidate.")
    parser.add_argument("--device", default="auto", help="auto, cpu, or cuda.")
    parser.add_argument("--debug", action="store_true", help="Show the oriented card crop.")
    parser.add_argument("--image-path", default="", help="Run once on an existing image and save/show the captioned result.")
    parser.add_argument(
        "--rank-model",
        default=str(Path(__file__).with_name("models") / "rank_cnn.pt"),
        help="Rank CNN checkpoint path.",
    )
    parser.add_argument(
        "--suit-model",
        default=str(Path(__file__).with_name("models") / "suit_cnn.pt"),
        help="Suit CNN checkpoint path.",
    )
    parser.add_argument(
        "--snapshot-dir",
        default=str(Path(__file__).with_name("detection_captures")),
        help="Directory for saved captioned frames.",
    )
    parser.add_argument("--save-image", default="", help="Output path for --image-path captioned result.")
    parser.add_argument("--no-window", action="store_true", help="Do not open a preview window for --image-path.")
    return parser.parse_args()


def card_full_name(label: str) -> str:
    if len(label) < 2:
        return label or "Unknown card"
    rank = label[:-1]
    suit = label[-1]
    rank_name = RANK_NAMES.get(rank, rank)
    suit_name = SUIT_NAMES.get(suit, suit)
    return f"{rank_name} of {suit_name}"


def classify_red_suit_shape(oriented_warped) -> str:
    x, y, width, height = SUIT_ROI
    suit_roi = oriented_warped[y : y + height, x : x + width]
    hsv = cv2.cvtColor(suit_roi, cv2.COLOR_BGR2HSV)
    hue, saturation, value = cv2.split(hsv)
    red_mask = (((hue <= 12) | (hue >= 165)) & (saturation >= 45) & (value <= 245)).astype(np.uint8) * 255
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, np.ones((3, 3), dtype=np.uint8), iterations=1)
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, np.ones((5, 5), dtype=np.uint8), iterations=1)
    contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = [contour for contour in contours if cv2.contourArea(contour) >= 80]
    if not contours:
        return ""

    contour = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    if perimeter <= 0:
        return ""

    hull_area = cv2.contourArea(cv2.convexHull(contour))
    if hull_area <= 0:
        return ""

    x0, y0, w0, h0 = cv2.boundingRect(contour)
    extent = area / float(w0 * h0)
    solidity = area / hull_area
    circularity = (4.0 * np.pi * area) / (perimeter * perimeter)
    approx = cv2.approxPolyDP(contour, 0.045 * perimeter, True)

    # Diamonds are compact, convex, and angular. Hearts have a top notch/lobes
    # and are usually less solid in the same corner ROI.
    if solidity >= 0.92 and extent >= 0.42 and circularity <= 0.78 and len(approx) <= 8:
        return "D"
    if solidity < 0.92 or circularity > 0.72:
        return "H"
    return ""


def apply_shape_suit_correction(label: str, warped: np.ndarray) -> tuple[str, str]:
    if len(label) < 2 or label[-1] not in ("D", "H"):
        return label, ""

    oriented = orient_card_to_corner(warped)
    shape_suit = classify_red_suit_shape(oriented)
    if shape_suit and shape_suit != label[-1]:
        return f"{label[:-1]}{shape_suit}", f"shape {label[-1]}->{shape_suit}"
    return label, ""


def draw_text_box(frame, lines: list[str], origin: tuple[int, int], color: tuple[int, int, int]) -> None:
    x, y = origin
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.72
    thickness = 2
    padding = 8
    line_height = 28
    widths = [cv2.getTextSize(line, font, font_scale, thickness)[0][0] for line in lines]
    box_width = max(widths, default=0) + (padding * 2)
    box_height = (line_height * len(lines)) + (padding * 2)
    x = max(0, min(x, frame.shape[1] - box_width - 1))
    y = max(box_height + 1, min(y, frame.shape[0] - 1))
    cv2.rectangle(frame, (x, y - box_height), (x + box_width, y), (25, 25, 25), -1)
    cv2.rectangle(frame, (x, y - box_height), (x + box_width, y), color, 2)
    for index, line in enumerate(lines):
        baseline = y - box_height + padding + 21 + (index * line_height)
        cv2.putText(frame, line, (x + padding, baseline), font, font_scale, color, thickness, cv2.LINE_AA)


def draw_detection_overlay(
    frame,
    quad,
    contour,
    label: str,
    rank_confidence: float,
    suit_confidence: float,
    status: str,
    correction: str,
):
    canvas = frame.copy()
    color = (0, 220, 0) if label else (0, 200, 255)
    if contour is not None:
        cv2.drawContours(canvas, [contour], -1, (0, 255, 255), 2)
    if quad is not None:
        points = quad.astype("int32")
        cv2.polylines(canvas, [points], True, color, 3)
        top_left = points[points[:, 1].argmin()]
        origin = (int(top_left[0]), int(top_left[1]) - 10)
    else:
        origin = (12, 74)

    if label:
        lines = [
            card_full_name(label),
            f"{label}  rank {rank_confidence:.2f}  suit {suit_confidence:.2f}",
        ]
        if correction:
            lines.append(correction)
    else:
        lines = [status]
    draw_text_box(canvas, lines, origin, color)
    return canvas


def load_models(args: argparse.Namespace):
    device = resolve_device(args.device)
    rank_model, rank_id_to_label, rank_target = load_checkpoint(Path(args.rank_model), device)
    suit_model, suit_id_to_label, suit_target = load_checkpoint(Path(args.suit_model), device)
    if rank_target != "rank" or suit_target != "suit":
        raise RuntimeError("checkpoint targets do not match expected rank/suit models")
    return device, rank_model, rank_id_to_label, suit_model, suit_id_to_label


def detect_frame(frame, min_area: int, rank_model, rank_id_to_label, suit_model, suit_id_to_label, device):
    quad, contour, area = find_card_quad(frame, min_area)
    if quad is None:
        return "", 0.0, 0.0, None, contour, None, area, "no card found"
    warped = warp_card(frame, quad)
    _rank, rank_confidence, _suit, suit_confidence, label = predict_card(
        warped, rank_model, rank_id_to_label, suit_model, suit_id_to_label, device
    )
    corrected_label, correction = apply_shape_suit_correction(label, warped)
    return corrected_label, rank_confidence, suit_confidence, quad, contour, warped, area, correction or "detected"


def save_snapshot(snapshot_dir: Path, frame, label: str) -> Path:
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    safe_label = label or "unknown"
    path = snapshot_dir / f"{stamp}_{safe_label}_captioned.jpg"
    cv2.imwrite(str(path), frame)
    return path


def run_image_once(args: argparse.Namespace, rank_model, rank_id_to_label, suit_model, suit_id_to_label, device) -> int:
    image_path = Path(args.image_path)
    frame = cv2.imread(str(image_path))
    if frame is None:
        print(f"Failed to read image: {image_path}")
        return 1

    label, rank_confidence, suit_confidence, quad, contour, warped, _area, status = detect_frame(
        frame, args.min_area, rank_model, rank_id_to_label, suit_model, suit_id_to_label, device
    )
    correction = status if status.startswith("shape ") else ""
    captioned = draw_detection_overlay(frame, quad, contour, label, rank_confidence, suit_confidence, status, correction)
    output_path = Path(args.save_image) if args.save_image else save_snapshot(Path(args.snapshot_dir), captioned, label)
    cv2.imwrite(str(output_path), captioned)

    print(f"Image: {image_path}")
    print(f"Detected: {card_full_name(label) if label else '-'}")
    if label:
        print(f"Label: {label}")
        print(f"Rank confidence: {rank_confidence:.4f}")
        print(f"Suit confidence: {suit_confidence:.4f}")
    print(f"Captioned image: {output_path}")

    if not args.no_window:
        cv2.imshow("card_detection", captioned)
        if args.debug and warped is not None:
            cv2.imshow("card_detection_warped", orient_card_to_corner(warped))
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return 0 if label else 2


def main() -> int:
    args = parse_args()
    try:
        device, rank_model, rank_id_to_label, suit_model, suit_id_to_label = load_models(args)
    except RuntimeError as exc:
        print(f"Failed to load models: {exc}")
        return 1

    if args.image_path:
        return run_image_once(args, rank_model, rank_id_to_label, suit_model, suit_id_to_label, device)

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
    print(f"Device: {device}")
    print("Controls: s save captioned frame, g debug, Esc quit")

    debug_enabled = args.debug
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Camera read failed.")
                return 1

            label, rank_confidence, suit_confidence, quad, contour, warped, _area, status = detect_frame(
                frame, args.min_area, rank_model, rank_id_to_label, suit_model, suit_id_to_label, device
            )
            correction = status if status.startswith("shape ") else ""
            preview = draw_detection_overlay(
                frame, quad, contour, label, rank_confidence, suit_confidence, status, correction
            )
            cv2.imshow("card_detection", preview)
            if debug_enabled and warped is not None:
                cv2.imshow("card_detection_warped", orient_card_to_corner(warped))

            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                break
            if key == ord("g"):
                debug_enabled = not debug_enabled
                if not debug_enabled:
                    cv2.destroyWindow("card_detection_warped")
                continue
            if key == ord("s"):
                saved_path = save_snapshot(Path(args.snapshot_dir), preview, label)
                print(f"Saved {saved_path}")
    finally:
        cap.release()
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
