from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cv.card_common.camera import CameraOptions, open_camera
from cv.card_dataset_tool.app import find_card_quad, is_valid_label, normalize_label, update_label, warp_card
from cv.card_dataset_tool.cnn_common import load_checkpoint
from cv.card_dataset_tool.live_patch_cnn import predict_patch, resolve_device
from cv.card_dataset_tool.patch_preprocess import SUIT_ROI, extract_roi, normalize_patch_image


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
    parser.add_argument("--expected-label", default="", help="Actual card label for saved debug captures, for example 7H.")
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
    parser.add_argument(
        "--log-path",
        default=str(Path(__file__).with_name("detection_captures") / "detection_log.csv"),
        help="CSV path for saved detection observations.",
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


def classify_black_suit_shape(oriented_warped) -> str:
    x, y, width, height = SUIT_ROI
    suit_roi = oriented_warped[y : y + height, x : x + width]
    hsv = cv2.cvtColor(suit_roi, cv2.COLOR_BGR2HSV)
    _hue, saturation, value = cv2.split(hsv)
    black_mask = ((value <= 145) & (saturation <= 135)).astype(np.uint8) * 255
    black_mask = cv2.morphologyEx(black_mask, cv2.MORPH_OPEN, np.ones((3, 3), dtype=np.uint8), iterations=1)
    black_mask = cv2.morphologyEx(black_mask, cv2.MORPH_CLOSE, np.ones((5, 5), dtype=np.uint8), iterations=1)
    contours, _ = cv2.findContours(black_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
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
    lower_half = contour[:, 0, 1] > (y0 + h0 * 0.58)
    lower_width = 0
    if np.any(lower_half):
        lower_points = contour[:, 0, :][lower_half]
        lower_width = int(lower_points[:, 0].max() - lower_points[:, 0].min() + 1)
    solidity = area / hull_area
    lower_width_ratio = lower_width / max(1.0, float(w0))

    # Spades usually have one broad upper body narrowing into a stem.
    # Clubs keep substantial width in the lower half because of the lower lobe.
    if lower_width_ratio <= 0.55 and solidity >= 0.78:
        return "S"
    if lower_width_ratio > 0.55 or solidity < 0.78:
        return "C"
    return ""


def apply_shape_suit_correction(label: str, oriented_warped: np.ndarray) -> tuple[str, str]:
    if len(label) < 2:
        return label, ""

    color_group = suit_color_group(oriented_warped)
    if color_group == "red":
        shape_suit = classify_red_suit_shape(oriented_warped)
        if label[-1] in ("C", "S") and shape_suit in ("D", "H"):
            return f"{label[:-1]}{shape_suit}", f"color {label[-1]}->{shape_suit}"
    if color_group == "black":
        shape_suit = classify_black_suit_shape(oriented_warped)
        if label[-1] in ("D", "H") and shape_suit in ("C", "S"):
            return f"{label[:-1]}{shape_suit}", f"color {label[-1]}->{shape_suit}"
    return label, ""


def suit_color_group(oriented_warped: np.ndarray) -> str:
    x, y, width, height = SUIT_ROI
    suit_roi = oriented_warped[y : y + height, x : x + width]
    return roi_color_group(suit_roi)


def roi_color_group(roi: np.ndarray) -> str:
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    hue, saturation, value = cv2.split(hsv)
    red_count = int(((((hue <= 12) | (hue >= 165)) & (saturation >= 45) & (value <= 245))).sum())
    black_count = int((((value <= 145) & (saturation <= 135))).sum())
    if red_count < 25 and black_count < 25:
        return "unknown"
    if red_count >= max(35, int(black_count * 0.35)):
        return "red"
    if black_count >= 35:
        return "black"
    return "unknown"


def suit_color_score(label: str, color_group: str) -> float:
    if len(label) < 2 or color_group == "unknown":
        return 1.0
    suit = label[-1]
    if suit in ("D", "H"):
        return 1.0 if color_group == "red" else 0.08
    if suit in ("C", "S"):
        return 1.0 if color_group == "black" else 0.08
    return 1.0


def rank_color_score(label: str, oriented_warped: np.ndarray) -> float:
    if len(label) < 2:
        return 1.0
    rank_roi = extract_roi(oriented_warped, "rank")
    color_group = roi_color_group(rank_roi)
    if color_group == "unknown":
        return 0.75
    suit = label[-1]
    if suit in ("D", "H"):
        return 1.0 if color_group == "red" else 0.18
    if suit in ("C", "S"):
        return 1.0 if color_group == "black" else 0.18
    return 1.0


def index_ink_score(oriented_warped: np.ndarray) -> float:
    rank_roi = extract_roi(oriented_warped, "rank")
    suit_roi = extract_roi(oriented_warped, "suit")
    rank_gray = cv2.cvtColor(rank_roi, cv2.COLOR_BGR2GRAY)
    suit_gray = cv2.cvtColor(suit_roi, cv2.COLOR_BGR2GRAY)
    rank_ink = int((rank_gray < 175).sum())
    suit_ink = int((suit_gray < 175).sum())
    if rank_ink < 70 or suit_ink < 50:
        return 0.20
    if rank_ink > 7800 or suit_ink > 9000:
        return 0.35
    return 1.0


def rank_layout_score(oriented_warped: np.ndarray) -> float:
    rank_roi = extract_roi(oriented_warped, "rank")
    gray = cv2.cvtColor(rank_roi, cv2.COLOR_BGR2GRAY)
    mask = gray < 175
    ys, xs = np.where(mask)
    if len(xs) < 70:
        return 0.20

    x0 = int(xs.min())
    y0 = int(ys.min())
    x1 = int(xs.max())
    y1 = int(ys.max())
    width = x1 - x0 + 1
    height = y1 - y0 + 1

    score = 1.0
    if x0 > 34:
        score *= 0.30
    if y0 > 34:
        score *= 0.30
    if width > 118 and height > 145:
        score *= 0.35
    if len(xs) > 6500:
        score *= 0.35
    return score


def predict_card_best_orientation(
    warped: np.ndarray,
    rank_model,
    rank_id_to_label,
    suit_model,
    suit_id_to_label,
    device,
):
    best = None
    for rotation in range(4):
        oriented = np.ascontiguousarray(np.rot90(warped, rotation))
        rank_patch = normalize_patch_image(extract_roi(oriented, "rank"), "rank")
        suit_patch = normalize_patch_image(extract_roi(oriented, "suit"), "suit")
        rank, rank_confidence = predict_patch(rank_model, rank_id_to_label, rank_patch, device)
        suit, suit_confidence = predict_patch(suit_model, suit_id_to_label, suit_patch, device)
        label = f"{rank}{suit}"
        score = (
            min(rank_confidence, suit_confidence)
            * ((rank_confidence + suit_confidence) / 2.0)
            * suit_color_score(label, suit_color_group(oriented))
            * rank_color_score(label, oriented)
            * index_ink_score(oriented)
            * rank_layout_score(oriented)
        )
        candidate = (score, rank_confidence, suit_confidence, label, oriented, rotation)
        if best is None or candidate[0] > best[0]:
            best = candidate

    if best is None:
        return "", 0.0, 0.0, warped, 0
    _score, rank_confidence, suit_confidence, label, oriented, rotation = best
    return label, rank_confidence, suit_confidence, oriented, rotation


def number_pip_mask(oriented_warped: np.ndarray, suit: str) -> np.ndarray:
    inner = oriented_warped[55:465, 45:315]
    hsv = cv2.cvtColor(inner, cv2.COLOR_BGR2HSV)
    hue, saturation, value = cv2.split(hsv)
    if suit in ("D", "H"):
        mask = (((hue <= 12) | (hue >= 165)) & (saturation >= 45) & (value <= 245)).astype(np.uint8) * 255
    else:
        mask = ((value <= 145) & (saturation <= 135)).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), dtype=np.uint8), iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7, 7), dtype=np.uint8), iterations=1)
    return mask


def estimate_number_rank_by_pips(oriented_warped: np.ndarray, suit: str) -> str:
    mask = number_pip_mask(oriented_warped, suit)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    large = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 120 or area > 4500:
            continue
        x, y, width, height = cv2.boundingRect(contour)
        if width < 8 or height < 8:
            continue
        large.append(contour)

    count = len(large)
    if 2 <= count <= 10:
        return str(count)
    return ""


def apply_number_rank_correction(label: str, oriented_warped: np.ndarray) -> tuple[str, str]:
    # Disabled until we have raw failing frames from the live camera. Simple
    # contour counts overcount/undercount pips depending on glare and overlap.
    return label, ""

    if len(label) < 2:
        return label, ""
    rank = label[:-1]
    suit = label[-1]
    if rank not in {"2", "3", "4", "5", "6", "7", "8", "9", "10"}:
        return label, ""

    pip_rank = estimate_number_rank_by_pips(oriented_warped, suit)
    if pip_rank and pip_rank != rank:
        return f"{pip_rank}{suit}", f"pips {rank}->{pip_rank}"
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
    expected_label: str,
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
        match_text = ""
        if expected_label:
            match_text = "match" if is_valid_label(expected_label) and expected_label == label else "mismatch"
        lines = [
            card_full_name(label),
            f"{label}  rank {rank_confidence:.2f}  suit {suit_confidence:.2f}",
        ]
        if expected_label:
            lines.append(f"actual {expected_label}  {match_text}")
        if correction:
            lines.append(correction)
    else:
        lines = [status]
        if expected_label:
            lines.append(f"actual {expected_label}")
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
    label, rank_confidence, suit_confidence, oriented, rotation = predict_card_best_orientation(
        warped, rank_model, rank_id_to_label, suit_model, suit_id_to_label, device
    )
    corrected_label, correction = apply_shape_suit_correction(label, oriented)
    corrected_label, rank_correction = apply_number_rank_correction(corrected_label, oriented)
    status = rank_correction or correction or f"detected rot={rotation * 90}"
    return corrected_label, rank_confidence, suit_confidence, quad, contour, oriented, area, status


def save_snapshot(snapshot_dir: Path, frame, expected_label: str, label: str, raw_frame=None) -> tuple[Path, Path | None]:
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    safe_expected = expected_label if is_valid_label(expected_label) else "unknown"
    safe_label = label or "unknown"
    captioned_path = snapshot_dir / f"{stamp}_actual_{safe_expected}_pred_{safe_label}_captioned.jpg"
    cv2.imwrite(str(captioned_path), frame)
    raw_path = None
    if raw_frame is not None:
        raw_path = snapshot_dir / f"{stamp}_actual_{safe_expected}_pred_{safe_label}_raw.jpg"
        cv2.imwrite(str(raw_path), raw_frame)
    return captioned_path, raw_path


def append_detection_log(
    log_path: Path,
    expected_label: str,
    predicted_label: str,
    rank_confidence: float,
    suit_confidence: float,
    status: str,
    contour_area: float,
    captioned_path: Path | None,
    raw_path: Path | None,
) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not log_path.exists()
    with log_path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        if write_header:
            writer.writerow(
                [
                    "timestamp",
                    "expected",
                    "predicted",
                    "match",
                    "rank_confidence",
                    "suit_confidence",
                    "status",
                    "contour_area",
                    "captioned_path",
                    "raw_path",
                ]
            )
        writer.writerow(
            [
                time.strftime("%Y%m%d_%H%M%S"),
                expected_label,
                predicted_label,
                expected_label == predicted_label if is_valid_label(expected_label) and predicted_label else "",
                f"{rank_confidence:.4f}",
                f"{suit_confidence:.4f}",
                status,
                f"{contour_area:.1f}",
                str(captioned_path or ""),
                str(raw_path or ""),
            ]
        )


def run_image_once(args: argparse.Namespace, rank_model, rank_id_to_label, suit_model, suit_id_to_label, device) -> int:
    image_path = Path(args.image_path)
    frame = cv2.imread(str(image_path))
    if frame is None:
        print(f"Failed to read image: {image_path}")
        return 1

    expected_label = normalize_label(args.expected_label)
    label, rank_confidence, suit_confidence, quad, contour, warped, _area, status = detect_frame(
        frame, args.min_area, rank_model, rank_id_to_label, suit_model, suit_id_to_label, device
    )
    correction = status if status.startswith("shape ") else ""
    captioned = draw_detection_overlay(
        frame, quad, contour, expected_label, label, rank_confidence, suit_confidence, status, correction
    )
    if args.save_image:
        output_path = Path(args.save_image)
        cv2.imwrite(str(output_path), captioned)
        raw_path = None
    else:
        output_path, raw_path = save_snapshot(Path(args.snapshot_dir), captioned, expected_label, label, frame)
    append_detection_log(
        Path(args.log_path),
        expected_label,
        label,
        rank_confidence,
        suit_confidence,
        status,
        _area,
        output_path,
        raw_path,
    )

    print(f"Image: {image_path}")
    print(f"Expected: {expected_label or '-'}")
    print(f"Detected: {card_full_name(label) if label else '-'}")
    if is_valid_label(expected_label) and label:
        print(f"Match: {expected_label == label}")
    if label:
        print(f"Label: {label}")
        print(f"Rank confidence: {rank_confidence:.4f}")
        print(f"Suit confidence: {suit_confidence:.4f}")
    print(f"Captioned image: {output_path}")

    if not args.no_window:
        cv2.imshow("card_detection", captioned)
        if args.debug and warped is not None:
            cv2.imshow("card_detection_warped", warped)
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
    print("Controls: type actual label, '-' clear, backspace edit, s save, g debug, Esc quit")

    debug_enabled = args.debug
    expected_label = normalize_label(args.expected_label)
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
                frame, quad, contour, expected_label, label, rank_confidence, suit_confidence, status, correction
            )
            cv2.imshow("card_detection", preview)
            if debug_enabled and warped is not None:
                cv2.imshow("card_detection_warped", warped)

            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                break
            if key == ord("g"):
                debug_enabled = not debug_enabled
                if not debug_enabled:
                    cv2.destroyWindow("card_detection_warped")
                continue
            if key == ord("s"):
                captioned_path, raw_path = save_snapshot(Path(args.snapshot_dir), preview, expected_label, label, frame)
                append_detection_log(
                    Path(args.log_path),
                    expected_label,
                    label,
                    rank_confidence,
                    suit_confidence,
                    status,
                    _area,
                    captioned_path,
                    raw_path,
                )
                match_text = ""
                if is_valid_label(expected_label) and label:
                    match_text = " match" if expected_label == label else " mismatch"
                print(f"Saved {captioned_path}{match_text}")
                continue
            if key != 255:
                expected_label = normalize_label(update_label(expected_label, key))
    finally:
        cap.release()
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
