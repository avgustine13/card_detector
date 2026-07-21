from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cv.card_common.camera import CameraOptions, open_camera
from cv.card_dataset_tool.app import find_card_quad, is_valid_label, normalize_label, update_label, warp_card
from cv.card_dataset_tool.cnn_common import load_checkpoint
from cv.card_dataset_tool.patch_preprocess import (
    extract_roi,
    normalize_patch_image,
    orient_card_to_corner,
    patch_to_tensor_array,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Live single-card CNN recognition test tool.")
    parser.add_argument("--camera", type=int, default=0, help="Camera index.")
    parser.add_argument("--backend", default="auto", help="Camera backend: auto, opencv, or rpicam.")
    parser.add_argument("--width", type=int, default=640, help="Requested capture width.")
    parser.add_argument("--height", type=int, default=480, help="Requested capture height.")
    parser.add_argument("--fps", type=int, default=15, help="Requested capture FPS.")
    parser.add_argument("--min-area", type=int, default=12000, help="Minimum contour area for a card candidate.")
    parser.add_argument("--device", default="auto", help="auto, cpu, or cuda.")
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
        "--log-path",
        default=str(Path(__file__).with_name("models") / "live_test_log.csv"),
        help="CSV path for manually logged observations.",
    )
    parser.add_argument(
        "--captures-dir",
        default=str(Path(__file__).with_name("live_captures")),
        help="Directory for saved raw/warped frames from logged observations.",
    )
    parser.add_argument("--expected-label", default="", help="Expected card label for one-shot image testing.")
    parser.add_argument("--image-path", default="", help="Run one prediction on an existing image and exit.")
    parser.add_argument("--save-on-log", action="store_true", help="Save raw and warped images when logging.")
    parser.add_argument("--debug", action="store_true", help="Show warped debug window.")
    return parser.parse_args()


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def predict_patch(
    model: torch.nn.Module,
    id_to_label: dict[int, str],
    patch: np.ndarray,
    device: torch.device,
) -> Tuple[str, float]:
    tensor_array = patch_to_tensor_array(patch)
    expected_channels = int(model.features[0].in_channels)
    if expected_channels == 1 and tensor_array.shape[0] == 3:
        gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
        tensor_array = patch_to_tensor_array(gray)
    tensor = torch.from_numpy(tensor_array[None, :, :, :]).to(device)
    with torch.no_grad():
        probabilities = torch.softmax(model(tensor), dim=1)[0]
    confidence, index = torch.max(probabilities, dim=0)
    return id_to_label[int(index.item())], float(confidence.item())


def predict_card(
    warped: np.ndarray,
    rank_model: torch.nn.Module,
    rank_id_to_label: dict[int, str],
    suit_model: torch.nn.Module,
    suit_id_to_label: dict[int, str],
    device: torch.device,
) -> Tuple[str, float, str, float, str]:
    oriented = orient_card_to_corner(warped)
    rank_patch = normalize_patch_image(extract_roi(oriented, "rank"), "rank")
    suit_patch = normalize_patch_image(extract_roi(oriented, "suit"), "suit")
    rank, rank_confidence = predict_patch(rank_model, rank_id_to_label, rank_patch, device)
    suit, suit_confidence = predict_patch(suit_model, suit_id_to_label, suit_patch, device)
    return rank, rank_confidence, suit, suit_confidence, f"{rank}{suit}"


def append_log(
    log_path: Path,
    expected_label: str,
    predicted_label: str,
    rank_confidence: float,
    suit_confidence: float,
    raw_path: Path | None,
    warped_path: Path | None,
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
                    "raw_path",
                    "warped_path",
                ]
            )
        writer.writerow(
            [
                time.strftime("%Y%m%d_%H%M%S"),
                expected_label,
                predicted_label,
                expected_label == predicted_label if is_valid_label(expected_label) else "",
                f"{rank_confidence:.4f}",
                f"{suit_confidence:.4f}",
                str(raw_path or ""),
                str(warped_path or ""),
            ]
        )


def save_observation(captures_dir: Path, expected_label: str, predicted_label: str, frame: np.ndarray, warped: np.ndarray) -> Tuple[Path, Path]:
    captures_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    safe_expected = expected_label if expected_label else "unknown"
    raw_path = captures_dir / f"{stamp}_{safe_expected}_as_{predicted_label}_raw.jpg"
    warped_path = captures_dir / f"{stamp}_{safe_expected}_as_{predicted_label}_warped.jpg"
    cv2.imwrite(str(raw_path), frame)
    cv2.imwrite(str(warped_path), warped)
    return raw_path, warped_path


def draw_overlay(
    frame: np.ndarray,
    quad: np.ndarray | None,
    contour: np.ndarray | None,
    expected_label: str,
    predicted_label: str,
    rank_confidence: float,
    suit_confidence: float,
    status: str,
) -> np.ndarray:
    canvas = frame.copy()
    if contour is not None:
        cv2.drawContours(canvas, [contour], -1, (0, 255, 255), 2)
    if quad is not None:
        cv2.polylines(canvas, [quad.astype(np.int32)], True, (0, 255, 0), 3)

    color = (0, 220, 0)
    if is_valid_label(expected_label) and predicted_label and expected_label != predicted_label:
        color = (0, 0, 255)

    cv2.putText(canvas, f"expected: {expected_label or '-'}", (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 220, 0), 2, cv2.LINE_AA)
    cv2.putText(canvas, f"predicted: {predicted_label or '-'}", (12, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.85, color, 2, cv2.LINE_AA)
    cv2.putText(
        canvas,
        f"rank:{rank_confidence:.2f} suit:{suit_confidence:.2f}  {status}",
        (12, 94),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return canvas


def run_image_once(
    image_path: Path,
    expected_label: str,
    min_area: int,
    rank_model: torch.nn.Module,
    rank_id_to_label: dict[int, str],
    suit_model: torch.nn.Module,
    suit_id_to_label: dict[int, str],
    device: torch.device,
    log_path: Path,
    captures_dir: Path,
    save_on_log: bool,
) -> int:
    frame = cv2.imread(str(image_path))
    if frame is None:
        print(f"Failed to read image: {image_path}")
        return 1

    quad, _contour, area = find_card_quad(frame, min_area)
    if quad is None:
        print("No card found.")
        return 2

    warped = warp_card(frame, quad)
    rank, rank_confidence, suit, suit_confidence, predicted_label = predict_card(
        warped, rank_model, rank_id_to_label, suit_model, suit_id_to_label, device
    )
    del rank, suit

    raw_path = None
    warped_path = None
    if save_on_log:
        raw_path, warped_path = save_observation(captures_dir, expected_label, predicted_label, frame, warped)

    append_log(log_path, expected_label, predicted_label, rank_confidence, suit_confidence, raw_path, warped_path)

    print(f"Image: {image_path}")
    print(f"Expected: {expected_label or '-'}")
    print(f"Predicted: {predicted_label}")
    if is_valid_label(expected_label):
        print(f"Match: {expected_label == predicted_label}")
    print(f"Rank confidence: {rank_confidence:.4f}")
    print(f"Suit confidence: {suit_confidence:.4f}")
    print(f"Contour area: {area:.1f}")
    if raw_path is not None:
        print(f"Saved raw: {raw_path}")
        print(f"Saved warped: {warped_path}")
    print(f"Logged: {log_path}")
    return 0


def main() -> int:
    args = parse_args()
    device = resolve_device(args.device)
    rank_model, rank_id_to_label, rank_target = load_checkpoint(Path(args.rank_model), device)
    suit_model, suit_id_to_label, suit_target = load_checkpoint(Path(args.suit_model), device)
    if rank_target != "rank" or suit_target != "suit":
        print("Checkpoint targets do not match expected rank/suit models.")
        return 1

    log_path = Path(args.log_path)
    captures_dir = Path(args.captures_dir)
    expected_label = normalize_label(args.expected_label)
    if args.image_path:
        return run_image_once(
            Path(args.image_path),
            expected_label,
            args.min_area,
            rank_model,
            rank_id_to_label,
            suit_model,
            suit_id_to_label,
            device,
            log_path,
            captures_dir,
            args.save_on_log,
        )

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
    print("Controls: type expected label, '-' clear, backspace edit, space log, s save+log, g debug, Esc quit")

    debug_enabled = args.debug

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Camera read failed.")
                return 1

            quad, contour, _area = find_card_quad(frame, args.min_area)
            warped = warp_card(frame, quad) if quad is not None else None

            predicted_label = ""
            rank_confidence = 0.0
            suit_confidence = 0.0
            status = "no card found"
            if warped is not None:
                rank, rank_confidence, suit, suit_confidence, predicted_label = predict_card(
                    warped, rank_model, rank_id_to_label, suit_model, suit_id_to_label, device
                )
                del rank, suit
                if not expected_label:
                    status = "type expected label"
                elif not is_valid_label(expected_label):
                    status = "expected label incomplete"
                elif expected_label == predicted_label:
                    status = "match"
                else:
                    status = "mismatch"

            preview = draw_overlay(frame, quad, contour, expected_label, predicted_label, rank_confidence, suit_confidence, status)
            cv2.imshow("card_live_patch_cnn", preview)
            if debug_enabled and warped is not None:
                cv2.imshow("card_live_warped", orient_card_to_corner(warped))

            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                break
            if key == ord("g"):
                debug_enabled = not debug_enabled
                if not debug_enabled:
                    cv2.destroyWindow("card_live_warped")
                continue
            if key in (32, ord("s")):
                if warped is None:
                    print("No card available to log.")
                    continue
                raw_path = None
                warped_path = None
                if args.save_on_log or key == ord("s"):
                    raw_path, warped_path = save_observation(captures_dir, expected_label, predicted_label, frame, warped)
                append_log(log_path, expected_label, predicted_label, rank_confidence, suit_confidence, raw_path, warped_path)
                print(f"Logged expected={expected_label or '-'} predicted={predicted_label or '-'} status={status}")
                continue
            if key != 255:
                expected_label = normalize_label(update_label(expected_label, key))
    finally:
        cap.release()
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
