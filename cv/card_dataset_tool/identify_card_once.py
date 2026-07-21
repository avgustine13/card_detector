from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

import cv2
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cv.card_dataset_tool.app import find_card_quad, warp_card
from cv.card_dataset_tool.cnn_common import load_checkpoint
from cv.card_dataset_tool.live_patch_cnn import predict_card, resolve_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Identify the single card currently under the camera.")
    parser.add_argument("--camera", type=int, default=0, help="Camera index.")
    parser.add_argument("--width", type=int, default=640, help="Capture width.")
    parser.add_argument("--height", type=int, default=480, help="Capture height.")
    parser.add_argument("--timeout-ms", type=int, default=1000, help="Still capture warmup timeout in milliseconds.")
    parser.add_argument("--min-area", type=int, default=12000, help="Minimum contour area for a card candidate.")
    parser.add_argument("--device", default="cpu", help="cpu, cuda, or auto.")
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
        "--output-dir",
        default=str(Path(__file__).with_name("live_captures")),
        help="Directory for saved raw and warped debug images.",
    )
    parser.add_argument("--no-save", action="store_true", help="Do not save raw/warped debug images.")
    parser.add_argument("--quiet", action="store_true", help="Print only the predicted card label.")
    return parser.parse_args()


def capture_still(output_path: Path, camera: int, width: int, height: int, timeout_ms: int) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "rpicam-still",
        "--camera",
        str(camera),
        "--nopreview",
        "--timeout",
        str(timeout_ms),
        "--width",
        str(width),
        "--height",
        str(height),
        "-o",
        str(output_path),
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        message = (result.stderr or result.stdout).strip()
        raise RuntimeError(message or f"rpicam-still exited with {result.returncode}")


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    raw_path = output_dir / f"{stamp}_identify_raw.jpg"
    warped_path = output_dir / f"{stamp}_identify_warped.jpg"

    try:
        capture_still(raw_path, args.camera, args.width, args.height, args.timeout_ms)
    except RuntimeError as exc:
        print(f"Camera capture failed: {exc}", file=sys.stderr)
        return 1

    frame = cv2.imread(str(raw_path))
    if frame is None:
        print(f"Failed to read captured image: {raw_path}", file=sys.stderr)
        return 1

    quad, _contour, area = find_card_quad(frame, args.min_area)
    if quad is None:
        print("No card found.", file=sys.stderr)
        if args.no_save:
            raw_path.unlink(missing_ok=True)
        return 2

    warped = warp_card(frame, quad)
    if not args.no_save:
        cv2.imwrite(str(warped_path), warped)

    device = resolve_device(args.device)
    rank_model, rank_id_to_label, rank_target = load_checkpoint(Path(args.rank_model), device)
    suit_model, suit_id_to_label, suit_target = load_checkpoint(Path(args.suit_model), device)
    if rank_target != "rank" or suit_target != "suit":
        print("Checkpoint targets do not match expected rank/suit models.", file=sys.stderr)
        return 1

    rank, rank_confidence, suit, suit_confidence, predicted_label = predict_card(
        warped, rank_model, rank_id_to_label, suit_model, suit_id_to_label, device
    )
    del rank, suit

    if args.no_save:
        raw_path.unlink(missing_ok=True)

    if args.quiet:
        print(predicted_label)
        return 0

    print(f"Card: {predicted_label}")
    print(f"Rank confidence: {rank_confidence:.4f}")
    print(f"Suit confidence: {suit_confidence:.4f}")
    print(f"Contour area: {area:.1f}")
    if not args.no_save:
        print(f"Raw image: {raw_path}")
        print(f"Warped image: {warped_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
