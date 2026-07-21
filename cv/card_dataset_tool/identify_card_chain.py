from __future__ import annotations

import argparse
import csv
import shutil
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cv.card_dataset_tool.app import find_card_quad, warp_card
from cv.card_dataset_tool.cnn_common import load_checkpoint
from cv.card_dataset_tool.live_patch_cnn import predict_card, resolve_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Recognize a sequence of cards placed under the camera.")
    parser.add_argument("--count", type=int, default=4, help="Number of cards to accept before exiting.")
    parser.add_argument("--camera", type=int, default=0, help="Camera index.")
    parser.add_argument("--width", type=int, default=640, help="Capture width.")
    parser.add_argument("--height", type=int, default=480, help="Capture height.")
    parser.add_argument("--capture-timeout-ms", type=int, default=700, help="Still capture warmup timeout.")
    parser.add_argument("--interval", type=float, default=1.2, help="Seconds between attempts.")
    parser.add_argument("--cooldown", type=float, default=2.0, help="Seconds to wait after accepting a card.")
    parser.add_argument("--stable-reads", type=int, default=2, help="Same prediction count needed for acceptance.")
    parser.add_argument("--session-timeout", type=float, default=180.0, help="Maximum session length in seconds.")
    parser.add_argument("--min-area", type=int, default=12000, help="Minimum contour area for a card candidate.")
    parser.add_argument("--min-confidence", type=float, default=0.70, help="Minimum rank and suit confidence to accept.")
    parser.add_argument(
        "--same-card-min-shift",
        type=float,
        default=25.0,
        help="Minimum contour-center movement needed to accept the same predicted card again.",
    )
    parser.add_argument(
        "--same-card-area-ratio",
        type=float,
        default=0.08,
        help="Minimum relative contour-area change needed to accept the same predicted card again.",
    )
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
        default=str(Path(__file__).with_name("chain_captures")),
        help="Directory for accepted raw/warped images and session log.",
    )
    parser.add_argument("--keep-attempts", action="store_true", help="Keep every attempted raw frame.")
    return parser.parse_args()


def quad_center(quad: np.ndarray) -> tuple[float, float]:
    center = quad.astype(np.float32).mean(axis=0)
    return float(center[0]), float(center[1])


def placement_changed(
    last_center: tuple[float, float] | None,
    last_area: float | None,
    center: tuple[float, float],
    area: float,
    min_shift: float,
    min_area_ratio: float,
) -> bool:
    if last_center is None or last_area is None:
        return True

    shift = float(np.hypot(center[0] - last_center[0], center[1] - last_center[1]))
    area_delta_ratio = abs(area - last_area) / max(last_area, 1.0)
    return shift >= min_shift or area_delta_ratio >= min_area_ratio


def capture_still(output_path: Path, camera: int, width: int, height: int, timeout_ms: int) -> bool:
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
        print(f"capture failed: {message or result.returncode}", file=sys.stderr)
        return False
    return True


def append_log(log_path: Path, row: list[str]) -> None:
    write_header = not log_path.exists()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        if write_header:
            writer.writerow(
                [
                    "timestamp",
                    "position",
                    "card",
                    "rank_confidence",
                    "suit_confidence",
                    "contour_area",
                    "center_x",
                    "center_y",
                    "raw_path",
                    "warped_path",
                ]
            )
        writer.writerow(row)


def main() -> int:
    args = parse_args()
    if args.count < 1:
        print("--count must be positive", file=sys.stderr)
        return 1

    output_dir = Path(args.output_dir)
    session_id = time.strftime("%Y%m%d_%H%M%S")
    session_dir = output_dir / session_id
    attempts_dir = session_dir / "attempts"
    log_path = session_dir / "chain_log.csv"
    current_path = session_dir / "current.jpg"

    device = resolve_device(args.device)
    rank_model, rank_id_to_label, rank_target = load_checkpoint(Path(args.rank_model), device)
    suit_model, suit_id_to_label, suit_target = load_checkpoint(Path(args.suit_model), device)
    if rank_target != "rank" or suit_target != "suit":
        print("Checkpoint targets do not match expected rank/suit models.", file=sys.stderr)
        return 1

    accepted: list[str] = []
    last_accepted_center: tuple[float, float] | None = None
    last_accepted_area: float | None = None
    candidate = ""
    candidate_count = 0
    started = time.monotonic()

    print(f"Session: {session_id}")
    print(f"Target count: {args.count}")
    print("Place the first card under the camera. Accepted cards will print as they stabilize.")
    print()

    while len(accepted) < args.count:
        elapsed = time.monotonic() - started
        if elapsed > args.session_timeout:
            print("Session timed out.", file=sys.stderr)
            break

        attempt_stamp = time.strftime("%Y%m%d_%H%M%S")
        if not capture_still(current_path, args.camera, args.width, args.height, args.capture_timeout_ms):
            time.sleep(args.interval)
            continue

        frame = cv2.imread(str(current_path))
        if frame is None:
            print("captured image unreadable")
            time.sleep(args.interval)
            continue

        quad, _contour, area = find_card_quad(frame, args.min_area)
        if quad is None:
            print("no card")
            candidate = ""
            candidate_count = 0
            time.sleep(args.interval)
            continue

        warped = warp_card(frame, quad)
        center = quad_center(quad)
        _rank, rank_confidence, _suit, suit_confidence, predicted = predict_card(
            warped, rank_model, rank_id_to_label, suit_model, suit_id_to_label, device
        )

        confidence_ok = rank_confidence >= args.min_confidence and suit_confidence >= args.min_confidence
        if predicted == candidate and confidence_ok:
            candidate_count += 1
        else:
            candidate = predicted if confidence_ok else ""
            candidate_count = 1 if confidence_ok else 0

        print(
            f"seen={predicted} rank={rank_confidence:.3f} suit={suit_confidence:.3f} "
            f"area={area:.0f} center=({center[0]:.0f},{center[1]:.0f}) stable={candidate_count}/{args.stable_reads}"
        )

        if args.keep_attempts:
            attempts_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(current_path, attempts_dir / f"{attempt_stamp}_{predicted}_raw.jpg")

        can_accept_same_prediction = placement_changed(
            last_accepted_center,
            last_accepted_area,
            center,
            area,
            args.same_card_min_shift,
            args.same_card_area_ratio,
        )

        if confidence_ok and candidate_count >= args.stable_reads and (
            not accepted or predicted != accepted[-1] or can_accept_same_prediction
        ):
            position = len(accepted) + 1
            raw_path = session_dir / f"{position:02d}_{predicted}_raw.jpg"
            warped_path = session_dir / f"{position:02d}_{predicted}_warped.jpg"
            shutil.copy2(current_path, raw_path)
            cv2.imwrite(str(warped_path), warped)
            append_log(
                log_path,
                [
                    attempt_stamp,
                    str(position),
                    predicted,
                    f"{rank_confidence:.4f}",
                    f"{suit_confidence:.4f}",
                    f"{area:.1f}",
                    f"{center[0]:.1f}",
                    f"{center[1]:.1f}",
                    str(raw_path),
                    str(warped_path),
                ],
            )
            accepted.append(predicted)
            last_accepted_center = center
            last_accepted_area = area
            print(f"ACCEPTED {position}/{args.count}: {predicted}")
            print("Place the next card so its visible contour shifts from the previous one.")
            print()
            candidate = ""
            candidate_count = 0
            time.sleep(args.cooldown)
        else:
            time.sleep(args.interval)

    current_path.unlink(missing_ok=True)

    print()
    if accepted:
        print(f"Recognized order: {' '.join(accepted)}")
        print(f"Log: {log_path}")
    else:
        print("No cards accepted.")
    return 0 if len(accepted) == args.count else 2


if __name__ == "__main__":
    raise SystemExit(main())
