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
    parser = argparse.ArgumentParser(description="Recognize cards added one-by-one to an overlapping pile.")
    parser.add_argument("--count", type=int, default=4, help="Number of cards to accept before exiting.")
    parser.add_argument("--camera", type=int, default=0, help="Camera index.")
    parser.add_argument("--width", type=int, default=640, help="Capture width.")
    parser.add_argument("--height", type=int, default=480, help="Capture height.")
    parser.add_argument("--capture-timeout-ms", type=int, default=500, help="Still capture warmup timeout.")
    parser.add_argument("--interval", type=float, default=0.8, help="Seconds between attempts.")
    parser.add_argument("--cooldown", type=float, default=1.0, help="Seconds after accepting before baseline refresh.")
    parser.add_argument("--stable-reads", type=int, default=2, help="Same prediction count needed for acceptance.")
    parser.add_argument("--session-timeout", type=float, default=240.0, help="Maximum session length in seconds.")
    parser.add_argument("--min-area", type=int, default=8000, help="Minimum contour area for a card candidate.")
    parser.add_argument("--min-confidence", type=float, default=0.55, help="Minimum rank and suit confidence to accept.")
    parser.add_argument("--min-change-area", type=int, default=2500, help="Minimum changed-pixel area to trigger recognition.")
    parser.add_argument("--diff-threshold", type=int, default=22, help="Frame difference threshold.")
    parser.add_argument(
        "--candidate-source",
        choices=("full", "changed"),
        default="full",
        help="Use frame difference only as a trigger, then recognize the full-frame top contour, or recognize inside the changed region.",
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
        default=str(Path(__file__).with_name("diff_chain_captures")),
        help="Directory for accepted raw/warped/diff images and session log.",
    )
    parser.add_argument("--keep-attempts", action="store_true", help="Keep every attempted raw frame.")
    return parser.parse_args()


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


def changed_mask(previous: np.ndarray, current: np.ndarray, threshold: int) -> np.ndarray:
    prev_gray = cv2.cvtColor(previous, cv2.COLOR_BGR2GRAY)
    curr_gray = cv2.cvtColor(current, cv2.COLOR_BGR2GRAY)
    diff = cv2.absdiff(prev_gray, curr_gray)
    diff = cv2.GaussianBlur(diff, (5, 5), 0)
    mask = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)[1]
    kernel = np.ones((5, 5), dtype=np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.dilate(mask, kernel, iterations=3)
    return mask


def mask_change_area(mask: np.ndarray) -> int:
    return int((mask > 0).sum())


def find_changed_card_quad(
    frame: np.ndarray,
    mask: np.ndarray,
    min_area: int,
    candidate_source: str,
) -> tuple[np.ndarray | None, float, str]:
    if candidate_source == "full":
        quad, _contour, area = find_card_quad(frame, min_area)
        if quad is not None:
            return quad, area, "full"

    masked = frame.copy()
    masked[mask == 0] = 255
    quad, _contour, area = find_card_quad(masked, min_area)
    if quad is not None:
        return quad, area, "changed"

    quad, _contour, area = find_card_quad(frame, min_area)
    return quad, area, "full"


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
                    "change_area",
                    "candidate_source",
                    "raw_path",
                    "warped_path",
                    "diff_path",
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
    current_path = session_dir / "current.jpg"
    baseline_path = session_dir / "baseline.jpg"
    log_path = session_dir / "diff_chain_log.csv"

    device = resolve_device(args.device)
    rank_model, rank_id_to_label, rank_target = load_checkpoint(Path(args.rank_model), device)
    suit_model, suit_id_to_label, suit_target = load_checkpoint(Path(args.suit_model), device)
    if rank_target != "rank" or suit_target != "suit":
        print("Checkpoint targets do not match expected rank/suit models.", file=sys.stderr)
        return 1

    print(f"Session: {session_id}")
    print(f"Target count: {args.count}")
    print("Keep the current pile still. Place each new card on top/overlapping when prompted.")
    print("Capturing initial baseline...")
    if not capture_still(baseline_path, args.camera, args.width, args.height, args.capture_timeout_ms):
        return 1
    baseline = cv2.imread(str(baseline_path))
    if baseline is None:
        print("baseline image unreadable", file=sys.stderr)
        return 1

    accepted: list[str] = []
    candidate = ""
    candidate_count = 0
    started = time.monotonic()
    print("Place card 1.")
    print()

    while len(accepted) < args.count:
        if time.monotonic() - started > args.session_timeout:
            print("Session timed out.", file=sys.stderr)
            break

        attempt_stamp = time.strftime("%Y%m%d_%H%M%S")
        if not capture_still(current_path, args.camera, args.width, args.height, args.capture_timeout_ms):
            time.sleep(args.interval)
            continue
        frame = cv2.imread(str(current_path))
        if frame is None:
            time.sleep(args.interval)
            continue

        mask = changed_mask(baseline, frame, args.diff_threshold)
        change_area = mask_change_area(mask)
        if change_area < args.min_change_area:
            print(f"waiting change area={change_area}")
            candidate = ""
            candidate_count = 0
            time.sleep(args.interval)
            continue

        quad, contour_area, candidate_source = find_changed_card_quad(
            frame,
            mask,
            args.min_area,
            args.candidate_source,
        )
        if quad is None:
            print(f"change seen but no card area={change_area}")
            candidate = ""
            candidate_count = 0
            time.sleep(args.interval)
            continue

        warped = warp_card(frame, quad)
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
            f"contour={contour_area:.0f} change={change_area} source={candidate_source} "
            f"stable={candidate_count}/{args.stable_reads}"
        )

        if args.keep_attempts:
            attempts_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(current_path, attempts_dir / f"{attempt_stamp}_{predicted}_raw.jpg")

        if confidence_ok and candidate_count >= args.stable_reads:
            position = len(accepted) + 1
            raw_path = session_dir / f"{position:02d}_{predicted}_raw.jpg"
            warped_path = session_dir / f"{position:02d}_{predicted}_warped.jpg"
            diff_path = session_dir / f"{position:02d}_{predicted}_diff.jpg"
            shutil.copy2(current_path, raw_path)
            cv2.imwrite(str(warped_path), warped)
            cv2.imwrite(str(diff_path), mask)
            append_log(
                log_path,
                [
                    attempt_stamp,
                    str(position),
                    predicted,
                    f"{rank_confidence:.4f}",
                    f"{suit_confidence:.4f}",
                    f"{contour_area:.1f}",
                    str(change_area),
                    candidate_source,
                    str(raw_path),
                    str(warped_path),
                    str(diff_path),
                ],
            )
            accepted.append(predicted)
            print(f"ACCEPTED {position}/{args.count}: {predicted}")
            print()

            time.sleep(args.cooldown)
            baseline = frame
            shutil.copy2(current_path, baseline_path)
            candidate = ""
            candidate_count = 0
            if len(accepted) < args.count:
                print(f"Place card {len(accepted) + 1}.")
                print()
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
