from __future__ import annotations

import argparse
import csv
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cv.card_dataset_tool.app import CARD_HEIGHT, CARD_WIDTH
from cv.card_dataset_tool.cnn_common import load_checkpoint
from cv.card_dataset_tool.identify_card_diff_chain import capture_still, changed_mask, mask_change_area
from cv.card_dataset_tool.live_patch_cnn import predict_card, resolve_device


CORNER_WIDTH = 150
CORNER_HEIGHT = 268


@dataclass(frozen=True)
class IndexPrediction:
    card: str
    rank_confidence: float
    suit_confidence: float
    score: float
    rect: tuple[int, int, int, int]
    rotation: int
    corner_image: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Recognize overlapped cards by classifying the newly changed index corner."
    )
    parser.add_argument("--count", type=int, default=4, help="Number of cards to accept before exiting.")
    parser.add_argument("--camera", type=int, default=0, help="Camera index.")
    parser.add_argument("--width", type=int, default=640, help="Capture width.")
    parser.add_argument("--height", type=int, default=480, help="Capture height.")
    parser.add_argument("--capture-timeout-ms", type=int, default=500, help="Still capture warmup timeout.")
    parser.add_argument("--interval", type=float, default=0.8, help="Seconds between attempts.")
    parser.add_argument("--cooldown", type=float, default=1.0, help="Seconds after accepting before baseline refresh.")
    parser.add_argument("--stable-reads", type=int, default=2, help="Same prediction count needed for acceptance.")
    parser.add_argument("--session-timeout", type=float, default=240.0, help="Maximum session length in seconds.")
    parser.add_argument("--min-confidence", type=float, default=0.50, help="Minimum rank and suit confidence to accept.")
    parser.add_argument("--min-score", type=float, default=0.35, help="Minimum combined index-corner score to accept.")
    parser.add_argument("--min-change-area", type=int, default=1800, help="Minimum changed-pixel area to trigger recognition.")
    parser.add_argument("--diff-threshold", type=int, default=22, help="Frame difference threshold.")
    parser.add_argument("--max-candidates", type=int, default=12, help="Maximum changed ink groups to classify per frame.")
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
        default=str(Path(__file__).with_name("index_chain_captures")),
        help="Directory for accepted raw/corner/diff images and session log.",
    )
    parser.add_argument("--keep-attempts", action="store_true", help="Keep every attempted raw frame.")
    return parser.parse_args()


def card_ink_mask(frame: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hue, saturation, value = cv2.split(hsv)
    card_stock = ((gray >= 145) & (saturation <= 95)).astype(np.uint8) * 255
    card_stock = cv2.dilate(card_stock, np.ones((17, 17), dtype=np.uint8), iterations=1)
    black = ((gray < 135) & (saturation < 125)).astype(np.uint8) * 255
    red = (((hue <= 12) | (hue >= 165)) & (saturation >= 45) & (value <= 245)).astype(np.uint8) * 255
    mask = cv2.bitwise_or(black, red)
    mask = cv2.bitwise_and(mask, card_stock)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), dtype=np.uint8), iterations=1)
    return mask


def clamp_rect(x: int, y: int, w: int, h: int, shape: tuple[int, ...]) -> tuple[int, int, int, int] | None:
    frame_h, frame_w = shape[:2]
    x0 = max(0, x)
    y0 = max(0, y)
    x1 = min(frame_w, x + w)
    y1 = min(frame_h, y + h)
    if x1 - x0 < 24 or y1 - y0 < 36:
        return None
    return x0, y0, x1 - x0, y1 - y0


def expand_to_corner_rect(x: int, y: int, w: int, h: int, shape: tuple[int, ...]) -> tuple[int, int, int, int] | None:
    center_x = x + (w / 2.0)
    center_y = y + (h / 2.0)
    target_h = max(72.0, h * 2.6, w * 1.8)
    target_w = max(48.0, target_h * (CORNER_WIDTH / CORNER_HEIGHT), w * 1.7)
    target_h = max(target_h, target_w * (CORNER_HEIGHT / CORNER_WIDTH))
    target_h = min(target_h, 260.0)
    target_w = min(target_w, 170.0)
    return clamp_rect(
        int(round(center_x - target_w / 2.0)),
        int(round(center_y - target_h / 2.0)),
        int(round(target_w)),
        int(round(target_h)),
        shape,
    )


def find_index_candidate_rects(frame: np.ndarray, diff_mask: np.ndarray, max_candidates: int) -> list[tuple[int, int, int, int]]:
    ink = card_ink_mask(frame)
    changed = cv2.dilate(diff_mask, np.ones((21, 21), dtype=np.uint8), iterations=1)
    changed_ink = cv2.bitwise_and(ink, changed)
    grouped = cv2.dilate(changed_ink, np.ones((23, 23), dtype=np.uint8), iterations=1)
    grouped = cv2.morphologyEx(grouped, cv2.MORPH_CLOSE, np.ones((9, 9), dtype=np.uint8), iterations=1)
    contours, _ = cv2.findContours(grouped, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    rects: list[tuple[int, int, int, int]] = []
    ranked: list[tuple[float, tuple[int, int, int, int]]] = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 40:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        if w > frame.shape[1] * 0.55 or h > frame.shape[0] * 0.65:
            continue
        rect = expand_to_corner_rect(x, y, w, h, frame.shape)
        if rect is None:
            continue
        _rx, _ry, rw, rh = rect
        if rw > 190 or rh > 285:
            continue
        rx, ry, rw, rh = rect
        changed_pixels = int((diff_mask[ry : ry + rh, rx : rx + rw] > 0).sum())
        ranked.append((float(changed_pixels + area), rect))

    for _score, rect in sorted(ranked, reverse=True):
        if not any(rect_overlap_ratio(rect, existing) > 0.65 for existing in rects):
            rects.append(rect)
        if len(rects) >= max_candidates:
            break
    return rects


def rect_overlap_ratio(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    x0 = max(ax, bx)
    y0 = max(ay, by)
    x1 = min(ax + aw, bx + bw)
    y1 = min(ay + ah, by + bh)
    intersection = max(0, x1 - x0) * max(0, y1 - y0)
    smaller = min(aw * ah, bw * bh)
    return intersection / smaller if smaller else 0.0


def rotate_image(image: np.ndarray, rotation: int) -> np.ndarray:
    if rotation == 0:
        return image
    if rotation == 90:
        return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    if rotation == 180:
        return cv2.rotate(image, cv2.ROTATE_180)
    if rotation == 270:
        return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    raise ValueError(f"unsupported rotation: {rotation}")


def corner_crop_to_card(crop: np.ndarray, rotation: int) -> np.ndarray:
    rotated = rotate_image(crop, rotation)
    corner = cv2.resize(rotated, (CORNER_WIDTH, CORNER_HEIGHT), interpolation=cv2.INTER_AREA)
    card = np.full((CARD_HEIGHT, CARD_WIDTH, 3), 255, dtype=np.uint8)
    card[:CORNER_HEIGHT, :CORNER_WIDTH] = corner
    return card


def corner_size_prior(rect: tuple[int, int, int, int]) -> float:
    _x, _y, width, height = rect
    width_prior = min(1.0, 90.0 / max(1.0, float(width)))
    height_prior = min(1.0, 135.0 / max(1.0, float(height)))
    return max(0.35, width_prior * height_prior)


def suit_color_consistent(corner: np.ndarray, suit: str) -> bool:
    hsv = cv2.cvtColor(corner, cv2.COLOR_BGR2HSV)
    hue, saturation, value = cv2.split(hsv)
    red_count = int(((((hue <= 12) | (hue >= 165)) & (saturation >= 45) & (value <= 245))).sum())
    black_count = int((((value <= 145) & (saturation <= 135))).sum())
    if suit in ("D", "H"):
        return red_count >= max(35, int(black_count * 0.25))
    if suit in ("C", "S"):
        return black_count >= 35 and red_count <= max(80, int(black_count * 0.75))
    return True


def predict_best_index(
    frame: np.ndarray,
    diff_mask: np.ndarray,
    rank_model,
    rank_id_to_label: dict[int, str],
    suit_model,
    suit_id_to_label: dict[int, str],
    device,
    max_candidates: int,
) -> IndexPrediction | None:
    best: IndexPrediction | None = None
    for rect in find_index_candidate_rects(frame, diff_mask, max_candidates):
        x, y, w, h = rect
        crop = frame[y : y + h, x : x + w]
        for rotation in (0, 90, 180, 270):
            corner_card = corner_crop_to_card(crop, rotation)
            _rank, rank_confidence, suit, suit_confidence, card = predict_card(
                corner_card,
                rank_model,
                rank_id_to_label,
                suit_model,
                suit_id_to_label,
                device,
            )
            if not suit_color_consistent(corner_card[:CORNER_HEIGHT, :CORNER_WIDTH], suit):
                continue
            score = (
                min(rank_confidence, suit_confidence)
                * ((rank_confidence + suit_confidence) / 2.0)
                * corner_size_prior(rect)
            )
            prediction = IndexPrediction(
                card=card,
                rank_confidence=rank_confidence,
                suit_confidence=suit_confidence,
                score=score,
                rect=rect,
                rotation=rotation,
                corner_image=corner_card[:CORNER_HEIGHT, :CORNER_WIDTH].copy(),
            )
            if best is None or prediction.score > best.score:
                best = prediction
    return best


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
                    "score",
                    "change_area",
                    "rect",
                    "rotation",
                    "raw_path",
                    "corner_path",
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
    log_path = session_dir / "index_chain_log.csv"

    device = resolve_device(args.device)
    rank_model, rank_id_to_label, rank_target = load_checkpoint(Path(args.rank_model), device)
    suit_model, suit_id_to_label, suit_target = load_checkpoint(Path(args.suit_model), device)
    if rank_target != "rank" or suit_target != "suit":
        print("Checkpoint targets do not match expected rank/suit models.", file=sys.stderr)
        return 1

    print(f"Session: {session_id}")
    print(f"Target count: {args.count}")
    print("Keep the current pile still. Place each new card with at least one index corner visible.")
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

        diff_mask = changed_mask(baseline, frame, args.diff_threshold)
        change_area = mask_change_area(diff_mask)
        if change_area < args.min_change_area:
            print(f"waiting change area={change_area}")
            candidate = ""
            candidate_count = 0
            time.sleep(args.interval)
            continue

        prediction = predict_best_index(
            frame,
            diff_mask,
            rank_model,
            rank_id_to_label,
            suit_model,
            suit_id_to_label,
            device,
            args.max_candidates,
        )
        if prediction is None:
            print(f"change seen but no changed index ink area={change_area}")
            candidate = ""
            candidate_count = 0
            time.sleep(args.interval)
            continue

        confidence_ok = (
            prediction.rank_confidence >= args.min_confidence
            and prediction.suit_confidence >= args.min_confidence
            and prediction.score >= args.min_score
        )
        if prediction.card == candidate and confidence_ok:
            candidate_count += 1
        else:
            candidate = prediction.card if confidence_ok else ""
            candidate_count = 1 if confidence_ok else 0

        print(
            f"seen={prediction.card} rank={prediction.rank_confidence:.3f} "
            f"suit={prediction.suit_confidence:.3f} score={prediction.score:.3f} "
            f"change={change_area} rect={prediction.rect} rot={prediction.rotation} "
            f"stable={candidate_count}/{args.stable_reads}"
        )

        if args.keep_attempts:
            attempts_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(current_path, attempts_dir / f"{attempt_stamp}_{prediction.card}_raw.jpg")

        if confidence_ok and candidate_count >= args.stable_reads:
            position = len(accepted) + 1
            raw_path = session_dir / f"{position:02d}_{prediction.card}_raw.jpg"
            corner_path = session_dir / f"{position:02d}_{prediction.card}_corner.jpg"
            diff_path = session_dir / f"{position:02d}_{prediction.card}_diff.jpg"
            shutil.copy2(current_path, raw_path)
            cv2.imwrite(str(corner_path), prediction.corner_image)
            cv2.imwrite(str(diff_path), diff_mask)
            append_log(
                log_path,
                [
                    attempt_stamp,
                    str(position),
                    prediction.card,
                    f"{prediction.rank_confidence:.4f}",
                    f"{prediction.suit_confidence:.4f}",
                    f"{prediction.score:.4f}",
                    str(change_area),
                    " ".join(str(value) for value in prediction.rect),
                    str(prediction.rotation),
                    str(raw_path),
                    str(corner_path),
                    str(diff_path),
                ],
            )
            accepted.append(prediction.card)
            print(f"ACCEPTED {position}/{args.count}: {prediction.card}")
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
