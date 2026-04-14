import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

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
RANK_ROI = (18, 16, 92, 132)
SUIT_ROI = (18, 126, 92, 96)


@dataclass
class TemplateMatch:
    label: str
    score: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Single-card detection and identification lab.")
    parser.add_argument("--camera", type=int, default=0, help="Camera index for cv2.VideoCapture.")
    parser.add_argument("--backend", default="auto", help="Camera backend: auto, opencv, or rpicam.")
    parser.add_argument("--width", type=int, default=640, help="Requested capture width.")
    parser.add_argument("--height", type=int, default=480, help="Requested capture height.")
    parser.add_argument("--fps", type=int, default=15, help="Requested capture FPS.")
    parser.add_argument("--min-area", type=int, default=12000, help="Minimum contour area for a card candidate.")
    parser.add_argument("--debug", action="store_true", help="Show debug windows at startup.")
    parser.add_argument(
        "--captures-dir",
        default=str(Path(__file__).with_name("captures")),
        help="Directory used for saved samples.",
    )
    parser.add_argument(
        "--templates-dir",
        default=str(Path(__file__).with_name("templates")),
        help="Directory containing rank/suit templates.",
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


def find_card_quad(frame: np.ndarray, min_area: int) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
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

    return best_quad, best_contour


def normalize_patch(patch: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    scaled = cv2.resize(gray, (96, 96), interpolation=cv2.INTER_AREA)
    return cv2.threshold(scaled, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]


def load_templates(template_root: Path) -> Dict[str, Dict[str, np.ndarray]]:
    loaded: Dict[str, Dict[str, np.ndarray]] = {"ranks": {}, "suits": {}}
    for group in ("ranks", "suits"):
        group_dir = template_root / group
        if not group_dir.exists():
            continue
        for file_path in sorted(group_dir.glob("*.png")):
            image = cv2.imread(str(file_path), cv2.IMREAD_GRAYSCALE)
            if image is None:
                continue
            normalized = cv2.resize(image, (96, 96), interpolation=cv2.INTER_AREA)
            loaded[group][file_path.stem.upper()] = cv2.threshold(
                normalized, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )[1]
    return loaded


def best_template_match(patch: np.ndarray, templates: Dict[str, np.ndarray]) -> Optional[TemplateMatch]:
    if not templates:
        return None

    best_label = ""
    best_score = -1.0
    for label, template in templates.items():
        result = cv2.matchTemplate(patch, template, cv2.TM_CCOEFF_NORMED)
        score = float(result[0][0])
        if score > best_score:
            best_label = label
            best_score = score

    return TemplateMatch(best_label, best_score)


def identify_card(warped: np.ndarray, templates: Dict[str, Dict[str, np.ndarray]]) -> Dict[str, Optional[TemplateMatch]]:
    rx, ry, rw, rh = RANK_ROI
    sx, sy, sw, sh = SUIT_ROI

    rank_patch = normalize_patch(warped[ry : ry + rh, rx : rx + rw])
    suit_patch = normalize_patch(warped[sy : sy + sh, sx : sx + sw])

    return {
        "rank": best_template_match(rank_patch, templates["ranks"]),
        "suit": best_template_match(suit_patch, templates["suits"]),
    }


def make_status_text(matches: Dict[str, Optional[TemplateMatch]]) -> str:
    rank = matches["rank"]
    suit = matches["suit"]
    if rank is None or suit is None:
        return "templates missing"
    return f"{rank.label}{suit.label}  rank:{rank.score:.2f} suit:{suit.score:.2f}"


def draw_overlay(
    frame: np.ndarray,
    quad: Optional[np.ndarray],
    contour: Optional[np.ndarray],
    status_text: str,
    overlay_enabled: bool,
) -> np.ndarray:
    canvas = frame.copy()
    if overlay_enabled and contour is not None:
        cv2.drawContours(canvas, [contour], -1, (0, 255, 255), 2)
    if overlay_enabled and quad is not None:
        cv2.polylines(canvas, [quad.astype(np.int32)], True, (0, 255, 0), 3)
    cv2.putText(canvas, status_text, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 220, 0), 2, cv2.LINE_AA)
    return canvas


def draw_detection_debug(warped: np.ndarray) -> np.ndarray:
    canvas = warped.copy()
    rx, ry, rw, rh = RANK_ROI
    sx, sy, sw, sh = SUIT_ROI
    cv2.rectangle(canvas, (rx, ry), (rx + rw, ry + rh), (0, 255, 0), 2)
    cv2.rectangle(canvas, (sx, sy), (sx + sw, sy + sh), (255, 0, 0), 2)
    return canvas


def save_capture(captures_dir: Path, frame: np.ndarray, warped: Optional[np.ndarray]) -> Tuple[Path, Optional[Path]]:
    captures_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    raw_path = captures_dir / f"{stamp}_raw.jpg"
    warped_path = captures_dir / f"{stamp}_warped.jpg"
    cv2.imwrite(str(raw_path), frame)
    if warped is not None:
        cv2.imwrite(str(warped_path), warped)
        return raw_path, warped_path
    return raw_path, None


def main() -> int:
    args = parse_args()
    templates = load_templates(Path(args.templates_dir))
    captures_dir = Path(args.captures_dir)

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

    overlay_enabled = True
    debug_enabled = args.debug

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Camera read failed.")
                return 1

            quad, contour = find_card_quad(frame, args.min_area)
            warped = warp_card(frame, quad) if quad is not None else None
            matches = identify_card(warped, templates) if warped is not None else {"rank": None, "suit": None}

            if warped is None:
                status_text = "no card found"
            else:
                status_text = make_status_text(matches)

            preview = draw_overlay(frame, quad, contour, status_text, overlay_enabled)
            cv2.imshow("card_identifier_lab", preview)

            if debug_enabled:
                _, edges = preprocess(frame)
                cv2.imshow("card_identifier_edges", edges)
                if warped is not None:
                    cv2.imshow("card_identifier_warped", draw_detection_debug(warped))

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("o"):
                overlay_enabled = not overlay_enabled
            if key == ord("d"):
                debug_enabled = not debug_enabled
                if not debug_enabled:
                    cv2.destroyWindow("card_identifier_edges")
                    if cv2.getWindowProperty("card_identifier_warped", cv2.WND_PROP_VISIBLE) >= 0:
                        cv2.destroyWindow("card_identifier_warped")
            if key == ord("s"):
                raw_path, warped_path = save_capture(captures_dir, frame, warped)
                if warped_path is None:
                    print(f"Saved raw frame: {raw_path}")
                else:
                    print(f"Saved raw frame: {raw_path}")
                    print(f"Saved warped card: {warped_path}")
    finally:
        cap.release()
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
