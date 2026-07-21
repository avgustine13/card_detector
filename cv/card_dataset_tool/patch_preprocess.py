from __future__ import annotations

import cv2
import numpy as np


RANK_ROI = (0, 0, 128, 160)
SUIT_ROI = (0, 88, 150, 180)
ORIENTATION_SUIT_ROI = (0, 96, 128, 128)
PATCH_SIZE = (96, 96)


def extract_roi(image: np.ndarray, target: str) -> np.ndarray:
    x, y, w, h = RANK_ROI if target == "rank" else SUIT_ROI
    return image[y : y + h, x : x + w]


def extract_orientation_suit_roi(image: np.ndarray) -> np.ndarray:
    x, y, w, h = ORIENTATION_SUIT_ROI
    return image[y : y + h, x : x + w]


def corner_ink_score(patch: np.ndarray) -> float:
    gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    return float((255 - gray).mean())


def orient_card_to_corner(image: np.ndarray) -> np.ndarray:
    best_image = image
    best_score = -1.0
    for rotation in range(4):
        rotated = np.ascontiguousarray(np.rot90(image, rotation))
        score = corner_ink_score(extract_roi(rotated, "rank")) + corner_ink_score(extract_orientation_suit_roi(rotated))
        if score > best_score:
            best_score = score
            best_image = rotated
    return best_image


def normalize_patch_image(patch: np.ndarray, target: str) -> np.ndarray:
    scaled = cv2.resize(patch, PATCH_SIZE, interpolation=cv2.INTER_AREA)
    if target == "suit":
        gray = cv2.cvtColor(scaled, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        hsv = cv2.cvtColor(scaled, cv2.COLOR_BGR2HSV)
        hue, saturation, value = cv2.split(hsv)
        red_mask = (((hue <= 12) | (hue >= 165)) & (saturation >= 45) & (value <= 245)).astype(np.uint8) * 255
        black_mask = ((value <= 135) & (saturation <= 105)).astype(np.uint8) * 255
        return np.dstack([gray, red_mask, black_mask])

    gray = cv2.cvtColor(scaled, cv2.COLOR_BGR2GRAY)
    return cv2.equalizeHist(gray)


def normalize_patch_feature(patch: np.ndarray, target: str) -> np.ndarray:
    return normalize_patch_image(patch, target).astype(np.float32).reshape(-1) / 255.0


def patch_channel_count(target: str) -> int:
    return 3 if target == "suit" else 1


def patch_to_tensor_array(patch: np.ndarray) -> np.ndarray:
    if patch.ndim == 2:
        return patch.astype(np.float32)[None, :, :] / 255.0
    return np.transpose(patch.astype(np.float32), (2, 0, 1)) / 255.0


def suit_color_group(patch: np.ndarray) -> str:
    if patch.ndim != 3:
        return "unknown"

    if patch.shape[2] >= 3:
        red_count = int((patch[:, :, 1] > 0).sum())
        black_count = int((patch[:, :, 2] > 0).sum())
    else:
        red_count = 0
        black_count = 0

    if red_count < 20 and black_count < 20:
        return "unknown"

    if red_count >= max(35, int(black_count * 0.35)):
        return "red"
    if black_count >= 35:
        return "black"
    return "unknown"
