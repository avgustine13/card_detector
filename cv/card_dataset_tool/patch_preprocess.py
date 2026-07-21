from __future__ import annotations

import cv2
import numpy as np


RANK_ROI = (0, 0, 128, 160)
SUIT_ROI = (0, 88, 150, 180)
PATCH_SIZE = (96, 96)


def extract_roi(image: np.ndarray, target: str) -> np.ndarray:
    x, y, w, h = RANK_ROI if target == "rank" else SUIT_ROI
    return image[y : y + h, x : x + w]


def corner_ink_score(patch: np.ndarray) -> float:
    gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    return float((255 - gray).mean())


def orient_card_to_corner(image: np.ndarray) -> np.ndarray:
    best_image = image
    best_score = -1.0
    for rotation in range(4):
        rotated = np.ascontiguousarray(np.rot90(image, rotation))
        score = corner_ink_score(extract_roi(rotated, "rank")) + corner_ink_score(extract_roi(rotated, "suit"))
        if score > best_score:
            best_score = score
            best_image = rotated
    return best_image


def normalize_patch_image(patch: np.ndarray, target: str) -> np.ndarray:
    scaled = cv2.resize(patch, PATCH_SIZE, interpolation=cv2.INTER_AREA)
    if target == "suit":
        return scaled

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

    b, g, r = cv2.split(patch)
    ink_mask = np.minimum.reduce([b, g, r]) < 215
    if not np.any(ink_mask):
        return "unknown"

    red_pixels = ink_mask & (r > 80) & ((r.astype(np.int16) - np.maximum(b, g).astype(np.int16)) > 25)
    dark_pixels = ink_mask & (np.maximum.reduce([b, g, r]) < 120)
    red_ratio = float(red_pixels.sum()) / float(ink_mask.sum())
    dark_ratio = float(dark_pixels.sum()) / float(ink_mask.sum())

    if red_ratio > 0.18:
        return "red"
    if dark_ratio > 0.35:
        return "black"
    return "unknown"
