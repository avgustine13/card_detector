from __future__ import annotations

import cv2
import numpy as np


RANK_ROI = (0, 0, 128, 160)
SUIT_ROI = (0, 96, 128, 128)
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
    del target
    gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    scaled = cv2.resize(gray, PATCH_SIZE, interpolation=cv2.INTER_AREA)
    return cv2.equalizeHist(scaled)


def normalize_patch_feature(patch: np.ndarray, target: str) -> np.ndarray:
    return normalize_patch_image(patch, target).astype(np.float32).reshape(-1) / 255.0
