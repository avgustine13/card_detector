from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import cv2
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset

from cv.card_dataset_tool.dataset_meta import include_image_path, load_meta_index
from cv.card_dataset_tool.patch_preprocess import PATCH_SIZE, extract_roi, normalize_patch_image, orient_card_to_corner


@dataclass(frozen=True)
class PatchSample:
    label: str
    rank: str
    suit: str
    path: Path
    rank_patch: np.ndarray
    suit_patch: np.ndarray


def split_label(label: str) -> Tuple[str, str]:
    return label[:-1], label[-1]


def normalized_patch_image(patch: np.ndarray) -> np.ndarray:
    return normalize_patch_image(patch, "rank")


def load_sample(image_path: Path, label: str) -> PatchSample | None:
    image = cv2.imread(str(image_path))
    if image is None:
        return None

    oriented = orient_card_to_corner(image)
    rank, suit = split_label(label)
    rank_patch = normalize_patch_image(extract_roi(oriented, "rank"), "rank")
    suit_patch = normalize_patch_image(extract_roi(oriented, "suit"), "suit")
    return PatchSample(label, rank, suit, image_path, rank_patch, suit_patch)


def load_grouped_dataset(dataset_dir: Path, min_contour_area: float = 0.0) -> Dict[str, List[PatchSample]]:
    warped_root = dataset_dir / "warped"
    grouped: Dict[str, List[PatchSample]] = defaultdict(list)
    meta_index = load_meta_index(dataset_dir)

    for label_dir in sorted(path for path in warped_root.iterdir() if path.is_dir()):
        label = label_dir.name.upper()
        for image_path in sorted(label_dir.glob("*_warped.jpg")):
            if not include_image_path(dataset_dir, image_path, meta_index, min_contour_area):
                continue
            sample = load_sample(image_path, label)
            if sample is not None:
                grouped[label].append(sample)
    return grouped


def split_grouped_dataset(
    grouped: Dict[str, List[PatchSample]], test_per_label: int, seed: int
) -> Tuple[List[PatchSample], List[PatchSample], List[str]]:
    train_samples: List[PatchSample] = []
    test_samples: List[PatchSample] = []
    skipped_labels: List[str] = []
    rng = np.random.default_rng(seed)

    for label in sorted(grouped):
        samples = list(grouped[label])
        if len(samples) <= test_per_label:
            skipped_labels.append(label)
            continue

        indices = np.arange(len(samples))
        rng.shuffle(indices)
        test_indices = set(indices[:test_per_label].tolist())
        for index, sample in enumerate(samples):
            if index in test_indices:
                test_samples.append(sample)
            else:
                train_samples.append(sample)

    return train_samples, test_samples, skipped_labels


class CardPatchDataset(Dataset):
    def __init__(self, samples: Sequence[PatchSample], target: str, label_to_id: Dict[str, int]) -> None:
        self.samples = list(samples)
        self.target = target
        self.label_to_id = label_to_id

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        sample = self.samples[index]
        patch = sample.rank_patch if self.target == "rank" else sample.suit_patch
        patch_tensor = torch.from_numpy((patch.astype(np.float32) / 255.0)[None, :, :])
        label_name = sample.rank if self.target == "rank" else sample.suit
        label_tensor = torch.tensor(self.label_to_id[label_name], dtype=torch.long)
        return patch_tensor, label_tensor


class TinyPatchCNN(nn.Module):
    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 12 * 12, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.classifier(x)


def build_label_maps(samples: Sequence[PatchSample], target: str) -> Tuple[Dict[str, int], Dict[int, str]]:
    names = sorted({sample.rank if target == "rank" else sample.suit for sample in samples})
    label_to_id = {name: index for index, name in enumerate(names)}
    id_to_label = {index: name for name, index in label_to_id.items()}
    return label_to_id, id_to_label


def accuracy_and_confusions(
    expected_labels: Sequence[str], predicted_labels: Sequence[str]
) -> Tuple[float, Counter[Tuple[str, str]]]:
    correct = sum(int(expected == predicted) for expected, predicted in zip(expected_labels, predicted_labels))
    accuracy = correct / len(expected_labels) if expected_labels else 0.0
    confusions: Counter[Tuple[str, str]] = Counter()
    for expected, predicted in zip(expected_labels, predicted_labels):
        if expected != predicted:
            confusions[(expected, predicted)] += 1
    return accuracy, confusions


def save_checkpoint(
    output_path: Path,
    model: TinyPatchCNN,
    target: str,
    id_to_label: Dict[int, str],
    patch_size: Tuple[int, int],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "target": target,
            "id_to_label": id_to_label,
            "patch_size": patch_size,
            "state_dict": model.state_dict(),
        },
        output_path,
    )


def load_checkpoint(checkpoint_path: Path, device: torch.device) -> Tuple[TinyPatchCNN, Dict[int, str], str]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    id_to_label = {int(key): value for key, value in checkpoint["id_to_label"].items()}
    model = TinyPatchCNN(len(id_to_label))
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval()
    return model, id_to_label, str(checkpoint["target"])


def patch_tensor_from_sample(sample: PatchSample, target: str, device: torch.device) -> torch.Tensor:
    patch = sample.rank_patch if target == "rank" else sample.suit_patch
    return torch.from_numpy((patch.astype(np.float32) / 255.0)[None, None, :, :]).to(device)


def write_metrics_json(output_path: Path, payload: dict) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def summarize_label_counts(title: str, samples: Sequence[PatchSample]) -> str:
    counts = Counter(sample.label for sample in samples)
    lines = [title]
    for label in sorted(counts):
        lines.append(f"  {label}: {counts[label]}")
    return "\n".join(lines)
