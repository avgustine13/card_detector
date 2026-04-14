from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import List, Tuple

import torch

from cv.card_dataset_tool.cnn_common import (
    accuracy_and_confusions,
    load_checkpoint,
    load_grouped_dataset,
    patch_tensor_from_sample,
    split_grouped_dataset,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate saved CNN checkpoints on the card patch dataset.")
    parser.add_argument("--dataset-dir", default=str(Path(__file__).with_name("dataset")), help="Dataset root.")
    parser.add_argument("--rank-model", default=str(Path(__file__).with_name("models") / "rank_cnn.pt"), help="Rank model path.")
    parser.add_argument("--suit-model", default=str(Path(__file__).with_name("models") / "suit_cnn.pt"), help="Suit model path.")
    parser.add_argument("--test-per-label", type=int, default=3, help="Held-out samples per label.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--device", default="auto", help="auto, cpu, or cuda.")
    parser.add_argument("--max-confusions", type=int, default=12, help="Maximum confusion pairs to print.")
    parser.add_argument(
        "--min-contour-area",
        type=float,
        default=0.0,
        help="Skip samples whose meta.jsonl contour_area is below this threshold.",
    )
    return parser.parse_args()


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def print_confusions(title: str, confusions: Counter[Tuple[str, str]], limit: int) -> None:
    print(title)
    if not confusions:
        print("  none")
        return
    for (expected, predicted), count in confusions.most_common(limit):
        print(f"  {expected} -> {predicted}: {count}")


def main() -> int:
    args = parse_args()
    device = resolve_device(args.device)
    grouped = load_grouped_dataset(Path(args.dataset_dir), min_contour_area=args.min_contour_area)
    _, test_samples, skipped_labels = split_grouped_dataset(grouped, args.test_per_label, args.seed)
    if not test_samples:
        print("Test split is empty.")
        return 1

    rank_model, rank_id_to_label, rank_target = load_checkpoint(Path(args.rank_model), device)
    suit_model, suit_id_to_label, suit_target = load_checkpoint(Path(args.suit_model), device)
    if rank_target != "rank" or suit_target != "suit":
        print("Checkpoint targets do not match expected rank/suit models.")
        return 1

    expected_rank: List[str] = []
    predicted_rank: List[str] = []
    expected_suit: List[str] = []
    predicted_suit: List[str] = []
    expected_card: List[str] = []
    predicted_card: List[str] = []

    for sample in test_samples:
        with torch.no_grad():
            rank_logits = rank_model(patch_tensor_from_sample(sample, "rank", device))
            suit_logits = suit_model(patch_tensor_from_sample(sample, "suit", device))
        rank_label = rank_id_to_label[int(torch.argmax(rank_logits, dim=1).item())]
        suit_label = suit_id_to_label[int(torch.argmax(suit_logits, dim=1).item())]

        expected_rank.append(sample.rank)
        predicted_rank.append(rank_label)
        expected_suit.append(sample.suit)
        predicted_suit.append(suit_label)
        expected_card.append(sample.label)
        predicted_card.append(f"{rank_label}{suit_label}")

    rank_accuracy, rank_confusions = accuracy_and_confusions(expected_rank, predicted_rank)
    suit_accuracy, suit_confusions = accuracy_and_confusions(expected_suit, predicted_suit)
    card_accuracy, card_confusions = accuracy_and_confusions(expected_card, predicted_card)

    print(f"Dataset dir: {args.dataset_dir}")
    print(f"Rank model: {args.rank_model}")
    print(f"Suit model: {args.suit_model}")
    print(f"Device: {device}")
    print(f"Min contour area: {args.min_contour_area}")
    print(f"Skipped labels: {len(skipped_labels)}")
    if skipped_labels:
        print(f"  {' '.join(skipped_labels)}")
    print()
    print(f"Rank accuracy: {rank_accuracy:.3f}")
    print(f"Suit accuracy: {suit_accuracy:.3f}")
    print(f"Card accuracy: {card_accuracy:.3f}")
    print()
    print_confusions("Rank Confusions", rank_confusions, args.max_confusions)
    print_confusions("Suit Confusions", suit_confusions, args.max_confusions)
    print_confusions("Card Confusions", card_confusions, args.max_confusions)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
