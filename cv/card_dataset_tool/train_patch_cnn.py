from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader

from cv.card_dataset_tool.cnn_common import (
    CardPatchDataset,
    PATCH_SIZE,
    TinyPatchCNN,
    accuracy_and_confusions,
    build_label_maps,
    load_checkpoint,
    load_grouped_dataset,
    patch_tensor_from_sample,
    save_checkpoint,
    split_grouped_dataset,
    summarize_label_counts,
    write_metrics_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train small CNNs for card rank and suit recognition.")
    parser.add_argument("--dataset-dir", default=str(Path(__file__).with_name("dataset")), help="Dataset root.")
    parser.add_argument("--output-dir", default=str(Path(__file__).with_name("models")), help="Output directory for checkpoints.")
    parser.add_argument("--test-per-label", type=int, default=3, help="Held-out samples per label.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--seeds",
        default="",
        help="Optional comma-separated seed list. When set, trains once per seed and promotes the best run.",
    )
    parser.add_argument("--epochs", type=int, default=12, help="Training epochs.")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Adam weight decay.")
    parser.add_argument("--device", default="auto", help="auto, cpu, or cuda.")
    parser.add_argument("--max-confusions", type=int, default=12, help="Maximum confusion pairs to print.")
    parser.add_argument(
        "--save-policy",
        choices=("if-better", "always"),
        default="if-better",
        help="Keep newly trained checkpoints only if they beat the currently saved ones, or always overwrite.",
    )
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


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_loader(samples, target, label_to_id, batch_size, shuffle):
    dataset = CardPatchDataset(samples, target, label_to_id)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0)


def train_classifier(
    train_samples,
    test_samples,
    target: str,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    device: torch.device,
) -> Tuple[TinyPatchCNN, Dict[int, str], float, Counter[Tuple[str, str]], List[Tuple[str, str, Path]]]:
    label_to_id, id_to_label = build_label_maps(train_samples + test_samples, target)
    model = TinyPatchCNN(len(label_to_id)).to(device)
    train_loader = build_loader(train_samples, target, label_to_id, batch_size, True)
    test_loader = build_loader(test_samples, target, label_to_id, batch_size, False)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    for _ in range(epochs):
        model.train()
        for inputs, targets in train_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(inputs)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()

    model.eval()
    expected_labels: List[str] = []
    predicted_labels: List[str] = []
    mistakes: List[Tuple[str, str, Path]] = []
    start = 0
    for inputs, targets in test_loader:
        batch_samples = test_samples[start : start + len(targets)]
        start += len(targets)
        inputs = inputs.to(device)
        with torch.no_grad():
            logits = model(inputs)
            predictions = torch.argmax(logits, dim=1).cpu().tolist()
        expected_ids = targets.tolist()
        for sample, expected_id, predicted_id in zip(batch_samples, expected_ids, predictions):
            expected_label = id_to_label[expected_id]
            predicted_label = id_to_label[predicted_id]
            expected_labels.append(expected_label)
            predicted_labels.append(predicted_label)
            if expected_label != predicted_label:
                mistakes.append((expected_label, predicted_label, sample.path))

    accuracy, confusions = accuracy_and_confusions(expected_labels, predicted_labels)
    return model, id_to_label, accuracy, confusions, mistakes


def evaluate_full_cards(
    test_samples,
    rank_model: TinyPatchCNN,
    rank_id_to_label: Dict[int, str],
    suit_model: TinyPatchCNN,
    suit_id_to_label: Dict[int, str],
    device: torch.device,
) -> Tuple[float, Counter[Tuple[str, str]], List[Tuple[str, str, Path]]]:
    rank_model.eval()
    suit_model.eval()
    expected_labels: List[str] = []
    predicted_labels: List[str] = []
    mistakes: List[Tuple[str, str, Path]] = []

    for sample in test_samples:
        rank_tensor = torch.from_numpy((sample.rank_patch.astype("float32") / 255.0)[None, None, :, :]).to(device)
        suit_tensor = torch.from_numpy((sample.suit_patch.astype("float32") / 255.0)[None, None, :, :]).to(device)
        with torch.no_grad():
            rank_logits = rank_model(rank_tensor)
            suit_logits = suit_model(suit_tensor)
        predicted_rank = rank_id_to_label[int(torch.argmax(rank_logits, dim=1).item())]
        predicted_suit = suit_id_to_label[int(torch.argmax(suit_logits, dim=1).item())]
        predicted_label = f"{predicted_rank}{predicted_suit}"
        expected_labels.append(sample.label)
        predicted_labels.append(predicted_label)
        if predicted_label != sample.label:
            mistakes.append((sample.label, predicted_label, sample.path))

    return accuracy_and_confusions(expected_labels, predicted_labels) + (mistakes,)


def print_confusions(title: str, confusions: Counter[Tuple[str, str]], limit: int) -> None:
    print(title)
    if not confusions:
        print("  none")
        return
    for (expected, predicted), count in confusions.most_common(limit):
        print(f"  {expected} -> {predicted}: {count}")


def print_mistakes(title: str, mistakes: Sequence[Tuple[str, str, Path]], limit: int) -> None:
    print(title)
    if not mistakes:
        print("  none")
        return
    for expected, predicted, path in mistakes[:limit]:
        print(f"  {expected} -> {predicted}  {path}")


def evaluate_saved_models(
    rank_model_path: Path,
    suit_model_path: Path,
    test_samples,
    device: torch.device,
) -> Tuple[float, float, float] | None:
    if not rank_model_path.exists() or not suit_model_path.exists():
        return None

    rank_model, rank_id_to_label, rank_target = load_checkpoint(rank_model_path, device)
    suit_model, suit_id_to_label, suit_target = load_checkpoint(suit_model_path, device)
    if rank_target != "rank" or suit_target != "suit":
        return None

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

    rank_accuracy, _ = accuracy_and_confusions(expected_rank, predicted_rank)
    suit_accuracy, _ = accuracy_and_confusions(expected_suit, predicted_suit)
    card_accuracy, _ = accuracy_and_confusions(expected_card, predicted_card)
    return rank_accuracy, suit_accuracy, card_accuracy


def parse_seed_values(seed: int, seeds_arg: str) -> List[int]:
    if not seeds_arg.strip():
        return [seed]

    values: List[int] = []
    for part in seeds_arg.split(","):
        item = part.strip()
        if not item:
            continue
        values.append(int(item))
    if not values:
        return [seed]
    return values


def is_better_result(
    candidate: Tuple[float, float, float],
    incumbent: Tuple[float, float, float],
) -> bool:
    # Prefer higher card accuracy first, then suit, then rank.
    return candidate > incumbent


def main() -> int:
    args = parse_args()
    device = resolve_device(args.device)

    dataset_dir = Path(args.dataset_dir)
    output_dir = Path(args.output_dir)
    grouped = load_grouped_dataset(dataset_dir, min_contour_area=args.min_contour_area)
    seed_values = parse_seed_values(args.seed, args.seeds)
    set_seed(seed_values[0])
    train_samples, test_samples, skipped_labels = split_grouped_dataset(grouped, args.test_per_label, seed_values[0])
    if not train_samples or not test_samples:
        print("Dataset split is empty. Reduce --test-per-label or add more samples.")
        return 1

    print(f"Dataset dir: {dataset_dir}")
    print(f"Output dir: {output_dir}")
    print(f"Device: {device}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print(f"Weight decay: {args.weight_decay}")
    print(f"Seeds: {', '.join(str(seed) for seed in seed_values)}")
    print(f"Min contour area: {args.min_contour_area}")
    print(f"Skipped labels: {len(skipped_labels)}")
    if skipped_labels:
        print(f"  {' '.join(skipped_labels)}")
    print()
    print(summarize_label_counts("Train Split", train_samples))
    print(summarize_label_counts("Test Split", test_samples))
    print()
    best_result = None

    for seed in seed_values:
        set_seed(seed)
        rank_model, rank_id_to_label, rank_accuracy, rank_confusions, rank_mistakes = train_classifier(
            train_samples, test_samples, "rank", args.epochs, args.batch_size, args.lr, args.weight_decay, device
        )
        suit_model, suit_id_to_label, suit_accuracy, suit_confusions, suit_mistakes = train_classifier(
            train_samples, test_samples, "suit", args.epochs, args.batch_size, args.lr, args.weight_decay, device
        )
        card_accuracy, card_confusions, card_mistakes = evaluate_full_cards(
            test_samples, rank_model, rank_id_to_label, suit_model, suit_id_to_label, device
        )

        print(f"Seed {seed}: rank={rank_accuracy:.3f} suit={suit_accuracy:.3f} card={card_accuracy:.3f}")
        result_key = (card_accuracy, suit_accuracy, rank_accuracy)
        if best_result is None or is_better_result(result_key, best_result["key"]):
            best_result = {
                "seed": seed,
                "key": result_key,
                "rank_model": rank_model,
                "rank_id_to_label": rank_id_to_label,
                "rank_accuracy": rank_accuracy,
                "rank_confusions": rank_confusions,
                "rank_mistakes": rank_mistakes,
                "suit_model": suit_model,
                "suit_id_to_label": suit_id_to_label,
                "suit_accuracy": suit_accuracy,
                "suit_confusions": suit_confusions,
                "suit_mistakes": suit_mistakes,
                "card_accuracy": card_accuracy,
                "card_confusions": card_confusions,
                "card_mistakes": card_mistakes,
            }

    assert best_result is not None
    rank_model = best_result["rank_model"]
    rank_id_to_label = best_result["rank_id_to_label"]
    rank_accuracy = best_result["rank_accuracy"]
    rank_confusions = best_result["rank_confusions"]
    rank_mistakes = best_result["rank_mistakes"]
    suit_model = best_result["suit_model"]
    suit_id_to_label = best_result["suit_id_to_label"]
    suit_accuracy = best_result["suit_accuracy"]
    suit_confusions = best_result["suit_confusions"]
    suit_mistakes = best_result["suit_mistakes"]
    card_accuracy = best_result["card_accuracy"]
    card_confusions = best_result["card_confusions"]
    card_mistakes = best_result["card_mistakes"]

    print()
    print(f"Best seed: {best_result['seed']}")
    print(f"Rank accuracy: {rank_accuracy:.3f}")
    print(f"Suit accuracy: {suit_accuracy:.3f}")
    print(f"Card accuracy: {card_accuracy:.3f}")
    print()
    print_confusions("Rank Confusions", rank_confusions, args.max_confusions)
    print_confusions("Suit Confusions", suit_confusions, args.max_confusions)
    print_confusions("Card Confusions", card_confusions, args.max_confusions)
    print()
    print_mistakes("Rank Mistakes", rank_mistakes, args.max_confusions)
    print_mistakes("Suit Mistakes", suit_mistakes, args.max_confusions)
    print_mistakes("Card Mistakes", card_mistakes, args.max_confusions)

    rank_model_path = output_dir / "rank_cnn.pt"
    suit_model_path = output_dir / "suit_cnn.pt"
    metrics = {
        "dataset_dir": str(dataset_dir),
        "device": str(device),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "seed": best_result["seed"],
        "candidate_seeds": seed_values,
        "min_contour_area": args.min_contour_area,
        "test_per_label": args.test_per_label,
        "rank_accuracy": rank_accuracy,
        "suit_accuracy": suit_accuracy,
        "card_accuracy": card_accuracy,
        "rank_confusions": [[a, b, c] for (a, b), c in rank_confusions.most_common(args.max_confusions)],
        "suit_confusions": [[a, b, c] for (a, b), c in suit_confusions.most_common(args.max_confusions)],
        "card_confusions": [[a, b, c] for (a, b), c in card_confusions.most_common(args.max_confusions)],
    }

    metrics_path = output_dir / "metrics.json"
    existing_scores = None
    if args.save_policy == "if-better":
        existing_scores = evaluate_saved_models(rank_model_path, suit_model_path, test_samples, device)
        if existing_scores is not None:
            existing_rank_accuracy, existing_suit_accuracy, existing_card_accuracy = existing_scores
            print()
            print(
                "Current saved checkpoints on this split: "
                f"rank={existing_rank_accuracy:.3f} suit={existing_suit_accuracy:.3f} card={existing_card_accuracy:.3f}"
            )
            if card_accuracy <= existing_card_accuracy:
                print(
                    "Keeping existing checkpoints. "
                    f"New card accuracy {card_accuracy:.3f} did not beat saved {existing_card_accuracy:.3f}."
                )
                return 0

    save_checkpoint(rank_model_path, rank_model.cpu(), "rank", rank_id_to_label, PATCH_SIZE)
    save_checkpoint(suit_model_path, suit_model.cpu(), "suit", suit_id_to_label, PATCH_SIZE)
    write_metrics_json(metrics_path, metrics)
    print()
    if existing_scores is not None:
        print(f"Promoted new checkpoints: new card accuracy {card_accuracy:.3f} beat saved {existing_scores[2]:.3f}.")
    print(f"Saved rank model: {rank_model_path}")
    print(f"Saved suit model: {suit_model_path}")
    print(f"Saved metrics: {metrics_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
