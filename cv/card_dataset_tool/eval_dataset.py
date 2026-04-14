import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import cv2
import numpy as np

from cv.card_dataset_tool.dataset_meta import include_image_path, load_meta_index
from cv.card_dataset_tool.patch_preprocess import extract_roi, normalize_patch_feature, normalize_patch_image, orient_card_to_corner


@dataclass(frozen=True)
class Sample:
    label: str
    rank: str
    suit: str
    path: Path
    rank_patch: np.ndarray
    suit_patch: np.ndarray
    rank_feature: np.ndarray
    suit_feature: np.ndarray


@dataclass(frozen=True)
class Mistake:
    sample: Sample
    predicted_rank: str
    predicted_suit: str
    rank_score: float
    suit_score: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the warped playing-card dataset with simple offline classifiers.")
    parser.add_argument(
        "--dataset-dir",
        default=str(Path(__file__).with_name("dataset")),
        help="Dataset root containing warped/<LABEL>/*.jpg.",
    )
    parser.add_argument(
        "--test-per-label",
        type=int,
        default=3,
        help="How many samples per card label to hold out for evaluation.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for the train/test split.",
    )
    parser.add_argument(
        "--mode",
        choices=("prototype", "nn", "mlp"),
        default="prototype",
        help="Classifier mode: class prototype matching, nearest-neighbor, or OpenCV MLP.",
    )
    parser.add_argument(
        "--knn-k",
        type=int,
        default=1,
        help="Neighbor count for nn mode. Use 1 to match plain nearest-neighbor behavior.",
    )
    parser.add_argument(
        "--max-confusions",
        type=int,
        default=12,
        help="Maximum number of confusion pairs to print.",
    )
    parser.add_argument(
        "--mistake-sheet",
        default="",
        help="Optional output image path for a contact sheet of misclassified test cards.",
    )
    parser.add_argument("--mlp-hidden", type=int, default=128, help="Hidden layer size for mlp mode.")
    parser.add_argument("--mlp-epochs", type=int, default=400, help="Maximum training iterations for mlp mode.")
    parser.add_argument("--mlp-lr", type=float, default=0.05, help="Backprop learning rate for mlp mode.")
    parser.add_argument(
        "--min-contour-area",
        type=float,
        default=0.0,
        help="Skip samples whose meta.jsonl contour_area is below this threshold.",
    )
    return parser.parse_args()


def split_label(label: str) -> Tuple[str, str]:
    return label[:-1], label[-1]


def load_sample(image_path: Path, label: str) -> Sample | None:
    image = cv2.imread(str(image_path))
    if image is None:
        return None

    oriented = orient_card_to_corner(image)
    rank, suit = split_label(label)
    rank_patch = extract_roi(oriented, "rank")
    suit_patch = extract_roi(oriented, "suit")
    rank_patch_image = normalize_patch_image(rank_patch, "rank")
    suit_patch_image = normalize_patch_image(suit_patch, "suit")
    return Sample(
        label=label,
        rank=rank,
        suit=suit,
        path=image_path,
        rank_patch=rank_patch_image,
        suit_patch=suit_patch_image,
        rank_feature=normalize_patch_feature(rank_patch, "rank"),
        suit_feature=normalize_patch_feature(suit_patch, "suit"),
    )


def load_dataset(dataset_dir: Path, min_contour_area: float = 0.0) -> Dict[str, List[Sample]]:
    warped_root = dataset_dir / "warped"
    grouped: Dict[str, List[Sample]] = defaultdict(list)
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


def split_dataset(
    grouped: Dict[str, List[Sample]], test_per_label: int, seed: int
) -> Tuple[List[Sample], List[Sample], List[str]]:
    train_samples: List[Sample] = []
    test_samples: List[Sample] = []
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


def cosine_score(left: np.ndarray, right: np.ndarray) -> float:
    denom = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denom == 0.0:
        return -1.0
    return float(np.dot(left, right) / denom)


def build_prototypes(samples: Sequence[Sample], attr_name: str) -> Dict[str, np.ndarray]:
    grouped: Dict[str, List[np.ndarray]] = defaultdict(list)
    for sample in samples:
        grouped[getattr(sample, attr_name)].append(getattr(sample, f"{attr_name}_feature"))

    prototypes: Dict[str, np.ndarray] = {}
    for label, features in grouped.items():
        stack = np.stack(features, axis=0)
        prototypes[label] = np.median(stack, axis=0)
    return prototypes


def build_nn_index(samples: Sequence[Sample], attr_name: str) -> Tuple[np.ndarray, List[str]]:
    features = np.stack([getattr(sample, f"{attr_name}_feature") for sample in samples], axis=0)
    labels = [getattr(sample, attr_name) for sample in samples]
    return features, labels


def build_mlp_training_set(
    samples: Sequence[Sample], attr_name: str
) -> Tuple[np.ndarray, np.ndarray, Dict[str, int], Dict[int, str]]:
    labels = sorted({getattr(sample, attr_name) for sample in samples})
    label_to_id = {label: index for index, label in enumerate(labels)}
    id_to_label = {index: label for label, index in label_to_id.items()}

    features = []
    targets = []
    for sample in samples:
        features.append(getattr(sample, f"{attr_name}_feature"))
        targets.append(label_to_id[getattr(sample, attr_name)])

    target_matrix = np.zeros((len(targets), len(labels)), dtype=np.float32)
    for row_index, label_id in enumerate(targets):
        target_matrix[row_index, label_id] = 1.0

    return (
        np.stack(features, axis=0).astype(np.float32),
        target_matrix,
        label_to_id,
        id_to_label,
    )


def train_mlp_classifier(
    train_features: np.ndarray,
    train_targets: np.ndarray,
    hidden_size: int,
    epochs: int,
    learning_rate: float,
) -> cv2.ml_ANN_MLP:
    input_size = int(train_features.shape[1])
    output_size = int(train_targets.shape[1])
    mlp = cv2.ml.ANN_MLP_create()
    mlp.setLayerSizes(np.array([input_size, hidden_size, output_size], dtype=np.int32))
    mlp.setActivationFunction(cv2.ml.ANN_MLP_SIGMOID_SYM, 2.0, 1.0)
    mlp.setTrainMethod(cv2.ml.ANN_MLP_BACKPROP, learning_rate, 0.1)
    mlp.setTermCriteria((cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, epochs, 1e-5))
    mlp.train(train_features, cv2.ml.ROW_SAMPLE, train_targets)
    return mlp


def predict_from_mlp(
    sample: Sample,
    attr_name: str,
    mlp: cv2.ml_ANN_MLP,
    id_to_label: Dict[int, str],
) -> Tuple[str, float]:
    feature = getattr(sample, f"{attr_name}_feature").reshape(1, -1)
    _, outputs = mlp.predict(feature)
    output_vector = outputs[0]
    label_id = int(np.argmax(output_vector))
    score = float(output_vector[label_id])
    return id_to_label[label_id], score


def predict_from_prototypes(feature: np.ndarray, prototypes: Dict[str, np.ndarray]) -> Tuple[str, float]:
    best_label = ""
    best_score = -2.0
    for label, prototype in prototypes.items():
        score = cosine_score(feature, prototype)
        if score > best_score:
            best_label = label
            best_score = score
    return best_label, best_score


def predict_from_nn(
    feature: np.ndarray,
    train_features: np.ndarray,
    train_labels: Sequence[str],
    k: int,
) -> Tuple[str, float]:
    scores = np.array([cosine_score(feature, other) for other in train_features], dtype=np.float32)
    k = max(1, min(k, len(train_labels)))
    top_indices = np.argsort(scores)[-k:][::-1]

    votes: Dict[str, float] = defaultdict(float)
    best_per_label: Dict[str, float] = {}
    for index in top_indices:
        label = train_labels[int(index)]
        score = float(scores[int(index)])
        votes[label] += max(score, 0.0)
        best_per_label[label] = max(score, best_per_label.get(label, -1.0))

    predicted = max(votes.items(), key=lambda item: (item[1], best_per_label[item[0]]))[0]
    return predicted, best_per_label[predicted]


def evaluate_axis_prototype(
    test_samples: Sequence[Sample],
    prototypes: Dict[str, np.ndarray],
    attr_name: str,
) -> Tuple[float, List[Tuple[str, str]]]:
    correct = 0
    confusions: List[Tuple[str, str]] = []

    for sample in test_samples:
        feature = getattr(sample, f"{attr_name}_feature")
        expected = getattr(sample, attr_name)
        predicted, _ = predict_from_prototypes(feature, prototypes)
        if predicted == expected:
            correct += 1
        else:
            confusions.append((expected, predicted))

    accuracy = correct / len(test_samples) if test_samples else 0.0
    return accuracy, confusions


def evaluate_axis_nn(
    test_samples: Sequence[Sample],
    train_features: np.ndarray,
    train_labels: Sequence[str],
    attr_name: str,
    k: int,
) -> Tuple[float, List[Tuple[str, str]]]:
    correct = 0
    confusions: List[Tuple[str, str]] = []

    for sample in test_samples:
        feature = getattr(sample, f"{attr_name}_feature")
        expected = getattr(sample, attr_name)
        predicted, _ = predict_from_nn(feature, train_features, train_labels, k)
        if predicted == expected:
            correct += 1
        else:
            confusions.append((expected, predicted))

    accuracy = correct / len(test_samples) if test_samples else 0.0
    return accuracy, confusions


def evaluate_cards_prototype(
    test_samples: Sequence[Sample],
    rank_prototypes: Dict[str, np.ndarray],
    suit_prototypes: Dict[str, np.ndarray],
) -> Tuple[float, List[Tuple[str, str]], List[Mistake]]:
    correct = 0
    confusions: List[Tuple[str, str]] = []
    mistakes: List[Mistake] = []

    for sample in test_samples:
        predicted_rank, rank_score = predict_from_prototypes(sample.rank_feature, rank_prototypes)
        predicted_suit, suit_score = predict_from_prototypes(sample.suit_feature, suit_prototypes)
        predicted_label = f"{predicted_rank}{predicted_suit}"
        if predicted_label == sample.label:
            correct += 1
        else:
            confusions.append((sample.label, predicted_label))
            mistakes.append(Mistake(sample, predicted_rank, predicted_suit, rank_score, suit_score))

    accuracy = correct / len(test_samples) if test_samples else 0.0
    return accuracy, confusions, mistakes


def evaluate_cards_nn(
    test_samples: Sequence[Sample],
    rank_train_features: np.ndarray,
    rank_train_labels: Sequence[str],
    suit_train_features: np.ndarray,
    suit_train_labels: Sequence[str],
    k: int,
) -> Tuple[float, List[Tuple[str, str]], List[Mistake]]:
    correct = 0
    confusions: List[Tuple[str, str]] = []
    mistakes: List[Mistake] = []

    for sample in test_samples:
        predicted_rank, rank_score = predict_from_nn(sample.rank_feature, rank_train_features, rank_train_labels, k)
        predicted_suit, suit_score = predict_from_nn(sample.suit_feature, suit_train_features, suit_train_labels, k)
        predicted_label = f"{predicted_rank}{predicted_suit}"
        if predicted_label == sample.label:
            correct += 1
        else:
            confusions.append((sample.label, predicted_label))
            mistakes.append(Mistake(sample, predicted_rank, predicted_suit, rank_score, suit_score))

    accuracy = correct / len(test_samples) if test_samples else 0.0
    return accuracy, confusions, mistakes


def evaluate_axis_mlp(
    test_samples: Sequence[Sample],
    attr_name: str,
    mlp: cv2.ml_ANN_MLP,
    id_to_label: Dict[int, str],
) -> Tuple[float, List[Tuple[str, str]]]:
    correct = 0
    confusions: List[Tuple[str, str]] = []

    for sample in test_samples:
        expected = getattr(sample, attr_name)
        predicted, _ = predict_from_mlp(sample, attr_name, mlp, id_to_label)
        if predicted == expected:
            correct += 1
        else:
            confusions.append((expected, predicted))

    accuracy = correct / len(test_samples) if test_samples else 0.0
    return accuracy, confusions


def evaluate_cards_mlp(
    test_samples: Sequence[Sample],
    rank_mlp: cv2.ml_ANN_MLP,
    rank_id_to_label: Dict[int, str],
    suit_mlp: cv2.ml_ANN_MLP,
    suit_id_to_label: Dict[int, str],
) -> Tuple[float, List[Tuple[str, str]], List[Mistake]]:
    correct = 0
    confusions: List[Tuple[str, str]] = []
    mistakes: List[Mistake] = []

    for sample in test_samples:
        predicted_rank, rank_score = predict_from_mlp(sample, "rank", rank_mlp, rank_id_to_label)
        predicted_suit, suit_score = predict_from_mlp(sample, "suit", suit_mlp, suit_id_to_label)
        predicted_label = f"{predicted_rank}{predicted_suit}"
        if predicted_label == sample.label:
            correct += 1
        else:
            confusions.append((sample.label, predicted_label))
            mistakes.append(Mistake(sample, predicted_rank, predicted_suit, rank_score, suit_score))

    accuracy = correct / len(test_samples) if test_samples else 0.0
    return accuracy, confusions, mistakes


def print_counts(title: str, samples: Sequence[Sample]) -> None:
    counts = Counter(sample.label for sample in samples)
    print(title)
    for label in sorted(counts):
        print(f"  {label}: {counts[label]}")


def print_confusions(title: str, confusions: Sequence[Tuple[str, str]], limit: int) -> None:
    print(title)
    if not confusions:
        print("  none")
        return

    for (expected, predicted), count in Counter(confusions).most_common(limit):
        print(f"  {expected} -> {predicted}: {count}")


def print_mistakes(mistakes: Sequence[Mistake], limit: int) -> None:
    print("Example Card Mistakes")
    if not mistakes:
        print("  none")
        return

    for mistake in mistakes[:limit]:
        print(
            f"  {mistake.sample.label} -> {mistake.predicted_rank}{mistake.predicted_suit}"
            f"  rank:{mistake.rank_score:.3f} suit:{mistake.suit_score:.3f}"
            f"  {mistake.sample.path}"
        )


def render_mistake_sheet(mistakes: Sequence[Mistake], output_path: Path, limit: int) -> None:
    selected = list(mistakes[:limit])
    if not selected:
        return

    thumb_width = 220
    thumb_height = 320
    caption_height = 54
    cols = 3
    rows = (len(selected) + cols - 1) // cols
    canvas = np.full((rows * (thumb_height + caption_height), cols * thumb_width, 3), 245, dtype=np.uint8)

    for index, mistake in enumerate(selected):
        row = index // cols
        col = index % cols
        x = col * thumb_width
        y = row * (thumb_height + caption_height)
        image = cv2.imread(str(mistake.sample.path))
        if image is None:
            continue
        thumb = cv2.resize(image, (thumb_width, thumb_height), interpolation=cv2.INTER_AREA)
        canvas[y : y + thumb_height, x : x + thumb_width] = thumb
        cv2.putText(
            canvas,
            f"{mistake.sample.label} -> {mistake.predicted_rank}{mistake.predicted_suit}",
            (x + 8, y + thumb_height + 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (15, 15, 15),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            mistake.sample.path.name,
            (x + 8, y + thumb_height + 42),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.46,
            (35, 35, 35),
            1,
            cv2.LINE_AA,
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    normalized_path = output_path
    if normalized_path.suffix.lower() not in {".png", ".bmp", ".tif", ".tiff", ".webp", ".jpg", ".jpeg"}:
        normalized_path = normalized_path.with_suffix(".png")

    wrote = cv2.imwrite(str(normalized_path), canvas)
    if not wrote and normalized_path.suffix.lower() in {".jpg", ".jpeg"}:
        normalized_path = normalized_path.with_suffix(".png")
        wrote = cv2.imwrite(str(normalized_path), canvas)
    if not wrote:
        raise RuntimeError(f"Failed to write mistake sheet to: {normalized_path}")

    if normalized_path != output_path:
        print(f"Requested mistake sheet path adjusted to: {normalized_path}")


def sort_mistakes(mistakes: Iterable[Mistake]) -> List[Mistake]:
    return sorted(mistakes, key=lambda item: (item.rank_score + item.suit_score), reverse=True)


def main() -> int:
    args = parse_args()
    dataset_dir = Path(args.dataset_dir)
    grouped = load_dataset(dataset_dir, min_contour_area=args.min_contour_area)
    if not grouped:
        print(f"No warped samples found under: {dataset_dir}")
        return 1

    train_samples, test_samples, skipped_labels = split_dataset(grouped, args.test_per_label, args.seed)
    if not train_samples or not test_samples:
        print("Dataset split is empty. Reduce --test-per-label or add more samples.")
        return 1

    if args.mode == "prototype":
        rank_prototypes = build_prototypes(train_samples, "rank")
        suit_prototypes = build_prototypes(train_samples, "suit")
        rank_accuracy, rank_confusions = evaluate_axis_prototype(test_samples, rank_prototypes, "rank")
        suit_accuracy, suit_confusions = evaluate_axis_prototype(test_samples, suit_prototypes, "suit")
        card_accuracy, card_confusions, mistakes = evaluate_cards_prototype(test_samples, rank_prototypes, suit_prototypes)
    elif args.mode == "nn":
        rank_train_features, rank_train_labels = build_nn_index(train_samples, "rank")
        suit_train_features, suit_train_labels = build_nn_index(train_samples, "suit")
        rank_accuracy, rank_confusions = evaluate_axis_nn(
            test_samples, rank_train_features, rank_train_labels, "rank", args.knn_k
        )
        suit_accuracy, suit_confusions = evaluate_axis_nn(
            test_samples, suit_train_features, suit_train_labels, "suit", args.knn_k
        )
        card_accuracy, card_confusions, mistakes = evaluate_cards_nn(
            test_samples,
            rank_train_features,
            rank_train_labels,
            suit_train_features,
            suit_train_labels,
            args.knn_k,
        )
    else:
        rank_train_features, rank_train_targets, _, rank_id_to_label = build_mlp_training_set(train_samples, "rank")
        suit_train_features, suit_train_targets, _, suit_id_to_label = build_mlp_training_set(train_samples, "suit")
        rank_mlp = train_mlp_classifier(
            rank_train_features, rank_train_targets, args.mlp_hidden, args.mlp_epochs, args.mlp_lr
        )
        suit_mlp = train_mlp_classifier(
            suit_train_features, suit_train_targets, args.mlp_hidden, args.mlp_epochs, args.mlp_lr
        )
        rank_accuracy, rank_confusions = evaluate_axis_mlp(test_samples, "rank", rank_mlp, rank_id_to_label)
        suit_accuracy, suit_confusions = evaluate_axis_mlp(test_samples, "suit", suit_mlp, suit_id_to_label)
        card_accuracy, card_confusions, mistakes = evaluate_cards_mlp(
            test_samples,
            rank_mlp,
            rank_id_to_label,
            suit_mlp,
            suit_id_to_label,
        )

    sorted_mistakes = sort_mistakes(mistakes)

    print(f"Dataset dir: {dataset_dir}")
    print(f"Min contour area: {args.min_contour_area}")
    print(f"Mode: {args.mode}")
    if args.mode == "nn":
        print(f"k-NN k: {args.knn_k}")
    if args.mode == "mlp":
        print(f"MLP hidden: {args.mlp_hidden}")
        print(f"MLP epochs: {args.mlp_epochs}")
        print(f"MLP lr: {args.mlp_lr}")
    print(f"Labels loaded: {len(grouped)}")
    print(f"Train samples: {len(train_samples)}")
    print(f"Test samples: {len(test_samples)}")
    print(f"Skipped labels: {len(skipped_labels)}")
    if skipped_labels:
        print(f"  {' '.join(skipped_labels)}")
    print()

    print_counts("Train Split", train_samples)
    print_counts("Test Split", test_samples)
    print()

    print(f"Rank accuracy: {rank_accuracy:.3f}")
    print(f"Suit accuracy: {suit_accuracy:.3f}")
    print(f"Card accuracy: {card_accuracy:.3f}")
    print()

    print_confusions("Rank Confusions", rank_confusions, args.max_confusions)
    print_confusions("Suit Confusions", suit_confusions, args.max_confusions)
    print_confusions("Card Confusions", card_confusions, args.max_confusions)
    print()
    print_mistakes(sorted_mistakes, args.max_confusions)

    if args.mistake_sheet:
        output_path = Path(args.mistake_sheet)
        render_mistake_sheet(sorted_mistakes, output_path, args.max_confusions)
        print()
        print(f"Mistake sheet: {output_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
