from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import cv2
import numpy as np

from cv.card_dataset_tool.patch_preprocess import extract_roi, normalize_patch_image, orient_card_to_corner


@dataclass(frozen=True)
class TriageSample:
    label: str
    rank: str
    suit: str
    path: Path
    rank_patch: np.ndarray
    suit_patch: np.ndarray
    combined_feature: np.ndarray
    rank_feature: np.ndarray
    suit_feature: np.ndarray


@dataclass(frozen=True)
class SampleIssue:
    label: str
    path: str
    outlier_score: float
    nearest_card_label: str
    nearest_card_distance: float
    card_margin: float
    nearest_rank: str
    nearest_rank_distance: float
    rank_margin: float
    nearest_suit: str
    nearest_suit_distance: float
    suit_margin: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Offline dataset triage for warped playing-card samples.")
    parser.add_argument("--dataset-dir", default=str(Path(__file__).with_name("dataset")), help="Dataset root.")
    parser.add_argument(
        "--focus-labels",
        default="8D,7D,10D,10H,6H,7C,6C,8C",
        help="Comma-separated card labels to include in the review sheet.",
    )
    parser.add_argument("--top-k", type=int, default=6, help="How many suspect samples per label to report.")
    parser.add_argument("--sheet-count", type=int, default=24, help="How many focused samples to place into the review sheet.")
    parser.add_argument(
        "--output-json",
        default=str(Path(__file__).with_name("models") / "triage_report.json"),
        help="Where to write the JSON triage report.",
    )
    parser.add_argument(
        "--output-sheet",
        default=str(Path(__file__).with_name("models") / "triage_sheet.jpg"),
        help="Where to write the review contact sheet.",
    )
    return parser.parse_args()


def split_label(label: str) -> Tuple[str, str]:
    return label[:-1], label[-1]


def load_sample(image_path: Path, label: str) -> TriageSample | None:
    image = cv2.imread(str(image_path))
    if image is None:
        return None

    oriented = orient_card_to_corner(image)
    rank_patch = normalize_patch_image(extract_roi(oriented, "rank"), "rank")
    suit_patch = normalize_patch_image(extract_roi(oriented, "suit"), "suit")
    rank_feature = rank_patch.astype(np.float32).reshape(-1) / 255.0
    suit_feature = suit_patch.astype(np.float32).reshape(-1) / 255.0
    combined_feature = np.concatenate([rank_feature, suit_feature], axis=0)
    rank, suit = split_label(label)
    return TriageSample(label, rank, suit, image_path, rank_patch, suit_patch, combined_feature, rank_feature, suit_feature)


def load_grouped_dataset(dataset_dir: Path) -> Dict[str, List[TriageSample]]:
    warped_root = dataset_dir / "warped"
    grouped: Dict[str, List[TriageSample]] = defaultdict(list)
    for label_dir in sorted(path for path in warped_root.iterdir() if path.is_dir()):
        label = label_dir.name.upper()
        for image_path in sorted(label_dir.glob("*_warped.jpg")):
            sample = load_sample(image_path, label)
            if sample is not None:
                grouped[label].append(sample)
    return grouped


def mean_feature(samples: Sequence[TriageSample], attr: str) -> np.ndarray:
    stack = np.stack([getattr(sample, attr) for sample in samples], axis=0)
    return stack.mean(axis=0)


def squared_distance(a: np.ndarray, b: np.ndarray) -> float:
    delta = a - b
    return float(np.mean(delta * delta))


def build_full_label_centroids(grouped: Dict[str, List[TriageSample]]) -> Dict[str, np.ndarray]:
    return {label: mean_feature(samples, "combined_feature") for label, samples in grouped.items() if samples}


def build_axis_centroids(grouped: Dict[str, List[TriageSample]], axis: str) -> Dict[str, np.ndarray]:
    bucketed: Dict[str, List[TriageSample]] = defaultdict(list)
    attr_name = f"{axis}_feature"
    for samples in grouped.values():
        for sample in samples:
            bucketed[getattr(sample, axis)].append(sample)
    return {name: mean_feature(samples, attr_name) for name, samples in bucketed.items() if samples}


def nearest_other(
    current_name: str,
    feature: np.ndarray,
    centroids: Dict[str, np.ndarray],
) -> Tuple[str, float, float]:
    candidates: List[Tuple[float, str]] = []
    own_distance = squared_distance(feature, centroids[current_name])
    for other_name, centroid in centroids.items():
        if other_name == current_name:
            continue
        candidates.append((squared_distance(feature, centroid), other_name))
    candidates.sort(key=lambda item: item[0])
    nearest_distance, nearest_name = candidates[0]
    return nearest_name, nearest_distance, nearest_distance - own_distance


def collect_issues(
    grouped: Dict[str, List[TriageSample]],
    full_label_centroids: Dict[str, np.ndarray],
    rank_centroids: Dict[str, np.ndarray],
    suit_centroids: Dict[str, np.ndarray],
) -> List[SampleIssue]:
    issues: List[SampleIssue] = []
    for label, samples in grouped.items():
        own_centroid = full_label_centroids[label]
        for sample in samples:
            outlier_score = squared_distance(sample.combined_feature, own_centroid)
            nearest_card_label, nearest_card_distance, card_margin = nearest_other(
                sample.label, sample.combined_feature, full_label_centroids
            )
            nearest_rank, nearest_rank_distance, rank_margin = nearest_other(sample.rank, sample.rank_feature, rank_centroids)
            nearest_suit, nearest_suit_distance, suit_margin = nearest_other(sample.suit, sample.suit_feature, suit_centroids)
            issues.append(
                SampleIssue(
                    label=label,
                    path=str(sample.path),
                    outlier_score=outlier_score,
                    nearest_card_label=nearest_card_label,
                    nearest_card_distance=nearest_card_distance,
                    card_margin=card_margin,
                    nearest_rank=nearest_rank,
                    nearest_rank_distance=nearest_rank_distance,
                    rank_margin=rank_margin,
                    nearest_suit=nearest_suit,
                    nearest_suit_distance=nearest_suit_distance,
                    suit_margin=suit_margin,
                )
            )
    return issues


def summarize_focus_labels(issues: Sequence[SampleIssue], focus_labels: Iterable[str], top_k: int) -> Dict[str, List[dict]]:
    by_label: Dict[str, List[SampleIssue]] = defaultdict(list)
    for issue in issues:
        by_label[issue.label].append(issue)

    summary: Dict[str, List[dict]] = {}
    for label in focus_labels:
        items = sorted(
            by_label.get(label, []),
            key=lambda issue: (issue.card_margin, issue.rank_margin, issue.suit_margin, -issue.outlier_score),
        )
        summary[label] = [
            {
                "path": item.path,
                "outlier_score": round(item.outlier_score, 6),
                "nearest_card_label": item.nearest_card_label,
                "card_margin": round(item.card_margin, 6),
                "nearest_rank": item.nearest_rank,
                "rank_margin": round(item.rank_margin, 6),
                "nearest_suit": item.nearest_suit,
                "suit_margin": round(item.suit_margin, 6),
            }
            for item in items[:top_k]
        ]
    return summary


def find_stale_meta_records(dataset_dir: Path) -> List[dict]:
    meta_path = dataset_dir / "meta.jsonl"
    if not meta_path.exists():
        return []

    stale: List[dict] = []
    for line in meta_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        warped_rel = record.get("warped_path")
        if not warped_rel:
            continue
        warped_path = dataset_dir / warped_rel
        if not warped_path.exists():
            stale.append(
                {
                    "label": record.get("label"),
                    "warped_path": warped_rel,
                    "raw_path": record.get("raw_path"),
                    "timestamp": record.get("timestamp"),
                }
            )
    return stale


def render_review_sheet(issues: Sequence[SampleIssue], output_path: Path, count: int) -> None:
    selected = sorted(
        issues,
        key=lambda issue: (issue.card_margin, issue.rank_margin, issue.suit_margin, -issue.outlier_score),
    )[:count]
    if not selected:
        return

    thumb_w = 220
    thumb_h = 320
    header_h = 54
    cols = 4
    rows = int(np.ceil(len(selected) / cols))
    canvas = np.full((rows * (thumb_h + header_h), cols * thumb_w, 3), 245, dtype=np.uint8)

    for index, issue in enumerate(selected):
        image = cv2.imread(issue.path)
        if image is None:
            continue
        image = orient_card_to_corner(image)
        scaled = cv2.resize(image, (thumb_w, thumb_h), interpolation=cv2.INTER_AREA)
        row = index // cols
        col = index % cols
        y0 = row * (thumb_h + header_h)
        x0 = col * thumb_w
        canvas[y0 : y0 + thumb_h, x0 : x0 + thumb_w] = scaled
        cv2.putText(
            canvas,
            f"{issue.label} -> {issue.nearest_card_label}",
            (x0 + 6, y0 + thumb_h + 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (0, 0, 180),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            f"r:{issue.nearest_rank}  s:{issue.nearest_suit}",
            (x0 + 6, y0 + thumb_h + 36),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (40, 40, 40),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            Path(issue.path).name,
            (x0 + 6, y0 + thumb_h + 52),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.36,
            (80, 80, 80),
            1,
            cv2.LINE_AA,
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), canvas)


def aggregate_top_confusions(issues: Sequence[SampleIssue], attr_expected: str, attr_predicted: str) -> List[Tuple[str, str, int]]:
    counts: Counter[Tuple[str, str]] = Counter()
    for issue in issues:
        counts[(getattr(issue, attr_expected), getattr(issue, attr_predicted))] += 1
    return [(a, b, count) for (a, b), count in counts.most_common(12)]


def main() -> int:
    args = parse_args()
    dataset_dir = Path(args.dataset_dir)
    focus_labels = [label.strip().upper() for label in args.focus_labels.split(",") if label.strip()]

    grouped = load_grouped_dataset(dataset_dir)
    if not grouped:
        print(f"No warped samples found under {dataset_dir}")
        return 1

    full_label_centroids = build_full_label_centroids(grouped)
    rank_centroids = build_axis_centroids(grouped, "rank")
    suit_centroids = build_axis_centroids(grouped, "suit")
    issues = collect_issues(grouped, full_label_centroids, rank_centroids, suit_centroids)
    stale_meta = find_stale_meta_records(dataset_dir)

    focus_summary = summarize_focus_labels(issues, focus_labels, args.top_k)
    render_review_sheet(
        [issue for issue in issues if issue.label in set(focus_labels)],
        Path(args.output_sheet),
        args.sheet_count,
    )

    nearest_card_confusions = Counter((issue.label, issue.nearest_card_label) for issue in issues)
    nearest_rank_confusions = Counter((issue.label[:-1], issue.nearest_rank) for issue in issues)
    nearest_suit_confusions = Counter((issue.label[-1], issue.nearest_suit) for issue in issues)

    report = {
        "dataset_dir": str(dataset_dir),
        "label_count": len(grouped),
        "sample_count": sum(len(samples) for samples in grouped.values()),
        "stale_meta_records": stale_meta,
        "nearest_card_confusions": [[a, b, c] for (a, b), c in nearest_card_confusions.most_common(12)],
        "nearest_rank_confusions": [[a, b, c] for (a, b), c in nearest_rank_confusions.most_common(12)],
        "nearest_suit_confusions": [[a, b, c] for (a, b), c in nearest_suit_confusions.most_common(12)],
        "focus_labels": focus_summary,
    }

    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"Dataset dir: {dataset_dir}")
    print(f"Labels: {report['label_count']}")
    print(f"Samples: {report['sample_count']}")
    print(f"Stale meta records: {len(stale_meta)}")
    if stale_meta:
        for record in stale_meta[:8]:
            print(f"  {record['label']} missing {record['warped_path']}")
    print()
    print("Nearest-card confusions")
    for expected, predicted, count in report["nearest_card_confusions"]:
        print(f"  {expected} -> {predicted}: {count}")
    print()
    print("Nearest-rank confusions")
    for expected, predicted, count in report["nearest_rank_confusions"]:
        print(f"  {expected} -> {predicted}: {count}")
    print()
    print("Nearest-suit confusions")
    for expected, predicted, count in report["nearest_suit_confusions"]:
        print(f"  {expected} -> {predicted}: {count}")
    print()
    for label in focus_labels:
        print(f"Focus label {label}")
        items = report["focus_labels"].get(label, [])
        if not items:
            print("  none")
            continue
        for item in items:
            print(
                f"  {Path(item['path']).name} "
                f"card->{item['nearest_card_label']} ({item['card_margin']:.5f}) "
                f"rank->{item['nearest_rank']} ({item['rank_margin']:.5f}) "
                f"suit->{item['nearest_suit']} ({item['suit_margin']:.5f})"
            )
    print()
    print(f"Saved report: {output_json}")
    print(f"Saved sheet: {args.output_sheet}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
