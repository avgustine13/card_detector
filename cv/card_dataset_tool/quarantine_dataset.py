from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Dict, List, Sequence

from cv.card_dataset_tool.triage_dataset import (
    SampleIssue,
    build_axis_centroids,
    build_full_label_centroids,
    collect_issues,
    find_stale_meta_records,
    load_grouped_dataset,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate or apply a quarantine list for suspect dataset samples.")
    parser.add_argument("--dataset-dir", default=str(Path(__file__).with_name("dataset")), help="Dataset root.")
    parser.add_argument(
        "--focus-labels",
        default="AS,AH,AC,AD,10H,10D,7C,6C,8C,6H,8D,7D",
        help="Comma-separated labels to target for quarantine review.",
    )
    parser.add_argument("--top-k", type=int, default=4, help="Maximum suspect samples per focus label.")
    parser.add_argument(
        "--min-keep-per-label",
        type=int,
        default=20,
        help="Do not quarantine below this remaining sample count per label.",
    )
    parser.add_argument(
        "--output-json",
        default=str(Path(__file__).with_name("models") / "quarantine_candidates.json"),
        help="Path for the quarantine manifest JSON.",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Move listed samples into dataset/quarantine and rewrite meta.jsonl without quarantined or stale records.",
    )
    return parser.parse_args()


def load_meta_records(dataset_dir: Path) -> List[dict]:
    meta_path = dataset_dir / "meta.jsonl"
    if not meta_path.exists():
        return []
    return [json.loads(line) for line in meta_path.read_text(encoding="utf-8").splitlines() if line.strip()]


def issue_sort_key(issue: SampleIssue) -> tuple[float, float, float, float]:
    return (issue.card_margin, issue.rank_margin, issue.suit_margin, -issue.outlier_score)


def select_quarantine_candidates(
    grouped: Dict[str, list],
    issues: Sequence[SampleIssue],
    focus_labels: Sequence[str],
    top_k: int,
    min_keep_per_label: int,
) -> List[SampleIssue]:
    by_label: Dict[str, List[SampleIssue]] = {}
    for issue in issues:
        by_label.setdefault(issue.label, []).append(issue)

    selected: List[SampleIssue] = []
    for label in focus_labels:
        label_issues = sorted(by_label.get(label, []), key=issue_sort_key)
        allowed = max(0, len(grouped.get(label, [])) - min_keep_per_label)
        for issue in label_issues[: min(top_k, allowed)]:
            selected.append(issue)
    return selected


def build_manifest_entries(dataset_dir: Path, meta_records: Sequence[dict], candidates: Sequence[SampleIssue]) -> List[dict]:
    by_warped: Dict[str, dict] = {}
    for record in meta_records:
        warped_path = record.get("warped_path")
        if warped_path:
            by_warped[str(Path(warped_path).as_posix())] = record

    entries: List[dict] = []
    for issue in candidates:
        rel_warped = Path(issue.path).relative_to(dataset_dir).as_posix()
        record = by_warped.get(rel_warped, {})
        entries.append(
            {
                "label": issue.label,
                "timestamp": record.get("timestamp"),
                "raw_path": record.get("raw_path"),
                "warped_path": rel_warped,
                "nearest_card_label": issue.nearest_card_label,
                "card_margin": issue.card_margin,
                "nearest_rank": issue.nearest_rank,
                "rank_margin": issue.rank_margin,
                "nearest_suit": issue.nearest_suit,
                "suit_margin": issue.suit_margin,
                "outlier_score": issue.outlier_score,
            }
        )
    return entries


def apply_quarantine(dataset_dir: Path, entries: Sequence[dict], stale_records: Sequence[dict], meta_records: Sequence[dict]) -> dict:
    quarantine_root = dataset_dir / "quarantine"
    moved_raw = 0
    moved_warped = 0

    quarantine_rel_paths = {entry["warped_path"] for entry in entries}
    stale_rel_paths = {record["warped_path"] for record in stale_records}

    for entry in entries:
        raw_rel = entry.get("raw_path")
        warped_rel = entry.get("warped_path")

        if raw_rel:
            src_raw = dataset_dir / raw_rel
            dst_raw = quarantine_root / raw_rel
            if src_raw.exists():
                dst_raw.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(src_raw), str(dst_raw))
                moved_raw += 1

        if warped_rel:
            src_warped = dataset_dir / warped_rel
            dst_warped = quarantine_root / warped_rel
            if src_warped.exists():
                dst_warped.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(src_warped), str(dst_warped))
                moved_warped += 1

    kept_records = []
    removed_meta = 0
    for record in meta_records:
        warped_rel = record.get("warped_path")
        if warped_rel in quarantine_rel_paths or warped_rel in stale_rel_paths:
            removed_meta += 1
            continue
        kept_records.append(record)

    meta_path = dataset_dir / "meta.jsonl"
    content = "\n".join(json.dumps(record) for record in kept_records)
    if content:
        content += "\n"
    meta_path.write_text(content, encoding="utf-8")

    return {
        "moved_raw": moved_raw,
        "moved_warped": moved_warped,
        "removed_meta_records": removed_meta,
        "quarantine_root": str(quarantine_root),
    }


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
    stale_records = find_stale_meta_records(dataset_dir)
    meta_records = load_meta_records(dataset_dir)

    candidates = select_quarantine_candidates(grouped, issues, focus_labels, args.top_k, args.min_keep_per_label)
    manifest_entries = build_manifest_entries(dataset_dir, meta_records, candidates)

    manifest = {
        "dataset_dir": str(dataset_dir),
        "focus_labels": focus_labels,
        "top_k": args.top_k,
        "min_keep_per_label": args.min_keep_per_label,
        "stale_meta_records": stale_records,
        "quarantine_candidates": manifest_entries,
    }

    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"Dataset dir: {dataset_dir}")
    print(f"Focus labels: {' '.join(focus_labels)}")
    print(f"Stale meta records: {len(stale_records)}")
    for record in stale_records:
        print(f"  stale {record['label']} {record['warped_path']}")
    print(f"Quarantine candidates: {len(manifest_entries)}")
    for entry in manifest_entries:
        print(
            f"  {entry['label']} {Path(entry['warped_path']).name} "
            f"card->{entry['nearest_card_label']} ({entry['card_margin']:.5f}) "
            f"rank->{entry['nearest_rank']} ({entry['rank_margin']:.5f}) "
            f"suit->{entry['nearest_suit']} ({entry['suit_margin']:.5f})"
        )
    print(f"Saved manifest: {output_json}")

    if args.apply:
        result = apply_quarantine(dataset_dir, manifest_entries, stale_records, meta_records)
        print("Applied quarantine")
        print(f"  moved raw: {result['moved_raw']}")
        print(f"  moved warped: {result['moved_warped']}")
        print(f"  removed meta records: {result['removed_meta_records']}")
        print(f"  quarantine root: {result['quarantine_root']}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
