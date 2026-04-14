from __future__ import annotations

import json
from pathlib import Path
from typing import Dict


def load_meta_index(dataset_dir: Path) -> Dict[str, dict]:
    meta_path = dataset_dir / "meta.jsonl"
    if not meta_path.exists():
        return {}

    index: Dict[str, dict] = {}
    for line in meta_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        warped_path = record.get("warped_path")
        if warped_path:
            index[str(Path(warped_path).as_posix())] = record
    return index


def include_image_path(
    dataset_dir: Path,
    image_path: Path,
    meta_index: Dict[str, dict],
    min_contour_area: float = 0.0,
) -> bool:
    if min_contour_area <= 0.0:
        return True

    rel_path = image_path.relative_to(dataset_dir).as_posix()
    record = meta_index.get(rel_path)
    if record is None:
        return True

    contour_area = float(record.get("contour_area", 0.0))
    return contour_area >= min_contour_area
