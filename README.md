# Card Detector

This repository contains the playing-card detection and recognition work extracted from `home_fortress`.

## Contents

- `cv/card_identifier_lab/`
  - single-card contour detection and perspective warp lab
- `cv/card_dataset_tool/`
  - labeled capture tool, dataset utilities, CNN training, evaluation, and quarantine flow
- `cv/card_common/`
  - shared camera helpers used by the card tools
- `AI_CONTEXT.md`
  - card-specific working context and latest known status

## Setup

Base capture and evaluation dependencies:

```powershell
python -m pip install -r cv/card_identifier_lab/requirements.txt
python -m pip install -r cv/card_dataset_tool/requirements.txt
```

Optional CNN dependencies:

```powershell
python -m pip install -r cv/card_dataset_tool/requirements-cnn.txt
```

`torch` is intentionally not pinned there. Install an appropriate CPU build for the target machine first.

## Main Workflows

Run the single-card contour and warp lab:

```powershell
python cv/card_identifier_lab/app.py
```

Run the labeled dataset capture tool:

```powershell
python cv/card_dataset_tool/app.py
```

Evaluate saved warped cards:

```powershell
python cv/card_dataset_tool/eval_dataset.py --test-per-label 3
```

Train the rank and suit CNNs:

```powershell
python cv/card_dataset_tool/train_patch_cnn.py --test-per-label 3
```

Evaluate saved CNN checkpoints:

```powershell
python cv/card_dataset_tool/eval_patch_cnn.py --test-per-label 3
```

Inspect weak samples and confusion clusters:

```powershell
python -m cv.card_dataset_tool.triage_dataset
python -m cv.card_dataset_tool.quarantine_dataset --top-k 4 --min-keep-per-label 20
```

## Dataset Notes

- Primary dataset lives under `cv/card_dataset_tool/dataset/`
- Older local snapshot is kept under `cv/card_dataset_tool/dataset_local_backup_2026-04-03/`
- Current checkpoints and reports live under `cv/card_dataset_tool/models/`

## Current Local Baseline

From `cv/card_dataset_tool/models/metrics.json`:

- rank accuracy: `0.8981`
- suit accuracy: `0.9444`
- full-card accuracy: `0.8519`
- promoted seed: `7`

## Next Improvement Loop

1. Inspect the exact mistakes in the eval output and triage sheet.
2. Replace only obviously weak captures for confusion-heavy labels.
3. Re-run triage, training, and evaluation.
4. Promote checkpoints only if `card_accuracy` improves beyond `0.8519`.
