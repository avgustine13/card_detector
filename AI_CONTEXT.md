# AI Context

## Scope

This repository is the extracted playing-card CV work from `home_fortress`.

Main modules:

- `cv/card_identifier_lab/`
- `cv/card_dataset_tool/`
- `cv/card_common/`

## Project Direction

The work started with the simplest useful path first:

1. detect one dominant playing card
2. rectify it into a stable top-down warp
3. extract rank and suit from the indexed corner
4. build a real dataset from the actual deck, camera, and lighting
5. train small offline models on warped card crops

Multi-card and overlay work were intentionally deferred until the single-card pipeline became reliable.

## Current Status

- Active workstream: `cv/card_dataset_tool/`
- Goal: offline card rank and suit recognition from warped card crops using corner patches
- Verified training environment:
  - Raspberry Pi 5
  - Python `3.11.2`
  - venv-based install
  - CPU-only `torch`
- Do not use Raspberry Pi Zero 2 W for `torch` install or CNN training

## Current Local Artifacts

- `cv/card_dataset_tool/models/rank_cnn.pt`
- `cv/card_dataset_tool/models/suit_cnn.pt`
- `cv/card_dataset_tool/models/metrics.json`
- `cv/card_dataset_tool/models/triage_report.json`
- `cv/card_dataset_tool/models/triage_sheet.jpg`
- `cv/card_dataset_tool/models/quarantine_candidates.json`

## Current Local Baseline

From `cv/card_dataset_tool/models/metrics.json`:

- rank accuracy: `0.8981481481481481`
- suit accuracy: `0.9444444444444444`
- full-card accuracy: `0.8518518518518519`
- chosen seed: `7`
- candidate seeds: `42, 7, 13, 21`

## Landed Implementation Changes

- Added the labeled capture tool and dataset metadata flow
- Added offline dataset evaluation modes
- Added separate rank and suit CNN training/evaluation scripts
- Extracted shared patch preprocessing into `cv/card_dataset_tool/patch_preprocess.py`
- Changed patch representation from hard-thresholded binary to grayscale plus histogram equalization
- Rotated warped cards to the corner with the strongest index ink before ROI extraction
- Widened corner ROIs toward the card edge to capture the actual rank and suit indices more reliably
- Added dataset triage and quarantine workflows
- Added guarded multi-seed training and checkpoint promotion

## Main Remaining Error Clusters

Current `card_confusions`:

- `10D -> 9D`
- `6C -> 7C`
- `7C -> 8C`
- `7H -> 6H`
- `7H -> 10H`
- `8C -> 8S`
- `8D -> 6D`
- `8D -> 7D`
- `8H -> 7H`
- `8H -> 6H`
- `9C -> 10C`
- `9C -> 9S`

Current `suit_confusions`:

- `C -> S`
- `D -> S`
- `H -> D`
- `H -> S`

## Resume Plan

When resuming work:

1. Inspect the exact misclassified warped samples from the latest eval output.
2. Separate bad warps and weak captures from true model weaknesses.
3. Replace or rescan only clearly weak examples for high-confusion labels such as:
   - `10D`
   - `6C`
   - `7C`
   - `8C`
   - `8D`
   - `7H`
   - `9C`
4. Re-run:

```powershell
python -m cv.card_dataset_tool.triage_dataset
python -m cv.card_dataset_tool.train_patch_cnn --test-per-label 3 --seeds 42,7,13,21
python -m cv.card_dataset_tool.eval_patch_cnn --test-per-label 3
```

5. Promote checkpoints only if `card_accuracy` beats `0.8519`.
