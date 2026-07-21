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
- Latest dataset update: July 21, 2026 recapture for `6C`, `7C`, `8C`, `8D`, `10D`, `7H`, and `9C`
- Dataset size after recapture: `1140` raw images, `1140` warped images, `1140` metadata records
- Verified training environment:
  - Raspberry Pi 5
  - Python `3.11.2`
  - venv-based install at `/home/avgustine/home_fortress/.venv-capture`
  - `torch` available in that venv
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

## Latest Recapture Test

On July 21, 2026, the next capture checklist was executed on the Raspberry Pi checkout at
`/home/avgustine/card_detector`.

Added samples:

- `6C`: `10`
- `7C`: `10`
- `8C`: `10`
- `8D`: `10`
- `10D`: `10`
- `7H`: `5`
- `9C`: `5`

Post-capture triage:

- samples: `1140`
- labels: `36`
- stale metadata records: `0`
- updated `cv/card_dataset_tool/models/triage_report.json`
- updated `cv/card_dataset_tool/models/triage_sheet.jpg`

Evaluation after training command completed:

- rank accuracy: `0.944`
- suit accuracy: `0.991`
- full-card accuracy: `0.944`

Remaining eval card confusions:

- `10D -> 8D`
- `10H -> 7C`
- `6C -> 8C`
- `7H -> 6H`
- `8D -> 7D`
- `8H -> 6H`

Important caveat:

- `rank_cnn.pt`, `suit_cnn.pt`, and `metrics.json` timestamps did not update after the July 21 training run.
- Treat `metrics.json` as the last promoted checkpoint baseline until the training-save/promotion behavior is checked.

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

Promoted-checkpoint `card_confusions` from `metrics.json`:

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

Promoted-checkpoint `suit_confusions` from `metrics.json`:

- `C -> S`
- `D -> S`
- `H -> D`
- `H -> S`

## Resume Plan

When resuming work:

1. Inspect why the July 21 training run did not update `rank_cnn.pt`, `suit_cnn.pt`, or `metrics.json`.
2. Confirm whether checkpoint promotion is intentionally guarded, writing elsewhere, or failing silently.
3. Re-run evaluation against the intended promoted checkpoints:

```powershell
python -m cv.card_dataset_tool.eval_patch_cnn --test-per-label 3
```

4. If promotion is fixed and `card_accuracy` remains above `0.8519`, update the promoted model artifacts and `metrics.json`.
5. Review the remaining July 21 eval mistakes listed above before another capture pass.
