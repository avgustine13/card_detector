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
- Checked on the Pi: this is expected with the current `--save-policy if-better` behavior.
- `train_patch_cnn.py` evaluates the existing saved checkpoints on the same held-out split and only saves if the newly trained candidate strictly beats the saved checkpoint's card accuracy.
- The July 21 `0.944` eval was from the already-saved checkpoints on the expanded dataset, so checkpoint files stayed unchanged because the new training run did not strictly beat the saved model on that split.
- Treat `metrics.json` as stale reporting metadata for the old dataset until it is refreshed intentionally; the checkpoint files themselves are still usable for live testing.

## Live Capture Plan

Next step is real live recognition testing rather than another dataset capture pass.

Recommended sequence:

1. Use the existing promoted CNN checkpoints for live one-card recognition testing.
2. Test real cards one at a time under the intended camera and lighting setup.
3. Record failures by actual card label and predicted label.
4. Save new samples only for repeatable misses, bad warps, or unstable index-corner crops.
5. After enough real misses are collected, retrain and promote only if the saved-checkpoint eval improves.

Live testing should focus first on the latest remaining eval mistakes:

- `10D`
- `10H`
- `6C`
- `7H`
- `8D`
- `8H`

Live observations from July 21, 2026:

- `KD` predicted as `KD` with rank confidence `1.0000`, suit confidence `1.0000`.
- One current-card run predicted `AS` with rank confidence `1.0000`, suit confidence `0.9996`.
- One current-card run predicted `QC` with rank confidence `0.9733`, suit confidence `0.9999`.
- `AC` predicted as `AC` with rank confidence `1.0000`, suit confidence `0.7039`; user noted it had previously been recognized as `AS`.
- Treat `AC` versus `AS` / clubs versus spades as an active live ambiguity to retest.
- Chain session `20260721_154843`: position 4 predicted `AS`, user confirmed actual `AH`; record as `AH -> AS`.
- Chain session `20260721_155645`: position 3 predicted `KS`, user confirmed actual `QH`; record as `QH -> KS`.
- Chain session `20260721_155645`: position 4 predicted `KH`, user confirmed actual `KS`; record as `KS -> KH`.
- Hearts/spades separation is now a confirmed live issue, especially around `AH`, `QH`, `KS`, and `KH`.
- Chain session `20260721_171929`: accepted order was `QC`, `6D`, `KH`, `AD`.
- User confirmed position 3 was actually `KD`, so record `KD -> KH`.
- The same session accepted `6D`, but before stabilizing it flickered through `9D`, `KD`, and `7D`; rank instability around `6D` is still visible live.

Color-aware suit model experiment:

- Added a wider suit ROI that keeps more of the lower suit symbol/stem.
- Suit preprocessing now uses three channels: equalized grayscale shape, red-symbol mask, and black-symbol mask.
- The first hard red/black prior was too brittle and was removed.
- New promoted Pi checkpoint after retraining with seeds `42,7,13,21`:
  - rank accuracy: `0.9167`
  - suit accuracy: `0.9630`
  - full-card accuracy: `0.8981`
  - seed: `42`
- This is internally consistent with the new preprocessing, but it is lower than the earlier `0.944` offline saved-checkpoint eval. Treat it as a live-test candidate, not a proven improvement.

Live tester:

```bash
cd /home/avgustine/card_detector
/home/avgustine/home_fortress/.venv-capture/bin/python cv/card_dataset_tool/live_patch_cnn.py --backend rpicam --debug
```

One-shot current-card identifier:

```bash
cd /home/avgustine/card_detector
/home/avgustine/home_fortress/.venv-capture/bin/python cv/card_dataset_tool/identify_card_once.py
```

Four-card order recognizer:

```bash
cd /home/avgustine/card_detector
/home/avgustine/home_fortress/.venv-capture/bin/python cv/card_dataset_tool/identify_card_chain.py --count 4
```

Overlapping-card chain behavior:

- The chain recognizer does not require clearing the table.
- It accepts a repeated predicted label only when the detected contour center or area changes enough from the last accepted card.
- When stacking/overlapping cards, place the new top card so its visible outline shifts from the previously accepted card.
- Chain logs include contour center coordinates to diagnose whether acceptance was blocked by insufficient movement.

For real gameplay-style overlapping, prefer `identify_card_diff_chain.py`. It captures the current pile as a baseline, waits for a newly placed card to change the frame, recognizes the current full-frame top-card contour by default, then refreshes the baseline before waiting for the next card. The older changed-region candidate mode can classify partial exposed regions and is less reliable for normal overlapping play.

Controls:

- type expected card label
- `space`: log current prediction to `cv/card_dataset_tool/models/live_test_log.csv`
- `s`: log and save raw/warped images under `cv/card_dataset_tool/live_captures/`
- `g`: toggle warped debug window
- `-`: clear expected label
- `Esc`: quit

## Landed Implementation Changes

- Added the labeled capture tool and dataset metadata flow
- Added offline dataset evaluation modes
- Added separate rank and suit CNN training/evaluation scripts
- Added live one-card CNN recognition tester with manual observation logging
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

1. Run live one-card recognition with the current saved checkpoints.
2. Capture notes for repeatable live failures, especially `10D`, `10H`, `6C`, `7H`, `8D`, and `8H`.
3. Re-run evaluation as a sanity check when needed:

```powershell
python -m cv.card_dataset_tool.eval_patch_cnn --test-per-label 3
```

4. Refresh `metrics.json` intentionally if we want it to reflect the current dataset eval rather than the original promoted-training run.
5. Do another capture pass only for repeatable real-world misses.
