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

Run live one-card CNN recognition:

```powershell
python cv/card_dataset_tool/live_patch_cnn.py --backend rpicam --debug
```

In the live tester, type the expected card label, press `space` to log the current prediction, press `s` to save raw/warped images with the log entry, and press `Esc` to quit.

Run the live detection display with full-name captions:

```powershell
python cv/card_dataset_tool/detect_card_app.py --backend rpicam --debug
```

It overlays the card contour and a caption such as `King of Diamonds`. Press `space` to save a captioned frame and `Esc` to quit.
Type the actual card label before pressing `space` to save matched debug files and a CSV row, for example `actual_7H_pred_10H`.

Identify the single card currently under the Pi camera:

```powershell
python cv/card_dataset_tool/identify_card_once.py
```

Recognize four cards in placement order:

```powershell
python cv/card_dataset_tool/identify_card_chain.py --count 4
```

For overlapping cards, place each new card so its visible outline shifts from the previous accepted card. The chain recognizer uses contour-center and area changes to avoid treating the old visible card as the next card.

For real gameplay-style overlapping, use frame-difference chain recognition. It captures the current pile as a baseline, waits for each newly placed card to change the image, recognizes the changed top card, then refreshes the baseline:
By default, frame difference is only used as the trigger; recognition still uses the full-frame card contour, which is better when the newly placed top card is fully visible and older cards remain partly visible underneath.

```powershell
python cv/card_dataset_tool/identify_card_diff_chain.py --count 4
```

For gameplay-style overlap where the full card outline is partly covered, use the index-corner chain recognizer. It waits for frame change, finds changed rank/suit ink groups, classifies the best visible index corner, then refreshes the baseline:

```powershell
python cv/card_dataset_tool/identify_card_index_chain.py --count 4
```

This mode requires at least one rank/suit index corner of each newly placed card to remain visible.

Inspect weak samples and confusion clusters:

```powershell
python -m cv.card_dataset_tool.triage_dataset
python -m cv.card_dataset_tool.quarantine_dataset --top-k 4 --min-keep-per-label 20
```

## Dataset Notes

- Primary dataset lives under `cv/card_dataset_tool/dataset/`
- Older local snapshot is kept under `cv/card_dataset_tool/dataset_local_backup_2026-04-03/`
- Current checkpoints and reports live under `cv/card_dataset_tool/models/`
- July 21, 2026 recapture added `60` samples across `6C`, `7C`, `8C`, `8D`, `10D`, `7H`, and `9C`
- Current dataset count after that recapture is `1140` raw images, `1140` warped images, and `1140` metadata records

## Current Local Baseline

From `cv/card_dataset_tool/models/metrics.json`:

- rank accuracy: `0.8981`
- suit accuracy: `0.9444`
- full-card accuracy: `0.8519`
- promoted seed: `7`

## Latest Eval

After the July 21, 2026 recapture and test pass on the Raspberry Pi checkout:

- rank accuracy: `0.944`
- suit accuracy: `0.991`
- full-card accuracy: `0.944`

Remaining card mistakes:

- `10D -> 8D`
- `10H -> 7C`
- `6C -> 8C`
- `7H -> 6H`
- `8D -> 7D`
- `8H -> 6H`

Note: the model checkpoints and `metrics.json` were not timestamp-updated by that run, so the saved model promotion path still needs to be checked before treating those artifacts as the new baseline.

## Next Improvement Loop

1. Run live one-card CNN recognition with `cv/card_dataset_tool/live_patch_cnn.py`.
2. Log repeatable live misses, especially `10D`, `10H`, `6C`, `7H`, `8D`, and `8H`.
3. Save raw/warped images only for repeatable misses or unstable warps.
4. Re-run training/eval after enough real misses are collected.
