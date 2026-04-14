# Next Capture Checklist

Date: 2026-04-14

Purpose: replace the most suspect samples in the current dataset before the next triage, training, and eval pass.

Current baseline to beat:

- `rank_accuracy`: `0.8981`
- `suit_accuracy`: `0.9444`
- `card_accuracy`: `0.8519`
- promoted seed: `7`

## Capture Order

1. `6C` and `7C`
2. `8C`
3. `8D`
4. `10D`
5. `7H`
6. `9C`

## Replace First

### `6C`

- `20260407_001030_6C`
- `20260403_005841_6C`
- `20260403_005835_6C`
- `20260409_185730_6C`

Main issue:

- drifting toward `7C`

### `7C`

- `20260409_185633_7C`
- `20260407_002655_7C`
- `20260409_200137_7C`
- `20260407_002636_7C`

Main issue:

- drifting toward `6C`

### `8C`

- `20260409_200113_8C`
- `20260409_184837_8C`
- `20260409_185659_8C`
- `20260409_200109_8C`

Main issue:

- drifting toward `6C`
- weak suit separation

Note:

- `20260409_200113_8C` is the strongest outlier and should be replaced first.

### `8D`

- `20260409_184544_8D`
- `20260403_180726_8D`
- `20260407_002843_8D`
- `20260407_002837_8D`

Main issue:

- drifting toward `6H`, `7H`, and `10D`

### `10D`

- `20260403_181214_10D`
- `20260409_195545_10D`
- `20260403_181222_10D`
- `20260409_185517_10D`

Main issue:

- drifting toward `10H` and `9D`-like neighbors
- weak diamond suit visibility

## Manual Review Needed

These labels are still in the active eval confusion list, but the saved quarantine manifest does not break out exact replacement samples for them:

- `7H`
- `9C`

Next step for those labels:

1. inspect the latest eval mistakes
2. identify exact warped filenames
3. replace only obviously weak captures

## Capture Rules

- keep the full card visible
- keep the index corner crisp
- reject glare over rank or suit
- reject borderline warps
- prefer clean variation over noisy hard samples
- rescan `6C` and `7C` under the same setup so their separation is clearer

## After Capture

Run:

```powershell
python -m cv.card_dataset_tool.triage_dataset
python -m cv.card_dataset_tool.train_patch_cnn --test-per-label 3 --seeds 42,7,13,21
python -m cv.card_dataset_tool.eval_patch_cnn --test-per-label 3
```

Promotion rule:

- only keep new checkpoints if `card_accuracy` beats `0.8519`
