# Tomorrow Plan

Date: 2026-04-10

Current committed baseline after guarded multi-seed training:

- `seed`: `7`
- `rank_accuracy`: `0.8981`
- `suit_accuracy`: `0.9444`
- `card_accuracy`: `0.8519`

## Goal

Improve the card CNN beyond the current committed baseline without regressing the saved checkpoints.

## Priority Labels

Replace or rescan only clearly weak captures for:

- `10D`
- `6C`
- `7C`
- `8C`
- `8D`
- `7H`
- `9C`

## Workflow

1. Inspect the exact hard samples from the latest eval mistakes before rescanning.
2. Replace only obviously weak captures instead of doing a bulk rescan.
3. Re-run:

```bash
python -m cv.card_dataset_tool.triage_dataset
python -m cv.card_dataset_tool.train_patch_cnn --test-per-label 3 --seeds 42,7,13,21
python -m cv.card_dataset_tool.eval_patch_cnn --test-per-label 3
```

4. Keep the guarded training flow:

- only promote checkpoints if the new `card_accuracy` beats `0.8519`

## Current Weak Patterns

- `10D -> 9D`
- `6C -> 7C`
- `7C -> 8C`
- `8C` suit confusion
- `8D -> 6D` / `8D -> 7D`
- `7H -> 6H` / `7H -> 10H`
- `9C -> 10C` / `9C -> 9S`
