# Card Dataset Tool

`card_dataset_tool` is a capture-and-label utility for building a real playing-card dataset from your camera setup.

It is designed to support the next steps after `cv/card_identifier_lab/`:

- capture raw camera frames
- detect the dominant card in view
- rectify the card to a normalized top-down image
- save warped samples into label folders such as `AS`, `10H`, or `KC`
- write metadata for later training or template generation

## Why this exists

Before training a detector, it is better to collect consistent real images from:

- your actual deck
- your actual table/background
- your actual camera
- your actual lighting

This tool makes that repeatable.

## Files

- `app.py`
  - live camera capture and labeling tool
- `requirements.txt`
  - Python dependencies
- `dataset/raw/`
  - saved full frames
- `dataset/warped/`
  - saved warped card crops grouped by label
- `dataset/meta.jsonl`
  - one JSON record per saved sample

## Dependencies

```powershell
python -m pip install -r cv/card_dataset_tool/requirements.txt
```

Optional CNN path:

```powershell
python -m pip install -r cv/card_dataset_tool/requirements-cnn.txt
```

`torch` is intentionally not pinned in `requirements-cnn.txt`.
Install it separately for the target machine first, then install the rest of the CNN dependencies.

On Raspberry Pi / Linux ARM64, do not assume `python -m pip install torch` is safe.
It can resolve to a very large wheel plus CUDA-related `nvidia-*` packages, which is a bad fit for a CPU-only Pi and may appear to freeze the machine during download or unpack.

Recommended Pi flow:

```bash
python3.11 -m venv .venv311
source .venv311/bin/activate
python -m pip install --upgrade pip
```

Then install a CPU-appropriate `torch` build for that exact Python version and platform, and only after that run:

```powershell
python -m pip install -r cv/card_dataset_tool/requirements-cnn.txt
```

If the Pi is only for inference or dataset capture, a simpler option is to train the CNN on a stronger machine and copy these files back to the Pi:

- `cv/card_dataset_tool/models/rank_cnn.pt`
- `cv/card_dataset_tool/models/suit_cnn.pt`
- `cv/card_dataset_tool/models/metrics.json`

## Run

```powershell
python cv/card_dataset_tool/app.py
```

Optional example:

```powershell
python cv/card_dataset_tool/app.py --camera 0 --label AS --debug
python cv/card_dataset_tool/app.py --backend rpicam --width 640 --height 480 --fps 15
```

On Raspberry Pi CSI cameras, prefer `--backend rpicam`.
For USB webcams, `--backend opencv` is usually the right choice.

## Baseline Evaluation

Use the saved warped cards to run a first offline recognition check:

```powershell
python cv/card_dataset_tool/eval_dataset.py
```

Useful options:

```powershell
python cv/card_dataset_tool/eval_dataset.py --test-per-label 3
python cv/card_dataset_tool/eval_dataset.py --dataset-dir cv/card_dataset_tool/dataset --seed 7
python cv/card_dataset_tool/eval_dataset.py --mode prototype --mistake-sheet cv/card_dataset_tool/mistakes.jpg
python cv/card_dataset_tool/eval_dataset.py --mode nn --knn-k 3
python cv/card_dataset_tool/eval_dataset.py --mode mlp --mlp-hidden 128 --mlp-epochs 400
python cv/card_dataset_tool/eval_dataset.py --mode nn --min-contour-area 15000
```

What it does:

- loads `dataset/warped/<LABEL>/*.jpg`
- extracts rank and suit patches from the normalized card corner
- supports `prototype`, `nn`, and `mlp` matching modes
- reports rank, suit, and full-card accuracy plus main confusion pairs
- can write a contact sheet of misclassified test samples for visual review

## CNN Training

Train separate rank and suit CNNs on the warped corner patches:

```powershell
python cv/card_dataset_tool/train_patch_cnn.py --test-per-label 3
```

Useful options:

```powershell
python cv/card_dataset_tool/train_patch_cnn.py --epochs 12 --batch-size 32 --lr 0.001
python cv/card_dataset_tool/train_patch_cnn.py --device cpu --output-dir cv/card_dataset_tool/models
python cv/card_dataset_tool/train_patch_cnn.py --min-contour-area 15000
```

This writes:

- `cv/card_dataset_tool/models/rank_cnn.pt`
- `cv/card_dataset_tool/models/suit_cnn.pt`
- `cv/card_dataset_tool/models/metrics.json`

Re-evaluate saved checkpoints later:

```powershell
python cv/card_dataset_tool/eval_patch_cnn.py --test-per-label 3
python cv/card_dataset_tool/eval_patch_cnn.py --test-per-label 3 --min-contour-area 15000
```

Inspect dataset outliers and likely ambiguity clusters:

```powershell
python -m cv.card_dataset_tool.triage_dataset
```

Generate a quarantine manifest for the worst samples in confusion-heavy labels:

```powershell
python -m cv.card_dataset_tool.quarantine_dataset --top-k 4 --min-keep-per-label 20
```

Apply the quarantine by moving those raw/warped pairs under `dataset/quarantine/` and cleaning `meta.jsonl`:

```powershell
python -m cv.card_dataset_tool.quarantine_dataset --top-k 4 --min-keep-per-label 20 --apply
```

If a capture session introduced partial-card warps, use `--min-contour-area` to ignore them based on `meta.jsonl`
without deleting the files from the dataset.

## Controls

- `q`
  - enters `Q` into the label
- `Esc`
  - quit
- `g`
  - toggle debug windows
- `o`
  - toggle overlay drawing
- `space`
  - save sample using the current label
- `u`
  - undo the last saved sample in the current run
- `backspace`
  - delete one character from the current label
- `0-9`, `A`, `J`, `Q`, `K`, `C`, `D`, `H`, `S`
  - edit the current label
- `-`
  - clear the current label

## Label format

Use compact card ids:

- `AS`
  - ace of spades
- `10H`
  - ten of hearts
- `KC`
  - king of clubs

Recommended rank set:

- `A`
- `2` to `10`
- `J`
- `Q`
- `K`

Recommended suit set:

- `C`
- `D`
- `H`
- `S`

## Workflow

1. Put one card in view.
2. Adjust position until the warp looks clean.
3. Type the label.
4. Press `space` to save.
5. Repeat across rotations, distances, lighting conditions, and backgrounds.

## Capture Quality Checklist

Use variation intentionally, but do not keep obviously bad captures.

- keep multiple rotations for each card
- keep small left/right/up/down placement changes
- keep a few distance changes
- keep a few mild perspective changes
- keep the real table, background, and lighting you expect in use
- prefer sharp rank/suit corners over perfectly sharp card edges elsewhere
- only save when the full card is visible
- only save when the warp preview looks rectangular and stable
- reject motion blur
- reject glare that washes out the corner symbols
- reject extreme distance where rank/suit details become too small
- reject ultra-close captures that distort the corner too heavily

Recommended target per card:

- around 25 to 30 total samples
- about 20 to 24 clean varied samples
- at most 4 to 6 harder samples

Practical rule:

- if the warp looks questionable, do not save it
- if the corner is crisp and the framing is meaningfully different, keep it

## Notes

- The current version assumes one dominant visible card.
- It is intended for dataset collection, not final gameplay overlay.
- Later we can derive templates from the saved warped images or train a small classifier on them.
