# Card Identifier Lab

`card_identifier_lab` is the first test project for playing-card recognition in `home_fortress`.

The goal of this lab is to solve the easiest useful version first:

- detect one dominant playing card in view
- rectify it into a flat top-down image
- identify rank and suit from the card corner when templates are available
- save samples for later dataset building and model training

This project is intentionally separate from `video/` and any future multi-card overlay work.

## Current scope

Version 1 focuses on:

- live camera preview
- largest card-like contour detection
- perspective warp to a normalized card image
- optional template matching for rank and suit
- debug overlays and sample capture

It does not yet do:

- multiple card detection
- overlap handling
- trained neural detection
- direct integration with the MJPEG video server

## Files

- `app.py`
  - live test app for camera input and card identification
- `requirements.txt`
  - Python packages for the lab
- `templates/`
  - optional rank/suit templates used by the identifier
- `captures/`
  - saved raw frames and warped card images for dataset building

## Dependencies

```powershell
python -m pip install -r cv/card_identifier_lab/requirements.txt
```

## Run

```powershell
python cv/card_identifier_lab/app.py
```

Optional arguments:

```powershell
python cv/card_identifier_lab/app.py --camera 0 --min-area 25000 --debug
python cv/card_identifier_lab/app.py --backend rpicam --width 640 --height 480 --fps 15
```

On Raspberry Pi CSI cameras, prefer `--backend rpicam`.
For USB webcams, `--backend opencv` is usually the right choice.

## Controls

- `q`
  - quit
- `s`
  - save current raw frame and warped card if available
- `o`
  - toggle overlay drawing
- `d`
  - toggle debug windows

## Template layout

Add grayscale template images if you want identification to work.

Expected files:

- `templates/ranks/A.png`
- `templates/ranks/2.png`
- `templates/ranks/3.png`
- ...
- `templates/ranks/K.png`
- `templates/suits/C.png`
- `templates/suits/D.png`
- `templates/suits/H.png`
- `templates/suits/S.png`

Recommended template source:

- use clean crops from your actual printed cards under your actual camera setup
- prefer the top-left corner rank/suit marks
- keep templates tightly cropped and high contrast

## Recommended next steps

1. Verify contour detection and perspective warp under your table lighting.
2. Build rank/suit templates from your real deck.
3. Save warped cards into `captures/` for dataset collection.
4. Split out a dedicated dataset tool after the card warp stage is stable.
5. Add multi-card detection only after the single-card pipeline is reliable.
