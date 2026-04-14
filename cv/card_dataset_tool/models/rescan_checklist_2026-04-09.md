# Card Rescan Checklist

Date: 2026-04-09

Purpose: refill the active dataset back to a balanced 30 samples per label after quarantine cleanup.

## Need 4 New Captures Each

- `10D`
- `10H`
- `6C`
- `6H`
- `7C`
- `7D`
- `8C`
- `8D`
- `AC`
- `AD`
- `AH`

## Current Active Counts

- `10D`: 26
- `10H`: 26
- `6C`: 26
- `6H`: 26
- `7C`: 26
- `7D`: 26
- `8C`: 26
- `8D`: 26
- `AC`: 26
- `AD`: 26
- `AH`: 26

## Notes For Capture Session

- Keep `AS` and all non-listed labels unchanged unless you are intentionally replacing weak samples.
- The strongest remaining confusion clusters are `10H/6H`, `6C/7C`, `8D/6H`, and ace suit confusion (`AC/AD/AH/AS`).
- Prioritize cleaner card corner alignment and stronger suit visibility for the listed labels.
- After rescanning, re-run:

```powershell
python -m cv.card_dataset_tool.triage_dataset
python -m cv.card_dataset_tool.eval_dataset --test-per-label 3
```
