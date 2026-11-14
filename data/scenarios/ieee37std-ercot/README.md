# ieee37std-ercot daily loads

This dataset provides archive training, validation, and reserved test splits for `ieee37std`.
Each hourly multiplier repeats across the feeder's 30 loads in compiled order.

Conversion and scaling follow the [IEEE13 ERCOT method](../ieee13-ercot/README.md).
Archive dropped days and chronological splits are identical to `ieee13-ercot`.
Each multiplier equals ERCOT load divided by its calendar-year peak.

Regenerate from the repository root:

```bash
uv run src/datasets/ercot.py archive --feeder ieee37std --dataset ieee37std-ercot
```

Add `--overwrite` to replace generated files and preserve this README.
