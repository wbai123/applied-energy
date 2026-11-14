# ieee37mod-ercotsup daily loads

This dataset provides two-day supervisor benchmark for `ieee37mod`.
Each hourly multiplier repeats across the feeder's 30 loads in compiled order.

Conversion and scaling follow the [IEEE13 ERCOT method](../ieee13-ercot/README.md).
Archive dropped days and chronological splits are identical to `ieee13-ercot`.
The benchmark retains October 8–9, 2026, matching `ieee13-ercotsup`.
The 2026 archive peak scales both days, including the permitted forecast fallback.

Regenerate from the repository root:

```bash
uv run src/datasets/ercot.py report --feeder ieee37mod --dataset ieee37mod-ercotsup
```

Add `--overwrite` to replace generated files and preserve this README.
