# IEEE13 ERCOT daily loads

## Sources

[ERCOT Hourly Load Data Archives](https://www.ercot.com/gridinfo/load/load_hist)
provide measured system load in MW from January 2002 through September 2026.
The 25 original files live in `data/sources/ercot/`.
`generation.json` records their SHA-256 checksums and the conversion settings.

Files from 2002–2016 use Excel datetimes in `Hour_End`, sometimes with millisecond jitter.
Files from 2017–2026 use text stamps in `Hour Ending` or `HourEnding`, except for one real Excel datetime cell in `Native_Load_2022.xlsx` (2022-12-01 HE1), which the datetime rule handles.
The converter reads both Excel formats with the pinned calamine engine.

## Conversion

The converter uses only the last column, `ERCOT`.
Hour ending 1 becomes hour 0; hour ending 24 becomes hour 23.
An old-format midnight stamp belongs to the preceding delivery date.
A new-format `24:00` stamp belongs to its stated delivery date.
Repeated autumn hours carry a ` DST` suffix only in the 2017–2026 text format.
Each hourly multiplier repeats across all IEEE13 loads in the configured load order.
ISO delivery dates identify days; rows follow date and hour order.

## Scaling

`multiplier = load_MW / calendar_year_peak_MW`

The reference uses finite values from every source row before dropping days.
Annual normalisation removes load growth: annual peaks rise from about 56 GW in 2002 to 91 GW in 2026.
Peak normalisation keeps each year's upper multiplier known.
Each year's peak hour equals the feeder's nominal case load.
Yearly lows fall near 0.32–0.46; CSV multipliers use six decimals.

The 2026 reference uses January–September only, with a peak of 91,133.727696 MW.
Later 2026 observations do not alter that reference.
A larger later peak would change scaling in a newly generated full-year dataset.
Annual references use the whole available calendar year, including hours after an individual day.

## Excluded days

The archive contains 9,039 delivery days.
The converter drops 49 daylight-saving days with 23 or 25 source rows.
Blank cells still count as source rows and do not contribute to annual peaks.
The blank on 2016-11-06 belongs to an excluded 25-row day.
Retained days require hours ending 1–24 once each and finite positive loads.

The converter also drops 2021-02-14 through 2021-02-19, during Winter Storm Uri.
Rolling outages make measured load differ from demand on those six days.
The metadata lists every excluded date and its reason.

## Chronological splits

| Split | Delivery dates | Days |
|---|---|---:|
| Train | 2002-01-01–2023-12-31 | 7,985 |
| Validation | 2024-01-01–2025-09-30 | 636 |
| Test | 2025-10-01–2026-09-30 | 363 |

Chronological blocks prevent interleaved neighbouring days from leaking across splits.
Validation covers 2024 and the first nine months of 2025; the test block covers the most recent year.
Do not use the archive test split for this round's controller evaluation or model selection.
The archive ends before the supervisor benchmark days.

## Supervisor benchmark

`ieee13-ercotsup` uses the report in `data/sources/ercotsup/`.
It retains October 8–9, 2026, using actual loads and one available current-day forecast for missing HE24.
The general rule permits at most one missing actual load per complete day with a valid forecast fallback.
October 10 has only forecasts and does not qualify.
The benchmark uses the 2026 archive peak with no further factor.

## Alternatives for new datasets

- Annual-mean reference.
- Trailing 365-day peak reference without future data.
- Weather-zone profiles for different load groups.
- Shape templates with noise.
- Other date splits or a recent-years-only window.

## Regeneration

Run these commands from the repository root for fresh dataset directories:

```bash
uv run src/datasets/ercot.py archive --feeder ieee13 --dataset ieee13-ercot
uv run src/datasets/ercot.py report --feeder ieee13 --dataset ieee13-ercotsup
```

Add `--overwrite` to regenerate existing generated files; the converter preserves this README.
Regenerated inputs require retraining controllers and revalidating outputs.
