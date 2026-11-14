"""Convert ERCOT archive loads and eligible report days into named multiplier datasets."""

import csv
import datetime as dt
import hashlib
import json
import math
import pathlib
import re
import sys
from collections import defaultdict

import pandas as pd

from common.cli import ArgumentParser, dataset_name
from feeder.loads import dataset_dir
from feeder.simulator import FeederSimulator

ROOT = pathlib.Path(__file__).resolve().parents[2]
ARCHIVE = "data/sources/ercot"
REPORT = (
    "data/sources/ercotsup/cdr.00013499.0000000000000000.20261009.232000.HOURLYLFvsACTUALGEN55.csv"
)
SPLITS = {
    "train": ("2002-01-01", "2023-12-31"),
    "validation": ("2024-01-01", "2025-09-30"),
    "test": ("2025-10-01", "2026-09-30"),
}
STORM_DAYS = {dt.date(2021, 2, day) for day in range(14, 20)}
TEXT_STAMP = re.compile(r"(\d{2}/\d{2}/\d{4}) (\d{2}):00(?: DST)?")


def parse_stamp(value):
    """Map an Excel datetime or archive text stamp to delivery date and hour ending."""
    if isinstance(value, dt.datetime) and not pd.isna(value):
        rounded = pd.Timestamp(value).round("s").to_pydatetime()
        if rounded.minute == rounded.second == rounded.microsecond == 0:
            if rounded.hour == 0:
                return rounded.date() - dt.timedelta(days=1), 24
            return rounded.date(), rounded.hour
    elif isinstance(value, str):
        match = TEXT_STAMP.fullmatch(value)
        if match and 1 <= int(match[2]) <= 24:
            return dt.datetime.strptime(match[1], "%m/%d/%Y").date(), int(match[2])
    raise ValueError(f"unrecognized ERCOT hour stamp: {value!r}")


def archive_rows(frame):
    """Parse one archive table while retaining blank load cells as rows."""
    if len(frame.columns) < 2 or frame.columns[-1] != "ERCOT":
        raise ValueError("archive last column must be ERCOT")
    if frame.columns[0] not in {"Hour_End", "Hour Ending", "HourEnding"}:
        raise ValueError("archive first column must name the hour ending")
    rows = []
    for stamp, load in zip(frame.iloc[:, 0], frame.iloc[:, -1], strict=True):
        day, hour = parse_stamp(stamp)
        value = math.nan if pd.isna(load) or load == "" else float(load)
        rows.append((day, hour, value))
    return rows


def read_archive(source):
    """Read every Excel archive with the pinned calamine engine."""
    source = pathlib.Path(source)
    files = sorted([*source.glob("*.xls"), *source.glob("*.xlsx")])
    if not files:
        raise ValueError(f"no Excel archive files in {source}")
    rows = []
    for path in files:
        try:
            rows.extend(archive_rows(pd.read_excel(path, engine="calamine")))
        except ValueError as error:
            raise ValueError(f"{path.name}: {error}") from error
    return rows, {path.name: sha256(path) for path in files}


def sha256(path):
    """Return a source file's byte checksum."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def annual_peaks(rows):
    """Find finite annual maxima before any delivery day is excluded."""
    peaks = {}
    for day, _hour, load in rows:
        if math.isfinite(load):
            peaks[day.year] = max(peaks.get(day.year, -math.inf), load)
    if not peaks or any(peak <= 0 for peak in peaks.values()):
        raise ValueError("archive requires positive finite annual peaks")
    return dict(sorted(peaks.items()))


def group_rows(rows):
    """Group every source row by delivery date, preserving duplicate hours."""
    days = defaultdict(list)
    for day, hour, load in rows:
        days[day].append((hour, load))
    return dict(sorted(days.items()))


def select_archive_days(rows):
    """Drop incomplete and storm days; reject invalid complete days."""
    kept, row_count, storm = {}, [], []
    for day, hours in group_rows(rows).items():
        if len(hours) != 24:
            row_count.append(day.isoformat())
            continue
        if sorted(hour for hour, _load in hours) != list(range(1, 25)):
            raise ValueError(f"{day}: expected hour endings 1..24 once each")
        if any(not math.isfinite(load) or load <= 0 for _hour, load in hours):
            raise ValueError(f"{day}: loads must be finite and positive")
        if day in STORM_DAYS:
            storm.append(day.isoformat())
            continue
        kept[day] = sorted(hours)
    if not kept:
        raise ValueError("archive has no eligible days")
    return kept, row_count, storm


def split_for(day):
    """Assign a delivery date to its fixed chronological split."""
    for split, (first, last) in SPLITS.items():
        if first <= day.isoformat() <= last:
            return split
    raise ValueError(f"delivery date outside archive split ranges: {day}")


def select_report_days(frame):
    """Keep complete report days with at most one actual load filled from forecast."""
    required = {"DeliveryDate", "HourEnding", "ActualLoad", "CurrentDayForecast"}
    if not required.issubset(frame.columns):
        raise ValueError("report is missing delivery, hour, actual, or forecast columns")
    kept = {}
    for date, group in frame.groupby("DeliveryDate", sort=True):
        day = dt.datetime.strptime(date, "%m/%d/%Y").date()
        if len(group) != 24:
            continue
        hours = pd.to_numeric(group["HourEnding"], errors="coerce")
        if sorted(hours) != list(range(1, 25)):
            continue
        actual = group["ActualLoad"]
        missing = actual.isna() | (actual == "")
        if missing.sum() > 1:
            continue
        resolved = actual.where(~missing, group["CurrentDayForecast"])
        values = pd.to_numeric(resolved, errors="coerce")
        if any(not math.isfinite(value) or value <= 0 for value in values):
            continue
        kept[day] = sorted(zip(hours.astype(int), values, strict=True))
    if not kept:
        raise ValueError("report has no eligible days")
    if len({day.year for day in kept}) != 1:
        raise ValueError("eligible report days span more than one year")
    return dict(sorted(kept.items()))


def relative_path(path, option):
    """Keep source metadata paths relative to the repository."""
    try:
        return pathlib.Path(path).resolve().relative_to(ROOT).as_posix()
    except ValueError:
        raise ValueError(f"--{option} must be a path inside the repository") from None


def generate_dataset(directory, *, mode, source, archive=ARCHIVE, feeder="ieee13", overwrite=False):
    """Validate sources and destinations before writing deterministic generated files."""
    output = pathlib.Path(directory)
    if output.exists() and not overwrite:
        raise ValueError(
            f"data/scenarios/{output.name} exists; use --overwrite to replace its generated files"
        )
    load_names = FeederSimulator(feeder).load_names
    source = pathlib.Path(source)
    metadata = {
        "feeder": feeder,
        "source": relative_path(source, "source"),
        "decimals": 6,
    }
    if mode == "report":
        archive_path = relative_path(archive, "archive")
    if mode == "archive":
        rows, checksums = read_archive(source)
        peaks = annual_peaks(rows)
        days, row_count, storm = select_archive_days(rows)
        splits = {split: {} for split in SPLITS}
        for day, hours in days.items():
            splits[split_for(day)][day] = hours
        metadata.update(
            files=checksums,
            column="ERCOT",
            reference="calendar-year peak",
            peaks_mw=peaks,
            splits={
                split: {"start": first, "end": last, "days": len(splits[split])}
                for split, (first, last) in SPLITS.items()
            },
            dropped={
                "row_count": {"dates": row_count, "reason": "Delivery day does not hold 24 rows."},
                "uri": {
                    "dates": storm,
                    "reason": "Rolling outages make measured load differ from demand.",
                },
            },
        )
    elif mode == "report":
        days = select_report_days(pd.read_csv(source, keep_default_na=False))
        year = next(iter(days)).year
        rows, _checksums = read_archive(archive)
        peaks = annual_peaks(rows)
        if year not in peaks:
            raise ValueError(f"archive has no finite loads for report year {year}")
        splits = {"test": days}
        metadata.update(
            sha256=sha256(source),
            archive=archive_path,
            reference="calendar-year peak",
            reference_year=year,
            peak_mw=peaks[year],
            kept_days=[day.isoformat() for day in days],
        )
    else:
        raise ValueError(f"unknown converter mode: {mode}")
    # Construct every multiplier before creating or replacing any output.
    serialized = {
        split: [
            (
                day.isoformat(),
                hour - 1,
                *([f"{load / peaks[day.year]:.6f}"] * len(load_names)),
            )
            for day, hours in days.items()
            for hour, load in hours
        ]
        for split, days in splits.items()
    }
    metadata_text = json.dumps(metadata, indent=2, allow_nan=False) + "\n"
    output.mkdir(parents=True, exist_ok=True)
    if overwrite:
        for split in SPLITS:
            if split not in serialized:
                (output / f"{split}.csv").unlink(missing_ok=True)
    for split, rows in serialized.items():
        with (output / f"{split}.csv").open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(("day_id", "hour", *load_names))
            writer.writerows(rows)
    (output / "generation.json").write_text(metadata_text)
    return metadata


def main():
    """Parse named conversion settings and refuse invalid inputs without writing."""
    parser = ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest="mode", required=True)
    for mode in ("archive", "report"):
        command = modes.add_parser(mode)
        command.add_argument(
            "--feeder", choices=("ieee13", "ieee37std", "ieee37mod"), default="ieee13"
        )
        command.add_argument("--dataset", type=dataset_name, required=True)
        command.add_argument("--source", default=ARCHIVE if mode == "archive" else REPORT)
        if mode == "report":
            command.add_argument("--archive", default=ARCHIVE)
        command.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    try:
        metadata = generate_dataset(
            dataset_dir(args.dataset),
            mode=args.mode,
            source=ROOT / args.source,
            archive=ROOT / getattr(args, "archive", ARCHIVE),
            feeder=args.feeder,
            overwrite=args.overwrite,
        )
    except (OSError, ValueError) as error:
        print(error, file=sys.stderr)
        return 2
    print(f"wrote data/scenarios/{args.dataset}")
    print(json.dumps(metadata.get("splits", metadata.get("kept_days"))))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
