#!/usr/bin/env python3
"""Validate the MAW 2.0 / GLOW measurement companion archive."""

from __future__ import annotations

import argparse
import csv
import sys
import zipfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

EXPECTED_CSVS = [
    "coffee_table_colocated_val.csv",
    "coffee_table_natural_val.csv",
    "window_sill_colocated_val.csv",
    "window_sill_natural_val.csv",
    "shoe_rack_colocated_val.csv",
    "shoe_rack_natural_val.csv",
]
EXPECTED_ROWS = 200
REQUIRED_ROW_SUFFIXES = (".npy", "_mask.png", "_albedo.png")
FORBIDDEN_PARTS = {
    ".ipynb_checkpoints",
    "evaluation",
    "label_me",
    "labels",
    "metrics",
    "parents",
    "raw",
    "results",
    "results_old",
    "source",
}
FORBIDDEN_SUFFIXES = (".ipynb", ".DS_Store")
FORBIDDEN_PREFIXES = ("._", ".smbdelete")
FORBIDDEN_NAME_FRAGMENTS = (
    "home_coffe_table_3",
    "home_staged_window",
    "home_window_sill_01_18",
    "shoe_rack_01_08",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("release_root", type=Path, help="Extracted glow_maw2_measurements_release root")
    parser.add_argument("--zip", dest="zip_path", type=Path, help="Optional ZIP archive to validate")
    return parser.parse_args()


def fail(message: str) -> None:
    raise RuntimeError(message)


def read_rows(path: Path) -> list[list[str]]:
    with path.open(newline="") as handle:
        rows = list(csv.reader(handle, delimiter="\t"))
    for line_no, row in enumerate(rows, start=1):
        if len(row) != 6:
            fail(f"{path}:{line_no}: expected 6 tab-separated columns, got {len(row)}")
    return rows


def check_forbidden_path(path: Path, root: Path) -> None:
    rel = path.relative_to(root)
    rel_text = rel.as_posix()
    if any(fragment in rel_text for fragment in FORBIDDEN_NAME_FRAGMENTS):
        fail(f"legacy split name in release path: {rel}")
    parts = set(rel.parts)
    if parts & FORBIDDEN_PARTS:
        fail(f"forbidden path in release: {rel}")
    name = rel.name
    if len(rel.parts) == 1 and name == "Dockerfile":
        fail(f"evaluation Dockerfile should not be in release: {rel}")
    if name.endswith(FORBIDDEN_SUFFIXES) or name.startswith(FORBIDDEN_PREFIXES):
        fail(f"forbidden temp/authoring file in release: {rel}")


def validate_tree(root: Path) -> Counter[str]:
    if not root.is_dir():
        fail(f"release root is not a directory: {root}")

    for path in root.rglob("*"):
        check_forbidden_path(path, root)

    meta_dir = root / "meta_2_0"
    csvs = sorted(path.name for path in meta_dir.glob("*_val.csv"))
    if csvs != sorted(EXPECTED_CSVS):
        fail(f"expected CSVs {sorted(EXPECTED_CSVS)}, found {csvs}")

    counts: Counter[str] = Counter()
    total_rows = 0
    seen_measurements: set[str] = set()
    for csv_name in EXPECTED_CSVS:
        csv_path = meta_dir / csv_name
        rows = read_rows(csv_path)
        counts[csv_name] = len(rows)
        total_rows += len(rows)
        for row in rows:
            color_lib, mask, split, image_id, gt_albedo, _whdr = row
            expected_split = csv_name.removesuffix(".csv")
            if split != expected_split:
                fail(f"{csv_path}: row for image {image_id} has split {split}, expected {expected_split}")
            npy_path = Path(color_lib)
            if npy_path.suffix != ".npy":
                fail(f"{csv_path}: first column must be .npy, got {color_lib}")
            expected_files = [
                npy_path,
                Path(mask),
                npy_path.with_name(npy_path.stem + "_albedo.png"),
            ]
            if Path(gt_albedo) != expected_files[2]:
                fail(f"{csv_path}: gt albedo mismatch for image {image_id}: {gt_albedo}")
            for rel in expected_files:
                if rel.is_absolute():
                    fail(f"{csv_path}: absolute path is not allowed: {rel}")
                if not (root / rel).is_file():
                    fail(f"{csv_path}: missing referenced file {rel}")
            seen_measurements.add(f"{split}/{image_id}")

    if total_rows != EXPECTED_ROWS:
        fail(f"expected {EXPECTED_ROWS} rows, found {total_rows}")
    if len(seen_measurements) != EXPECTED_ROWS:
        fail(f"expected {EXPECTED_ROWS} unique measurement ids, found {len(seen_measurements)}")

    suffix_counts = Counter()
    for path in (root / "phase_2_0" / "masks").rglob("*"):
        if not path.is_file():
            continue
        name = path.name
        if name.endswith("_albedo_gray.png"):
            fail(f"grayscale output file should not be in release: {path.relative_to(root)}")
        elif name.endswith("_albedo.png"):
            suffix_counts["_albedo.png"] += 1
        elif name.endswith("_mask.png"):
            suffix_counts["_mask.png"] += 1
        elif name.endswith(".npy"):
            suffix_counts[".npy"] += 1
        else:
            fail(f"unexpected mask payload file: {path.relative_to(root)}")
    for suffix in REQUIRED_ROW_SUFFIXES:
        if suffix_counts[suffix] != EXPECTED_ROWS:
            fail(f"expected {EXPECTED_ROWS} {suffix} files, found {suffix_counts[suffix]}")

    counts["rows"] = total_rows
    counts.update(suffix_counts)
    return counts


def validate_zip(zip_path: Path, root_name: str) -> int:
    if not zip_path.is_file():
        fail(f"ZIP archive does not exist: {zip_path}")
    with zipfile.ZipFile(zip_path) as archive:
        names = archive.namelist()
    for name in names:
        rel = Path(name)
        if rel.parts and rel.parts[0] == root_name:
            rel = Path(*rel.parts[1:]) if len(rel.parts) > 1 else Path("")
        if not rel.parts:
            continue
        rel_text = rel.as_posix()
        if any(fragment in rel_text for fragment in FORBIDDEN_NAME_FRAGMENTS):
            fail(f"legacy split name in ZIP path: {name}")
        parts = set(rel.parts)
        if parts & FORBIDDEN_PARTS:
            fail(f"forbidden path in ZIP: {name}")
        filename = rel.name
        if len(rel.parts) == 1 and filename == "Dockerfile":
            fail(f"evaluation Dockerfile should not be in ZIP: {name}")
        if filename.endswith(FORBIDDEN_SUFFIXES) or filename.startswith(FORBIDDEN_PREFIXES):
            fail(f"forbidden temp/authoring file in ZIP: {name}")
    return len(names)


def main() -> None:
    args = parse_args()
    counts = validate_tree(args.release_root.resolve())
    print(f"validated {args.release_root} at {datetime.now(timezone.utc).isoformat()}")
    print(f"csv_files={len(EXPECTED_CSVS)} rows={counts['rows']}")
    print(
        "sidecars="
        f"npy:{counts['.npy']} "
        f"mask_png:{counts['_mask.png']} "
        f"albedo_png:{counts['_albedo.png']}"
    )
    if args.zip_path:
        zip_entries = validate_zip(args.zip_path.resolve(), args.release_root.name)
        print(f"zip_entries={zip_entries}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1)
