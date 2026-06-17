#!/usr/bin/env python3
"""Evaluate linear EXR albedo predictions on the MAW 2.0 / GLOW splits."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

import numerical_albedo


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate linear EXR albedo predictions on MAW 2.0 / GLOW measurements."
    )
    parser.add_argument(
        "predictions_dir",
        type=Path,
        help="Directory containing prediction EXRs named {sorted_image_index}_{method}.exr.",
    )
    parser.add_argument("output_path", type=Path, help="Path to write the average score.")
    parser.add_argument(
        "--measurements-root",
        type=Path,
        default=Path("."),
        help="Extracted MAW 2.0 measurement archive root. Default: current directory.",
    )
    parser.add_argument(
        "--meta",
        type=Path,
        required=True,
        help="Metadata CSV path, relative to --measurements-root unless absolute.",
    )
    parser.add_argument(
        "--method",
        required=True,
        help="Method suffix used in prediction filenames, e.g. nerad for 0_nerad.exr.",
    )
    parser.add_argument(
        "--loss",
        default="si",
        choices=numerical_albedo.LOSS_CHOICES,
        help="MAW albedo loss. Use si for intensity, per_si for chromaticity.",
    )
    parser.add_argument(
        "--metric",
        default="mean",
        choices=numerical_albedo.METRIC_CHOICES,
        help="MAW albedo metric. Use mean for intensity, deltae for chromaticity.",
    )
    return parser.parse_args()


def resolve_under(root: Path, path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else root / path


def load_rows(measurements_root: Path, meta: Path) -> list[list[str]]:
    meta_path = resolve_under(measurements_root, meta)
    with meta_path.open(newline="") as handle:
        rows = list(csv.reader(handle, delimiter="\t"))
    for line_no, row in enumerate(rows, start=1):
        if len(row) != 6:
            raise ValueError(f"{meta_path}:{line_no}: expected 6 tab-separated columns")
    return rows


def image_id_to_prediction_index(rows: list[list[str]]) -> dict[str, int]:
    image_ids = sorted(row[3] for row in rows)
    return {image_id: idx for idx, image_id in enumerate(image_ids)}


def evaluate_row(
    row: list[str],
    evaluator: numerical_albedo.AlbedoEvaluator,
    measurements_root: Path,
    predictions_dir: Path,
    method: str,
    mapper: dict[str, int],
) -> float:
    color_lib, mask, _split, image_id, _gt_albedo, _whdr = row
    pred_idx = mapper[image_id]
    prediction = predictions_dir / f"{pred_idx}_{method}.exr"
    metric, weights = evaluator.evaluate(
        str(resolve_under(measurements_root, color_lib)),
        str(resolve_under(measurements_root, mask)),
        str(prediction),
        "linear",
    )
    return float((metric * weights).sum() / weights.sum())


def main() -> None:
    args = parse_args()
    measurements_root = args.measurements_root.resolve()
    predictions_dir = args.predictions_dir.resolve()
    rows = load_rows(measurements_root, args.meta)
    mapper = image_id_to_prediction_index(rows)
    evaluator = numerical_albedo.AlbedoEvaluator(
        loss=args.loss,
        metric=args.metric,
        write_visualizations=False,
    )
    scores = [
        evaluate_row(row, evaluator, measurements_root, predictions_dir, args.method, mapper)
        for row in rows
    ]
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    args.output_path.write_text(f"{np.mean(scores)}\n")


if __name__ == "__main__":
    main()
