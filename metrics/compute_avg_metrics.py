
#!/usr/bin/env python3
"""
compute_avg_metrics.py

Add an averaged worksheet (default name "promedio") to an Excel workbook that
contains one sheet per language with identical metric columns.

NEW IN THIS VERSION
-------------------
* --exclude / -x flag lets you skip one or more sheets from the aggregation.
  Example: `-x canario -x catalan`

Usage
-----
python compute_avg_metrics.py input.xlsx [-o output.xlsx]
                                         [--name promedio]
                                         [-x canario] [-x otra]

If -o/--output is omitted, the script overwrites the original file
(a temporary backup copy named <file>.bak is created).

Assumptions
-----------
- All included language sheets share the same column layout.
- A column called "model" identifies the rows to aggregate.
- All other columns hold numeric metrics *except* an optional
  "times(sec)" column formatted like "1.95 ± 0.14".
  The script extracts the first number from that column and treats it as numeric.
"""
import argparse
import re
import shutil
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute average metrics across language sheets."
    )
    parser.add_argument("input", help="Path to the original Excel workbook.")
    parser.add_argument(
        "-o",
        "--output",
        help=(
            "Path for the updated workbook. "
            "If omitted, the original file is overwritten."
        ),
    )
    parser.add_argument(
        "--name",
        default="promedio",
        help="Name for the newly‑created sheet (default: 'promedio').",
    )
    parser.add_argument(
        "-x",
        "--exclude",
        action="append",
        default=[],
        metavar="SHEET",
        help=(
            "Sheet name(s) to exclude from the averaging. "
            "Use multiple -x flags to skip more than one sheet."
        ),
    )
    return parser.parse_args()


def extract_numeric_time(series: pd.Series) -> pd.Series:
    """Extract the first floating number from strings like '1.95 ± 0.14'."""
    return (
        series.astype(str)
        .str.extract(r"([-+]?[0-9]*\.?[0-9]+)", expand=False)
        .astype(float)
    )


def compute_average(df_list):
    """Concatenate data frames and compute the mean of each numeric metric per model."""
    combined = (
        pd.concat(df_list, keys=range(len(df_list)), names=["lang", "row"])
        .copy()
    )

    # Identify numeric columns (everything except 'model' & the original 'times(sec)')
    non_numeric = {"model"}
    if "times(sec)" in combined.columns:
        non_numeric.add("times(sec)")
        combined["times_sec_val"] = extract_numeric_time(combined["times(sec)"])

    numeric_cols = [c for c in combined.columns if c not in non_numeric]

    # Convert safely to numeric
    combined[numeric_cols] = combined[numeric_cols].apply(
        pd.to_numeric, errors="coerce"
    )

    # Average each metric across languages for every model
    avg = (
        combined.groupby("model", sort=False)[numeric_cols]
        .mean()
        .reset_index()
        .sort_values("model")
        .reset_index(drop=True)
    )

    # Optional: rename the parsed time column back to 'times(sec)' for clarity
    if "times_sec_val" in avg.columns:
        avg.rename(columns={"times_sec_val": "times(sec)_avg"}, inplace=True)

    return avg


def main():
    args = parse_args()
    input_path = Path(args.input).expanduser()
    output_path = Path(args.output).expanduser() if args.output else input_path

    # Make a backup if overwriting
    if output_path == input_path:
        backup = input_path.with_suffix(input_path.suffix + ".bak")
        shutil.copy2(input_path, backup)

    # Read the workbook
    xls = pd.ExcelFile(input_path)
    excluded_lower = {s.lower() for s in args.exclude}
    language_sheets = [
        s
        for s in xls.sheet_names
        if s.lower() not in {args.name.lower(), "average", "promedio"}
        and s.lower() not in excluded_lower
    ]

    if not language_sheets:
        raise ValueError(
            "No language sheets were found to aggregate (after applying exclusions)."
        )

    # Load selected language sheets
    frames = [pd.read_excel(xls, sheet_name=sh) for sh in language_sheets]

    # Compute averages
    avg_df = compute_average(frames)

    # Write result
    with pd.ExcelWriter(
        output_path,
        engine="openpyxl",
        mode="a" if output_path.exists() else "w",
        if_sheet_exists="replace",
    ) as writer:
        avg_df.to_excel(writer, sheet_name=args.name, index=False)

    print(f"✅ Added sheet '{args.name}' to {output_path}")


if __name__ == "__main__":
    main()
