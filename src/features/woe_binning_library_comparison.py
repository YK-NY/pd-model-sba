"""
Sanity-check ONLY: re-bins the same candidate predictors using the
`optbinning` library (OptimalBinning), so the hand-written bins in
woe_binning.py can be eyeballed against a well-established library's
output. Not part of the modeling pipeline, not imported anywhere else --
results saved separately and are for review only.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import TRAIN_YEARS_MAX
from woe_binning import (
    build_modeling_population, NUMERIC_CANDIDATES, CATEGORICAL_CANDIDATES, MIN_BIN_N,
)

import pandas as pd
from optbinning import OptimalBinning

ARTIFACTS_DIR = str(Path(__file__).resolve().parents[2] / "outputs" / "model_artifacts")


def bin_with_library(df, col, dtype, target_col="target_60mo"):
    work = df[[col, target_col]].dropna()
    optb = OptimalBinning(name=col, dtype=dtype, solver="cp", min_bin_size=MIN_BIN_N / len(work))
    optb.fit(work[col].values, work[target_col].values)
    table = optb.binning_table.build()
    return optb, table


def main():
    df = build_modeling_population()
    df = df[df["ApprovalFY"] <= TRAIN_YEARS_MAX]   # bins are judged on training loans only
    print(f"modeling population (training years only): {len(df)} rows\n")

    report_lines = []
    for col in NUMERIC_CANDIDATES:
        optb, table = bin_with_library(df, col, "numerical")
        iv = optb.binning_table.iv
        header = f"--- {col} (numeric, library): {len(table) - 2} bins, IV={iv:.3f} ---"
        print(header)
        print(table.to_string())
        print()
        report_lines.append(header + "\n" + table.to_string() + "\n")

    for col in CATEGORICAL_CANDIDATES:
        optb, table = bin_with_library(df, col, "categorical")
        iv = optb.binning_table.iv
        header = f"--- {col} (categorical, library): {len(table) - 2} bins, IV={iv:.3f} ---"
        print(header)
        print(table.to_string())
        print()
        report_lines.append(header + "\n" + table.to_string() + "\n")

    Path(ARTIFACTS_DIR).mkdir(parents=True, exist_ok=True)
    out_path = f"{ARTIFACTS_DIR}/woe_binning_library_comparison.txt"
    with open(out_path, "w") as f:
        f.write("\n".join(report_lines))
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
