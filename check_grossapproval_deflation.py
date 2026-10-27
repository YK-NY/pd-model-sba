"""
One-off comparison: nominal GrossApproval vs. CPI-deflated GrossApproval
(deflated to today's dollars using FRED CPIAUCNS), across all three FOIA
files, to decide whether loan size should be nominal or deflated (the final model uses nominal).

Builds a temp table per file -- does not modify load_sba.py or write
anything back into the main pipeline.

Run: python check_grossapproval_deflation.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent / "src" / "data_prep"))
import load_sba

import pandas as pd

CPI_PATH = str(load_sba._DATA_RAW / "CPIAUCNS.csv")

FILES = load_sba.FILES
ORDER = ["2000_2009", "2010_2019", "2020_Curr"]


def load_cpi():
    cpi = pd.read_csv(CPI_PATH, parse_dates=["observation_date"])
    cpi["year_month"] = cpi["observation_date"].dt.to_period("M")
    return cpi.set_index("year_month")["CPIAUCNS"]


def main():
    cpi = load_cpi()
    base_cpi = cpi.iloc[-1]  # most recent month = "today's dollars" base
    print(f"CPI series: {cpi.index.min()} to {cpi.index.max()}, {len(cpi)} monthly points")
    print(f"Base period (deflate TO this): {cpi.index[-1]}, CPIAUCNS={base_cpi:.3f}\n")

    for name in ORDER:
        df = load_sba.load_raw(FILES[name])

        temp = df[["ApprovalDate", "GrossApproval"]].copy()
        temp["year_month"] = temp["ApprovalDate"].dt.to_period("M")
        temp["cpi_at_approval"] = temp["year_month"].map(cpi)
        temp["deflator"] = base_cpi / temp["cpi_at_approval"]
        temp["GrossApproval_real"] = temp["GrossApproval"] * temp["deflator"]

        n_total = len(temp)
        n_cpi_missing = temp["cpi_at_approval"].isna().sum()  # ApprovalDate outside CPI series range

        print(f"=== {name} ===")
        print(f"rows: {n_total}  |  ApprovalDate outside CPI series range: {n_cpi_missing}")
        print(f"deflator range applied: {temp['deflator'].min():.3f} to {temp['deflator'].max():.3f}")
        print()
        print("nominal GrossApproval:")
        print(temp["GrossApproval"].describe(percentiles=[.25, .5, .75, .95]).apply(lambda x: f"{x:,.0f}"))
        print()
        print("CPI-deflated GrossApproval_real (today's dollars):")
        print(temp["GrossApproval_real"].describe(percentiles=[.25, .5, .75, .95]).apply(lambda x: f"{x:,.0f}"))
        print()
        pct_shift = (temp["GrossApproval_real"].median() / temp["GrossApproval"].median() - 1) * 100
        print(f"median shifts by {pct_shift:+.1f}% after deflation\n")


if __name__ == "__main__":
    main()
