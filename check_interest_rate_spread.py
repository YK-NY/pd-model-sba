"""
One-off comparison: InitialInterestRate vs. Prime rate (FRED MPRIME) at
ApprovalDate, across all three FOIA files, to inform the decision on
whether an implied-spread-over-Prime feature should replace the raw rate
everywhere (not just patch the FY2000-2009 gap).

Builds a temp merged table per file (does not modify the loaded dataframes
in load_sba.py, and nothing here is written back into the main pipeline).

Run: python check_interest_rate_spread.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent / "src" / "data_prep"))
import load_sba

import pandas as pd

MPRIME_PATH = str(load_sba._DATA_RAW / "MPRIME.csv")

FILES = load_sba.FILES
ORDER = ["2000_2009", "2010_2019", "2020_Curr"]


def load_prime():
    prime = pd.read_csv(MPRIME_PATH, parse_dates=["observation_date"])
    prime["year_month"] = prime["observation_date"].dt.to_period("M")
    return prime.set_index("year_month")["MPRIME"]


def main():
    prime = load_prime()
    print(f"Prime rate series: {prime.index.min()} to {prime.index.max()}, {len(prime)} monthly points\n")

    for name in ORDER:
        df = load_sba.load_raw(FILES[name])

        # temp table only -- ApprovalDate + InitialInterestRate, joined to Prime
        # at the ApprovalDate's month. Doesn't touch/modify the loaded df.
        temp = df[["ApprovalDate", "InitialInterestRate"]].copy()
        temp["year_month"] = temp["ApprovalDate"].dt.to_period("M")
        temp["prime_at_approval"] = temp["year_month"].map(prime)
        temp["spread"] = temp["InitialInterestRate"] - temp["prime_at_approval"]

        n_total = len(temp)
        n_rate_missing = temp["InitialInterestRate"].isna().sum()
        n_prime_missing = temp["prime_at_approval"].isna().sum()  # ApprovalDate outside MPRIME.csv range
        spread = temp["spread"].dropna()

        print(f"=== {name} ===")
        print(f"rows: {n_total}  |  InitialInterestRate missing: {n_rate_missing} "
              f"({n_rate_missing/n_total*100:.1f}%)  |  ApprovalDate outside Prime series range: {n_prime_missing}")
        print(f"spread = InitialInterestRate - Prime, n={len(spread)}")
        print(spread.describe(percentiles=[.01, .05, .25, .5, .75, .95, .99]))
        print()


if __name__ == "__main__":
    main()
