"""
Reusable sector-level EDA -- run this instead of re-deriving the same
groupby logic in a notebook each time.

load_clean_sector_df(name) is the canonical "give me a ready-to-analyze
dataframe" entry point: load -> normalize categoricals -> apply confirmed cleanup -> assign + merge NAICS sector -> reconcile
BusinessAge buckets -> assign binary target. Every other function here
takes that dataframe as input.

Run directly (`python sector_eda.py`) to print, for every file:
  - sector x LoanStatus summary (mean/median GrossApproval, TermInMonths,
    both chargeoff-rate definitions)
  - the FULL sparse-sector table (all 21 sectors, not just the 4 flagged
    earlier) -- to actually answer "should we run this for every sector"
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent / "src" / "data_prep"))
import load_sba
import naics_grouping
import business_age_mapping

import pandas as pd

ORDER = ["2000_2009", "2010_2019", "2020_Curr"]


def load_clean_sector_df(name):
    """
    Canonical loader: raw -> normalized -> confirmed-cleanup -> NAICS
    sector assigned + sparse sectors merged -> BusinessAge reconciled ->
    binary target assigned. This is the decided, pipeline --
    if you need the sector breakdown BEFORE the sparse-sector merge (e.g.
    to re-check that decision later), call naics_grouping.assign_naics_sector
    directly and skip merge_sparse_sectors.
    """
    df = load_sba.load_raw(load_sba.FILES[name])
    load_sba.normalize_categoricals(df)
    df = load_sba.clean_confirmed_issues(df, verbose=False)
    naics_grouping.assign_naics_sector(df)
    naics_grouping.merge_sparse_sectors(df)
    business_age_mapping.reclassify_business_age(df)
    load_sba.assign_target(df)
    return df


def sector_status_summary(df):
    """
    Group by NaicsSectorName x LoanStatus: mean/median GrossApproval and
    TermInMonths. Separate from the chargeoff-rate table below, since rate
    is a sector-level (not sector x status) stat.
    """
    return df.groupby(["NaicsSectorName", "LoanStatus"]).agg(
        n=("LoanStatus", "size"),
        GrossApproval_mean=("GrossApproval", "mean"),
        GrossApproval_median=("GrossApproval", "median"),
        TermInMonths_mean=("TermInMonths", "mean"),
        TermInMonths_median=("TermInMonths", "median"),
    ).round(0)


def sector_chargeoff_table(df):
    """
    Full sparse-sector table -- every sector, both chargeoff-rate
    definitions (of resolved PIF+CHGOFF only, vs. of all loans including
    CANCLD/EXEMPT/COMMIT), sorted by CHGOFF event count (not loan count -- the number of charge-off events is what makes a sector's rate reliable).
    """
    total = df.groupby("NaicsSectorName").size()
    pif = df[df["LoanStatus"] == "PIF"].groupby("NaicsSectorName").size()
    chgoff = df[df["LoanStatus"] == "CHGOFF"].groupby("NaicsSectorName").size()

    out = pd.DataFrame({"total_loans": total, "PIF": pif, "CHGOFF": chgoff}).fillna(0)
    out["chgoff_rate_of_resolved_pct"] = (out["CHGOFF"] / (out["PIF"] + out["CHGOFF"]) * 100).round(2)
    out["chgoff_rate_of_total_pct"] = (out["CHGOFF"] / out["total_loans"] * 100).round(2)
    return out.sort_values("CHGOFF")


def main():
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 15)
    pd.set_option("display.max_rows", 30)

    for name in ORDER:
        df = load_clean_sector_df(name)
        print(f"\n{'='*30} {name}: sector x LoanStatus summary {'='*30}")
        print(sector_status_summary(df))

        print(f"\n{'='*30} {name}: sector chargeoff table (sorted by CHGOFF event count) {'='*30}")
        print(sector_chargeoff_table(df).to_string())


if __name__ == "__main__":
    main()
