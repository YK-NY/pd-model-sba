"""
For TermInMonths==0 loans: how do they actually resolve, and when?
Breaks down CHGOFF and PIF outcomes by resolution year, sector, and
Fixed/Variable -- and directly tests the leakage hypothesis (does
term=0 correlate with unusually FAST chargeoffs, suggesting the term
field gets zeroed as a side effect of default rather than being an
independent origination-time value).

Run per file separately (not pooled) so results are comparable to
manual notebook checks: python term_zero_resolution_eda.py <name>
e.g. python term_zero_resolution_eda.py 2010_2019
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from sector_eda import load_clean_sector_df, ORDER

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "outputs/figures"


def analyze(name):
    df = load_clean_sector_df(name)
    tz = df[df["TermInMonths"] == 0].copy()
    print(f"\n{'='*20} {name}: TermInMonths==0, n={len(tz)} {'='*20}")
    print(tz["LoanStatus"].value_counts())

    chgoff = tz[tz["LoanStatus"] == "CHGOFF"].copy()
    pif = tz[tz["LoanStatus"] == "PIF"].copy()

    chgoff["MonthsToChargeoff"] = (chgoff["ChargeOffDate"] - chgoff["ApprovalDate"]).dt.days / 30.44
    pif["MonthsToPIF"] = (pif["PaidInFullDate"] - pif["ApprovalDate"]).dt.days / 30.44

    print(f"\nterm=0 CHGOFF: n={len(chgoff)}, MonthsToChargeoff describe:")
    print(chgoff["MonthsToChargeoff"].describe(percentiles=[.1, .25, .5, .75, .9]).round(1))

    print(f"\nterm=0 PIF: n={len(pif)}, MonthsToPIF describe:")
    print(pif["MonthsToPIF"].describe(percentiles=[.1, .25, .5, .75, .9]).round(1))

    # resolution year x outcome x Fixed/Variable
    chgoff["resolution_year"] = chgoff["ChargeOffDate"].dt.year
    pif["resolution_year"] = pif["PaidInFullDate"].dt.year
    print("\nCHGOFF resolution year x FixedorVariableInterestInd:")
    print(pd.crosstab(chgoff["resolution_year"], chgoff["FixedorVariableInterestInd"]))
    print("\nPIF resolution year x FixedorVariableInterestInd:")
    print(pd.crosstab(pif["resolution_year"], pif["FixedorVariableInterestInd"]))

    print("\nterm=0 by sector (CHGOFF vs PIF counts):")
    print(pd.crosstab(tz["NaicsSectorName"], tz["LoanStatus"]))

    # chart: MonthsToChargeoff for term=0 vs all CHGOFF (baseline), same file
    all_chgoff = df[df["LoanStatus"] == "CHGOFF"].copy()
    all_chgoff["MonthsToChargeoff"] = (all_chgoff["ChargeOffDate"] - all_chgoff["ApprovalDate"]).dt.days / 30.44

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(all_chgoff["MonthsToChargeoff"].dropna(), bins=40, alpha=0.5, density=True,
            label=f"all CHGOFF (n={all_chgoff['MonthsToChargeoff'].notna().sum()})", color="#4C72B0")
    ax.hist(chgoff["MonthsToChargeoff"].dropna(), bins=40, alpha=0.6, density=True,
            label=f"term=0 CHGOFF (n={chgoff['MonthsToChargeoff'].notna().sum()})", color="#C44E52")
    ax.set_xlabel("Months from approval to chargeoff")
    ax.set_ylabel("density")
    ax.set_title(f"{name}: MonthsToChargeoff, term=0 vs all CHGOFF loans")
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"{OUT}/28_term_zero_timing_{name}.png", dpi=150)
    plt.close(fig)
    print(f"\nsaved {OUT}/28_term_zero_timing_{name}.png")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        analyze(sys.argv[1])
    else:
        for name in ORDER:
            analyze(name)
