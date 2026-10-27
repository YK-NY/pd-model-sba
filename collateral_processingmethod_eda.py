"""
Collateral rate and chargeoff rate: Express-type ProcessingMethod
(SBAEXPRESSPROGRAM, COMMUNITYEXPRESS) vs. everything else. Checks whether
the elevated risk in Express-type/short-term products lines up with lower
collateralization -- ties the ProcessingMethod and rate/term findings
together with the working-capital-loan-risk literature.

Run: python collateral_processingmethod_eda.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from sector_eda import load_clean_sector_df, ORDER

import pandas as pd

EXPRESS_TYPE_METHODS = {"SBAEXPRESSPROGRAM", "COMMUNITYEXPRESS"}


def main():
    frames = [load_clean_sector_df(name) for name in ORDER]
    pooled = pd.concat(frames, ignore_index=True)

    pooled["method_group"] = pooled["ProcessingMethod"].apply(
        lambda m: "Express-type" if m in EXPRESS_TYPE_METHODS else "Other"
    )

    resolved = pooled[pooled["LoanStatus"].isin(["PIF", "CHGOFF"])]
    summary = resolved.groupby("method_group").agg(
        n=("target", "size"),
        chgoff_rate_pct=("target", lambda s: s.mean() * 100),
        pct_collateral_Y=("CollateralInd", lambda s: (s == "Y").mean() * 100),
    ).round(2)

    print("Express-type (SBAEXPRESSPROGRAM/COMMUNITYEXPRESS) vs Other ProcessingMethod:")
    print(summary)


if __name__ == "__main__":
    main()
