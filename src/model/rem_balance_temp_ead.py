"""
TRIAL RUN ONLY -- standalone, not imported by or integrated into any other
script. Purpose: decide whether an amortization-based "theoretical
remaining balance at charge-off" is worth adopting as an EAD proxy or LGD
input. Nothing here changes any existing model or artifact.

Method: for every charged-off loan, compute what the outstanding balance
WOULD BE after MonthsToChargeoff months of on-schedule payments, using the
loan's own stated terms (GrossApproval, InitialInterestRate, TermInMonths)
-- a standard closed-form amortization formula, no payment history needed
or available. Capped at TermInMonths (a loan failing after its own term
should theoretically be near $0 if paid on schedule).

Known, stated bias: assumes on-schedule payments right up until default,
which is almost certainly optimistic -- loans typically go delinquent for
a stretch before the FORMAL charge-off (for loans with terms of 36 months or less, 94-98% take longer than their own term to reach formal charge-off). So this is a plausible LOWER BOUND on true
exposure, not a precise estimate -- worth comparing against GrossApproval
(the current, deliberately conservative UPPER BOUND) rather than trusted
alone.

Population: the 60-month-seasoned CHGOFF population (train_lgd.py's
apply_seasoning_filter=True), for consistency with the rest of the LGD
work.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "features"))
from target_construction import load_model_base

import numpy as np
import pandas as pd


def build_seasoned_chgoff_population():
    raw = load_model_base()
    chgoff = raw[raw["LoanStatus"] == "CHGOFF"].copy()
    months_since_approval = (chgoff["AsOfDate"] - chgoff["ApprovalDate"]).dt.days / 30.44
    chgoff = chgoff[months_since_approval >= 60].copy()
    chgoff["realized_lgd"] = chgoff["GrossChargeOffAmount"] / chgoff["GrossApproval"]
    chgoff["MonthsToChargeoff"] = (chgoff["ChargeOffDate"] - chgoff["ApprovalDate"]).dt.days / 30.44
    return chgoff


def theoretical_remaining_balance(principal, annual_rate_pct, term_months, elapsed_months):
    """Standard amortization remaining-balance formula. Returns NaN where
    term is undefined (<=0) -- these loans are excluded from the trial,
    not silently zeroed."""
    n = term_months
    t = np.minimum(elapsed_months, n)  # cap at term -- can't owe negative theoretical balance
    r = annual_rate_pct / 12 / 100

    with np.errstate(invalid="ignore", divide="ignore"):
        # r > 0: standard amortization formula
        monthly_payment = principal * r * (1 + r) ** n / ((1 + r) ** n - 1)
        balance_amortizing = principal * (1 + r) ** t - monthly_payment * ((1 + r) ** t - 1) / r
        # r == 0: straight-line paydown, no interest
        balance_zero_rate = principal - (principal / n) * t

    balance = np.where(r > 0, balance_amortizing, balance_zero_rate)
    balance = np.where(n > 0, balance, np.nan)
    return np.clip(balance, 0, None)


def main():
    df = build_seasoned_chgoff_population()
    print(f"seasoned CHGOFF population: {len(df)} loans")

    df["theoretical_remaining_balance"] = theoretical_remaining_balance(
        df["GrossApproval"].values, df["InitialInterestRate"].values,
        df["TermInMonths"].values, df["MonthsToChargeoff"].values,
    )
    df["theoretical_remaining_pct"] = df["theoretical_remaining_balance"] / df["GrossApproval"]

    valid = df.dropna(subset=["theoretical_remaining_pct"]).copy()
    excluded = len(df) - len(valid)
    print(f"excluded (undefined term, e.g. TermInMonths<=0): {excluded} loans")
    print(f"valid for comparison: {len(valid)} loans\n")

    print("=== theoretical_remaining_pct vs realized_lgd, overall ===")
    print(valid[["theoretical_remaining_pct", "realized_lgd"]].describe().round(3).to_string())

    valid["gap"] = valid["realized_lgd"] - valid["theoretical_remaining_pct"]
    print(f"\nmean gap (realized_lgd - theoretical_remaining_pct): {valid['gap'].mean():.3f}")
    print(f"median gap: {valid['gap'].median():.3f}")
    print(f"% of loans where realized_lgd > theoretical (i.e. lost MORE than on-schedule balance): "
          f"{(valid['gap'] > 0).mean()*100:.1f}%")

    print("\n=== gap by RevolverStatus ===")
    print(valid.groupby("RevolverStatus")["gap"].agg(["size", "mean", "median"]).round(3).to_string())

    print("\n=== gap by CollateralInd ===")
    print(valid.groupby("CollateralInd")["gap"].agg(["size", "mean", "median"]).round(3).to_string())

    term_bucket = pd.cut(valid["TermInMonths"], bins=[0, 36, 60, 84, 120, 180, 300, 500])
    print("\n=== gap by term bucket ===")
    print(valid.groupby(term_bucket, observed=True)["gap"].agg(["size", "mean", "median"]).round(3).to_string())

    print("\n=== correlation: theoretical_remaining_pct vs realized_lgd ===")
    print(valid[["theoretical_remaining_pct", "realized_lgd"]].corr().iloc[0, 1].round(3))


if __name__ == "__main__":
    main()
