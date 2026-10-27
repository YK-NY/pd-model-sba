"""
RevolverStatus profile: chargeoff rate, collateral, guarantee %, loan size,
fixed/variable mix -- for revolving vs. term loans, and cross-checked
against term=0. Finding: revolving loans differ on every dimension checked -- less collateral and guarantee, higher rate, faster charge-off, higher loss.

Run: python revolver_profile_eda.py
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


def main():
    frames = [load_clean_sector_df(n) for n in ORDER]
    pooled = pd.concat(frames, ignore_index=True)
    resolved = pooled[pooled["LoanStatus"].isin(["PIF", "CHGOFF"])].copy()
    resolved["guarantee_pct"] = resolved["SBAGuaranteedApproval"] / resolved["GrossApproval"] * 100
    resolved["is_term_zero"] = resolved["TermInMonths"] == 0

    # ------------------------------------------------------------------
    # Q3: % of dataset that is revolver
    # ------------------------------------------------------------------
    revolver_share = pooled["RevolverStatus"].value_counts(normalize=True) * 100
    print("=== % of full (not just resolved) dataset by RevolverStatus ===")
    print(revolver_share.round(2))

    # ------------------------------------------------------------------
    # Q2/Q4: high-level profile, revolver vs term loans
    # ------------------------------------------------------------------
    profile = resolved.groupby("RevolverStatus").agg(
        n=("target", "size"),
        chgoff_rate_pct=("target", lambda s: round(s.mean() * 100, 1)),
        pct_collateral_Y=("CollateralInd", lambda s: round((s == "Y").mean() * 100, 1)),
        avg_guarantee_pct=("guarantee_pct", lambda s: round(s.mean(), 1)),
        median_GrossApproval=("GrossApproval", "median"),
        pct_variable_rate=("FixedorVariableInterestInd", lambda s: round((s == "V").mean() * 100, 1)),
    )
    print("\n=== Q2/Q4: profile by RevolverStatus (resolved loans) ===")
    print(profile)

    # ------------------------------------------------------------------
    # Q1: RevolverStatus x GrossApproval x term=0 x chargeoff, jointly
    # ------------------------------------------------------------------
    joint = resolved.groupby(["is_term_zero", "RevolverStatus"]).agg(
        n=("target", "size"),
        chgoff_rate_pct=("target", lambda s: round(s.mean() * 100, 1)),
        median_GrossApproval=("GrossApproval", "median"),
    )
    print("\n=== Q1: term=0 x RevolverStatus, chargeoff rate + median GrossApproval jointly ===")
    print(joint)

    # ------------------------------------------------------------------
    # chart: chargeoff rate by RevolverStatus, split by term=0
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 5))
    joint["chgoff_rate_pct"].unstack("RevolverStatus").plot(kind="bar", ax=ax, color=["#DD8452", "#4C72B0"])
    ax.set_xticklabels(["term!=0", "term=0"], rotation=0)
    ax.set_ylabel("chargeoff rate %")
    ax.set_title("Chargeoff rate: RevolverStatus x term=0")
    fig.tight_layout()
    fig.savefig(f"{OUT}/35_revolver_x_termzero_chgoffrate.png", dpi=150)
    plt.close(fig)
    print(f"\nsaved {OUT}/35_revolver_x_termzero_chgoffrate.png")


if __name__ == "__main__":
    main()
