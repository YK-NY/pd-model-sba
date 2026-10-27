"""
term=0 cross-checked against LocationID (lender), geography (zip3 tier),
CollateralInd, RevolverStatus -- plus GrossApproval/SBAGuaranteedApproval
by lender. Term=0 loans default at roughly 77% (resolved basis), which is why they are examined closely.

Run: python term_zero_crosscheck_eda.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from sector_eda import load_clean_sector_df, ORDER
from geography_lender_eda import build_geo_tiers

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "outputs/figures"


def main():
    frames = [load_clean_sector_df(n) for n in ORDER]
    pooled = pd.concat(frames, ignore_index=True)
    resolved = pooled[pooled["LoanStatus"].isin(["PIF", "CHGOFF"])].copy()
    resolved["is_term_zero"] = resolved["TermInMonths"] == 0

    # ------------------------------------------------------------------
    # 1. term=0 rate by lender (top 20 by volume), with that lender's
    #    overall chargeoff rate and avg loan size / guarantee % alongside
    # ------------------------------------------------------------------
    resolved["guarantee_pct"] = resolved["SBAGuaranteedApproval"] / resolved["GrossApproval"] * 100
    g = resolved.groupby("LocationID")
    lender_table = pd.DataFrame({
        "n": g.size(),
        "chgoff_rate_pct": (g["target"].mean() * 100).round(1),
        "term_zero_pct": (g["is_term_zero"].mean() * 100).round(2),
        "n_term_zero": g["is_term_zero"].sum(),
        "avg_GrossApproval": g["GrossApproval"].mean().round(0),
        "avg_guarantee_pct": g["guarantee_pct"].mean().round(1),
    }).sort_values("n", ascending=False)
    top20 = lender_table.head(20)
    print("=== top 20 lenders: chargeoff rate, term=0 rate, avg loan size, avg guarantee % ===")
    print(top20.to_string())

    fig, ax = plt.subplots(figsize=(9, 7))
    ax.barh(top20.index.astype(str), top20["term_zero_pct"], color="#C44E52")
    ax.set_xlabel("% of lender's loans with TermInMonths==0")
    ax.set_title("term=0 rate, top 20 lenders by volume")
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(f"{OUT}/31_termzero_by_lender.png", dpi=150)
    plt.close(fig)
    print(f"\nsaved {OUT}/31_termzero_by_lender.png")

    # lenders where term=0 actually concentrates (not just top-20 by volume)
    by_termzero_count = lender_table[lender_table["n"] >= 30].sort_values("n_term_zero", ascending=False).head(20)
    print("\n=== lenders with the most term=0 loans (min 30 total loans) ===")
    print(by_termzero_count.to_string())

    # ------------------------------------------------------------------
    # 2. term=0 rate by geography tier
    # ------------------------------------------------------------------
    geo = build_geo_tiers(resolved, min_events=30, n_tiers=4)
    resolved["zip3"] = resolved["BorrZip"].str[:3]
    resolved = resolved.merge(geo[["risk_tier"]], left_on="zip3", right_index=True, how="left")

    geo_termzero = resolved.groupby("risk_tier", observed=True)["is_term_zero"].agg(["mean", "sum", "size"])
    geo_termzero.columns = ["term_zero_pct", "n_term_zero", "n_total"]
    geo_termzero["term_zero_pct"] = (geo_termzero["term_zero_pct"] * 100).round(2)
    print("\n=== term=0 rate by geography risk tier ===")
    print(geo_termzero)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(geo_termzero.index.astype(str), geo_termzero["term_zero_pct"], color="#4C72B0")
    ax.set_xlabel("geography risk tier")
    ax.set_ylabel("% of loans with TermInMonths==0")
    ax.set_title("term=0 rate by geography risk tier")
    fig.tight_layout()
    fig.savefig(f"{OUT}/32_termzero_by_geotier.png", dpi=150)
    plt.close(fig)
    print(f"saved {OUT}/32_termzero_by_geotier.png")

    # ------------------------------------------------------------------
    # 3. term=0 chargeoff rate x CollateralInd, x RevolverStatus
    # ------------------------------------------------------------------
    collat = resolved.groupby(["is_term_zero", "CollateralInd"])["target"].agg(["mean", "size"])
    collat.columns = ["chgoff_rate", "n"]
    collat["chgoff_rate"] = (collat["chgoff_rate"] * 100).round(1)
    print("\n=== chargeoff rate: term=0 x CollateralInd ===")
    print(collat)

    revolv = resolved.groupby(["is_term_zero", "RevolverStatus"])["target"].agg(["mean", "size"])
    revolv.columns = ["chgoff_rate", "n"]
    revolv["chgoff_rate"] = (revolv["chgoff_rate"] * 100).round(1)
    print("\n=== chargeoff rate: term=0 x RevolverStatus ===")
    print(revolv)

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    collat.unstack("CollateralInd")["chgoff_rate"].plot(kind="bar", ax=axes[0], color=["#DD8452", "#4C72B0"])
    axes[0].set_title("chargeoff rate: term=0 x CollateralInd")
    axes[0].set_xticklabels(["term!=0", "term=0"], rotation=0)
    revolv.unstack("RevolverStatus")["chgoff_rate"].plot(kind="bar", ax=axes[1], color=["#DD8452", "#4C72B0"])
    axes[1].set_title("chargeoff rate: term=0 x RevolverStatus")
    axes[1].set_xticklabels(["term!=0", "term=0"], rotation=0)
    fig.tight_layout()
    fig.savefig(f"{OUT}/33_termzero_x_collateral_revolver.png", dpi=150)
    plt.close(fig)
    print(f"\nsaved {OUT}/33_termzero_x_collateral_revolver.png")


if __name__ == "__main__":
    main()
